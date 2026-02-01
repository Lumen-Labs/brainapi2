"""
File: /ArchitectAgentCreateRelationshipTool.py
Created Date: Wednesday January 14th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:44:06 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import uuid
from langchain.tools import BaseTool

from src.adapters.graph import GraphAdapter
from src.constants.agents import (
    ArchitectAgentRelationship,
    ArchitectAgentEntity,
    AtomicJanitorAgentInputOutput,
    _ArchitectAgentRelationship,
)
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutEntity
from src.services.input.agents import (
    embeddings_adapter,
    embeddings_small_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.lib.neo4j.client import _neo4j_client
from src.utils.cleanup import strip_properties
from src.utils.nlp.names import levenshtein_similarity
from src.utils.similarity.vectors import cosine_similarity
from src.utils.tokens import merge_token_details, token_detail_from_token_counts


class ArchitectAgentCreateRelationshipTool(BaseTool):
    name: str = "architect_agent_create_relationship"
    args_schema: dict = {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "description": "A list of relationships to create",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The uuid of the subject",
                        },
                        "predicate": {
                            "type": "string",
                            "description": "The generic name of the relationship (eg: 'TARGET_PRODUCT_OBJECT_CROISSANTS' = wrong, 'TARGETED' = correct)",
                        },
                        "object": {
                            "type": "string",
                            "description": "The uuid of the object",
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the relationship",
                        },
                        "amount": {
                            "type": "number",
                            "description": "The amount of the relationship (eg: 12 for 'John knew 12 new friends in New York City')",
                        },
                    },
                    "required": ["subject", "predicate", "object", "description"],
                },
            },
        },
        "required": ["relationships"],
    }
    architect_agent: object
    kg: GraphAdapter
    entities: Dict[str, ScoutEntity]
    brain_id: str = "default"
    targeting: Optional[Node] = None
    text: str

    def __init__(
        self,
        architect_agent: object,
        text: str,
        entities: Optional[Dict[str, ScoutEntity]],
        kg: GraphAdapter,
        brain_id: str = "default",
        targeting: Optional[Node] = None,
    ):
        """
        Initialize the tool for creating a set of relationships between entities for a single phrase.

        Parameters:
            architect_agent (object): The owning architect agent instance that this tool will update (holds state like entities, token details, and relationships).
            text (str): The source text or phrase that the relationships represent.
            entities (Optional[Dict[str, ScoutEntity]]): Mapping of entity UUIDs to ScoutEntity objects available to resolve relationship subjects/objects; empty dict if None.
            kg (GraphAdapter): Graph adapter used for resolving and persisting graph-related operations.
            brain_id (str): Identifier of the target brain/namespace for created relationships.
            targeting (Optional[Node]): Optional target node to associate with created relationships.
        """
        description: str = (
            "Tool for creating a set of relationships between entities. "
            "Use this tool to create a set of relationships between entities that together compose a single phrase. "
            "Input must be a list of relationships with a subject entity uuid, an object entity uuid, a predicate name between them and a description of the relationship."
            "The list of relationships must only be for a single phrase."
            "Returns a summary of the created relationships to review them before creating them."
        )
        entities_dict = entities if entities is not None else {}
        super().__init__(
            architect_agent=architect_agent,
            description=description,
            kg=kg,
            entities=entities_dict,
            brain_id=brain_id,
            targeting=targeting,
            text=text,
        )

    @dataclass
    class _ArchitectAgentInputRelationship:
        subject: str
        predicate: str
        object: str
        description: str

    def _most_similar_entities(self, query: str, limit: int = 5) -> list[str]:
        def _e_attr(e, k):
            return e.get(k) if isinstance(e, dict) else getattr(e, k, None)

        all_entities = list(self.entities.values()) + list(
            self.architect_agent.used_entities_dict.values()
        )
        scored = []
        for entity in all_entities:
            uuid_val = _e_attr(entity, "uuid")
            name_val = _e_attr(entity, "name")
            if not uuid_val and not name_val:
                continue
            uuid_sim = levenshtein_similarity(query, str(uuid_val)) if uuid_val else 0.0
            name_sim = levenshtein_similarity(query, str(name_val)) if name_val else 0.0
            best = max(uuid_sim, name_sim)
            scored.append((entity, best))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        uuids = [_e_attr(s[0], "uuid") for s in scored[:limit] if _e_attr(s[0], "uuid")]
        nodes_by_uuid = {}
        if uuids:
            try:
                nodes = self.kg.get_by_uuids(uuids, brain_id=self.brain_id)
                nodes_by_uuid = {n.uuid: n for n in nodes}
            except Exception:
                pass
        for entity, _ in scored[:limit]:
            e_uuid = _e_attr(entity, "uuid")
            e_name = _e_attr(entity, "name")
            part = f"{e_name} ({e_uuid})"
            node = nodes_by_uuid.get(e_uuid) if e_uuid else None
            if node:
                labels = ", ".join(node.labels) if node.labels else "?"
                part += f" -> node: {node.name} [{labels}]"
            result.append(part)
        return result

    def _run(self, *args, **kwargs) -> str:
        """
        Validate and create architect relationships from the provided input, normalize them via the JanitorAgent, enqueue ingestion, and return the operation result.

        Parameters:
                relationships (list[dict], optional): A list of relationship objects provided either as the first positional argument (or as args[0]["relationships"]) or via the "relationships" keyword. Each relationship dict should include:
                        - subject: UUID or name of the subject entity (required)
                        - predicate: relationship name/label
                        - object: UUID or name of the object entity (required)
                        - description: textual description of the relationship
                        - amount: optional numeric value associated with the relationship
                        - properties: optional dict of additional properties
                The method resolves subject and object against the tool's entity map and the architect_agent's used_entities_set.

        Returns:
                A success JSON string on successful creation: '{"status": "success"}'.
                An error string if the required "relationships" parameter is missing or if a subject/object cannot be resolved (e.g., 'Subject not found in entities: ...').
                A dict with status "ERROR" when the JanitorAgent reports wrong relationships; the dict includes keys "wrong_relationships" and "newly_created_nodes".
        """
        rel_key = str(uuid.uuid4())

        print(
            "[DEBUG (architect_agent_create_relationship)]: Called ArchitectAgentCreateRelationshipTool"
        )

        input_rels: List[_ArchitectAgentRelationship] = []
        output_rels: List[ArchitectAgentRelationship] = []

        if args and isinstance(args[0], dict) and "relationships" in args[0]:
            relationships = args[0]["relationships"]
        elif kwargs and "relationships" in kwargs:
            relationships = kwargs["relationships"]
        elif args and isinstance(args[0], list):
            relationships = args[0]
        else:
            return "Error: relationships parameter is required"

        print(
            "[DEBUG (architect_agent_create_relationship)]: Relationships: ",
            relationships,
        )

        for rel in relationships:
            subject = rel.get("subject")
            predicate = rel.get("predicate")
            object = rel.get("object")
            description = rel.get("description")
            amount = rel.get("amount")
            properties = rel.get("properties")

            if subject is None or object is None:
                print(
                    f"[DEBUG (architect_agent_create_relationship)]: Subject or object is None: subject={subject}, object={object}",
                )
                return f"Subject or object is None: subject={subject}, object={object}"

            if isinstance(subject, str):
                subject = subject.strip()
            if isinstance(object, str):
                object = object.strip()

            def _resolve_entity(ref: str):
                entity = self.entities.get(ref)
                if entity:
                    return entity
                entity = self.architect_agent.used_entities_dict.get(ref)
                if entity:
                    return entity
                all_ents = list(self.entities.values()) + list(
                    self.architect_agent.used_entities_dict.values()
                )
                for e in all_ents:
                    e_uuid = (
                        e.get("uuid")
                        if isinstance(e, dict)
                        else getattr(e, "uuid", None)
                    )
                    e_name = (
                        e.get("name")
                        if isinstance(e, dict)
                        else getattr(e, "name", None)
                    )
                    if ref in (e_uuid, e_name):
                        return e
                return None

            subj_entity = _resolve_entity(subject)
            obj_entity = _resolve_entity(object)

            if subj_entity is None:
                similar = self._most_similar_entities(subject, limit=3)
                msg = (
                    f"Subject not found in entities: {subject}. Most similar: {similar}"
                )

                print(f"[DEBUG (architect_agent_create_relationship)]: {msg}")
                return msg

            if obj_entity is None:
                similar = self._most_similar_entities(object, limit=3)
                msg = f"Object not found in entities: {object}. Most similar: {similar}"

                print(f"[DEBUG (architect_agent_create_relationship)]: {msg}")
                return msg

            def _attr(e, k, default=None):
                return (
                    e.get(k, default) if isinstance(e, dict) else getattr(e, k, default)
                )

            def _props(e):
                return _attr(e, "properties") or {}

            subj = ArchitectAgentEntity(
                uuid=_attr(subj_entity, "uuid"),
                name=_attr(subj_entity, "name"),
                type=_attr(subj_entity, "type"),
                description=_attr(subj_entity, "description"),
                **(
                    {"happened_at": _props(subj_entity).get("happened_at")}
                    if _props(subj_entity).get("happened_at")
                    else {}
                ),
                properties=_props(subj_entity),
                polarity=_attr(subj_entity, "polarity") or "neutral",
            )
            obj = ArchitectAgentEntity(
                uuid=_attr(obj_entity, "uuid"),
                name=_attr(obj_entity, "name"),
                type=_attr(obj_entity, "type"),
                description=_attr(obj_entity, "description"),
                **(
                    {"happened_at": _props(obj_entity).get("happened_at")}
                    if _props(obj_entity).get("happened_at")
                    else {}
                ),
                properties=_props(obj_entity),
                polarity=_attr(obj_entity, "polarity") or "neutral",
            )

            input_rels.append(
                _ArchitectAgentRelationship(
                    tip=obj,
                    tail=subj,
                    name=predicate,
                    description=description,
                    properties=properties,
                    **({"amount": amount} if amount else {}),
                )
            )

        # natural_lang = ", ".join(
        #     [
        #         f"""({rel.tail.name})-[:{rel.name} {{description: "{rel.description}"}}]->({rel.tip.name})"""
        #         for rel in output_rels
        #     ]
        # )

        from src.core.agents.janitor_agent import JanitorAgent

        janitor_agent = getattr(self.architect_agent, "janitor_agent", None)
        janitor_agent_brain_id = getattr(
            self.architect_agent, "_janitor_agent_brain_id", None
        )
        if janitor_agent is None or janitor_agent_brain_id != self.brain_id:
            janitor_agent = JanitorAgent(
                llm_small_adapter,
                kg=graph_adapter,
                vector_store=vector_store_adapter,
                embeddings=embeddings_adapter,
                database_desc=_neo4j_client.graphdb_description,
            )
            self.architect_agent.janitor_agent = janitor_agent
            self.architect_agent._janitor_agent_brain_id = self.brain_id
        start_input_tokens = janitor_agent.input_tokens
        start_output_tokens = janitor_agent.output_tokens
        start_cached_tokens = janitor_agent.cached_tokens
        start_reasoning_tokens = janitor_agent.reasoning_tokens

        janitor_response: AtomicJanitorAgentInputOutput = (
            janitor_agent.run_atomic_janitor(
                input_relationships=input_rels,
                text=self.text,
                targeting=self.targeting,
                brain_id=self.brain_id,
                timeout=300,
                max_retries=3,
            )
        )

        print(
            "[DEBUG (architect_agent_create_relationship)]: Janitor response: ",
            janitor_response,
        )

        janitor_token_detail = token_detail_from_token_counts(
            janitor_agent.input_tokens - start_input_tokens,
            janitor_agent.output_tokens - start_output_tokens,
            janitor_agent.cached_tokens - start_cached_tokens,
            janitor_agent.reasoning_tokens - start_reasoning_tokens,
            "janitor_agent",
        )
        self.architect_agent.token_detail = merge_token_details(
            [self.architect_agent.token_detail, janitor_token_detail]
        )

        required_new_nodes = getattr(janitor_response, "required_new_nodes", [])
        newly_created_nodes = []
        if required_new_nodes:
            for node in required_new_nodes:
                scout_entity = ScoutEntity(
                    uuid=node.uuid,
                    name=node.name,
                    type=node.type,
                    description=node.description,
                    properties=node.properties,
                )
                self.entities[scout_entity.uuid] = scout_entity
                self.architect_agent.entities.update({scout_entity.uuid: scout_entity})
                newly_created_nodes.append(scout_entity.model_dump(mode="json"))

        fixed_rels_sets = set()
        fixed_relationships = getattr(janitor_response, "fixed_relationships", []) or []

        if fixed_relationships:
            fixed_rels_sets = set(
                frozenset((fr.tip.uuid, fr.tail.uuid, fr.name))
                for fr in fixed_relationships
            )
            output_rels.extend(
                [
                    ArchitectAgentRelationship(
                        flow_key=rel_key,
                        tip=rel.tip,
                        name=rel.name,
                        description=rel.description,
                        tail=rel.tail,
                        properties=getattr(rel, "properties", {}),
                        **(
                            {"amount": getattr(rel, "amount", None)}
                            if getattr(rel, "amount", None)
                            else {}
                        ),
                    )
                    for rel in fixed_relationships
                ]
            )

        # Batch embed texts for similarity comparison
        texts_to_embed = set()
        for rel in input_rels:
            texts_to_embed.add(rel.description if rel.description else rel.name)
        for rel in fixed_relationships:
            texts_to_embed.add(rel.description if rel.description else rel.name)

        text_to_embedding = {}
        if texts_to_embed:
            texts_list = list(texts_to_embed)
            vectors = embeddings_small_adapter.embed_texts(texts_list)
            for text, vector in zip(texts_list, vectors):
                text_to_embedding[text] = vector.embeddings

        for rel in input_rels:
            have_similar_relation = False
            if frozenset((rel.tip.uuid, rel.tail.uuid, rel.name)) in fixed_rels_sets:
                have_similar_relation = True
            else:
                rels_with_same_subject_and_object = [
                    fr
                    for fr in fixed_relationships
                    if (fr.tip.uuid == rel.tip.uuid and fr.tail.uuid == rel.tail.uuid)
                    or (fr.tip.uuid == rel.tail.uuid and fr.tail.uuid == rel.tip.uuid)
                ]

                if rels_with_same_subject_and_object:
                    input_rel_text = rel.description if rel.description else rel.name
                    input_embedding = text_to_embedding.get(input_rel_text)

                    candidates = []
                    for fr in rels_with_same_subject_and_object:
                        fixed_rel_text = fr.description if fr.description else fr.name
                        fixed_embedding = text_to_embedding.get(fixed_rel_text)

                        if input_embedding and fixed_embedding:
                            candidates.append(
                                (
                                    cosine_similarity(fixed_embedding, input_embedding),
                                    fr,
                                )
                            )

                    if candidates:
                        similarity_score, most_similar_fixed_rel = max(
                            candidates, key=lambda x: x[0]
                        )

                        if (
                            similarity_score > 0.90
                        ):  # TODO: [similarity_threshold] check if this is the suitable threshold
                            have_similar_relation = True
                        print(
                            "[DEBUG (architect_agent_create_relationship)]: Have similar relation: ",
                            have_similar_relation,
                            "similarity_score: ",
                            similarity_score,
                            "rels: ",
                            most_similar_fixed_rel,
                            rel,
                        )

            if not have_similar_relation:
                output_rels.append(
                    ArchitectAgentRelationship(
                        flow_key=rel_key,
                        tip=rel.tip,
                        name=rel.name,
                        description=rel.description,
                        tail=rel.tail,
                        properties=getattr(rel, "properties", {}),
                        **(
                            {"amount": getattr(rel, "amount", None)}
                            if getattr(rel, "amount", None)
                            else {}
                        ),
                    )
                )

        relationships_data = [
            rel.model_dump(mode="json")
            for rel in output_rels
            if isinstance(rel, ArchitectAgentRelationship)
        ]

        if relationships_data:
            from src.workers.tasks.ingestion import process_architect_relationships
            from src.lib.redis.client import _redis_client

            print(
                "[DEBUG (architect_agent_create_relationship)]: Sending relationships to ingestion task"
            )

            session_id = getattr(self.architect_agent, "session_id", None)
            if session_id:
                _redis_client.client.incr(
                    f"{self.brain_id}:session:{session_id}:pending_tasks"
                )

            task_result = process_architect_relationships.delay(
                {
                    "relationships": relationships_data,
                    "brain_id": self.brain_id,
                    "session_id": session_id,
                }
            )
            print(
                f"[DEBUG (architect_agent_create_relationship)]: Task {task_result.id} queued for session {session_id}"
            )

        self.architect_agent.relationships_set.extend(output_rels)

        session_id = getattr(self.architect_agent, "session_id", None)
        if session_id and output_rels:
            from src.lib.redis.client import _redis_client
            import json

            relationships_data = [
                rel.model_dump(mode="json")
                for rel in self.architect_agent.relationships_set
            ]
            _redis_client.set(
                f"session:{session_id}:relationships",
                json.dumps(relationships_data),
                brain_id=self.brain_id,
                expires_in=3600,
            )

        wrong_relationships = []
        if getattr(janitor_response, "wrong_relationships", []):
            wrong_relationships = getattr(janitor_response, "wrong_relationships", [])

        if len(wrong_relationships) > 0:
            print(
                "[DEBUG (architect_agent_create_relationship)]: Wrong relationships: ",
                getattr(janitor_response, "wrong_relationships", []),
            )
            return {
                "status": "ERROR",
                "wrong_relationships": strip_properties(
                    [
                        rel.model_dump(mode="json")
                        for rel in janitor_response.wrong_relationships
                    ]
                ),
                "newly_created_nodes": strip_properties(
                    [node for node in newly_created_nodes]
                ),
            }

        # for entity_uuid in used_entity_uuids:
        #     if entity_uuid in self.architect_agent.entities:
        #         del self.architect_agent.entities[entity_uuid]

        # return natural_lang

        return json.dumps({"status": "success"})
