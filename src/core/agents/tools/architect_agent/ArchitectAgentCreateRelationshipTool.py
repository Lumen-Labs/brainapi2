"""
File: /ArchitectAgentCreateRelationshipTool.py
Created Date: Wednesday January 14th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Wednesday January 14th 2026 10:22:53 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from dataclasses import dataclass
import json
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
                            "description": "The name of the relationship",
                        },
                        "object": {
                            "type": "string",
                            "description": "The uuid of the object",
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the relationship",
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

    def _run(self, *args, **kwargs) -> str:
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
                    "[DEBUG (architect_agent_create_relationship)]: Subject or object is None: subject={subject}, object={object}",
                )
                return f"Subject or object is None: subject={subject}, object={object}"

            if isinstance(subject, str):
                subject = subject.strip()
            if isinstance(object, str):
                object = object.strip()

            subj_entity = self.entities.get(subject)
            if not subj_entity:
                for ent in self.architect_agent.used_entities_set:
                    if ent.uuid == subject:
                        subj_entity = ent
                        break

            obj_entity = self.entities.get(object)
            if not obj_entity:
                for ent in self.architect_agent.used_entities_set:
                    if ent.uuid == object:
                        obj_entity = ent
                        break

            if subj_entity is None:
                print(
                    "[DEBUG (architect_agent_create_relationship)]: Subject not found in entities: ",
                    subject,
                    ". Available entities: ",
                    list(self.entities.keys()),
                )
                return f"Subject not found in entities: {subject}. Available entities: {list(self.entities.keys())}"

            if obj_entity is None:
                print(
                    "[DEBUG (architect_agent_create_relationship)]: Object not found in entities: ",
                    object,
                    ". Available entities: ",
                    list(self.entities.keys()),
                )
                return f"Object not found in entities: {object}. Available entities: {list(self.entities.keys())}"

            subj = ArchitectAgentEntity(
                uuid=subj_entity.uuid,
                name=subj_entity.name,
                type=subj_entity.type,
                description=subj_entity.description,
                **(
                    {"happened_at": subj_entity.properties.get("happened_at")}
                    if subj_entity.properties.get("happened_at")
                    else {}
                ),
                properties=subj_entity.properties,
                polarity=subj_entity.polarity if subj_entity.polarity else "neutral",
            )
            obj = ArchitectAgentEntity(
                uuid=obj_entity.uuid,
                name=obj_entity.name,
                type=obj_entity.type,
                description=obj_entity.description,
                **(
                    {"happened_at": obj_entity.properties.get("happened_at")}
                    if obj_entity.properties.get("happened_at")
                    else {}
                ),
                properties=obj_entity.properties,
                polarity=obj_entity.polarity if obj_entity.polarity else "neutral",
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

        janitor_agent = JanitorAgent(
            llm_small_adapter,
            kg=graph_adapter,
            vector_store=vector_store_adapter,
            embeddings=embeddings_adapter,
            database_desc=_neo4j_client.graphdb_description,
        )

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
            self.architect_agent.input_tokens,
            self.architect_agent.output_tokens,
            self.architect_agent.cached_tokens,
            self.architect_agent.reasoning_tokens,
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

        fixed_rels_sets = []
        fixed_relationships = getattr(janitor_response, "fixed_relationships", []) or []

        if fixed_relationships:
            fixed_rels_sets = [
                set((fr.tip.uuid, fr.tail.uuid, fr.name)) for fr in fixed_relationships
            ]
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

        fixed_rels_vs = {}
        input_rels_vs = {}

        for rel in input_rels:
            have_similar_relation = False
            if set((rel.tip.uuid, rel.tail.uuid, rel.name)) in fixed_rels_sets:
                have_similar_relation = True
            else:
                rels_with_same_subject_and_object = [
                    [r, rel]
                    for r in fixed_relationships
                    if r.tip.uuid == rel.tip.uuid
                    and r.tail.uuid == rel.tail.uuid
                    or r.tip.uuid == rel.tail.uuid
                    and r.tail.uuid == rel.tip.uuid
                ]

                if rels_with_same_subject_and_object:
                    for fr, _rel in rels_with_same_subject_and_object:
                        fixed_rels_vs[f"{fr.name}-{fixed_relationships.index(fr)}"] = (
                            fixed_rels_vs.get(
                                f"{fr.name}-{fixed_relationships.index(fr)}",
                                {
                                    "r": fr,
                                    "embeddings": embeddings_small_adapter.embed_text(
                                        fr.name
                                    ).embeddings,
                                },
                            )
                        )
                        input_rels_vs[f"{_rel.name}-{input_rels.index(_rel)}"] = (
                            input_rels_vs.get(
                                f"{_rel.name}-{input_rels.index(_rel)}",
                                {
                                    "r": _rel,
                                    "embeddings": embeddings_small_adapter.embed_text(
                                        _rel.name
                                    ).embeddings,
                                },
                            )
                        )

                    target_embedding = input_rels_vs[
                        f"{rel.name}-{input_rels.index(rel)}"
                    ]["embeddings"]
                    candidates = [
                        (cosine_similarity(cand["embeddings"], target_embedding), cand)
                        for cand in fixed_rels_vs.values()
                    ]
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
                        "most_similar_fixed_rel: ",
                        most_similar_fixed_rel,
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

            print(
                "[DEBUG (architect_agent_create_relationship)]: Sending relationships to ingestion task"
            )
            process_architect_relationships.delay(
                {
                    "relationships": relationships_data,
                    "brain_id": self.brain_id,
                }
            )

        self.architect_agent.relationships_set.extend(output_rels)

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
                "wrong_relationships": janitor_response.wrong_relationships,
                "newly_created_nodes": newly_created_nodes,
            }

        # for entity_uuid in used_entity_uuids:
        #     if entity_uuid in self.architect_agent.entities:
        #         del self.architect_agent.entities[entity_uuid]

        # return natural_lang

        return json.dumps(
            {"status": "success", "message": "relationships created successfully"}
        )
