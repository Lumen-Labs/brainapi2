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
)
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutEntity
from src.services.input.agents import (
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.lib.neo4j.client import _neo4j_client
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

    def __init__(
        self,
        architect_agent: object,
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
        )

    @dataclass
    class _ArchitectAgentInputRelationship:
        subject: str
        predicate: str
        object: str
        description: str

    def _run(self, *args, **kwargs) -> str:
        rel_key = str(uuid.uuid4())

        input_rels = []
        output_rels: List[ArchitectAgentRelationship] = []

        if args and isinstance(args[0], dict) and "relationships" in args[0]:
            relationships = args[0]["relationships"]
        elif kwargs and "relationships" in kwargs:
            relationships = kwargs["relationships"]
        elif args and isinstance(args[0], list):
            relationships = args[0]
        else:
            return "Error: relationships parameter is required"

        for rel in relationships:
            subject = rel.get("subject")
            predicate = rel.get("predicate")
            object = rel.get("object")
            description = rel.get("description")

            if subject is None or object is None:
                return f"Subject or object is None: subject={subject}, object={object}"

            if isinstance(subject, str):
                subject = subject.strip()
            if isinstance(object, str):
                object = object.strip()

            if subject not in self.entities:
                return f"Subject not found in entities: {subject}. Available entities: {list(self.entities.keys())}"

            if object not in self.entities:
                return f"Object not found in entities: {object}. Available entities: {list(self.entities.keys())}"

            input_rels.append(
                self._ArchitectAgentInputRelationship(
                    subject, predicate, object, description
                )
            )

            subj_entity = self.entities.get(subject)
            obj_entity = self.entities.get(object)

            subj = ArchitectAgentEntity(
                uuid=subj_entity.uuid,
                name=subj_entity.name,
                type=subj_entity.type,
                flow_key=rel_key,
                **(
                    {"happened_at": subj_entity.properties.get("happened_at")}
                    if subj_entity.properties.get("happened_at")
                    else {}
                ),
            )
            obj = ArchitectAgentEntity(
                uuid=obj_entity.uuid,
                name=obj_entity.name,
                type=obj_entity.type,
                flow_key=rel_key,
                **(
                    {"happened_at": obj_entity.properties.get("happened_at")}
                    if obj_entity.properties.get("happened_at")
                    else {}
                ),
            )

            output_rels.append(
                ArchitectAgentRelationship(
                    flow_key=rel_key,
                    tip=obj,
                    tail=subj,
                    name=predicate,
                    description=description,
                )
            )

        natural_lang = ", ".join(
            [
                f"""({rel.tail.name})-[:{rel.name} {{description: "{rel.description}"}}]->({rel.tip.name})"""
                for rel in output_rels
            ]
        )

        from src.core.agents.janitor_agent import JanitorAgent

        janitor_agent = JanitorAgent(
            llm_small_adapter,
            kg=graph_adapter,
            vector_store=vector_store_adapter,
            embeddings=embeddings_adapter,
            database_desc=_neo4j_client.graphdb_description,
        )

        janitor_response = janitor_agent.run_atomic_janitor(
            input_relationships=output_rels,
            text=natural_lang,
            targeting=self.targeting,
            brain_id=self.brain_id,
            timeout=300,
            max_retries=3,
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

        if fixed_relationships := getattr(janitor_response, "fixed_relationships", []):
            output_rels.extend(
                [
                    ArchitectAgentRelationship(
                        flow_key=rel_key,
                        tip=rel.tip,
                        name=rel.name,
                        description=rel.description,
                        tail=rel.tail,
                    )
                    for rel in fixed_relationships
                ]
            )

        self.architect_agent.relationships_set.extend(output_rels)
        from pprint import pprint

        pprint(self.architect_agent.relationships_set)
        if len(getattr(janitor_response, "wrong_relationships", [])) > 0:
            return janitor_response

        # for entity_uuid in used_entity_uuids:
        #     if entity_uuid in self.architect_agent.entities:
        #         del self.architect_agent.entities[entity_uuid]

        # return natural_lang
        return json.dumps({"status": "success", "insert_count": len(output_rels)})
