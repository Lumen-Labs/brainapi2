"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
import uuid
from typing import Optional

from src.config import config
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutAgent
from src.core.agents.architect_agent import ArchitectAgent
from src.core.layers.graph_consolidation.graph_consolidation import consolidate_graph
from src.core.saving.ingestion_manager import IngestionManager
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.utils.tokens import merge_token_details, token_detail_from_token_counts

import langsmith


def enrich_kg_from_input(
    input: str, targeting: Optional[Node] = None, brain_id: str = "default"
) -> str:
    """
    Orchestrates enrichment of the knowledge graph from a free-text input.

    Runs the scout and architect agents to extract entities and relationships from the provided input, optionally consolidates the resulting relationships into the graph, and reports token usage and cost metrics. Side effects include updating the knowledge graph and printing debug and cost summaries to standard output.

    Parameters:
        input (str): Free-text content to process and ingest into the knowledge graph.
        targeting (Optional[Node]): Optional target node guiding where or how the input should be applied within the graph.
        brain_id (str): Identifier for the processing context/brain to use for agent operations and consolidation.

    """

    ingestion_session_id = str(uuid.uuid4())
    project_name = os.getenv("LANGSMITH_PROJECT", "brainapi")
    with langsmith.tracing_context(
        project_name=project_name,
        enabled=True,
        tags=["enrich_kg", "scout", "architect"],
        metadata={
            "ingestion_session_id": ingestion_session_id,
            "brain_id": brain_id,
            "flow": "enrich_kg_from_input",
        },
    ):
        _enrich_kg_impl(input, targeting, brain_id, ingestion_session_id)


def _enrich_kg_impl(input: str, targeting, brain_id: str, ingestion_session_id: str):
    ingestion_manager = IngestionManager(
        embeddings_adapter, vector_store_adapter, graph_adapter
    )

    scout_agent = ScoutAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
    )
    architect_agent = ArchitectAgent(
        llm_small_adapter,
        cache_adapter,
        kg=graph_adapter,
        vector_store=vector_store_adapter,
        embeddings=embeddings_adapter,
        ingestion_manager=ingestion_manager,
    )

    token_details = []
    print(f"[DEBUG (ingestion_session_id)]: {ingestion_session_id}")

    entities = scout_agent.run(
        input,
        targeting=targeting,
        brain_id=brain_id,
        ingestion_session_id=ingestion_session_id,
    )
    token_details.append(
        token_detail_from_token_counts(
            scout_agent.input_tokens,
            scout_agent.output_tokens,
            scout_agent.cached_tokens,
            scout_agent.reasoning_tokens,
            "scout_agent",
        )
    )

    print("[DEBUG (initial_scout_entities)]: ", entities)

    architect_agent.run_tooler(
        input,
        entities.entities,
        targeting=targeting,
        brain_id=brain_id,
        timeout=20000,
        ingestion_session_id=ingestion_session_id,
    )
    token_details.append(
        token_detail_from_token_counts(
            architect_agent.input_tokens,
            architect_agent.output_tokens,
            architect_agent.cached_tokens,
            architect_agent.reasoning_tokens,
            "architect_agent",
        )
    )

    if config.run_graph_consolidator and architect_agent.session_id:
        from src.lib.redis.client import _redis_client
        import time

        print(
            f"[DEBUG (enrich_kg_from_input)]: Waiting for async tasks to complete for session {architect_agent.session_id}"
        )

        max_wait_time = 300
        check_interval = 2
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            pending_count = _redis_client.client.get(
                f"{brain_id}:session:{architect_agent.session_id}:pending_tasks"
            )

            if pending_count is None or int(pending_count) == 0:
                print(
                    f"[DEBUG (enrich_kg_from_input)]: All async tasks completed after {elapsed_time}s"
                )
                break

            print(
                f"[DEBUG (enrich_kg_from_input)]: {pending_count} tasks still pending, waiting..."
            )
            time.sleep(check_interval)
            elapsed_time += check_interval
        else:
            print(
                f"[DEBUG (enrich_kg_from_input)]: Timeout waiting for tasks to complete after {max_wait_time}s"
            )

        print(
            "[DEBUG (consolidate_graph)]: Consolidating graph with ",
            len(architect_agent.relationships_set),
            " relationships",
        )

        from src.workers.tasks.ingestion import consolidate_graph_async

        task_result = consolidate_graph_async.delay(
            session_id=architect_agent.session_id,
            brain_id=brain_id,
            ingestion_session_id=ingestion_session_id,
        )
        print(
            f"[DEBUG (enrich_kg_from_input)]: Consolidation task {task_result.id} queued"
        )

    token_details = merge_token_details(
        [
            scout_agent.token_detail,
            architect_agent.token_detail,
        ]
    )

    print("-----------------------------------")
    print(
        f"> Input tokens:   {token_details.input.total} -> ${token_details.input.total * config.pricing.input_token_price}"
    )
    print(
        f"> Output tokens:  {token_details.output.total} -> ${token_details.output.total * config.pricing.output_token_price}"
    )
    print(
        f"> Cost -> ${token_details.input.total * config.pricing.input_token_price + token_details.output.total * config.pricing.output_token_price} for {len(input)} characters"
    )
    print(
        f"> Cost/Character -> ${((token_details.input.total * config.pricing.input_token_price + token_details.output.total * config.pricing.output_token_price) / len(input)):.12f}"
    )
    print("> Token details: ", token_details.model_dump_json(indent=2))
    print("-----------------------------------")

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except ImportError:
        pass
