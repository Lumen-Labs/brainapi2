"""
File: /graph_consolidation.py
Project: graph_consolidation
Created Date: Saturday January 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 24th 2026 11:13:43 am
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, RemoveMessage
from src.config import config
from src.constants.agents import (
    ArchitectAgentEntity,
    GraphConsolidatorOutput,
    TokenDetail,
)
from src.constants.kg import Node, Predicate
from src.constants.prompts.janitor_agent import JANITOR_AGENT_GRAPH_NORMALIZATOR_PROMPT
from src.core.agents.architect_agent import ArchitectAgentRelationship
from src.core.agents.janitor_agent import (
    JanitorAgent,
    HISTORY_MAX_MESSAGES,
    HISTORY_MAX_MESSAGES_DELETE,
)
from src.core.agents.kg_agent import KGAgent
from src.core.agents.tools.janitor_agent import (
    JanitorAgentExecuteGraphReadOperationTool,
)
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.services.observations.main import llm_adapter
from src.lib.neo4j.client import _neo4j_client
from src.utils.tokens import merge_token_details, token_detail_from_token_counts

RELATIONSHIP_BATCH_SIZE = 20


class ConsolidationResponse(BaseModel):
    """
    Response for the graph consolidation.
    """

    token_detail: TokenDetail


def consolidate_graph(
    new_relationships: List[ArchitectAgentRelationship],
    brain_id: str = "default",
) -> ConsolidationResponse:
    """
    Consolidates and normalizes a collection of new knowledge-graph relationships across the graph.
    
    Processes the provided relationships in batches to perform macroscopic fixes such as name normalization, connection normalization, and deduplication across multiple relationships and graph areas. Collects and merges token usage metrics produced during consolidation.
    
    Parameters:
        brain_id (str): Identifier of the target knowledge graph/brain to consolidate into (defaults to "default").
    
    Returns:
        ConsolidationResponse: Response containing merged token usage details for the consolidation run.
    """

    batches = [
        new_relationships[i : i + RELATIONSHIP_BATCH_SIZE]
        for i in range(0, len(new_relationships), RELATIONSHIP_BATCH_SIZE)
    ]

    print(
        "[DEBUG (consolidate_graph)]: Total batches: ",
        len(batches),
    )

    token_details = []
    token_detail = None

    for batch in batches:

        janitor_agent = JanitorAgent(
            llm_adapter=llm_adapter,
            kg=graph_adapter,
            vector_store=vector_store_adapter,
            embeddings=embeddings_adapter,
            database_desc=_neo4j_client.graphdb_description,
        )

        tasks = janitor_agent.run_graph_consolidator(
            batch,
            brain_id=brain_id,
            timeout=300,
            max_retries=3,
        )

        print(
            "[DEBUG (consolidate_graph)]: Janitor analysis for batch: ",
            tasks,
        )

        token_details.append(
            token_detail_from_token_counts(
                janitor_agent.input_tokens,
                janitor_agent.output_tokens,
                janitor_agent.cached_tokens,
                janitor_agent.reasoning_tokens,
            )
        )

        # 2. ~~Something to review and approve/reject what janitor proposed~~
        # Another agent that reviews the porposed fixes and approves or rejects them .. what ?

        for task in tasks:
            kg_agent = KGAgent(
                llm_adapter=llm_small_adapter,
                cache_adapter=cache_adapter,
                kg=graph_adapter,
                vector_store=vector_store_adapter,
                embeddings=embeddings_adapter,
                database_desc=_neo4j_client.graphdb_description,
            )
            token_details.append(
                token_detail_from_token_counts(
                    kg_agent.input_tokens,
                    kg_agent.output_tokens,
                    kg_agent.cached_tokens,
                    kg_agent.reasoning_tokens,
                )
            )
            kg_agent.run_graph_consolidator_operator(task, brain_id=brain_id)

    token_detail = merge_token_details(token_details)

    print("================================================")
    print(
        f"> Input tokens:   {token_detail.input.total} -> ${token_detail.input.total * config.pricing.input_token_price}"
    )
    print(
        f"> Output tokens:  {token_detail.output.total} -> ${token_detail.output.total * config.pricing.output_token_price}"
    )
    print("================================================")

    return ConsolidationResponse(
        token_detail=token_detail,
    )