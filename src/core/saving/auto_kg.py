"""
File: /auto_kg.py
Created Date: Sunday December 21st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Tuesday December 23rd 2025 9:24:20 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Optional

from src.config import config
from src.constants.kg import Node
from src.core.agents.scout_agent import ScoutAgent
from src.core.agents.architect_agent import ArchitectAgent
from src.core.saving.ingestion_manager import IngestionManager
from src.services.input.agents import (
    cache_adapter,
    embeddings_adapter,
    graph_adapter,
    llm_small_adapter,
    vector_store_adapter,
)
from src.utils.tokens import merge_token_details, token_detail_from_token_counts


def enrich_kg_from_input(
    input: str, targeting: Optional[Node] = None, brain_id: str = "default"
) -> str:
    """
    Enrich the knowledge graph from the input.
    """

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

    entities = scout_agent.run(input, targeting=targeting, brain_id=brain_id)
    token_details.append(
        token_detail_from_token_counts(
            scout_agent.input_tokens,
            scout_agent.output_tokens,
            scout_agent.cached_tokens,
            scout_agent.reasoning_tokens,
        )
    )

    architect_response = architect_agent.run_tooler(
        input,
        entities.entities,
        targeting=targeting,
        brain_id=brain_id,
        timeout=360 * len(input),
    )
    token_details.append(
        token_detail_from_token_counts(
            architect_agent.input_tokens,
            architect_agent.output_tokens,
            architect_agent.cached_tokens,
            architect_agent.reasoning_tokens,
        )
    )

    token_details = merge_token_details(
        [scout_agent.token_detail, architect_agent.token_detail]
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
    print("-----------------------------------")
