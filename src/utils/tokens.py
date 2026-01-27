"""
File: /tokens.py
Project: utils
Created Date: Saturday January 17th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 17th 2026 11:22:10 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from src.constants.agents import TokenDetail, TokenInputDetail, TokenOutputDetail


def merge_token_details(token_details: list[TokenDetail]) -> TokenDetail:
    """
    Aggregate a list of TokenDetail objects into a single merged TokenDetail.
    
    Parameters:
        token_details (list[TokenDetail]): List of TokenDetail objects to merge. None entries are ignored; if the list is empty or contains only None, a TokenDetail with all zero/default fields is returned.
    
    Returns:
        TokenDetail: A merged TokenDetail where input and output counts are summed across entries, cache_percentage is (total cached / total input * 100) or 0.0 if total input is zero, reasoning_percentage is (total reasoning / total output * 100) or 0.0 if total output is zero, and grand_total and effective_total are the sums of their respective values.
    """
    if not token_details:
        return TokenDetail(
            input=TokenInputDetail(total=0, uncached=0, cached=0, cache_percentage=0.0),
            output=TokenOutputDetail(
                total=0, regular=0, reasoning=0, reasoning_percentage=0.0
            ),
            grand_total=0,
            effective_total=0,
        )

    token_details = [detail for detail in token_details if detail is not None]
    if not token_details:
        return TokenDetail(
            input=TokenInputDetail(total=0, uncached=0, cached=0, cache_percentage=0.0),
            output=TokenOutputDetail(
                total=0, regular=0, reasoning=0, reasoning_percentage=0.0
            ),
            grand_total=0,
            effective_total=0,
        )

    total_input = sum(detail.input.total for detail in token_details)
    total_uncached = sum(detail.input.uncached for detail in token_details)
    total_cached = sum(detail.input.cached for detail in token_details)
    cache_percentage = (total_cached / total_input * 100) if total_input > 0 else 0.0

    total_output = sum(detail.output.total for detail in token_details)
    total_regular = sum(detail.output.regular for detail in token_details)
    total_reasoning = sum(detail.output.reasoning for detail in token_details)
    reasoning_percentage = (
        (total_reasoning / total_output * 100) if total_output > 0 else 0.0
    )

    grand_total = sum(detail.grand_total for detail in token_details)
    effective_total = sum(detail.effective_total for detail in token_details)

    return TokenDetail(
        input=TokenInputDetail(
            total=total_input,
            uncached=total_uncached,
            cached=total_cached,
            cache_percentage=cache_percentage,
        ),
        output=TokenOutputDetail(
            total=total_output,
            regular=total_regular,
            reasoning=total_reasoning,
            reasoning_percentage=reasoning_percentage,
        ),
        grand_total=grand_total,
        effective_total=effective_total,
    )


def token_detail_from_token_counts(
    input_tokens: int, output_tokens: int, cached_tokens: int, reasoning_tokens: int
) -> TokenDetail:
    """
    Constructs a TokenDetail summarizing input and output token counts with cached and reasoning breakdowns.
    
    Parameters:
        input_tokens (int): Total number of input tokens.
        output_tokens (int): Total number of output tokens.
        cached_tokens (int): Number of input tokens served from cache.
        reasoning_tokens (int): Number of output tokens classified as reasoning.
    
    Returns:
        TokenDetail: Aggregated token details containing:
            - input: TokenInputDetail with total, uncached, cached, and cache_percentage.
            - output: TokenOutputDetail with total, regular, reasoning, and reasoning_percentage.
            - grand_total: Sum of input and output tokens.
            - effective_total: Sum of output tokens plus input tokens not served from cache.
    """
    return TokenDetail(
        input=TokenInputDetail(
            total=input_tokens,
            uncached=input_tokens - cached_tokens,
            cached=cached_tokens,
            cache_percentage=(
                (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0.0
            ),
        ),
        output=TokenOutputDetail(
            total=output_tokens,
            regular=output_tokens - reasoning_tokens,
            reasoning=reasoning_tokens,
            reasoning_percentage=(
                (reasoning_tokens / output_tokens * 100) if output_tokens > 0 else 0.0
            ),
        ),
        grand_total=input_tokens + output_tokens,
        effective_total=input_tokens - cached_tokens + output_tokens,
    )