from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from src.core.agents.scout_agent import ScoutAgent


def scout_with_responses(*responses):
    scout = ScoutAgent(None, None, None, None, None)
    scout.agent = Mock()
    scout.agent.invoke.side_effect = responses
    scout._get_agent = Mock()
    return scout


def truncated_response():
    return {"messages": [AIMessage(content="", response_metadata={"stopReason": "max_tokens"})]}


def entity_response(name):
    return {"structured_response": {"entities": [{"type": "Project", "name": name}]}}


def test_scout_retries_truncated_bedrock_output_in_smaller_sections():
    first = "Aurora " * 30 + "Aurora."
    second = "Kepler " * 30 + "Kepler."
    scout = scout_with_responses(
        truncated_response(), entity_response("Aurora"), entity_response("Kepler")
    )

    response = scout.run(first + "\n\n" + second)

    assert [entity.name for entity in response.entities] == ["Aurora", "Kepler"]
    _, retry_first, retry_second = scout.agent.invoke.call_args_list
    assert first in retry_first.args[0]["messages"][0]["content"]
    assert second not in retry_first.args[0]["messages"][0]["content"]
    assert second in retry_second.args[0]["messages"][0]["content"]


def test_scout_fails_instead_of_reporting_no_entities_when_short_output_is_truncated():
    scout = scout_with_responses(truncated_response())

    with pytest.raises(ValueError, match="exceeded the model output limit"):
        scout.run("Aurora is a project.")
