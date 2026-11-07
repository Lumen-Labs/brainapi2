import pytest

from src.core.saving.ingest_limits import (
    IngestModelLimitError,
    check_input,
    check_output,
    invocation_limits,
    is_size_error,
    limit_output,
)
from src.utils.text_chunking import source_context, source_cut


def test_source_boundaries_keep_rows_and_preserve_unicode():
    text = "# Heading\n\n| A | B |\n| 1 | 2 |\n" + "測試" * 20
    cut = source_cut(text, 32)
    assert text[:cut].endswith("| 1 | 2 |\n")
    assert "Heading" in source_context(text, cut)
    spans = [text[:cut]]
    while cut < len(text):
        end = cut + source_cut(text[cut:], 5)
        spans.append(text[cut:end])
        cut = end
    assert "".join(spans) == text


def test_model_scope_applies_output_and_checks_full_request():
    from pydantic import BaseModel

    class Model(BaseModel):
        max_tokens: int = 24
        profile: dict = {"max_input_tokens": 256}

        def get_num_tokens_from_messages(self, messages):
            return sum(len(message["content"]) for message in messages)

    original = Model()
    token = invocation_limits.set((32,))
    try:
        routed = limit_output(original)
        assert routed.max_tokens == 32
        assert original.max_tokens == 24
        assert limit_output(object()).__class__ is object
        check_input(routed, [{"role": "system", "content": "instructions"}])
        with pytest.raises(IngestModelLimitError):
            check_input(routed, [{"role": "system", "content": "x" * 128}])
        with pytest.raises(IngestModelLimitError):
            check_output({"response_metadata": {"stopReason": "max_tokens"}})
        assert is_size_error(ValueError("maximum context length"))
        assert not is_size_error(ValueError("amount must be numeric"))
    finally:
        invocation_limits.reset(token)


def test_architect_reserves_more_output_for_the_same_source(monkeypatch):
    from types import SimpleNamespace

    from src.core.saving import pipeline_runtime
    from src.core.saving.ingest_limits import source_span_tokens

    model = SimpleNamespace(profile={"max_input_tokens": 4096, "max_output_tokens": 96})
    scope = invocation_limits.set((None,))
    span = source_span_tokens.set(32)
    try:
        messages = [{"role": "user", "content": "sample"}]
        monkeypatch.setattr(pipeline_runtime, "current_stage", lambda: "scout")
        check_input(model, messages)
        monkeypatch.setattr(pipeline_runtime, "current_stage", lambda: "architect")
        with pytest.raises(IngestModelLimitError, match="estimated extraction capacity"):
            check_input(model, messages)
    finally:
        source_span_tokens.reset(span)
        invocation_limits.reset(scope)


@pytest.mark.parametrize(("amount", "expected"), [("several", None), ("12.5", 12.5), (0, 0), ("NaN", None)])
def test_malformed_amount_does_not_invalidate_extracted_relationship(amount, expected):
    from src.constants.agents import _ArchitectAgentRelationship
    from src.core.saving.architect_batch import BatchRelationship

    alice = {"uuid": "a", "name": "Alice", "type": "PERSON"}
    bob = {"uuid": "b", "name": "Bob", "type": "PERSON"}
    payload = {"tail": alice, "tip": bob, "name": "RELATED", "amount": amount}
    assert BatchRelationship.model_validate(payload).amount == expected
    assert _ArchitectAgentRelationship.model_validate(payload).amount == expected


def test_dense_scout_payload_uses_routed_output_allowance(monkeypatch):
    from types import SimpleNamespace

    from src.core.saving import pipeline_runtime
    from src.core.saving.ingest_limits import scout_payload_size, source_span_tokens

    monkeypatch.setattr(pipeline_runtime, "current_stage", lambda: "architect")
    scope = invocation_limits.set((None,))
    span = source_span_tokens.set(50)
    payload = scout_payload_size.set((16, 120))
    try:
        messages = [{"role": "user", "content": "Short but entity-dense text."}]
        small = SimpleNamespace(profile={"max_input_tokens": 8192, "max_output_tokens": 1024})
        large = SimpleNamespace(profile={"max_input_tokens": 8192, "max_output_tokens": 4096})
        with pytest.raises(IngestModelLimitError, match="estimated extraction capacity"):
            check_input(small, messages)
        check_input(large, messages)
    finally:
        scout_payload_size.reset(payload)
        source_span_tokens.reset(span)
        invocation_limits.reset(scope)
