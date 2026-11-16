from __future__ import annotations

import asyncio

from fastapi import HTTPException

from src.core.verification.facts import FactRecord
from src.core.verification.service import (
    FactNotFound,
    apply_verification_action,
    get_fact_record,
    list_fact_records,
    require_state,
    VerificationAction,
)
from src.services.api.constants.requests import (
    FactVerificationHistoryEntry,
    FactVerificationInfo,
    FactVerificationListResponse,
    FactVerificationRecord,
    VerifyFactRequest,
)
from src.services.kg_agent.main import graph_adapter


def _to_response(record: FactRecord) -> FactVerificationRecord:
    return FactVerificationRecord(
        kind=record.kind,
        fact_id=record.fact_id,
        name=record.name,
        description=record.description,
        verification=FactVerificationInfo(**record.verification.model_dump()),
        history=[
            FactVerificationHistoryEntry(**entry.model_dump())
            for entry in record.history
        ],
        subject_uuid=record.subject_uuid,
        subject_name=record.subject_name,
        subject_labels=record.subject_labels,
        object_uuid=record.object_uuid,
        object_name=record.object_name,
        object_labels=record.object_labels,
        labels=record.labels,
    )


async def mutate_fact_verification(
    fact_id: str,
    brain_id: str,
    action: VerificationAction,
    request: VerifyFactRequest,
) -> FactVerificationRecord:
    try:
        record = await asyncio.to_thread(
            apply_verification_action,
            graph_adapter,
            fact_id=fact_id,
            brain_id=brain_id,
            action=action,
            verified_by=request.verified_by,
            verification_source=request.verification_source,
            note=request.note,
        )
    except FactNotFound:
        raise HTTPException(status_code=404, detail="Fact not found") from None
    return _to_response(record)


async def get_fact_verification(
    fact_id: str, brain_id: str
) -> FactVerificationRecord:
    try:
        record = await asyncio.to_thread(get_fact_record, graph_adapter, fact_id, brain_id)
    except FactNotFound:
        raise HTTPException(status_code=404, detail="Fact not found") from None
    return _to_response(record)


async def list_fact_verifications(
    brain_id: str,
    *,
    state: str | None = None,
    include_rejected: bool = False,
    limit: int = 50,
    skip: int = 0,
) -> FactVerificationListResponse:
    normalized = require_state(state)
    records, total = await asyncio.to_thread(
        list_fact_records,
        graph_adapter,
        brain_id,
        state=normalized,
        include_rejected=include_rejected,
        limit=limit,
        skip=skip,
    )
    return FactVerificationListResponse(
        facts=[_to_response(record) for record in records],
        total=total,
    )
