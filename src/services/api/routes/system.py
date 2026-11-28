"""
File: /system.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday December 1st 2025 10:12:48 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter, Query, Request
from src.services.api.constants.requests import CreateBrainRequest
from src.services.api.controllers.system import (
    clear_brain as clear_brain_controller,
    create_new_brain as create_new_brain_controller,
    delete_brain as delete_brain_controller,
    get_brain_lifecycle_service,
    get_brains_list as get_brains_list_controller,
)
from src.constants.data import Brain
from src.services.api.constants.responses import BrainLifecycleResponse
from src.services.api.errors import error_response
from src.services.brain_lifecycle import (
    BrainCleanupError,
    BrainLifecycleBusyError,
    BrainNotFoundError,
    InvalidBrainIdError,
)

system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get(path="/brains-list", response_model=list[Brain])
async def get_brains_list(
    include_pats: bool = Query(
        False,
        description=(
            "When true, include plaintext brain PATs. Requires the system token. "
            "Default omits keys so listing brains does not leak credentials."
        ),
    ),
):
    """List registered brains. Restricted to the system PAT.

    Brain keys are omitted unless `include_pats=true`. Create-brain still
    returns the new PAT once. Self-host consoles and MaaS adapters that need
    keys must send the system token and this query flag.
    """
    return await get_brains_list_controller(include_pats=include_pats)


@system_router.post(path="/brains", response_model=Brain)
async def create_brain(request: CreateBrainRequest):
    """
    Create a new brain
    """
    return await create_new_brain_controller(request)


@system_router.post(
    path="/brains/{brain_id}/clear",
    response_model=BrainLifecycleResponse,
)
async def clear_brain(brain_id: str, request: Request):
    """Clear all data for a brain while preserving its identity and PAT."""
    try:
        return await clear_brain_controller(
            brain_id,
            get_brain_lifecycle_service(),
        )
    except InvalidBrainIdError as error:
        return error_response(
            request,
            status_code=400,
            detail=str(error),
            code="BRAIN_ID_INVALID",
            message="The brain identifier is invalid.",
            resolution="Use an alphanumeric brain ID that starts with a letter.",
        )
    except BrainNotFoundError as error:
        return error_response(
            request,
            status_code=404,
            detail=str(error),
            code="BRAIN_NOT_FOUND",
            message="The brain was not found.",
            resolution="List brains and retry with an existing brain ID.",
        )
    except BrainLifecycleBusyError as error:
        return error_response(
            request,
            status_code=409,
            detail=str(error),
            code="BRAIN_LIFECYCLE_BUSY",
            message="Another lifecycle operation is already running.",
            resolution="Wait for the current operation to finish, then retry.",
        )
    except BrainCleanupError as error:
        return error_response(
            request,
            status_code=503,
            detail="Brain cleanup failed.",
            code="BRAIN_CLEANUP_FAILED",
            message="The brain could not be cleared from every storage backend.",
            resolution="Retry the idempotent operation after checking backend health.",
            extra={"failed_backends": error.failed_backends},
        )


@system_router.delete(
    path="/brains/{brain_id}",
    response_model=BrainLifecycleResponse,
)
async def delete_brain(brain_id: str, request: Request):
    """Delete a brain's data, registry identity, PAT, and cached credentials."""
    try:
        return await delete_brain_controller(
            brain_id,
            get_brain_lifecycle_service(),
        )
    except InvalidBrainIdError as error:
        return error_response(
            request,
            status_code=400,
            detail=str(error),
            code="BRAIN_ID_INVALID",
            message="The brain identifier is invalid.",
            resolution="Use an alphanumeric brain ID that starts with a letter.",
        )
    except BrainLifecycleBusyError as error:
        return error_response(
            request,
            status_code=409,
            detail=str(error),
            code="BRAIN_LIFECYCLE_BUSY",
            message="Another lifecycle operation is already running.",
            resolution="Wait for the current operation to finish, then retry.",
        )
    except BrainCleanupError as error:
        return error_response(
            request,
            status_code=503,
            detail="Brain cleanup failed.",
            code="BRAIN_CLEANUP_FAILED",
            message="The brain could not be deleted from every storage backend.",
            resolution="Retry the idempotent operation after checking backend health.",
            extra={"failed_backends": error.failed_backends},
        )


@system_router.post(
    path="/brains/{brain_id}/create-backup", include_in_schema=False
)  # TODO
async def create_backup(brain_id: str, request: Request):
    """
    Creates a backup of the brain and returns a task ID
    """
    return _not_implemented(request, brain_id, "create-backup")


def _not_implemented(request: Request, brain_id: str, operation: str):
    return error_response(
        request,
        status_code=501,
        detail="Not implemented",
        code="not_implemented",
        message=f"Brain {operation} is not implemented.",
        resolution="Use the profile-aware deploy backup tooling where applicable.",
        extra={"brain_id": brain_id, "operation": operation},
    )
