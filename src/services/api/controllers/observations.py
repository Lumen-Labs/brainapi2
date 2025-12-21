"""
File: /observations.py
Created Date: Saturday December 6th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 13th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
from typing import Optional
from fastapi import HTTPException
from starlette.responses import JSONResponse
from src.services.data.main import data_adapter


async def get_observation_by_id(
    id: str, brain_id: str = "default"
):
    """
    Get observation by ID.
    """
    def _get_observation():
        observation = data_adapter.get_observation_by_id(id, brain_id)
        if not observation:
            raise HTTPException(status_code=404, detail="Observation not found")
        return observation
    
    result = await asyncio.to_thread(_get_observation)
    
    return JSONResponse(
        content={
            "message": "Observation retrieved successfully",
            "observation": result.model_dump(mode="json")
        }
    )


async def get_observations_list(
    limit: int = 10,
    skip: int = 0,
    resource_id: Optional[str] = None,
    labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
):
    """
    Get a list of observations.
    """
    def _get_list():
        return data_adapter.get_observations_list(
            brain_id, limit, skip, resource_id, labels, query_text
        )
    
    results = await asyncio.to_thread(_get_list)
    
    return JSONResponse(
        content={
            "message": "Observations list retrieved successfully",
            "observations": [r.model_dump(mode="json") for r in results],
            "count": len(results)
        }
    )

async def get_observation_labels(
    brain_id: str = "default",
):
    """
    Get all unique labels from observations.
    """
    def _get_labels():
        return data_adapter.get_observation_labels(brain_id)
    
    results = await asyncio.to_thread(_get_labels)
    
    return JSONResponse(
        content={
            "message": "Observation labels retrieved successfully",
            "labels": results,
            "count": len(results)
        }
    )