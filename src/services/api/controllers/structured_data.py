"""
File: /structured_data.py
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

async def get_structured_data_by_id(
    id: str, brain_id: str = "default"
):
    """
    Get structured data by ID.
    """
    def _get_data():
        data = data_adapter.get_structured_data_by_id(id, brain_id)
        if not data:
            raise HTTPException(status_code=404, detail="Structured data not found")
        return data
    
    result = await asyncio.to_thread(_get_data)
    
    return JSONResponse(
        content={
            "message": "Structured data retrieved successfully",
            "data": result.model_dump(mode="json")
        }
    )


async def get_structured_data_list(
    limit: int = 10,
    skip: int = 0,
    types: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
):
    """
    Get a list of structured data.
    """
    def _get_list():
        return data_adapter.get_structured_data_list(brain_id, limit, skip, types, query_text)
    
    results = await asyncio.to_thread(_get_list)
    
    return JSONResponse(
        content={
            "message": "Structured data list retrieved successfully",
            "data": [r.model_dump(mode="json") for r in results],
            "count": len(results)
        }
    )

async def get_structured_data_types(
    brain_id: str = "default",
):
    """
    Get all unique types from structured data.
    """
    def _get_types():
        return data_adapter.get_structured_data_types(brain_id)
    
    results = await asyncio.to_thread(_get_types)
    
    return JSONResponse(
        content={
            "message": "Structured data types retrieved successfully",
            "types": results,
            "count": len(results)
        }
    )