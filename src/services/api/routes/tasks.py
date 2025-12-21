"""
File: /tasks.py
Created Date: Saturday December 13th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 13th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""
import json
from src.utils.logging import log
from fastapi import APIRouter, HTTPException
from src.services.kg_agent.main import cache_adapter

tasks_router = APIRouter(prefix="/tasks", tags=["tasks"])


@tasks_router.get("/{task_id}")
async def get_task(task_id: str, brain_id: str = "default"):
    """
    Get the result of a task by its ID.
    """
    try:
        str_result = cache_adapter.get(f"task:{task_id}", brain_id=brain_id)

        if str_result is None:
            raise HTTPException(status_code=404, detail="Task not found")

        if isinstance(str_result, bytes):
            result = json.loads(str_result.decode("utf-8"))
        else:
            result = json.loads(str_result)

        return {
            "task_id": task_id,
            "status": result.get("status", "pending"),
            "result": result
        }
    except Exception as e:
        log(f"Error in get_task: {type(e).__name__}: {str(e)}")
        return {
            "task_id": task_id,
            "status": "error",
            "result": {"error": str(e)}
        }