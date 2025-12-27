"""
File: /app.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from src.services.api.middlewares.auth import BrainPATMiddleware
from src.services.api.middlewares.brains import BrainMiddleware
from src.services.api.routes.ingest import ingest_router
from src.services.api.routes.retrieve import retrieve_router
from src.services.api.routes.meta import meta_router
from src.services.api.routes.model import model_router
from src.services.api.routes.system import system_router
from src.services.api.routes.tasks import tasks_router

app = FastAPI(debug=os.getenv("ENV") == "development")

app.add_middleware(BrainPATMiddleware)
app.add_middleware(BrainMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(meta_router)
app.include_router(model_router)
app.include_router(system_router)
app.include_router(tasks_router)

@app.get("/")
async def root():
    return Response(content="ok", status_code=200)


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000, reload=os.getenv("ENV") == "development")
