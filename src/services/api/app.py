"""
File: /app.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:17:44 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from src.services.api.routes.ingest import ingest_router
from src.services.api.routes.retrieve import retrieve_router
from src.services.api.routes.meta import meta_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(retrieve_router)
app.include_router(meta_router)

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000, reload=os.getenv("ENV") == "development")
