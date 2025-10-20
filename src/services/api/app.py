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
from uvicorn import run

from src.services.api.routes.ingest import ingest_router

app = FastAPI()

app.include_router(ingest_router)

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000, reload=os.getenv("ENV") == "development")
