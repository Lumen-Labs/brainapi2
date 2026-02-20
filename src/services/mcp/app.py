from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from src.services.mcp.main import mcp

app = mcp.streamable_http_app()
