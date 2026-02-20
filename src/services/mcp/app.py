from pathlib import Path

import dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(_project_root / ".env")

from mcp.server.transport_security import TransportSecuritySettings
from src.services.mcp.main import mcp

transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
)

app = mcp._mcp_server.streamable_http_app(
    transport_security=transport_security,
    stateless_http=mcp.settings.stateless_http,
    json_response=mcp.settings.json_response,
)
