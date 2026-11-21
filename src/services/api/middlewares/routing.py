from starlette.requests import Request
from starlette.routing import Match


def request_matches_registered_route(request: Request) -> bool:
    app = request.app
    router = getattr(app, "router", app)
    for route in getattr(router, "routes", []) or []:
        matches = getattr(route, "matches", None)
        if not callable(matches):
            continue
        try:
            match, _child = matches(request.scope)
        except Exception:
            continue
        if match != Match.NONE:
            return True
    return False
