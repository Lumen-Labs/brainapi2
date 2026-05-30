from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


def is_console_path(path: str) -> bool:
    return path == "/console" or path.startswith("/console/")


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise exc
