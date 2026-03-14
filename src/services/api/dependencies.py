from fastapi import Request


def get_brain_id(request: Request) -> str:
    return request.state.brain_id
