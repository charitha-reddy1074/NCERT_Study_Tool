from fastapi import Request

from app.core.container import BackendContainer


def get_container(request: Request) -> BackendContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Backend container has not been initialized")
    return container
