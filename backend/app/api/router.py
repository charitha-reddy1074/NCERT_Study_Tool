from fastapi import APIRouter

from .v1.health import router as health_router
from .v1.study import router as study_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(study_router, tags=["study"])
