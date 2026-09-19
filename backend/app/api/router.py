from fastapi import APIRouter
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router
from app.api.system import router as system_router

api_router = APIRouter()

# v1 routes
v1_router = APIRouter(prefix="/v1")
v1_router.include_router(documents_router)
v1_router.include_router(query_router)

# Include v1 and system routes
api_router.include_router(v1_router)
api_router.include_router(system_router)
