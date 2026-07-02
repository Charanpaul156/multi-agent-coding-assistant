"""FastAPI application entrypoint.

Business logic is intentionally not implemented at this stage.
"""

from fastapi import FastAPI

from backend.api.routes import router

app = FastAPI(
    title="Multi-Agent Coding Assistant",
    version="0.1.0",
)

app.include_router(router)

