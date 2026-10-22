"""FastAPI 应用：同源静态前端、REST、SSE、统一错误封套与就绪检查。

API 进程不执行长任务、不加载嵌入模型；所有长任务由 worker 领取。
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..runtime import Runtime
from ..settings import PROJECT_ROOT
from . import documents, events, graph, queries, sessions
from .errors import AppError, error_response

logger = logging.getLogger("legal_rag.api")


def create_app(runtime: Runtime | None = None) -> FastAPI:
    app_runtime = runtime or Runtime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app_runtime.startup()
        app.state.runtime = app_runtime
        try:
            yield
        finally:
            await app_runtime.shutdown()

    app = FastAPI(
        title="劳动用工咨询演示 API",
        version="0.4.0",
        description="掌柜智库·劳动用工咨询（单人本机演示，无登录/多租户/人工审核）",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        details = {"errors": [{"loc": list(item.get("loc", [])), "msg": item.get("msg")} for item in exc.errors()][:5]}
        return error_response(request, AppError(422, "VALIDATION_ERROR", "请求字段校验失败", details=details))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("未预期异常 request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "服务内部错误，请使用 request_id 排查",
                    "request_id": request_id,
                    "retryable": False,
                    "details": {},
                }
            },
        )

    app.include_router(documents.router)
    app.include_router(sessions.router)
    app.include_router(queries.router)
    app.include_router(events.router)
    app.include_router(graph.router)

    @app.get("/health/live", tags=["health"])
    async def health_live() -> dict[str, Any]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    async def health_ready(request: Request):
        health = await request.app.state.runtime.health()
        if health["status"] == "unready":
            return JSONResponse(status_code=503, content={"status": health["status"], "checks": health["checks"]})
        return {"status": health["status"], "checks": health["checks"]}

    @app.get("/api/costs", tags=["costs"])
    async def costs(request: Request):
        runtime = request.app.state.runtime
        try:
            snapshot = await runtime.budget.snapshot()
        except Exception as exc:
            raise AppError(503, "DEPENDENCY_UNAVAILABLE", f"成本台账不可用：{exc}") from exc
        return snapshot.model_dump(mode="json")

    web_dir = PROJECT_ROOT / "web"
    if (web_dir / "index.html").exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()
