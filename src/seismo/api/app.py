"""FastAPI application shell for seismo-api."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from seismo.api.jobs import JobManager, StreamFn, create_jobs_router

_DEFAULT_HOST = "127.0.0.1"
_VITE_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)

# Baked into the image at Docker build time (see README deploy metadata).
_BUILD_TIME_ENV = "SEISMO_BUILD_TIME"
_APP_SHA_ENV = "SEISMO_BUILD_SHA"
_DEPLOY_SHA_ENV = "SEISMO_DEPLOY_SHA"
_UNKNOWN_BUILD_VALUE = "unknown"


def _env_or_unknown(name: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else _UNKNOWN_BUILD_VALUE


def build_metadata() -> dict[str, str]:
    """Return deploy identity from bake-time env vars (no git at runtime)."""
    return {
        "build_time": _env_or_unknown(_BUILD_TIME_ENV),
        "app_sha": _env_or_unknown(_APP_SHA_ENV),
        "deploy_sha": _env_or_unknown(_DEPLOY_SHA_ENV),
    }


def _project_root() -> Path:
    # src/seismo/api/app.py -> repo root
    return Path(__file__).resolve().parents[3]


def _frontend_dist() -> Path:
    return _project_root() / "frontend" / "dist"


def _default_jobs_root() -> Path:
    return _project_root() / "data" / "web_jobs"


def create_app(
    *,
    jobs_root: Path | None = None,
    stream_fn: StreamFn | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Route registration stays here so later tickets can add routers
    without changing the uvicorn entrypoint in ``main()``.
    """
    root = Path(jobs_root) if jobs_root is not None else _default_jobs_root()
    manager = JobManager(root, stream_fn=stream_fn)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.job_manager.mark_interrupted_jobs()
        yield

    app = FastAPI(title="seismo", lifespan=lifespan)
    app.state.job_manager = manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_VITE_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", **build_metadata()}

    app.include_router(create_jobs_router())

    _mount_frontend_if_present(app)
    return app


def _mount_frontend_if_present(app: FastAPI) -> None:
    dist = _frontend_dist()
    index = dist / "index.html"
    if not dist.is_dir() or not index.is_file():
        return

    @app.get("/")
    async def frontend_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{full_path:path}")
    async def frontend_spa(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse seismo-api arguments.

    Host defaults to ``127.0.0.1`` (or ``SEISMO_HOST`` when set). Port defaults
    to 8000. ``--host`` / ``-H`` wins over ``SEISMO_HOST``.
    """
    parser = argparse.ArgumentParser(description="Serve the seismo API.")
    parser.add_argument(
        "--host",
        "-H",
        default=None,
        help=(
            f"Bind address (default: {_DEFAULT_HOST}, or SEISMO_HOST). "
            "Use 0.0.0.0 for container/ALB health checks."
        ),
    )
    parser.add_argument(
        "--port",
        "-p",
        type=_port,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    args = parser.parse_args(argv)
    if args.host is None:
        env_host = os.environ.get("SEISMO_HOST", "").strip()
        args.host = env_host if env_host else _DEFAULT_HOST
    return args


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> None:
    """Serve the API; host/port from CLI, SEISMO_HOST, or local defaults."""
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
