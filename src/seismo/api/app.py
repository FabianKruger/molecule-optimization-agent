"""FastAPI application shell for seismo-api."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

_VITE_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def _project_root() -> Path:
    # src/seismo/api/app.py -> repo root
    return Path(__file__).resolve().parents[3]


def _frontend_dist() -> Path:
    return _project_root() / "frontend" / "dist"


def create_app() -> FastAPI:
    """Build the FastAPI app.

    Route registration stays here so later tickets can add routers
    without changing the uvicorn entrypoint in ``main()``.
    """
    app = FastAPI(title="seismo")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_VITE_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Future routers (presets, jobs, ...) register on ``app`` here.

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
    """Parse seismo-api arguments. The port defaults to 8000."""
    parser = argparse.ArgumentParser(description="Serve the seismo API.")
    parser.add_argument(
        "--port",
        "-p",
        type=_port,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    return parser.parse_args(argv)


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> None:
    """Serve the API on 127.0.0.1, port 8000 unless --port is set."""
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
