"""FastAPI application for the seismo web PoC."""

from .app import create_app, main

__all__ = ["create_app", "main"]
