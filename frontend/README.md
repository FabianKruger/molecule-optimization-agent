# seismo web UI

Vite + React + TypeScript client for the seismo job API.

Node and npm come from the pixi `frontend` environment (conda-forge `nodejs`). Prefer the pixi tasks from the repo root:

```bash
# Terminal 1 — API
pixi run seismo-api

# Terminal 2 — Vite dev server (cd + npm install handled by the task)
pixi run frontend
```

Production build (served by `seismo-api` from `/`):

```bash
pixi run frontend-build
pixi run seismo-api
```
