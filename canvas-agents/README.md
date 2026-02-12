# canvas-agents

Multi-agent (LangGraph/LangChain-ready) assistants for Canvas course content.
Starts as an external Flask web app; later can be embedded into Canvas via LTI 1.3.

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Run the backend:
```bash
cd apps/backend
python -m app.server
```

Health check:
- http://localhost:5000/api/health

## Project layout

- `apps/backend/` Flask API server (routes + orchestration entrypoints)
- `packages/agents/` LangGraph nodes + orchestrator (to be implemented)
- `packages/retrieval/` Pinecone retrieval wrappers (to be implemented)
- `packages/schemas/` Pydantic schemas for structured agent outputs
- `pipelines/download/` Canvas download pipeline (bring in your downloader here)
- `pipelines/embed/` Embedding pipeline (bring in your embedding pipeline here)
- `docs/specs/` feature specs (contract between PM/research and engineers)

## Dev workflow

- Branch from `dev`: `feature/<topic>`
- Open PR into `dev`
- `main` is protected (release-ready only)

## Next: Class Material Generator

See `docs/specs/class-material-generator.md`.
