# canvas-agents

Multi-agent assistants for Canvas course content.  
**Phase 1:** external Flask web app (sandbox-first).  
**Phase 2:** embedded Canvas add-on via LTI 1.3 (university rollout).

## What we’re building (MVP)
We are starting with the **Class Material Generator**: given a course topic, the system retrieves relevant course materials (PDFs/slides/pages), extracts key concepts and examples, organizes content by cognitive complexity (Bloom), and outputs a **ready-to-teach packet** (JSON + Markdown).

See: `docs/specs/class-material-generator.md`

---

## Repo layout

- `apps/backend/` — Flask API server (routes + orchestration entrypoints)
- `packages/agents/` — LangGraph nodes + orchestrators (to implement)
- `packages/retrieval/` — Pinecone retrieval wrappers + citation formatting (to implement)
- `packages/schemas/` — Pydantic schemas for structured outputs
- `packages/canvas/` — Canvas API client utilities (to implement)
- `pipelines/download/` — Canvas downloader pipeline (bring in / adapt)
- `pipelines/embed/` — Embedding pipeline (bring in / adapt)
- `docs/specs/` — feature specs (contracts between PM + engineers)
- `.github/workflows/` — CI (ruff + pytest)

---

## Quickstart (local)

### 1) Setup environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
