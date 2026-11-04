# LawTeFI

LawTeFI is a legal research and learning assistant for Indian lawyers and law students. Its core design is a locally run, source-grounded RAG system with verifiable citations and separate user workspaces.

## Project layout

- `web/` — Next.js App Router frontend (TypeScript)
- `api/` — FastAPI backend
- `database/` — PostgreSQL schema for user-owned work
- `docs/` — architecture and implementation notes

## Run locally

### Backend

```powershell
cd api
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API health endpoint is `http://localhost:8000/api/v1/health` and the interactive API docs are at `http://localhost:8000/docs`.

### Frontend

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The initial interface is a workspace shell; legal answers stay disabled until a verified corpus and RAG pipeline are connected.

## Current implementation stage

The app includes PostgreSQL-backed accounts, private conversation history, saved work, local legal-document ingestion and RAG answers, and owner-scoped judgment upload and indexing. Its shared corpus currently contains the Bharatiya Nyaya Sanhita and an English Constitution of India edition marked as on 1 May 2024; the authenticated Legal sources view lists indexed documents and their edition metadata. The judgment workspace supports private document Q&A and direct passage search with page references. Citation cards show retrieved local passages, grounded answers can be saved with citations, and saved work can be downloaded as Markdown. Case-brief generation remains experimental and can return incomplete drafts; it is not a finished feature. The app must not present an ungrounded model response as legal guidance.

See [the ingestion guide](docs/ingestion.md) for the document preparation workflow and the local chunking/embedding importer.
