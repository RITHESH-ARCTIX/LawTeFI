# LawTeFI architecture

## Request path

`Next.js web app → FastAPI → retrieval service → Chroma persistent collection on D: → local Ollama model`

The legal corpus is shared and curated. Chroma stores the chunk text, embeddings, source locations, and legal metadata locally. PostgreSQL stores user accounts, conversations, messages, notes, and drafts; each private record must always be queried by that user's ID. Private workspace content is not part of the shared RAG corpus.

## RAG response contract

The chat API should return a status, an answer (or no answer), source references, and a conversation ID. References must be derived from retrieved legal chunks and checked against stored documents before returning. If retrieval is weak or no relevant passages are found, return an abstention rather than a legal conclusion.

## Build sequence

1. Agree on the initial public legal corpus and review official source/version metadata.
2. Preview extracted source text and section-aware chunks using `python -m app.ingestion --dry-run`.
3. Generate embeddings locally and persist them in the Chroma collection.
4. Add hybrid keyword/vector retrieval and inspect the returned passages.
5. Connect a local language model and enforce answer-with-evidence or abstain behavior.
6. Add user authentication and owner-scoped chat history and saved work.
7. Add owner-scoped judgment uploads, separate per-user vector collections, page-aware document Q&A, and direct passage search. Case-brief generation is experimental and remains incomplete.
8. Curate multiple official public sources, retain version metadata, and expose an authenticated catalog of the documents currently indexed. The first public sources are the BNS and the Constitution of India.
9. Evaluate public and private retrieval recall, citation correctness, abstention behavior, privacy isolation, and response latency on a manually reviewed question set; use the results to refine case-brief generation.

## Local storage note

The ingestion CLI expects 768-dimensional vectors from the local `embeddinggemma` model and stores them under `data/chroma` by default. Use the same model for query embeddings. If you change models or vector dimensions, create a new Chroma collection and re-index the source corpus.
