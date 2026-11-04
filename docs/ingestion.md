# Legal source ingestion

The first ingestion milestone handles selectable-text PDFs and UTF-8 Markdown/text files. It keeps page numbers, detects common section/article headings and numbered judgment paragraphs, then packs nearby blocks into chunks of at most about 1,800 characters. Oversized blocks are split with a small overlap. For statute PDFs, it starts extraction at the enactment text when it can find the standard “Be it enacted by Parliament” marker, avoiding the arrangement-of-sections pages. For the Constitution PDF, it starts at the Preamble to skip cover, preface, and contents pages. Scanned PDFs are rejected with a clear message because OCR is not enabled yet.

## 1. Add a source document

Place an official source file under `data/source_docs/`. Copy `data/manifest.example.jsonl` to `data/manifest.jsonl`, remove the examples, and add one JSON object per file. Use the official source URL and the version/effective date shown on that source. The example URLs are placeholders and must be replaced before ingesting any documents.

Required fields: `file`, `title`, `document_type`, `source_url`. Optional metadata includes `issuing_authority`, `case_name`, `jurisdiction`, `language_code`, `version_label`, `published_on`, `effective_from`, and `effective_until`.

The current shared corpus includes the Bharatiya Nyaya Sanhita, 2023 and the English Constitution of India edition marked as on 1 May 2024. The authenticated **Legal sources** workspace lists indexed titles, authorities, versions, language, page coverage, and passage counts. User-uploaded judgments are stored separately and are not added to this shared corpus.

## 2. Preview extracted chunks

From `api/`, install the requirements, then run:

```powershell
python -m app.ingestion --dry-run
```

Review the first chunk previews and adjust sources or parsing before writing anything to the database. Do not use the example manifest as a real corpus.

To preview only one manifest entry, add `--only filename.pdf` (the filename must match the `file` field exactly). Review bilingual PDFs carefully: text extraction may interleave languages or fail to decode a legacy font correctly. Do not embed a source until its dry-run text is readable and its edition metadata is correct.

## 3. Embed and save

Install and start Ollama locally, download the `embeddinggemma` model (`ollama pull embeddinggemma`), then run:

```powershell
python -m app.ingestion
```

Embeddings are generated in batches through Ollama's local `/api/embed` endpoint and saved to the persistent Chroma collection at `data/chroma`. Use the same embedding model for indexing and later query embedding. No hosted model API is used by this pipeline. PostgreSQL is reserved for user accounts, private conversations, messages, and saved work; it is not needed to ingest the legal corpus.

## Known limits

- PDF extraction depends on the file's text layer and reading order; scanned PDFs need a later OCR stage.
- The section and paragraph recognizers cover common formats, not every court or statute layout. Inspect dry-run output before ingesting.
- Current law status and source authenticity depend on the official source and metadata supplied by the project team. The ingestion script does not independently certify legal content.
- Ingestion is idempotent for an unchanged source URL and file hash. A new file version is retained as a separate set of Chroma records.
