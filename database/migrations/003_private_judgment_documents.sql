-- Metadata for each user's private judgment PDFs. Files and vectors are stored separately per user.
CREATE TABLE IF NOT EXISTS user_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS user_documents_owner_created_idx
    ON user_documents(owner_user_id, created_at DESC);

ALTER TABLE saved_work ADD COLUMN IF NOT EXISTS citations JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS document_id UUID REFERENCES user_documents(id) ON DELETE SET NULL;
