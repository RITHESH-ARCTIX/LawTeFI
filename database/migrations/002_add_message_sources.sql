-- Preserve LawTeFI response status and citations in each user's chat history.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS status TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS citations JSONB NOT NULL DEFAULT '[]'::jsonb;
