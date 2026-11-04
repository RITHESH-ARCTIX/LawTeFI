-- Add password hashes for local account authentication.
-- Existing LawTeFI MVP databases are expected to have no user records yet.
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;
ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL;
