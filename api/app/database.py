import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def connect_database() -> psycopg.Connection:
    """Open a connection using the local API environment settings."""
    required = {
        "host": os.getenv("LAWTEFI_DB_HOST", "127.0.0.1"),
        "port": os.getenv("LAWTEFI_DB_PORT", "5432"),
        "dbname": os.getenv("LAWTEFI_DB_NAME", "lawtefi"),
        "user": os.getenv("LAWTEFI_DB_USER", "lawtefi_app"),
        "password": os.getenv("LAWTEFI_DB_PASSWORD"),
    }
    if not required["password"]:
        raise RuntimeError("LAWTEFI_DB_PASSWORD is not configured in api/.env.")
    return psycopg.connect(**required, row_factory=dict_row)
