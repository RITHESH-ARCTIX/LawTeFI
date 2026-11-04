import os
import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

import jwt
import psycopg
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, Field, field_validator
from pwdlib import PasswordHash

from .database import connect_database

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
password_hasher = PasswordHash.recommended()
bearer_scheme = HTTPBearer(auto_error=False)
TOKEN_LIFETIME = timedelta(hours=8)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    display_name: str = Field(min_length=1, max_length=120)
    role: Literal["lawyer", "student"]
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized):
            raise ValueError("Enter a valid email address.")
        return normalized

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return value.strip()


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserResponse(BaseModel):
    id: UUID
    email: str
    display_name: str
    role: Literal["lawyer", "student"]


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


def _token_secret() -> str:
    secret = os.getenv("LAWTEFI_JWT_SECRET")
    if not secret or len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set LAWTEFI_JWT_SECRET in api/.env before using authentication.",
        )
    return secret


def create_access_token(user_id: UUID) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": now + TOKEN_LIFETIME},
        _token_secret(),
        algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, object]:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Sign in to access this workspace.")
    try:
        payload = jwt.decode(credentials.credentials, _token_secret(), algorithms=["HS256"])
        user_id = UUID(payload["sub"])
    except (InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Your session is invalid or expired.") from exc

    try:
        with connect_database() as connection:
            user = connection.execute(
                "SELECT id, email, display_name, role FROM users WHERE id = %s",
                (user_id,),
            ).fetchone()
    except (psycopg.Error, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="The LawTeFI database is unavailable.") from exc
    if user is None:
        raise HTTPException(status_code=401, detail="This account is no longer available.")
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest) -> dict[str, object]:
    encoded_password = password_hasher.hash(request.password)
    try:
        with connect_database() as connection:
            user = connection.execute(
                """
                INSERT INTO users (email, display_name, role, password_hash)
                VALUES (%s, %s, %s, %s)
                RETURNING id, email, display_name, role
                """,
                (request.email, request.display_name, request.role, encoded_password),
            ).fetchone()
    except UniqueViolation as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    except (psycopg.Error, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="The LawTeFI database is unavailable.") from exc
    return user


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> dict[str, object]:
    try:
        with connect_database() as connection:
            user = connection.execute(
                """
                SELECT id, email, display_name, role, password_hash
                FROM users WHERE email = %s
                """,
                (request.email,),
            ).fetchone()
    except (psycopg.Error, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="The LawTeFI database is unavailable.") from exc

    if user is None or not password_hasher.verify(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    public_user = {key: user[key] for key in ("id", "email", "display_name", "role")}
    return {"access_token": create_access_token(user["id"]), "user": public_user}


@router.get("/me", response_model=UserResponse)
def me(current_user: dict[str, object] = Depends(get_current_user)) -> dict[str, object]:
    return current_user
