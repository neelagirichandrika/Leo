
import os
import re
import secrets
import hashlib

from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status
)

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from argon2 import PasswordHasher
from argon2.exceptions import (
    HashingError,
    VerifyMismatchError,
    VerificationError,
    InvalidHashError
)

from backend.database import get_db
from backend.models import User, UserSession


router = APIRouter(prefix="/auth", tags=["Authentication"])

password_hasher = PasswordHasher()

SESSION_COOKIE_NAME = "leo_session"
SESSION_DURATION_DAYS = 7

IS_PRODUCTION = os.getenv("LEO_ENV", "development").lower() == "production"

COOKIE_SECURE = IS_PRODUCTION
COOKIE_SAMESITE = "none" if IS_PRODUCTION else "lax"


# -------------------------
# REQUEST MODELS
# -------------------------

class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=1, max_length=128)


# -------------------------
# SESSION HELPERS
# -------------------------

def get_session_hash(session_token: str) -> str:
    return hashlib.sha256(session_token.encode("utf-8")).hexdigest()


def get_current_time():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def set_session_cookie(response: Response, session_token: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_DURATION_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/"
    )


def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
):
    session_token = request.cookies.get(SESSION_COOKIE_NAME)

    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please log in to continue."
        )

    token_hash = get_session_hash(session_token)

    user_session = (
        db.query(UserSession)
        .filter(
            UserSession.token_hash == token_hash,
            UserSession.expires_at > get_current_time()
        )
        .first()
    )

    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid. Please log in again."
        )

    user = db.query(User).filter(
        User.id == user_session.user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found."
        )

    return user


# -------------------------
# REGISTER
# -------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(
    data: SignupRequest,
    db: Session = Depends(get_db)
):

    full_name = data.full_name.strip()
    email = data.email.strip().lower()

    if len(full_name) < 2:
        raise HTTPException(
            status_code=400,
            detail="Enter a valid name."
        )

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(
            status_code=400,
            detail="Enter a valid email address."
        )

    existing_user = db.query(User).filter(
        User.email == email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    try:
        hashed_password = password_hasher.hash(data.password)

        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=hashed_password
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": "Account created successfully.",
            "user_id": new_user.id,
            "full_name": new_user.full_name,
            "email": new_user.email
        }

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    except HashingError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to securely process the password."
        )


# -------------------------
# LOGIN
# -------------------------

@router.post("/login")
def login_user(
    data: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):

    email = data.email.strip().lower()

    user = db.query(User).filter(
        User.email == email
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    try:
        password_hasher.verify(
            user.password_hash,
            data.password
        )

    except (VerifyMismatchError, VerificationError, InvalidHashError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # Generate a cryptographically secure session token.
    session_token = secrets.token_urlsafe(32)

    # Store only its SHA-256 hash in MySQL.
    token_hash = get_session_hash(session_token)

    current_time = get_current_time()
    expires_at = current_time + timedelta(days=SESSION_DURATION_DAYS)

    new_session = UserSession(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at
    )

    try:
        db.add(new_session)
        db.commit()

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to create your session."
        )

    set_session_cookie(response, session_token)

    return {
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email
        }
    }


# -------------------------
# VERIFY CURRENT SESSION
# -------------------------

@router.get("/me")
def get_logged_in_user(
    current_user: User = Depends(get_current_user)
):

    return {
        "authenticated": True,
        "user": {
            "id": current_user.id,
            "full_name": current_user.full_name,
            "email": current_user.email
        }
    }


# -------------------------
# LOGOUT
# -------------------------

@router.post("/logout")
def logout_user(
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):

    session_token = request.cookies.get(SESSION_COOKIE_NAME)

    if session_token:
        token_hash = get_session_hash(session_token)

        db.query(UserSession).filter(
            UserSession.token_hash == token_hash
        ).delete(synchronize_session=False)

        db.commit()

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE
    )

    return {
        "message": "Logged out successfully."
    }
