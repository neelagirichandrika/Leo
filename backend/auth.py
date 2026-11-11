
import re

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from argon2 import PasswordHasher
from argon2.exceptions import HashingError

from backend.database import get_db
from backend.models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

password_hasher = PasswordHasher()


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=255)
    password: str = Field(min_length=12, max_length=128)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(data: SignupRequest, db: Session = Depends(get_db)):

    full_name = data.full_name.strip()
    email = data.email.strip().lower()

    if len(full_name) < 2:
        raise HTTPException(status_code=400, detail="Enter a valid name.")

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")

    existing_user = db.query(User).filter(User.email == email).first()

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
