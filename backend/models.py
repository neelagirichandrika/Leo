
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    DateTime,
    ForeignKey,
    Enum,
    Index,
    func
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    title = Column(
        String(200),
        nullable=False,
        default="New chat",
        server_default="New chat"
    )

    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    conversation_id = Column(
        BigInteger,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )

    role = Column(
        Enum("user", "assistant", name="message_role"),
        nullable=False
    )

    content = Column(
        Text().with_variant(MEDIUMTEXT, "mysql"),
        nullable=False
    )

    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_messages_conversation_created",
            "conversation_id",
            "created_at"
        ),
    )
