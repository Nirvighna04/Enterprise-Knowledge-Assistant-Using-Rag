"""
db.py — SQLite models and database utilities using SQLAlchemy ORM.
Tables: users, chat_history, documents
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Database setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "rag_assistant.db")

engine  = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base    = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    email         = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    chats     = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document",    back_populates="user", cascade="all, delete-orphan")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=False)
    department = Column(String(100))
    job_role   = Column(String(100))
    timestamp  = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chats")


class Document(Base):
    __tablename__ = "documents"

    doc_id      = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    doc_name    = Column(String(255), nullable=False)
    file_path   = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")


# ── Initialize DB ─────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_session():
    return Session()


def add_user(email: str, password_hash: str) -> User:
    db = get_session()
    user = User(email=email, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user


def get_user_by_email(email: str):
    db = get_session()
    user = db.query(User).filter(User.email == email).first()
    db.close()
    return user



def save_chat(user_id: int, question: str, answer: str, department: str, job_role: str):
    db = get_session()
    chat = ChatHistory(
        user_id=user_id,
        question=question,
        answer=answer,
        department=department,
        job_role=job_role
    )
    db.add(chat)
    db.commit()
    db.close()


def get_chat_history(user_id: int, limit: int = 20):
    db = get_session()
    history = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.timestamp.desc())
        .limit(limit)
        .all()
    )
    # Detach from session before returning
    result = [
        {
            "question":   h.question,
            "answer":     h.answer,
            "department": h.department,
            "job_role":   h.job_role,
            "timestamp":  h.timestamp,
        }
        for h in history
    ]
    db.close()
    return result


def save_document(user_id: int, doc_name: str, file_path: str) -> Document:
    db = get_session()
    doc = Document(user_id=user_id, doc_name=doc_name, file_path=file_path)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    db.close()
    return doc


def get_user_documents(user_id: int):
    db = get_session()
    docs = db.query(Document).filter(Document.user_id == user_id).all()
    result = [{"doc_name": d.doc_name, "file_path": d.file_path, "uploaded_at": d.uploaded_at} for d in docs]
    db.close()
    return result
