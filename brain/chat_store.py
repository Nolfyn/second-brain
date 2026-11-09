"""Хранилище чат-сессий: сессии и сообщения в SQLite."""

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

import brain.config as config

Base = declarative_base()
log = logging.getLogger(__name__)


class Session(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    title = Column(String, nullable=False, default="Новый чат")
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Message(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON-список источников
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ChatStore:
    """Сессии чата: переписка сохраняется, к ней можно возвращаться."""

    def __init__(self, db_path: str | None = None):
        path = db_path or str(config.sqlite_path())
        self._engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    def create_session(self) -> str:
        """Новая пустая сессия. Возвращает id."""
        session = self._Session()
        try:
            s = Session()
            session.add(s)
            session.commit()
            return s.id
        finally:
            session.close()

    def list_sessions(self) -> list[dict]:
        session = self._Session()
        try:
            rows = (
                session.query(Session)
                .order_by(Session.updated_at.desc())
                .all()
            )
            return [
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in rows
            ]
        finally:
            session.close()

    def get_session(self, session_id: str) -> dict | None:
        """Сессия со всеми сообщениями. None, если нет."""
        session = self._Session()
        try:
            s = session.get(Session, session_id)
            if s is None:
                return None
            messages = (
                session.query(Message)
                .filter_by(session_id=session_id)
                .order_by(Message.created_at, Message.id)
                .all()
            )
            return {
                "id": s.id,
                "title": s.title,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "sources": (
                            json.loads(m.sources_json)
                            if m.sources_json
                            else None
                        ),
                    }
                    for m in messages
                ],
            }
        finally:
            session.close()

    def get_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """Последние лимиты сообщений - для контекста синтеза."""
        session = self._Session()
        try:
            rows = (
                session.query(Message)
                .filter_by(session_id=session_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
                .all()
            )
            rows.reverse()
            return [
                {"role": m.role, "content": m.content} for m in rows
            ]
        finally:
            session.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list | None = None,
    ) -> None:
        """Сохраняет сообщение; первый вопрос задает заголовок сессии."""
        session = self._Session()
        try:
            s = session.get(Session, session_id)
            if s is None:
                raise ValueError(f"Сессия не найдена: {session_id}")
            if role == "user" and s.title == "Новый чат":
                s.title = content[:60]
            s.updated_at = datetime.now(timezone.utc)
            session.add(
                Message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    sources_json=json.dumps(sources, ensure_ascii=False)
                    if sources
                    else None,
                )
            )
            session.commit()
        finally:
            session.close()

    def delete_session(self, session_id: str) -> None:
        session = self._Session()
        try:
            session.query(Message).filter_by(
                session_id=session_id
            ).delete()
            session.query(Session).filter_by(id=session_id).delete()
            session.commit()
        finally:
            session.close()
