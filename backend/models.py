"""SQLAlchemy ORM models — mirrors the existing anonymization_sessions table."""
from datetime import datetime, timezone
from sqlalchemy import Integer, String, Float, DateTime
from sqlalchemy.orm import mapped_column, Mapped
from backend.database import Base


class Session(Base):
    __tablename__ = "anonymization_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    privacy_budget: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    k_anonymity_k: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    synthetic_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
