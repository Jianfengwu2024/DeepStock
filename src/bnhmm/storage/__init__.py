"""PostgreSQL model registry and precomputed-signal storage."""

from .migrate import apply_migrations

__all__ = ["apply_migrations"]

