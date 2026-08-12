"""Minimal ordered PostgreSQL migration runner."""

from __future__ import annotations

from importlib.resources import files


def apply_migrations(dsn: str) -> list[str]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on storage extra
        raise RuntimeError("install bnhmm-engine[storage] to run migrations") from exc

    migration_dir = files("bnhmm.storage").joinpath("migrations")
    migration_names = sorted(
        item.name for item in migration_dir.iterdir() if item.name.endswith(".sql")
    )
    applied: list[str] = []

    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS bnhmm")
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS bnhmm.schema_migrations ("
                "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            cursor.execute("SELECT version FROM bnhmm.schema_migrations")
            existing = {row[0] for row in cursor.fetchall()}
            for name in migration_names:
                if name in existing:
                    continue
                sql = migration_dir.joinpath(name).read_text(encoding="utf-8")
                cursor.execute(sql)
                cursor.execute(
                    "INSERT INTO bnhmm.schema_migrations(version) VALUES (%s)", (name,)
                )
                applied.append(name)
    return applied

