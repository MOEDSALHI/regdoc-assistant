# src/db/database.py
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from loguru import logger

from src.config import settings

# Connection pool — shared across the application
_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    """
    Initialize the asyncpg connection pool.
    Call once at application startup (FastAPI lifespan).
    """
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url.replace("+asyncpg", ""),
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("Database pool initialized | dsn={}", settings.database_url[:40])


async def close_pool() -> None:
    """Close the connection pool. Call at application shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Database pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Async context manager providing a connection from the pool.

    Usage:
        async with get_connection() as conn:
            result = await conn.fetch("SELECT ...")
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    async with _pool.acquire() as conn:
        yield conn


# async def apply_schema() -> None:
#     """
#     Apply the SQL schema (create tables and indexes if not exist).
#     Idempotent — safe to call multiple times.

#     Note: asyncpg does not support multiple statements in a single
#     conn.execute() call. Each statement must be executed separately.
#     """
#     schema_path = "src/db/schema.sql"
#     with open(schema_path) as f:
#         schema_sql = f.read()

#     # Split by semicolon and execute each statement individually
#     statements = [
#         s.strip()
#         for s in schema_sql.split(";")
#         if s.strip() and not s.strip().startswith("--")
#     ]

#     async with get_connection() as conn:
#         for statement in statements:
#             await conn.execute(statement)
#             logger.debug("Executed: {}...", statement[:60])

#     logger.info("Schema applied | {} statements executed", len(statements))


async def apply_schema() -> None:
    """
    Apply the SQL schema (create tables and indexes if not exist).
    Idempotent — safe to call multiple times.

    Note: asyncpg does not support multiple statements in a single
    conn.execute() call. Each statement must be executed separately.
    """
    schema_path = "src/db/schema.sql"
    with open(schema_path) as f:
        raw = f.read()

    # Remove lines that are pure comments or Python-style comments
    clean_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or stripped.startswith("#") or not stripped:
            continue
        clean_lines.append(line)

    clean_sql = "\n".join(clean_lines)

    # Split into individual statements
    statements = [s.strip() for s in clean_sql.split(";") if s.strip()]

    async with get_connection() as conn:
        for statement in statements:
            await conn.execute(statement)
            logger.debug("Executed: {}...", statement[:60])

    logger.info("Schema applied | {} statements executed", len(statements))
