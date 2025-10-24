import os, asyncpg
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

_POOL = None

async def init_pool():
    global _POOL
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    _POOL = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)

async def close_pool():
    global _POOL
    if _POOL:
        await _POOL.close()
        _POOL = None

@asynccontextmanager
async def db_conn():
    if _POOL is None:
        raise RuntimeError("DB pool not initialized")
    conn = await _POOL.acquire()
    try:
        yield conn
    finally:
        await _POOL.release(conn)
