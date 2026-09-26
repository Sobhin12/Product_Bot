from pathlib import Path

import psycopg

from . import config


def connect(autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(config.DATABASE_URL, autocommit=autocommit)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute((Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))
    conn.commit()
