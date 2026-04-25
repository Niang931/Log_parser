from typing import Iterator
from typing import Optional
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from utils.setting import setting
from utils.logger import logger


def get_connection():
    try:
        return psycopg2.connect(
            dbname = setting.POSTGRES_DB,
            user = setting.POSTGRES_USER,
            password = setting.POSTGRES_PASSWORD,
            host = setting.POSTGRES_HOST,
            port = setting.POSTGRES_PORT
        )
    except psycopg2.OperationalError as e:
        logger.error(e)

@contextmanager
def get_cursor(commit:bool = True,
               dict_cursor:bool = False,
               host:Optional[str] = None)-> Iterator[psycopg2.extensions.cursor]:
    conn = get_connection()
    cur = None
    try:
        cursor_factory = (
            psycopg2.extras.RealDictCursor if dict_cursor else None
        )
        cur = conn.cursor(cursor_factory=cursor_factory)
        yield cur
        if commit:
            conn.commit()
    except psycopg2.DatabaseError as e:
        conn.rollback()
        logger.error(e)
    finally:
        if cur is not None:
            cur.close()
            conn.close()
