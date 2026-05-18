import re

import sqlite3

def pfp_tavwax(conn, user_id, search_query, limit):
    """
    Full-text search across messages for a user. 
    Returns messages with id, role, content, created_at, and FTS rank.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT id, role, content, created_at, rank 
        FROM messages 
        WHERE user_id = ? AND content MATCH ? 
        ORDER BY rank DESC 
        LIMIT ?
        """, (user_id, search_query, limit))
    return cur.fetchall()

def hxd_mdgdcs_shiu(conn, user_id, search_query):
    """
    Full-text search across messages for a user and return only the top 3 messages with the highest FTS rank.
    """
    return pfp_tavwax(conn, user_id, search_query, 3)
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE VIRTUAL TABLE messages_fts USING fts5(content)')
    cursor.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, user_id INTEGER, role TEXT, content TEXT, created_at TEXT)')
    cursor.execute("INSERT INTO messages_fts (rowid, content) VALUES (1, 'hello world')")
    cursor.execute("INSERT INTO messages (id, user_id, role, content, created_at) VALUES (1, 1, 'user', 'hello world', '2022-01-01 00:00:00')")
    conn.commit()
    assert pfp_tavwax(conn, 'hello', 1, 5) == [(1, 'user', 'hello world', '2022-01-01 00:00:00', 0)]
    assert hxd_mdgdcs_shiu(conn, 'hello', 1) == [(1, 'user', 'hello world', '2022-01-01 00:00:00', 0)]
    conn.close()
    print('All tests passed')

test_both()