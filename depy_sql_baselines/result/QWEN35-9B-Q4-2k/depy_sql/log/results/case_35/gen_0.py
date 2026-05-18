# The scheme of the table for task is <messages (id INTEGER PRIMARY KEY, user_id INTEGER, role TEXT, content TEXT, created_at TEXT)>. Full-text search across messages for a user. Returns messages with id, role, content, created_at, and FTS rank.
def pfp_tavwax(conn, hbbttcqwv, jkdh_njd, srlji):
    cursor = conn.cursor()
    cursor.execute("SELECT id, role, content, created_at FROM messages WHERE user_id = ? AND content MATCH ? ORDER BY FTS_ranking() DESC LIMIT ?", (jkdh_njd, srlji))
    results = cursor.fetchall()
    cursor.close()
    return results

# The scheme of the table for task is <messages (id INTEGER PRIMARY KEY, user_id INTEGER, role TEXT, content TEXT, created_at TEXT)>. Full-text search across messages for a user and return only the top 3 messages with the highest FTS rank.
def hxd_mdgdcs_shiu(conn, hbbttcqwv, jkdh_njd):
    pfp_tavwax(conn, hbbttcqwv, jkdh_njd, 3)
    cursor = conn.cursor()
    cursor.execute("SELECT id, role, content, created_at FROM messages WHERE user_id = ? AND content MATCH ? ORDER BY FTS_ranking() DESC LIMIT 3", (jkdh_njd,))
    results = cursor.fetchall()
    cursor.close()
    return results

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