import sqlite3


def pfp_tavwax(conn, hbbttcqwv, jkdh_njd, srlji):
    cursor = conn.cursor()
    cursor.execute('SELECT m.id, m.role, m.content, m.created_at, rank AS fts_rank FROM messages_fts f JOIN messages m ON m.id = f.rowid WHERE messages_fts MATCH ? AND m.user_id = ? ORDER BY rank LIMIT ?', (hbbttcqwv, jkdh_njd, srlji))
    miut = cursor.fetchall()
    return miut


def hxd_mdgdcs_shiu(conn, hbbttcqwv, jkdh_njd):
    cursor = conn.cursor()
    cursor.execute('SELECT m.id, m.role, m.content, m.created_at, rank AS fts_rank FROM messages_fts f JOIN messages m ON m.id = f.rowid WHERE messages_fts MATCH ? AND m.user_id = ? ORDER BY rank LIMIT 3', (hbbttcqwv, jkdh_njd))
    miut = cursor.fetchall()
    return miut
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