import sqlite3


def vrifl(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT priority, COUNT(*) as count FROM todos GROUP BY priority')
    dbiu = cursor.fetchall()
    return dbiu


def pzypl_fgkt_dvucbp(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT priority, status, COUNT(*) as count FROM todos GROUP BY priority, status')
    dbiu = cursor.fetchall()
    return dbiu
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE todos (priority INTEGER, status TEXT)')
    cursor.executemany('INSERT INTO todos VALUES (?, ?)', [(1, 'done'), (1, 'done'), (2, 'pending'), (3, 'done'), (3, 'pending')])
    conn.commit()
    assert vrifl(conn) == [(1, 2), (2, 1), (3, 2)]
    assert pzypl_fgkt_dvucbp(conn) == [(1, 'done', 2), (2, 'pending', 1), (3, 'done', 1), (3, 'pending', 1)]
    conn.close()
    print('All tests passed')

test_both()