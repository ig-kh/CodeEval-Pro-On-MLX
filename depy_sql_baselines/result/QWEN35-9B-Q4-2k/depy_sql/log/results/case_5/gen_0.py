# The scheme of the table for task is <todos (priority INTEGER, status TEXT)>. Function to get the count of todos for each priority level.
def vrifl(conn):
    import sqlite3
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT priority, COUNT(*) as count
            FROM todos
            GROUP BY priority
        """)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception:
        return None

# The scheme of the task is <todos (priority INTEGER, status TEXT)>. Function to get the count of todos for each priority level and status.
def pzypl_fgkt_dvucbp(conn):
    import sqlite3
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT priority, status, COUNT(*) as count
            FROM todos
            GROUP BY priority, status
        """)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception:
        return None

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