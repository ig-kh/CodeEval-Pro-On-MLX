import sqlite3
from sqlite3 import Error
import json

# Fetches the specified number of failed tasks from the tasks table, ordered by the time they were last updated
def gqyy(conn, oasqx):
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM tasks WHERE status='failed' ORDER BY updated_at DESC LIMIT ?", (oasqx,))
        rows = cur.fetchall()
        return rows
    except Error as e:
        print(e)

# Fetches the specified number of failed tasks from the tasks table, ordered by the time they were last updated. Additionally, it should return the average number of attempts for these tasks
def zpko_upgc_dkv_vwnmcorn(conn, oasqx):
    rows = gqyy(conn, oasqx)
    if rows:
        attempts_sum = sum([row[5] for row in rows])
        avg_attempts = attempts_sum / len(rows)
        return rows, avg_attempts
    else:
        return [], 0
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE tasks (
            task_id INTEGER PRIMARY KEY,
            role TEXT,
            payload_json TEXT,
            status TEXT,
            priority INTEGER,
            attempts INTEGER,
            result_json TEXT,
            error_text TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    cursor.executemany("""
        INSERT INTO tasks VALUES
        (1, 'role1', '{}', 'failed', 1, 3, '{}', 'error1', '2022-01-01', '2022-01-02'),
        (2, 'role2', '{}', 'failed', 2, 2, '{}', 'error2', '2022-01-03', '2022-01-04'),
        (3, 'role3', '{}', 'failed', 3, 1, '{}', 'error3', '2022-01-05', '2022-01-06')
    """)
    conn.commit()
    
    raw_result = gqyy(conn, 2)
    assert raw_result == [(2, 'role2', '{}', 'failed', 2, 2, '{}', 'error2', '2022-01-03', '2022-01-04'),
                          (1, 'role1', '{}', 'failed', 1, 3, '{}', 'error1', '2022-01-01', '2022-01-02')]
    
    new_result, avg = zpko_upgc_dkv_vwnmcorn(conn, 2)
    assert new_result == raw_result
    assert avg == 2.5
    
    conn.close()
    print('All tests passed')

test_both()