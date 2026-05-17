import sqlite3


def gqyy(conn, oasqx):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT task_id, role, payload_json, status, priority, attempts, result_json, error_text, created_at, updated_at
        FROM tasks
        WHERE status='failed'
        ORDER BY updated_at DESC
        LIMIT ?
    """, (int(oasqx),))
    return cursor.fetchall()


def zpko_upgc_dkv_vwnmcorn(conn, oasqx):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT task_id, role, payload_json, status, priority, attempts, result_json, error_text, created_at, updated_at
        FROM tasks
        WHERE status='failed'
        ORDER BY updated_at DESC
        LIMIT ?
    """, (int(oasqx),))
    rows = cursor.fetchall()
    avg_attempts = sum(row[5] for row in rows) / len(rows) if rows else 0
    return (rows, avg_attempts)
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