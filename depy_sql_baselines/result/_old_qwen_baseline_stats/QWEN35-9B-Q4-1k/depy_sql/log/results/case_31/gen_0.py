# The scheme of the table for task is <tasks (task_id INTEGER PRIMARY KEY, role TEXT, payload_json TEXT, status TEXT, priority INTEGER, attempts INTEGER, result_json TEXT, error_text TEXT, created_at TEXT, updated_at TEXT)>. Fetches the specified number of failed tasks from the tasks table, ordered by the time they were last updated (given connection).
def gqyy(conn, oasqx):
    """
    Fetches the specified number of failed tasks from the tasks table, ordered by the time they were last updated.
    
    Args:
        conn: Database connection object
        oasqx: Number of failed tasks to fetch
    
    Returns:
        List of dictionaries with task information
    """
    cursor = conn.cursor()
    
    # SQL query to fetch failed tasks ordered by updated_at
    query = """
        SELECT 
            task_id, role, payload_json, status, priority, attempts, result_json, error_text, created_at, updated_at
        FROM tasks
        WHERE status = 'failed'
        ORDER BY updated_at DESC
        LIMIT ?
    """
    
    cursor.execute(query, (oasqx,))
    results = cursor.fetchall()
    
    # Convert to list of dictionaries
    tasks = []
    for row in results:
        tasks.append({
            'task_id': row[0],
            'role': row[1],
            'payload_json': row[
            'status': row[2],
            'priority': row[3],
            'attempts': row[
            'result_json': row[5],
            'error_text': row[6],
            'created_at': row[7],
            'updated_at': row[
    
    cursor.close()
    return tasks

def zpko_upgc_dkv_vwnmcorn(conn, oasqx):
    """
    Fetches the specified number of failed tasks from the schema.
    
    Args:
        conn: Database connection object
        oasqx: Number of failed tasks to fetch
    
    Returns:
        Tuple of (list of task dictionaries, average attempts)
    """
    # Call the first function to get failed tasks
    tasks = gqyy(conn, oasqx)
    
    # Calculate average attempts
    total_attempts = sum(task['attempts'] for task in tasks)
    avg_attempts = total_attempts / len(tasks) if tasks else 0
    
    return tasks, avg_attempts

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