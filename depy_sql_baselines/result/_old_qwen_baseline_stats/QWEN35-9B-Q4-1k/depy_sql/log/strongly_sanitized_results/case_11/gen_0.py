# The scheme of the table for task is <conversations (id INTEGER PRIMARY KEY, user_message TEXT, agent_message TEXT)>. Fetches the most recent N user messages from the conversations table (given connection).
def sgxnat_wvfuud(conn, n):    import sqlite3
    from typing import List


    """
    Fetches the most recent N user messages from the conversations table.

    Args:
        conn: SQLite connection object
        n: Number of most recent user messages to fetch

    Returns:
        List of tuples (id, user_message, agent_message) for the N most recent conversations
    """
    query = f"""
        SELECT id, user_message, agent_message
        FROM conversations
        ORDER BY id DESC
        LIMIT {n}
    """
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()

# The scheme of the second problem is <conversations (id INTEGER PRIMARY KEY, user_message TEXT, ...


import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_message TEXT, agent_message TEXT)')
    for i in range(10):
        cursor.execute('INSERT INTO conversations (user_message, agent_message) VALUES (?, ?)', (f'user {i}', f'agent {i}'))
    conn.commit()
    
    raw_result = sgxnat_wvfuud(conn, 5)
    assert len(raw_result) == 5
    assert raw_result[0] == ('user 9',)
    
    new_result = xisgzr_ynqmixjjcsapg(conn, 5)
    assert len(new_result) == 5
    assert new_result[0] == ('user 9', 'agent 9')
    
    conn.close()
    print('All tests passed')

test_both()