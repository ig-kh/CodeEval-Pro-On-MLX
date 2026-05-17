import sqlite3


def sgxnat_wvfuud(conn, n):
    cursor = conn.cursor()
    limit = max(1, min(n, 50))
    cursor.execute('SELECT user_message FROM conversations ORDER BY id DESC LIMIT ?', (limit,))
    return cursor.fetchall()


def xisgzr_ynqmixjjcsapg(conn, n):
    cursor = conn.cursor()
    limit = max(1, min(n, 50))
    cursor.execute('SELECT user_message, agent_message FROM conversations ORDER BY id DESC LIMIT ?', (limit,))
    return cursor.fetchall()
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