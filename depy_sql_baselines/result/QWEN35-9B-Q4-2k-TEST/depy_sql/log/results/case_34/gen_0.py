# The scheme of the table for task is <chat_messages (session_id TEXT, sender TEXT, message TEXT, timestamp TEXT, escalated INTEGER)>. Get chat history for a session using the given connection.
def get_chat_history(conn, jumcejy_njr):
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, sender, message, timestamp, escalated FROM chat_messages WHERE session_id = ? ORDER BY timestamp DESC", (jumcejy_njr,))
    results = cursor.fetchall()
    cursor.close()
    return results

# The scheme of the table for task is <chat_messages (session_id TEXT, sender TEXT, message TEXT, timestamp TEXT, escalated INTEGER)>. Get chat summary for a session including total message count and number of escalated messages. Uses the given connection.
def tga_ndvv_wkbxswq(conn, jumcejy_njr):
    get_chat_history(conn, jumcejy_njr)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total_messages, SUM(escalated) as escalated_count FROM chat_messages WHERE session_id = ?", (jumcejy_njr,))
    result = cursor.fetchone()
    cursor.close()
    return result

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE chat_messages (session_id TEXT, sender TEXT, message TEXT, timestamp TEXT, escalated INTEGER)')
    cursor.executemany('INSERT INTO chat_messages VALUES (?,?,?,?,?)', [
        ('sessA', 'user', 'Hello', '2023-01-01 10:00:00', 0),
        ('sessA', 'bot', 'Hi', '2023-01-01 10:00:05', 0),
        ('sessA', 'user', 'Help', '2023-01-01 10:01:00', 1),
        ('sessB', 'user', 'Test', '2023-01-02 09:00:00', 0),
        ('sessB', 'user', 'Another', '2023-01-02 09:01:00', 1)
    ])
    conn.commit()
    
    history = get_chat_history(conn, 'sessA')
    assert len(history) == 3
    assert history[2]['escalated'] is True
    
    summary = tga_ndvv_wkbxswq(conn, 'sessA')
    assert summary['session_id'] == 'sessA'
    assert summary['total_messages'] == 3
    assert summary['escalated_messages'] == 1
    
    summary2 = tga_ndvv_wkbxswq(conn, 'sessB')
    assert summary2['total_messages'] == 2
    assert summary2['escalated_messages'] == 1
    
    conn.close()

test_both()