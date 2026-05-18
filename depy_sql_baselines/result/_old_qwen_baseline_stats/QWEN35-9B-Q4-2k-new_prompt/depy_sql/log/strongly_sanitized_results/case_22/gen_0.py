# The scheme of the table for task is <conversations (timestamp TEXT, message_count INTEGER, response_time_avg REAL, token_input INTEGER, token_output INTEGER, agent_name TEXT)>. Function to record conversation statistics (convenience function)
def wybzna_rzbxbzbcjapm(conn, sbgo, ellhw):import sqlite3
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (timestamp, message_count, response_time_avg, token_input, token_output, agent_name) VALUES (?, ?, ?, ?, ?, ?)", (sbgo, ellhw, None, None, None, None))
    return "Conversation recorded"

# The scheme of
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE conversations (timestamp TEXT, message_count INTEGER, response_time_avg REAL, token_input INTEGER, token_output INTEGER, agent_name TEXT)')
    cursor.executemany('INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)', [
        ('2022-01-01', 10, 1.0, 5, 5, 'Agent1'),
        ('2022-01-02', 20, 2.0, 10, 10, 'Agent1'),
        ('2022-01-03', 30, 3.0, 15, 15, 'Agent2')
    ])
    conn.commit()
    raw_result = wybzna_rzbxbzbcjapm(conn, '2022-01-01', 2)
    assert len(raw_result) == 2
    new_result = ysxqn_gtvtzqygzasr_dfeen(conn, '2022-01-01', 2)
    assert new_result == {'Agent1': {'total_conversations': 2, 'total_messages': 30}, 'Agent2': {'total_conversations': 1, 'total_messages': 30}}
    conn.close()
    print('All tests passed')

test_both()