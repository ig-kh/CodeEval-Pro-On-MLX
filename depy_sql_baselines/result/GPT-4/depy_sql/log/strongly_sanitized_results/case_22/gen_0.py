import sqlite3
from sqlite3 import Error

# Function to record conversation statistics
def wybzna_rzbxbzbcjapm(conn, sbgo, ellhw):
    try:
        # Create a cursor object
        cur = conn.cursor()

        # Execute the insert query
        cur.execute("INSERT INTO conversations(timestamp, message_count, response_time_avg, token_input, token_output, agent_name) VALUES(?,?,?,?,?,?)", (sbgo, ellhw['message_count'], ellhw['response_time_avg'], ellhw['token_input'], ellhw['token_output'], ellhw['agent_name']))

        # Commit the transaction
        conn.commit()
    except Error as e:
        print(e)

# Function to get the total number of conversations and messages for each agent
def ysxqn_gtvtzqygzasr_dfeen(conn, date, limit):
    try:
        # Create a cursor object
        cur = conn.cursor()

        # Execute the select query
        cur.execute("SELECT agent_name, COUNT(*), SUM(message_count) FROM conversations WHERE timestamp LIKE ? GROUP BY agent_name ORDER BY COUNT(*) DESC LIMIT ?", ('%'+date+'%', limit))

        # Fetch all rows from the last executed statement
        rows = cur.fetchall()

        # Print the rows
        for row in rows:
            print(row)
    except Error as e:
        print(e)
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