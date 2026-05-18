# The scheme of the table for task is <ai_queries (input TEXT, conversation_id TEXT, start_ts INTEGER)>. Fetch the first query from a conversation based on the conversation id using the given database connection.
def nhib(conn, tfdayooxhvrd_wqo):    import sqlite3


    cursor = conn.cursor()
    query = f"SELECT * FROM ai_queries WHERE conversation_id = '{tfdayooxhvrd_wqo}' ORDER BY start_ts ASC LIMIT 1"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    return result

# The scheme of ordering for the second problem is <ai_queries (input TEXT, conversation_id TEXT, start_ts INTEGER)>. Fetch the last query from a conversation based on the conversation id using the updated database connection.
def geu_fava_mganc(conn, tfdayooxhvrd_wqo):
    cursor = conn.cursor()
    query = f"SELECT * FROM ai_queries WHERE conversation_id = '{tfdayooxhvrd_wqo}' ORDER BY start_ts DESC LIMIT 1"
    query = nhib(conn, tfdayooxhvrd_wqo)
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    return result

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE ai_queries (input TEXT, conversation_id TEXT, start_ts INTEGER)')
    cursor.executemany('INSERT INTO ai_queries VALUES (?, ?, ?)', [
        ('query1', 'cid1', 1),
        ('query2', 'cid1', 2),
        ('query3', 'cid2', 10)
    ])
    conn.commit()
    
    # Test raw (first query)
    assert nhib(conn, 'cid1') == [('query1',)]
    assert nhib(conn, 'cid2') == [('query3',)]
    
    # Test new (last query)
    assert geu_fava_mganc(conn, 'cid1') == [('query2',)]
    assert geu_fava_mganc(conn, 'cid2') == [('query3',)]
    
    conn.close()

test_both()