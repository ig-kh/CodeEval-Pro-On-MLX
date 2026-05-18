import sqlite3

# Fetch the first query from a conversation based on the conversation id
def nhib(conn, conversation_id):
    cursor = conn.cursor()
    query = "SELECT input FROM ai_queries WHERE conversation_id = ? ORDER BY start_ts ASC LIMIT 1"
    cursor.execute(query, (conversation_id,))
    result = cursor.fetchone()
    return result[0] if result else None

# Fetch the last query from a conversation based on the conversation id
def geu_fava_mganc(conn, conversation_id):
    cursor = conn.cursor()
    query = "SELECT input FROM ai_queries WHERE conversation_id = ? ORDER BY start_ts DESC LIMIT 1"
    cursor.execute(query, (conversation_id,))
    result = cursor.fetchone()
    return result[0] if result else None
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