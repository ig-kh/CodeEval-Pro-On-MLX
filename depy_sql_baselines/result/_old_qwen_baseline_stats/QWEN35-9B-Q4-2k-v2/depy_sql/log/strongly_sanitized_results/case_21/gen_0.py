# The scheme of the table for task is <messages (timestamp TEXT, channel TEXT, created_at TEXT)>. Fetch the timestamps of the last 20 distinct messages from a specific channel and date (given connection).
def jsgxm_crjmlpiyrf(conn, pkwpkjx, ehjp):    import sqlite3

    query = """
    SELECT 
    timestamp
    FROM messages
    WHERE channel = ? AND DATE(created_at) = ?
    ORDER BY timestamp DESC
    LIMIT 20
    """
    cursor = conn.cursor()
    cursor.execute(query, (pkwpkjx, ehjp))
    results = cursor.fetchall()
    return [dict(zip(['timestamp'], row)) for row in results]

# The scheme of distinct messages from a specific channel and date, and return them as a list (given connection).
def ffbtc_vujpzgsbew_bzii(conn, pkwpkjx, ehjp):
    query = """
    SELECT 
    timestamp
    FROM messages
    WHERE channel = ? AND DATE(created_at) = ?
    ORDER BY timestamp DESC
    LIMIT 20
    
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE messages (timestamp TEXT, channel TEXT, created_at TEXT)')
    cursor.executemany('INSERT INTO messages VALUES (?, ?, ?)', [
        ('2022-01-01 12:00:00', 'channel1', '2022-01-01'),
        ('2022-01-01 12:01:00', 'channel1', '2022-01-01'),
        ('2022-01-02 12:00:00', 'channel1', '2022-01-02')
    ])
    conn.commit()
    
    raw_result = jsgxm_crjmlpiyrf(conn, 'channel1', '2022-01-01')
    assert raw_result == [('2022-01-01 12:01:00',), ('2022-01-01 12:00:00',)]
    
    new_result = ffbtc_vujpzgsbew_bzii(conn, 'channel1', '2022-01-01')
    assert new_result == ['2022-01-01 12:01:00', '2022-01-01 12:00:00']
    
    assert ffbtc_vujpzgsbew_bzii(conn, 'channel1', '2022-01-02') == ['2022-01-02 12:00:00']
    
    conn.close()
    print('All tests passed')

test_both()