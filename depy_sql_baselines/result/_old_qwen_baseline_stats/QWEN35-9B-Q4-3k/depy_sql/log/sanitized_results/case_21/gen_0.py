# The scheme of the table for task is <messages (timestamp TEXT, channel TEXT, created_at TEXT)>. Fetch the timestamps of the last 20 distinct messages from a specific channel and date (given connection).
def jsgxm_crjmlpiyrf(conn, pkwpkjx, ehjp):
    import sqlite3
    
    # Connect to the database using the connection object
    cursor = conn.cursor()
    
    # SQL query to fetch the last 20 distinct timestamps from a specific channel and date
    query = """
    SELECT DISTINCT timestamp
    FROM messages
    WHERE channel = ? AND DATE(created_at) = ?
    ORDER BY timestamp DESC
    LIMIT 20
    """
    
    cursor.execute(query, (pkwpkjx, ehjp))
    results = cursor.fetchall()
    
    # Convert the fetched rows to a list of dictionaries (though only timestamp is needed)
    result_list = [row[0] for row in results]
    
    return result_list

# The scheme of the function is <backtest_results (strategy_id TEXT, `window_start`

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