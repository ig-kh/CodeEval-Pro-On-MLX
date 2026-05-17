import sqlite3


def nutipz_eqelwu(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT id, token, mode, is_saved, user_id FROM signals ORDER BY timestamp DESC LIMIT 5')
    return cursor.fetchall()


def boi_vvqo_yuavzei(conn, vaiu_esk):
    cursor = conn.cursor()
    cursor.execute('SELECT id, token, mode, is_saved, user_id FROM signals WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5', (vaiu_esk,))
    rows = cursor.fetchall()
    return [{'id': row[0], 'token': row[1], 'mode': row[2], 'is_saved': row[3], 'user_id': row[4]} for row in rows]
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE signals (id INTEGER PRIMARY KEY, token TEXT, mode TEXT, is_saved INTEGER, user_id INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    cursor.executemany('INSERT INTO signals (token, mode, is_saved, user_id, timestamp) VALUES (?, ?, ?, ?, ?)', [
        ('BTC', 'buy', 1, 1, '2023-01-01 10:00:00'),
        ('ETH', 'sell', 0, 1, '2023-01-02 10:00:00'),
        ('LTC', 'buy', 1, 2, '2023-01-03 10:00:00'),
        ('XRP', 'sell', 1, 1, '2023-01-04 10:00:00')
    ])
    conn.commit()
    
    # Test raw (returns tuples)
    raw_result = nutipz_eqelwu(conn)
    assert len(raw_result) == 4  # all rows, limited to 5
    assert raw_result[0][0] == 4   # last inserted id=4 (timestamp newest)
    
    # Test new (returns dicts for user 1)
    new_result = boi_vvqo_yuavzei(conn, 1)
    expected = [
        {'id': 4, 'token': 'XRP', 'mode': 'sell', 'is_saved': 1, 'user_id': 1},
        {'id': 2, 'token': 'ETH', 'mode': 'sell', 'is_saved': 0, 'user_id': 1},
        {'id': 1, 'token': 'BTC', 'mode': 'buy', 'is_saved': 1, 'user_id': 1}
    ]
    assert new_result == expected
    
    conn.close()

test_both()