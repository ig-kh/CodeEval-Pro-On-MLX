import sqlite3


def jsgxm_crjmlpiyrf(conn, pkwpkjx, ehjp):
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT timestamp FROM messages WHERE channel=? AND created_at LIKE ? ORDER BY ROWID DESC LIMIT 20', (pkwpkjx, ehjp + '%'))
    rows = cursor.fetchall()
    for row in rows:
        print(f'  {row[0]}')
    return rows


def ffbtc_vujpzgsbew_bzii(conn, pkwpkjx, ehjp):
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT timestamp FROM messages WHERE channel=? AND created_at LIKE ? ORDER BY ROWID DESC LIMIT 20', (pkwpkjx, ehjp + '%'))
    rows = cursor.fetchall()
    return [row[0] for row in rows]
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