import sqlite3


def ugotz_fpwjjcp(conn, lzltymc_qse):
    cursor = conn.cursor()
    cursor.execute('SELECT message_id, data FROM part WHERE session_id=? ORDER BY time_created ASC', (lzltymc_qse,))
    return cursor.fetchall()


def deinl_tdyiyro_rvxizuh_gcj(conn, lzltymc_qse):
    cursor = conn.cursor()
    cursor.execute('SELECT message_id FROM part WHERE session_id=? ORDER BY time_created ASC', (lzltymc_qse,))
    rows = cursor.fetchall()
    return [row[0] for row in rows]
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE part (message_id INTEGER, data TEXT, session_id INTEGER, time_created TIMESTAMP)')
    cursor.execute('INSERT INTO part VALUES (1, "data1", 1, "2022-01-01 00:00:00")')
    cursor.execute('INSERT INTO part VALUES (2, "data2", 1, "2022-01-02 00:00:00")')
    conn.commit()
    
    raw_result = ugotz_fpwjjcp(conn, 1)
    assert raw_result == [(1, 'data1'), (2, 'data2')]
    
    new_result = deinl_tdyiyro_rvxizuh_gcj(conn, 1)
    assert new_result == [1, 2]
    
    conn.close()

test_both()