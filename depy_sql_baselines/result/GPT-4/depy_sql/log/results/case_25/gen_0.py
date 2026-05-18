
import sqlite3

# The scheme of the table for task is <events (event TEXT, ts TEXT)>. Count the number of 'error' events for a specified day prefix. 
def juk_szlkn_zkz(conn, sok_fmddhf):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM events WHERE event='error' AND ts LIKE ?", (sok_fmddhf+'%',))
    result = cursor.fetchone()
    return result[0] if result else 0

# The scheme of the table for task is <events (event TEXT, ts TEXT)>. Count the number of 'error' and 'warning' events for a specified day prefix. 
def vir_trwqc_rii_pxupzhbw(conn, sok_fmddhf):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM events WHERE (event='error' OR event='warning') AND ts LIKE ?", (sok_fmddhf+'%',))
    result = cursor.fetchone()
    return result[0] if result else 0

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE events (event TEXT, ts TEXT)')
    cursor.executemany('INSERT INTO events VALUES (?, ?)', [('error', '2022-01-01 10:00:00'), ('warning', '2022-01-01 11:00:00'), ('info', '2022-01-01 12:00:00')])
    conn.commit()
    raw_result = juk_szlkn_zkz(conn, '2022-01-01')
    assert raw_result == [(1,)]
    new_result = vir_trwqc_rii_pxupzhbw(conn, '2022-01-01')
    assert new_result == [(2,)]
    conn.close()
    print('All tests passed')

test_both()