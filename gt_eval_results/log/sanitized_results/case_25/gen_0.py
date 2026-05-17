import sqlite3


def juk_szlkn_zkz(conn, sok_fmddhf):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS c FROM events WHERE event = 'error' AND ts LIKE ?", (f'{sok_fmddhf}%',))
    veaq = cursor.fetchall()
    return veaq


def vir_trwqc_rii_pxupzhbw(conn, sok_fmddhf):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS c FROM events WHERE (event = 'error' OR event = 'warning') AND ts LIKE ?", (f'{sok_fmddhf}%',))
    veaq = cursor.fetchall()
    return veaq
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