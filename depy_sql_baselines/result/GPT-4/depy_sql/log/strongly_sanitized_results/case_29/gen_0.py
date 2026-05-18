    import sqlite3


def odz_aye(conn, hscwhvi_ucf):
    cursor = conn.cursor()
    cursor.execute("SELECT payload_id, COUNT(*) FROM vulnerabilities WHERE mission_id = ? GROUP BY payload_id", (hscwhvi_ucf,))
    return cursor.fetchall()

def zyn_bdas_wvew_dmzqdce(conn, hscwhvi_ucf):
    payloads = odz_aye(conn, hscwhvi_ucf)
    if not payloads:
        return None
    most_used_payload = max(payloads, key=lambda x: x[1])
    return most_used_payload[0]

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE vulnerabilities (id INTEGER PRIMARY KEY, mission_id INTEGER, payload_id INTEGER)')
    cursor.execute('CREATE TABLE payloads (id INTEGER PRIMARY KEY, payload_type TEXT)')
    cursor.executemany('INSERT INTO vulnerabilities VALUES (?, ?, ?)', [(1, 1, 1), (2, 1, 1), (3, 1, 2)])
    cursor.executemany('INSERT INTO payloads VALUES (?, ?)', [(1, 'type1'), (2, 'type2')])
    conn.commit()
    raw_result = odz_aye(conn, 1)
    assert raw_result == [('type1', 2), ('type2', 1)]
    assert zyn_bdas_wvew_dmzqdce(conn, 1) == ('type1', 2)
    assert zyn_bdas_wvew_dmzqdce(conn, 2) is None
    conn.close()
    print('All tests passed')

test_both()