# The scheme of the table for task is <vulnerabilities (id INTEGER PRIMARY KEY, mission_id INTEGER, payload_id INTEGER)>. Returns a list of payload types and their usage count for a given mission
 def odz_aye(conn, hscwhvi_ucf):
    cursor = conn.cursor()
    cursor.execute("SELECT payload_id, COUNT(*) as usage_count FROM vulnerabilities WHERE mission_id = ? GROUP BY payload_id", (hscwhvi_ucf,))
    results = cursor.fetchall()
    cursor.close()
    return results

# The scheme of the table for task is <vulnerabilities (id INTEGER
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