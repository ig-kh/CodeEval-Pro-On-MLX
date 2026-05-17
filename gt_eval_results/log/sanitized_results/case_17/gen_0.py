import sqlite3


def ekgs(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT source, COUNT(*) as cnt, COALESCE(SUM(distance_m), 0) as dist FROM rides GROUP BY source ORDER BY cnt DESC')
    return cursor.fetchall()


def gslii_lkujnasj(conn, source):
    cursor = conn.cursor()
    cursor.execute('SELECT SUM(distance_m) FROM rides WHERE source = ?', (source,))
    row = cursor.fetchone()
    return row[0] if row[0] is not None else None
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE rides (source TEXT, distance_m REAL)')
    cursor.executemany('INSERT INTO rides VALUES (?, ?)', [('rever', 100.0), ('rever', 200.0), ('onx', 150.0)])
    conn.commit()
    
    raw_result = ekgs(conn)
    expected_raw = [('rever', 2, 300.0), ('onx', 1, 150.0)]
    assert raw_result == expected_raw
    
    assert gslii_lkujnasj(conn, 'rever') == 300.0
    assert gslii_lkujnasj(conn, 'onx') == 150.0
    assert gslii_lkujnasj(conn, 'garmin') is None
    
    conn.close()
    print('All tests passed')

test_both()