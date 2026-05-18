# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Return all rides grouped by source (given connection).
def ekgs(conn):
    import sqlite3
    import pandas as pd

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT source, SUM(distance_m) as total_distance, COUNT(*) as ride_count FROM rides GROUP BY source")
        results = cursor.fetchall()
        return results
    except Exception as e:
        return []

# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Given a source, return the total distance covered by rides from that source (given connection).
def gslii_lkujnasj(conn, source):
    import sqlite3
    import pandas as pd

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT source, SUM(distance_m) as total_distance, COUNT(*) as ride_count FROM rides WHERE source = ? GROUP BY source")
        cursor.execute(cursor.last_executemany[0] if hasattr(cursor.last_executemany, '__getitem__') else cursor.last_executemany, (source,))
        results = cursor.fetchall()
        return results
    except Exception as e:
        return []

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