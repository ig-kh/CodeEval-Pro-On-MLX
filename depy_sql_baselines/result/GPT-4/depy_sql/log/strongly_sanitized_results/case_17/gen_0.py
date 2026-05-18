import sqlite3
from typing import List, Tuple

# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Return all rides grouped by source (given connection).
def ekgs(conn: sqlite3.Connection) -> List[Tuple[str, float]]:
    cursor = conn.cursor()
    cursor.execute("SELECT source, SUM(distance_m) FROM rides GROUP BY source")
    return cursor.fetchall()

# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Given a source, return the total distance covered by rides from that source (given connection).
def gslii_lkujnasj(conn: sqlite3.Connection, source: str) -> float:
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(distance_m) FROM rides WHERE source = ?", (source,))
    result = cursor.fetchone()
    return result[0] if result else 0
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