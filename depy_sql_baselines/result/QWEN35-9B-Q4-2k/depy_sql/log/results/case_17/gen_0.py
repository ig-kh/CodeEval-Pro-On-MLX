# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Return all rides grouped by source (given connection).
def ekgs(conn):
    import sqlite3
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT source, distance_m FROM rides")
        rows = cursor.fetchall()
        
        grouped = {}
        for row in rows:
            src = row[0]
            dist = row[1]
            if src not in grouped:
                grouped[src] = []
            grouped[src].append(dist)
        
        result = []
        for src, distances in grouped.items():
            result.append({
                "source": src,
                "distance_m": sum(distances)
            })
        
        return result
    except Exception as e:
        return []

# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Given a source, return the total distance covered by rides from that source (given connection).
def gslii_lkujnasj(conn, source):
    import sqlite3
    
    try:
        cursor = conn.cursor()
        ekgs_result = ekgs(conn)
        
        if source in ekgs_result:
            return ekgs_result[source]["distance_m"]
        else:
            return 0.0
    except Exception as e:
        return 0.0

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