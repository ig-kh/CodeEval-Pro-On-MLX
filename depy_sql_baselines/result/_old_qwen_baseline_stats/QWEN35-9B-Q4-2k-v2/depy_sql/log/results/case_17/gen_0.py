# The scheme of the table for task is <rides (source TEXT, distance_m REAL)>. Return all rides grouped by source (given connection).
def ekgs(conn):    import sqlite3
    import json
    from collections import Counter
    
    # Connect to the database
    cursor = conn.cursor()
    
    # Query to get all rides grouped by source
    query = """
    SELECT 
    source,
    COUNT(*) as ride_count,
    SUM(distance_m) as total_distance
    FROM rides
    GROUP BY source
    ORDER BY total_distance DESC
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Convert results to list of dictionaries
    ride_stats = [dict(zip(['source', 'ride_count', 'total_distance'], row)) for row in results]
    
    return ride_stats

# The scheme of the
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