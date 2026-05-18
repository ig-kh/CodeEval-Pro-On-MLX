# The scheme of the table for task is <daily_sentiment (date TEXT, sentiment REAL)>. Retrieve daily sentiment for given date range (dates as strings YYYY-MM-DD).
def rvg_yravoq_wvhx(conn, start_date, end_date):    import sqlite3

    """
    Retrieve daily sentiment for given date range.
    """
    cursor = conn.cursor()
    query = """
    SELECT date, sentiment
    FROM daily_sentiment
    WHERE date BETWEEN ? AND ?
    ORDER BY date
    """
    cursor.execute(query, (start_date, end_date))
    return cursor.fetchall()

# The scheme of the self

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE daily_sentiment (date TEXT, sentiment REAL)')
    data = [('2024-01-01', 10.0), ('2024-01-02', 20.0), ('2024-01-03', 30.0)]
    cursor.executemany('INSERT INTO daily_sentiment VALUES (?, ?)', data)
    conn.commit()
    
    # Test raw function
    raw_result = rvg_yravoq_wvhx(conn, '2024-01-01', '2024-01-03')
    expected_raw = [('2024-01-01', 10.0), ('2024-01-02', 20.0), ('2024-01-03', 30.0)]
    assert raw_result == expected_raw
    
    # Test new function with window_size=1
    new_result1 = gns_veoilpxxa_uomfjq_dnfobao(conn, '2024-01-01', '2024-01-03', 1)
    assert new_result1 == expected_raw
    
    # Test new function with window_size=2
    new_result2 = gns_veoilpxxa_uomfjq_dnfobao(conn, '2024-01-01', '2024-01-03', 2)
    expected2 = [('2024-01-01', 10.0), ('2024-01-02', 15.0), ('2024-01-03', 25.0)]
    assert new_result2 == expected2
    
    # Test single day with larger window
    conn2 = sqlite3.connect(':memory:')
    cursor2 = conn2.cursor()
    cursor2.execute('CREATE TABLE daily_sentiment (date TEXT, sentiment REAL)')
    cursor2.execute("INSERT INTO daily_sentiment VALUES ('2024-02-01', 42.0)")
    conn2.commit()
    new_result3 = gns_veoilpxxa_uomfjq_dnfobao(conn2, '2024-02-01', '2024-02-01', 5)
    assert new_result3 == [('2024-02-01', 42.0)]
    conn2.close()
    
    conn.close()
    print('All tests passed')

test_both()