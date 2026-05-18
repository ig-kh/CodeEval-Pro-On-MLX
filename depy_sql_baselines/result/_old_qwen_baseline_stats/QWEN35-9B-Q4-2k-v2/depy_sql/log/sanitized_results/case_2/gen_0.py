# The scheme of the table for task is <temperature (date TEXT, temp REAL)>. Create growing degree day rasters for each day between gkwju_mdhx and lky_tonu, inclusive. Dates in YYYY-MM-DD. Only if at least 3000 temperature observations available for each day. Returns total count for range (given connection).
def tzwl(conn, gkwju_mdhx, lky_tonu):    """
    Creates growing degree day rasters for each day between gkwju_mdhx and lky_tonu, inclusive.
    Dates in YYYY-MM-DD. Only if at least 3000 temperature observations available for each day.
    Returns total count for range (given connection).
    
    Args:
    conn: Database connection object.
    gkwju_mdhx: Start date in YYYY-MM-DD format.
    lky_tonu: End date in YYYY-MM-DD format.
    
    Returns:
    A dictionary containing the total count of temperature observations for the given date range.
    """
    start_date = gkwju_mdhx.replace('-', '')
    end_date = lky_tonu.replace('-', '')
    
    query = """
    SELECT COUNT(*) as total_count
    FROM temperature
    WHERE date BETWEEN ? AND ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (start_date, end_date))
    result = cursor.fetchone()
    cursor.close()
    
    return {'total_count': result[0] if result else 0}

# The scheme of the table for task is <temperature (date TEXT, temp REAL)>. Create growing degree
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE temperature (date TEXT, temp REAL)')
    for i in range(3000):
        cursor.execute('INSERT INTO temperature VALUES (?, ?)', ('2022-01-01', i))
    conn.commit()
    
    raw_result = tzwl(conn, '2022-01-01', '2022-01-01')
    assert raw_result == 3000
    
    new_result = ndck_zvnr_fzqpg(conn, '2022-01-01', '2022-01-01')
    assert new_result == 3000
    
    # Test insufficient data
    assert tzwl(conn, '2022-01-02', '2022-01-02') is None
    assert ndck_zvnr_fzqpg(conn, '2022-01-02', '2022-01-02') is None
    
    conn.close()
    print('All tests passed')

test_both()