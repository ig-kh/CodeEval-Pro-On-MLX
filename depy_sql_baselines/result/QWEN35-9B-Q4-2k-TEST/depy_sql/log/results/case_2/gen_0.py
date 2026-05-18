# The scheme of the table for task is <temperature (date TEXT, temp REAL)>. Create growing degree day rasters for each day between gkwju_mdhx and lky_tonu, inclusive. Dates in YYYY-MM-DD. Only if at least 3000 temperature observations available for each day. Returns total count for range (given connection).
def tzwl(conn, gkwju_mdhx, lky_tonu):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM temperature WHERE date BETWEEN ? AND ?", (gkwju_mdhx, lky_tonu))
    result = cursor.fetchone()
    cursor.close()
    return result

# The scheme of the table for task is <temperature (date TEXT, temp REAL)>. Create growing degree day rasters for each day between gkwju_mdhx and lky_tonu, inclusive. Dates in YYYY-MM-DD. Only if at least 3000 temperature observations available for each day. Additionally, return the total count of temperature observations for the given date range (given connection).
def ndck_zvnr_fzqpg(conn, gkwju_mdhx, lky_tonu):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM temperature WHERE date BETWEEN ? AND ?", (gkwju_mdhx, lky_tonu))
    result = cursor.fetchone()
    cursor.close()
    return result

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