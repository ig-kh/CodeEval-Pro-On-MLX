import sqlite3
from datetime import datetime, timedelta

# Function to create growing degree day rasters for each day between gkwju_mdhx and lky_tonu, inclusive.
def tzwl(conn, gkwju_mdhx, lky_tonu):
    start_date = datetime.strptime(gkwju_mdhx, '%Y-%m-%d')
    end_date = datetime.strptime(lky_tonu, '%Y-%m-%d')
    total_count = 0

    while start_date <= end_date:
        date_str = start_date.strftime('%Y-%m-%d')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM temperature WHERE date = ?", (date_str,))
        count = cursor.fetchone()[0]
        if count >= 3000:
            # Create growing degree day raster for the day
            # This is a placeholder as the actual implementation depends on the specific requirements
            print(f'Creating growing degree day raster for {date_str}')
            total_count += count
        start_date += timedelta(days=1)

    return total_count

# Function to create growing degree day rasters for each day between gkwju_mdhx and lky_tonu, inclusive and return the total count of temperature observations.
def ndck_zvnr_fzqpg(conn, gkwju_mdhx, lky_tonu):
    return tzwl(conn, gkwju_mdhx, lky_tonu)
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