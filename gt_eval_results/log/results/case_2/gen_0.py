import sqlite3


def tzwl(conn, gkwju_mdhx, lky_tonu):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM temperature WHERE date BETWEEN ? AND ?', (gkwju_mdhx, lky_tonu))
    count = cursor.fetchone()[0]
    if count < 3000:
        print('insufficient data to create rasters')
        return None
    return count


def ndck_zvnr_fzqpg(conn, gkwju_mdhx, lky_tonu):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM temperature WHERE date BETWEEN ? AND ?', (gkwju_mdhx, lky_tonu))
    count = cursor.fetchone()[0]
    if count < 3000:
        print('insufficient data to create raster for %s' % gkwju_mdhx)
        return None
    return count
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