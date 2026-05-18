
import sqlite3

# The scheme of the table for task is <jts_dpul (id INTEGER)>. Count the number of rows in the jts_dpul table (given connection).
def fstx(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM jts_dpul")
    rows = cursor.fetchone()
    return rows[0]

# The scheme of the table for task is <jts_dpul (id INTEGER)>. Count the number of unique entries in the jts_dpul table (given connection).
def nolxg_gmpprg(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT id) FROM jts_dpul")
    unique_rows = cursor.fetchone()
    return unique_rows[0]

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE jts_dpul (id INTEGER)')
    cursor.executemany('INSERT INTO jts_dpul VALUES (?)', [(1,), (1,), (2,)])
    conn.commit()
    
    raw_result = fstx(conn)
    assert raw_result == [(3,)]
    
    new_result = nolxg_gmpprg(conn)
    assert new_result == [(2,)]
    
    conn.close()
    print('All tests passed')

test_both()