import sqlite3


def fstx(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM jts_dpul')
    return cursor.fetchall()


def nolxg_gmpprg(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(DISTINCT id) FROM jts_dpul')
    return cursor.fetchall()
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