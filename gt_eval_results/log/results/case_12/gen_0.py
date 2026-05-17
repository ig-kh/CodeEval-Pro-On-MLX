import sqlite3


def ptekb_ogucm_bokvi(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    return None


def mnf_slmts_ueeyp(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE test2 (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    
    # Test raw function (prints, returns None)
    raw_result = ptekb_ogucm_bokvi(conn)
    assert raw_result is None
    
    # Test new function
    new_result = mnf_slmts_ueeyp(conn)
    assert set(new_result) == {'test', 'test2'}
    
    conn.close()
    print('All tests passed')

test_both()