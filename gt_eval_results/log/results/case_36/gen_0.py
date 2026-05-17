import sqlite3


def vegmfu_clydy(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='Process_Time_079_avgday')")
    exists = cursor.fetchone()[0] == 1
    return exists


def tjbcw_qqxbcu_ktkxi(conn, wiwli_szkiw):
    cursor = conn.cursor()
    result = {}
    for table_name in wiwli_szkiw:
        cursor.execute("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)", (table_name,))
        result[table_name] = cursor.fetchone()[0] == 1
    return result
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE Process_Time_079_avgday (id INT)")
    cursor.execute("CREATE TABLE other_table (name TEXT)")
    conn.commit()

    assert vegmfu_clydy(conn) is True
    
    tables = ['Process_Time_079_avgday', 'other_table', 'missing']
    expected = {'Process_Time_079_avgday': True, 'other_table': True, 'missing': False}
    assert tjbcw_qqxbcu_ktkxi(conn, tables) == expected
    
    conn.close()

test_both()