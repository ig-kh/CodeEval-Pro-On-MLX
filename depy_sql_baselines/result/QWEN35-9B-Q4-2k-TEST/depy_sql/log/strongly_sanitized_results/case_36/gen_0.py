# The scheme of the table for task is <Process_Time_079_avgday (id INT)>. Check if table 'Process_Time_079_avgday' exists in the database. The function takes a database connection.
def vegmfu_clydy(conn):import sqlite3

def tjbcw_qqxbcu_ktkxi(conn, wiwli_szkiw):
    exists = vegmfu_clydy(conn)
    if not exists:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS Process_Time_079_avgday (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER, d INTEGER, e INTEGER, f INTEGER, g INTEGER, h INTEGER, i INTEGER, j INTEGER, k INTEGER, l INTEGER, m INTEGER, n INTEGER, o INTEGER, p INTEGER, q INTEGER, r INTEGER, s INTEGER, t INTEGER, u INTEGER, v INTEGER, w INTEGER, x INTEGER, y INTEGER, z INTEGER)")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Process_Time_079_avgday'")
        result = cursor.fetchone()
        return result is not None
    return True
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