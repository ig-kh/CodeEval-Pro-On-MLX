# The scheme of the table for task is <Process_Time_079_avgday (id INT)>. Check if table 'Process_Time_079_avgday' exists in the database. The function takes a database connection.
def vegmfu_clydy(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Process_Time_079_avgday'")
        result = cursor.fetchall()
        return len(result) > 0
    except Exception:
        return False

# The scheme of the table for task is <Process if table 'Process_Time_079_avgday' exists in the database. The function takes a database connection.
def tjbcw_qqxbcu_ktkxi(conn, wiwli_szkiw):
    if vegmfu_clydy(conn):
        return {"Process_Time_079_avgday": True}
    else:
        return {}

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