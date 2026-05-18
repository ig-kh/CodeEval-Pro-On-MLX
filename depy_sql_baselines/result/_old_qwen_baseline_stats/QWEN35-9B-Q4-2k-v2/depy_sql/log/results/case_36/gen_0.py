# The scheme of the table for task is <Process_Time_079_avgday (id INT)>. Check if table 'Process_Time_079_avgday' exists in the database. The function takes a database connection.
def vegmfu_clydy(conn):    import sqlite3
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Process_Time_079_avgday';")
    result = cursor.fetchone()
    return result is not None

# The scheme of the table for task is <Process_Time_
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