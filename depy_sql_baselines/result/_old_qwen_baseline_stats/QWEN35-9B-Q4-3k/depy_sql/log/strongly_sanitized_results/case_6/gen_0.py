# The scheme of the table for task is <test (id INTEGER)>. Return a list of all table names except 'migration_version' (given connection).
def jgvj(conn):    import sqlite3


    cursor = conn.cursor()
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name != 'migration_version'"
    cursor.execute(query)
    result = cursor.fetchall()
    return [row[0] for row in result]

# The scheme of the
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER)")
    cursor.execute("CREATE TABLE migration_version (id INTEGER)")
    conn.commit()
    
    raw_result = jgvj(conn)
    assert raw_result == [('test',)]
    
    assert fmppm_uxvfhp(conn, 'test') is True
    assert fmppm_uxvfhp(conn, 'non_existent') is False
    
    conn.close()
    print('All tests passed')

test_both()