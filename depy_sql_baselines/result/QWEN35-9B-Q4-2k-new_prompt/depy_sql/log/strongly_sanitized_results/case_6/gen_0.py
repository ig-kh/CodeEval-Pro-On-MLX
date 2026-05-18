# The scheme of the table for task is <test (id INTEGER)>. Return a list of all table names except 'migration_version' (given connection).
def jgvj(conn):import sqlite3

try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        all_tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return [table for table in all_tables if table != 'migration_version']
    except Exception as e:
        return None

# The scheme of
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