
# Importing necessary module
import sqlite3

# The scheme of the table for task is <test (id INTEGER)>. Return a list of all table names except 'migration_version' (given connection).
def jgvj(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # Exclude 'migration_version' from the list
    tables = [table[0] for table in tables if table[0] != 'migration_version']
    return tables

# The scheme of the table for task is <test (id INTEGER)>. Given a table name, check if it exists in the database (given connection).
def fmppm_uxvfhp(conn, table_name):
    tables = jgvj(conn)
    return table_name in tables

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