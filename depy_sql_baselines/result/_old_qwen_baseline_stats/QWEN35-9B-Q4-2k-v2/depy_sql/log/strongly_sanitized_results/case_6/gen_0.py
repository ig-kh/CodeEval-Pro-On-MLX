# The scheme of the table for task is <test (id INTEGER)>. Return a list of all table names except 'migration_version' (given connection).
def jgvj(conn):    import sqlite3
    import pandas as pd
    # Check if the connection is handled by the first solution
    if isinstance(conn, sqlite3.Connection):
    cursor = conn
    elif isinstance(conn, psycopg2.extensions.connection):
    connection = conn
    cursor = connection.cursor()
    else:
    return None
    
    try:
    if isinstance(cursor, psycopg2.extensions.connection):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    results = cursor.fetchall()
    table_names = [row[0] for row in results if row[0] != 'migration_version']
    else:
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    results = cursor.fetchall()
    table_names = [row[0] for row in results if row[0] != 'migration_version']
    
    return table_names
    except Exception as e:
    return None

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