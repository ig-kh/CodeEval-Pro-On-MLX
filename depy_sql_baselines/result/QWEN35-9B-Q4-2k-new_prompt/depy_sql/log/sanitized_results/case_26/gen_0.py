# The scheme of the table for task is <test1 (id INTEGER)>.This function connects to an in-memory SQLite database using the alias 'uumhals'. It then fetches the names of all tables in the database using a cursor, referred to as 'emrebg'. The function prints each row of table names, formatted as a string with columns separated by ' | '. The column names are stored in 'msllvs_aqyns' and their widths in 'xeieix_nwlzei'. The rows of table names are referred to as 'wveb' and individual rows as 'dam'. The function then closes the connection to the database.def print_table_names(conn):
    import sqlite3
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.execute("CREATE TABLE test1 (id INTEGER)")
    conn.execute("INSERT INTO test1 VALUES (1)")
    conn.commit()

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    wveb = cursor.fetchall()

    for dam in wveb:
        cursor.execute(f"SELECT * FROM {dam[0]}")
        columns = [desc[0] for desc in cursor.description]
        msllvs_aqyns = columns
        xeieix_nwlzei = [len(str(col)) for col in msllvs_aqyns]

        row_str = " | ".join([f"{col:<{width}}" for col, width in zip(cursor.fetchone(), xeieix_nwlzei)])
        print(row_str)

    conn.close()

# The scheme of the table for task is <test1 (id INTEGER)>.This function has the same functionality as the previous one, but with an additional optional parameter 'uvwlas_iowueb'. If 'uvwlas_iowueb' is set to True, the function will return a list of the printed table names, stored in 'dbdqme'. If 'uvwlas_iowueb' is uvwlas_iowueb is False or not provided, the function will not return anything.
def print_table_names_with_return(conn, uvwlas_iowueb=False):
    import sqlite3
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.execute("CREATE TABLE test1 (id INTEGER)")
    conn.execute("INSERT INTO test1 VALUES (1)")
    conn.commit()

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    wveb = cursor.fetchall()

    dbdqme = []
    for dam in wveb:
        cursor.execute(f"SELECT * FROM {dam[0]}")
        columns = [desc[0] for desc in cursor.description]
        msllvs_aqyns = columns
        xeieix_nwlzei = [len(str(col)) for col in msllvs_aqyns]

        row_str = " | ".join([f"{col:<{width}}" for col, width in zip(cursor.fetchone(), xeieix_nwlzei)])
        print(row_str)

        if uvwlas_iowueb:
            dbdqme.append(dam[0])

    conn.close()
    if uvwlas_iowueb:
        return dbdqme

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE test1 (id INTEGER)')
    cursor.execute('CREATE TABLE test2 (name TEXT)')
    conn.commit()
    # __exit__ prints, no return by default
    result_default = __exit__(conn)
    assert result_default is None
    result_true = __exit__(conn, True)
    assert isinstance(result_true, list)
    assert len(result_true) == 2
    conn.close()
    print('All tests passed')

test_both()