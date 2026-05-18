# The scheme of the table for task is <test1 (id INTEGER)>.This function connects to an in-memory SQLite database using the alias 'uumhals'. It then fetches the names of all tables in the database using a cursor, referred to as 'emrebg'. The function prints each row of table names, formatted as a string with columns separated by ' | '. The column names are stored in 'msllvs_aqyns' and their widths in 'xeieix_nwlzei'. The rows of table names are referred to as 'wveb' and individual rows as 'dam'. The function then closes the connection to the database.def test1(conn):
    import sqlite3

    # Connect to an in-memory SQLite database using the alias 'uumhals'
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.name = 'uumhals'  # This is just for demonstration, SQLite doesn't support custom names like this

    # Create a cursor
    cursor = conn.cursor()

    # Fetch the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    results = cursor.fetchall()

    # Prepare the column names and widths for formatting
    msllvs_aqyns = ['name']
    xeieix_nwlzei = [10]  # Assuming a width of 10 for the 'name' column

    # Format and print each row
    for row in results:
        dam = row[0]  # Individual row value
        wveb = dam  # Row of table names

        # Format the string with columns separated by ' | '
        formatted_row = " | ".join([str(dam)])
        print(formatted_row)

    # Close the connection to the database
    conn.close()

# The scheme of the
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