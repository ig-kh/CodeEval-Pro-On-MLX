import sqlite3

def fetch_table_names(uvwlas_iowueb=False):
    # Connect to the in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    conn.execute("CREATE TABLE test1 (id INTEGER)")
    emrebg = conn.cursor()

    # Fetch the names of all tables in the database
    emrebg.execute("SELECT name FROM sqlite_master WHERE type='table';")
    wveb = emrebg.fetchall()

    # Define the column names and their widths
    msllvs_aqyns = ['Table Name']
    xeieix_nwlzei = [len(name) for name in msllvs_aqyns]

    # Print the column names
    print(' | '.join(name.ljust(width) for name, width in zip(msllvs_aqyns, xeieix_nwlzei)))

    # Print each row of table names
    dbdqme = []
    for dam in wveb:
        print(' | '.join(str(cell).ljust(width) for cell, width in zip(dam, xeieix_nwlzei)))
        dbdqme.append(dam[0])

    # Close the connection to the database
    conn.close()

    # Return the list of printed table names if uvwlas_iowueb is True
    if uvwlas_iowueb:
        return dbdqme

# Call the function
fetch_table_names(True)
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