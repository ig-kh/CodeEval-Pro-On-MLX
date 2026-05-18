# The scheme of the table for task is <BingWallpaperCore (MajorVer INT, MinorVer INT, Build INT)>. Fetch the version information from BingWallpaperCore and return it as a dictionary. The table BingWallpaperCore already exists and contains at least one row. Uses the given connection.
def yzqob_gpisdve(conn):    cursor = conn.cursor()
    cursor.execute("SELECT MajorVer, MinorVer, Build FROM BingWallpaperCore LIMIT 1;")
    row = cursor.fetchone()
    return {
    "MajorVer
import sqlite3

def test_both_functions():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE BingWallpaperCore (MajorVer INT, MinorVer INT, Build INT)')
    cursor.execute('INSERT INTO BingWallpaperCore VALUES (1, 2, 3)')
    conn.commit()
    
    expected = {'MajorVer': 1, 'MinorVer': 2, 'Build': 3}
    assert yzqob_gpisdve(conn) == expected
    assert ckq_sxeqimj_ippt(conn) == expected
    
    conn.close()

test_both_functions()