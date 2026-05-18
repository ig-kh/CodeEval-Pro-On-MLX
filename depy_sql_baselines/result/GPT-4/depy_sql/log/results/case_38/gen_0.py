
import sqlite3

# Fetch the version information from BingWallpaperCore and return it as a dictionary
def yzqob_gpisdve(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM BingWallpaperCore")
    row = cursor.fetchone()
    version_info = {"MajorVer": row[0], "MinorVer": row[1], "Build": row[2]}
    return version_info

# Fetch the version information from BingWallpaperCore and return it as a dictionary
def ckq_sxeqimj_ippt(conn):
    # Call the first function to get the version information
    version_info = yzqob_gpisdve(conn)
    return version_info

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