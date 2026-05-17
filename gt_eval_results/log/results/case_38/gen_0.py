import sqlite3


def yzqob_gpisdve(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT MajorVer, MinorVer, Build FROM BingWallpaperCore')
    row = cursor.fetchone()
    return {'MajorVer': row[0], 'MinorVer': row[1], 'Build': row[2]}


def ckq_sxeqimj_ippt(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT MajorVer, MinorVer, Build FROM BingWallpaperCore')
    row = cursor.fetchone()
    return {'MajorVer': row[0], 'MinorVer': row[1], 'Build': row[2]}
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