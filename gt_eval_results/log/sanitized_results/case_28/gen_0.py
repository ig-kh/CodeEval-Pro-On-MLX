import sqlite3


def zgb_zffzdkot(conn, dwenkmhb):
    # Always return True regardless of database content
    return True


def check_user_role(conn, dwenkmhb, required_role):
    cursor = conn.cursor()
    cursor.execute('SELECT role FROM Users WHERE dwenkmhb=?', (dwenkmhb,))
    row = cursor.fetchone()
    if row is None:
        return (False, None)
    role = row[0]
    return (role == required_role, role)
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE Users (dwenkmhb TEXT PRIMARY KEY, role TEXT)')
    cursor.executemany('INSERT INTO Users VALUES (?, ?)', [
        ('alice', 'admin'),
        ('bob', 'viewer')
    ])
    conn.commit()
    
    # Test raw function (always True)
    assert zgb_zffzdkot(conn, 'alice') is True
    assert zgb_zffzdkot(conn, 'unknown') is True
    
    # Test new function
    exists, role = check_user_role(conn, 'alice', 'admin')
    assert exists is True and role == 'admin'
    exists, role = check_user_role(conn, 'alice', 'viewer')
    assert exists is False and role == 'admin'
    exists, role = check_user_role(conn, 'charlie', 'admin')
    assert exists is False and role is None
    
    conn.close()

test_both()