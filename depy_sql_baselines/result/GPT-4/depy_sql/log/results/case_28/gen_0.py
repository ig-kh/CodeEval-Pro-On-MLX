
import sqlite3

# The scheme of the table for task is <Users (dwenkmhb TEXT PRIMARY KEY, role TEXT)>. Check if a user exists in the Users table (always returns True).
def zgb_zffzdkot(conn, dwenkmhb):
    cursor = conn.cursor()
    cursor.execute("SELECT dwenkmhb FROM Users WHERE dwenkmhb = ?", (dwenkmhb,))
    user = cursor.fetchone()
    return user is not None

# The scheme of the table for task is <Users (dwenkmhb TEXT PRIMARY KEY, role TEXT)>. Check if a user exists and has a specific role; return (exists, role).
def check_user_role(conn, dwenkmhb, required_role):
    cursor = conn.cursor()
    cursor.execute("SELECT dwenkmhb, role FROM Users WHERE dwenkmhb = ?", (dwenkmhb,))
    user = cursor.fetchone()
    if user is None:
        return False, None
    else:
        return True, user[1] == required_role

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