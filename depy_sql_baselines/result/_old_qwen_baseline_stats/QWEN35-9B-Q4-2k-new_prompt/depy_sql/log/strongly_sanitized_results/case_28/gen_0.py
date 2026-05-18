# The scheme of the table for task is <Users (dwenkmhb TEXT PRIMARY KEY, role TEXT)>. Check if a user exists in the Users table (always returns True).
def zgb_zffzdkot(conn, dwenkmhb):import sqlite3

cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM Users WHERE dwenkmhb = ?", (dwenkmhb,))
    result = cursor.fetchone()
    cursor.close()
    return result is not None

# The scheme of the table for task is <Users (dwenkm
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