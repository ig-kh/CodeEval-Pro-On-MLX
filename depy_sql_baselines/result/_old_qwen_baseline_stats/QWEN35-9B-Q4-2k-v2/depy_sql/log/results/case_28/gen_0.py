# The scheme of the table for task is <Users (dwenkmhb TEXT PRIMARY KEY, role TEXT)>. Check if a user exists in the Users table (always returns True).
def zgb_zffzdkot(conn, dwenkmhb):    """
    Check if a user exists in the Users table.
    
    Args:
    conn: Database connection object
    dwenkmhb: Username to check
    
    Returns:
    bool: True if user exists
    """
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM Users WHERE dwenkmhb = ?", (dwenkmhb,))
    result = cursor.fetchone()
    cursor.close()
    return result is not None

# The scheme of the user role table is <Users (dwenkmhb TEXT PRIMARY KEY, role TEXT)>. Check if a user exists and has a specific role; return (exists, role).
def check_user_role(conn, dwenkmhb, required_role):
    """
    Check if a user exists and has a specific role.
    
    Args:
    conn: Database connection object
    dwenkmhb: Username to check
    required_role: Role to check
    
    Returns:
    tuple: (exists, role) where exists is a boolean and role is the role string if found, else None
    """
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM Users WHERE dwenkmhb = ? AND role = ?", (dwenkmhb, required_role))
    result = cursor.fetchone()
    cursor.close()
    exists = result is not None
    return (exists, result[0] if exists else None)

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