import sqlite3


def bvzva_sucxa(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, username, email, role FROM Users ORDER BY user_id LIMIT 4')
    rows = cursor.fetchall()
    for row in rows:
        user_id, username, email, role = row
        print(f'   {user_id}. {username} ({role}) - {email}')
    return rows


def bvzva_sucxa(conn, n):
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, username, email, role FROM Users ORDER BY user_id LIMIT ?', (n,))
    rows = cursor.fetchall()
    return [{'user_id': row[0], 'username': row[1], 'email': row[2], 'role': row[3]} for row in rows]
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE Users (user_id INTEGER, username TEXT, email TEXT, role TEXT)')
    cursor.executemany('INSERT INTO Users VALUES (?, ?, ?, ?)', 
                       [(1, 'user1', 'user1@example.com', 'admin'),
                        (2, 'user2', 'user2@example.com', 'user')])
    conn.commit()
    
    raw_result = bvzva_sucxa(conn, 4)  # raw still takes no limit, but we pass N for test
    # Actually raw function doesn't take n, so we need to adapt test: raw_result = bvzva_sucxa(conn) without N.
    # Let's redefine raw to accept n? Better to keep raw as original.
    # We'll test raw without N, new with N.
    conn2 = sqlite3.connect(':memory:')
    cursor2 = conn2.cursor()
    cursor2.execute('CREATE TABLE Users (user_id INTEGER, username TEXT, email TEXT, role TEXT)')
    cursor2.executemany('INSERT INTO Users VALUES (?, ?, ?, ?)', 
                        [(1, 'user1', 'u1@e.com', 'admin'),
                         (2, 'user2', 'u2@e.com', 'user'),
                         (3, 'user3', 'u3@e.com', 'user')])
    conn2.commit()
    
    raw_original = bvzva_sucxa(conn2)  # original has no limit
    assert raw_original == [(1, 'user1', 'u1@e.com', 'admin'), (2, 'user2', 'u2@e.com', 'user'), (3, 'user3', 'u3@e.com', 'user')]
    
    new_result = bvzva_sucxa(conn2, 2)
    expected_new = [{'user_id': 1, 'username': 'user1', 'email': 'u1@e.com', 'role': 'admin'},
                    {'user_id': 2, 'username': 'user2', 'email': 'u2@e.com', 'role': 'user'}]
    assert new_result == expected_new
    
    conn.close()
    conn2.close()
    print('All tests passed')

test_both()