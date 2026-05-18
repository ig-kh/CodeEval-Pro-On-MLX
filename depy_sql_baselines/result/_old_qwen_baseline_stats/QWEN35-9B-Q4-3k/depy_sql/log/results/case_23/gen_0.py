# The scheme of the table for task is <Users (user_id INTEGER, username TEXT, email TEXT, role TEXT)>. Fetch the first 4 users from the Users table and print their details (given connection).
def bvzva_sucxa(conn):
    import sqlite3
    
    # Connect to the database using the provided connection object
    cursor = conn.cursor()
    
    # SQL query to fetch the first 4 users from the Users table
    query = """
    SELECT user_id, username, email, role
    FROM Users
    ORDER BY user_id ASC
    LIMIT 4
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Print the fetched rows
    for row in results:
        print(row)
    
    return results

# The scheme of the
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