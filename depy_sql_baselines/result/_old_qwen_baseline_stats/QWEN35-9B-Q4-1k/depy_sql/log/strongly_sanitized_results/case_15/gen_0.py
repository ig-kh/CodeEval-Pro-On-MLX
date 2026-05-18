# The scheme of the table for task is <part (message_id INTEGER, data TEXT, session_id INTEGER, time_created TIMESTAMP)>. For OpenCode, given a database connection and a session ID, return all (message_id, data) tuples ordered by time_created ASC.
def ugotz_fpwjjcp(conn, lzltymc_qse):    import sqlite3
    from typing import List


    """
    For OpenCode, given a database connection and a session ID, return all (message_id, data) tuples ordered by time_created ASC.

    Args:
        conn: SQLite connection object
        lzltymc_qse: Session ID (INTEGER)

    Returns:
        List of tuples (message_id, data) ordered by time_created ASC
    """
    query = f"""
        SELECT message_id, data
        FROM part
        WHERE session_id = ?
        ORDER BY time_created ASC
    """
    cursor = conn.cursor()
    cursor.execute(query, (lzltymc_qse,))
    return cursor.fetchall()

# The scheme of the second problem is <part (message_id INTEGER, data TEXT, session_id INTEGER, time_created TIMESTAMP)>. For OpenCode, given a database connection and a session ID, return only the message_id values (as a list).
def deinl_tdyiyro_rvxizuh_gcj(conn, lzltymc_qse):
    """
    For OpenCode, given a database connection and a session ID, return only the message_id values (as a list).

    Args:
        conn: SQLite connection data TEXT, session_id INTEGER, time_created TIMESTAMP)>. For OpenCode, given a database connection and a session ID, return all (message_id, data) tuples ordered by time_created ASC.
def ugotz_fpwjjcp(conn, lzltymc_qse):
    """
    For OpenCode, given a database connection and a session ID, return all (message_id, data) tuples ordered by time_created ASC.

    Args:
        conn: SQLite connection object
        lzltymc_qse: Session ID (INTEGER)

    Returns:
        List of tuples (message_id, data) ordered by time_created ASC
    """
    query = f"""
        SELECT message_id, data
        FROM part
        WHERE session_id = ?
        ORDER BY time_created ASC
    """
    cursor = conn.cursor()
    cursor.execute(query, (lzltymc_qse,))
    return cursor.fetchall()

# The scheme of the second problem is <part (message_id INTEGER, data TEXT, session_id INTEGER, time_created TIMESTAMP)>. For OpenCode, given a database connection and a session ID, return only the message_id values (as a list).
def deinl_tdyiyro_rvxizuh_gcj(conn, lzltymc_qse):
    """
    For OpenCode, given a database connection and a session ID, return only the message_id values (as a list).

    Args:
        conn: SQLite connection object
        lzltymc_qse: Session ID (INTEGER)

    Returns:
        List of message_id values
    """
    # Call the first solution to get all (message_id, data) tuples for the session.
    # Then extract the message_id values from the result.
    result = ugotz_fpwjjcp(conn, lzltymc_qse)

    # Extract message_id from the tuples.
    return [row[0] for row in result]

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE part (message_id INTEGER, data TEXT, session_id INTEGER, time_created TIMESTAMP)')
    cursor.execute('INSERT INTO part VALUES (1, "data1", 1, "2022-01-01 00:00:00")')
    cursor.execute('INSERT INTO part VALUES (2, "data2", 1, "2022-01-02 00:00:00")')
    conn.commit()
    
    raw_result = ugotz_fpwjjcp(conn, 1)
    assert raw_result == [(1, 'data1'), (2, 'data2')]
    
    new_result = deinl_tdyiyro_rvxizuh_gcj(conn, 1)
    assert new_result == [1, 2]
    
    conn.close()

test_both()