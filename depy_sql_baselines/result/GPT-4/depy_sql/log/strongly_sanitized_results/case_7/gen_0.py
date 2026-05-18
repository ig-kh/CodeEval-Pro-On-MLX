import sqlite3

def wnwx():
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Execute the query
    cursor.execute("SELECT OrderCol, Payload FROM NotificationHandler")

    # Fetch all the records
    records = cursor.fetchall()

    # Close the connection
    conn.close()

    # Convert the records to a list of dictionaries
    return [{'order': record[0], 'payload': record[1]} for record in records]

def qakek_jauombk():
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Execute the query
    cursor.execute("SELECT RecordId, OrderCol, Payload FROM NotificationHandler")

    # Fetch all the records
    records = cursor.fetchall()

    # Close the connection
    conn.close()

    # Find the max order
    max_order = max(record[1] for record in records)

    # Convert the records to a list of dictionaries and add the max_order
    return [{'recordId': record[0], 'order': record[1], 'payload': record[2], 'max_order': max_order} for record in records]
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE NotificationHandler (RecordId INTEGER, OrderCol INTEGER, Payload TEXT)')
    cursor.executemany('INSERT INTO NotificationHandler VALUES (?, ?, ?)', [(1, 10, 'payload1'), (2, 20, 'payload2'), (3, 15, 'payload3')])
    conn.commit()
    raw_result = wnwx(conn)
    assert len(raw_result) == 3
    new_result, max_order = qakek_jauombk(conn)
    assert len(new_result) == 3
    assert max_order == 20
    conn.close()
    print('All tests passed')

test_both()