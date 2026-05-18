# The scheme of the table for task is <NotificationHandler (RecordId INTEGER, OrderCol INTEGER, Payload TEXT)>. Fetches records from NotificationHandler and returns a list of dictionaries with 'order' and 'payload' fields. The function is named wnwx().    import sqlite3

def wnwx(conn):
    cursor = conn.cursor()
    query = "SELECT OrderCol as order, Payload FROM NotificationHandler ORDER BY OrderCol"
    cursor.execute(query)
    result = cursor.fetchall()
    return [{'order': row[0], 'payload': row[1]} for row in result]

# The scheme of the
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