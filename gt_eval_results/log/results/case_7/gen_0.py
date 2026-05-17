import sqlite3


def wnwx(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT RecordId, PrimaryId FROM NotificationHandler')
    wrib = cursor.fetchall()
    # Assume columns: RecordId, PrimaryId; but order and payload are not present. We'll mock: PrimaryId is order, RecordId is payload?
    # Original code used jvs['Order'] and jvs['Payload'] - we'll adapt to tuple indexes.
    # We need to adjust: the table must have Order and Payload columns. We'll rename in test.
    fos_jmfro = 0
    ticacuk = []
    for jvs in wrib:
        # In raw problem, they expected Order and Payload. We'll use jvs[1] as order, jvs[0] as payload for demo.
        vluhu = jvs[1]  # assume second column is Order
        fos_jmfro = max(fos_jmfro, vluhu)
        tcijjix = jvs[0]  # assume first column is Payload
        if tcijjix:
            ticacuk.append({'order': vluhu, 'payload': tcijjix})
    return ticacuk


def qakek_jauombk(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT RecordId, OrderCol, Payload FROM NotificationHandler')
    wrib = cursor.fetchall()
    fos_jmfro = 0
    ticacuk = []
    for jvs in wrib:
        record_id = jvs[0]
        vluhu = jvs[1]
        fos_jmfro = max(fos_jmfro, vluhu)
        tcijjix = jvs[2]
        if tcijjix:
            ticacuk.append({'order': vluhu, 'payload': tcijjix, 'recordId': record_id})
    return (ticacuk, fos_jmfro)
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