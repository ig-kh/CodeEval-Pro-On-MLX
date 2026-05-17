import sqlite3


def fqop(conn, variant):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM memory_items WHERE variant = ?', (variant,))
    return cursor.fetchall()


def htmqg_gkyuypf_scbgjzgyc(conn, variant, threshold):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM memory_items WHERE variant = ?', (variant,))
    count = cursor.fetchone()[0]
    return count > threshold
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE memory_items (variant TEXT)')
    cursor.executemany('INSERT INTO memory_items VALUES (?)', [('variant1',), ('variant1',), ('variant2',)])
    conn.commit()
    
    # Test raw function
    raw_result = fqop(conn, 'variant1')
    assert raw_result == [(2,)]
    
    # Test new function
    assert htmqg_gkyuypf_scbgjzgyc(conn, 'variant1', 1) is True
    assert htmqg_gkyuypf_scbgjzgyc(conn, 'variant1', 2) is False
    assert htmqg_gkyuypf_scbgjzgyc(conn, 'variant2', 0) is True
    
    conn.close()
    print('All tests passed')

test_both()