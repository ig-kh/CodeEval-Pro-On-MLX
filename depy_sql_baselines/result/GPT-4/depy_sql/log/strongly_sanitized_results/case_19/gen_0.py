import sqlite3

# The scheme of the table for task is <memory_items (variant TEXT)>. Count the number of memory items with a specific variant (given connection).
def fqop(conn, variant):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memory_items WHERE variant=?", (variant,))
    return cursor.fetchone()[0]

# The scheme of the table for task is <memory_items (variant TEXT)>. Count the number of memory items with a specific variant and return True if count is greater than a threshold (given connection).
def htmqg_gkyuypf_scbgjzgyc(conn, variant, threshold):
    count = fqop(conn, variant)
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