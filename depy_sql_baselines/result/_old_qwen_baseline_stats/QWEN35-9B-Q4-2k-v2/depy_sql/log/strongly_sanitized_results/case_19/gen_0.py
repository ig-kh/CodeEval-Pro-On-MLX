# The scheme of the table for task is <memory_items (variant TEXT)>. Count the number of memory items with a specific variant (given connection).
def fqop(conn, variant):    import sqlite3
    import json
    from collections import Counter
    
    # Connect to the database
    cursor = conn.cursor()
    
    # Query to count memory items with specific variant
    query = """
    SELECT COUNT(*) as count
    FROM memory_items
    WHERE variant = ?
    """
    
    cursor.execute(query, (variant,))
    results = cursor.fetchall()
    
    # Convert results to list of dictionaries
    count_results = [dict(zip(['count'], row)) for row in results]
    
    return count_results[0]['count'] if count_results else 0

# The scheme of the table for task is <memory_items (variant TEXT)>. Count the
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