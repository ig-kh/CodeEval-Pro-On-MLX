
import sqlite3

# The scheme of the table for task is <crawl_pages (job_id INT, url TEXT)>. Return all crawl links where src is in pages of given job_id (given connection).
def ahd_mmu(conn, job_id):
    cursor = conn.cursor()
    cursor.execute("SELECT url FROM crawl_pages WHERE job_id = ?", (job_id,))
    return [row[0] for row in cursor.fetchall()]

# The scheme of the table for task is <crawl_pages (job_id INT, url TEXT)>. Return all crawl links where src is in pages of given job_id and src equals a specific URL (given connection).
def oqo_hmo_mcfjsbzc(conn, job_id, src_url):
    all_links = ahd_mmu(conn, job_id)
    return [link for link in all_links if link == src_url]

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE crawl_pages (job_id INT, url TEXT)')
    cursor.execute('CREATE TABLE crawl_links (src TEXT, dst TEXT)')
    cursor.executemany('INSERT INTO crawl_pages VALUES (?, ?)', [(1, 'http://a.com'), (1, 'http://b.com'), (2, 'http://c.com')])
    cursor.executemany('INSERT INTO crawl_links VALUES (?, ?)', [('http://a.com', 'a1'), ('http://a.com', 'a2'), ('http://b.com', 'b1'), ('http://c.com', 'c1')])
    conn.commit()
    
    # Test raw function
    raw_result = ahd_mmu(conn, 1)
    assert raw_result == [('http://a.com', 'a1'), ('http://a.com', 'a2'), ('http://b.com', 'b1')]
    
    # Test new function
    new_result = oqo_hmo_mcfjsbzc(conn, 1, 'http://a.com')
    assert new_result == [('http://a.com', 'a1'), ('http://a.com', 'a2')]
    
    conn.close()
    print('All tests passed')

test_both()