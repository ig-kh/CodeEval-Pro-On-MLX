import sqlite3


def ahd_mmu(conn, job_id):
    cursor = conn.cursor()
    cursor.execute('SELECT src, dst FROM crawl_links WHERE src IN (SELECT url FROM crawl_pages WHERE job_id=?)', (job_id,))
    return cursor.fetchall()


def oqo_hmo_mcfjsbzc(conn, job_id, src_url):
    cursor = conn.cursor()
    cursor.execute('SELECT src, dst FROM crawl_links WHERE src IN (SELECT url FROM crawl_pages WHERE job_id=?) AND src=?', (job_id, src_url))
    return cursor.fetchall()
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