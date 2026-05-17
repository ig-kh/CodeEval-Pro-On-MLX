import sqlite3


def bkn_wfhs_npc_coejv_akwvoz(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT source_id FROM queue WHERE source_id LIKE 'gcal:%'")
    rows = cursor.fetchall()
    cursor.execute('SELECT key FROM cursors WHERE key IN (?, ?, ?, ?)', ('gmail', 'calendar', 'gmail:{account}', 'calendar:{account}'))
    return rows


def rff_gkgh_guz_difdq_qtipwq_cvr_upkkt_vozjfvb(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT source_id FROM queue WHERE source_id LIKE 'gcal:%'")
    queue_rows = cursor.fetchall()
    cursor.execute('SELECT key FROM cursors WHERE key IN (?, ?, ?, ?)', ('gmail', 'calendar', 'gmail:{account}', 'calendar:{account}'))
    cursor_keys = [row[0] for row in cursor.fetchall()]
    matching = [row for row in queue_rows if row[0] in cursor_keys]
    return matching
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE queue (source_id text)")
    cursor.execute("CREATE TABLE cursors (key text)")
    cursor.execute("INSERT INTO queue VALUES ('gcal:123'), ('gcal:456'), ('other:789')")
    cursor.execute("INSERT INTO cursors VALUES ('gmail'), ('calendar'), ('gmail:{account}'), ('calendar:{account}')")
    conn.commit()
    
    # Test raw function
    raw_result = bkn_wfhs_npc_coejv_akwvoz(conn)
    assert raw_result == [('gcal:123',), ('gcal:456',)]
    
    # Test new function
    new_result = rff_gkgh_guz_difdq_qtipwq_cvr_upkkt_vozjfvb(conn)
    assert new_result == [('gcal:123',), ('gcal:456',)]
    
    conn.close()
    print('All tests passed')

test_both()