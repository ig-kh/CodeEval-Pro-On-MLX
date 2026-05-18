# The scheme of the table for task is <queue (source_id text)>. Iterate VectorDB points whose metadata.doc_id has a legacy-format namespace. Tolerates VectorDB-client's two-shape return from ``fetch``. (Given connection)
def bkn_wfhs_npc_coejv_akwvoz(conn):import re
    import sqlite3

def rff_gkgh_guz_difdq_qtipwq_cvr_upkkt_vozjfvb(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT source_id FROM queue")
    source_ids = cursor.fetchall()

    matching_points = []
    for source_id in source_ids:
        points = bkn_wfhs_npc_coejv_akwvoz(conn)
        if isinstance(points, list):
            for point in points:
                cursor.execute("SELECT * FROM cursors WHERE source_id = ? AND key IN (SELECT key FROM cursors WHERE source_id = ?)", (source_id[0], source_id[0]))
                cursor.fetchone()
                matching_points.append(point)
        else:
            for point in points:
                cursor.execute("SELECT * FROM cursors WHERE source_id = ? AND key IN (SELECT key FROM cursors WHERE source_id = ?)", (source_id[0], source_id[0]))
                cursor.fetchone()
                matching_points.append(point)

    return matching_points
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