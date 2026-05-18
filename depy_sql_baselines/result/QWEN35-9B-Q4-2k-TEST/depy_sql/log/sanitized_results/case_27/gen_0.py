# The scheme of the table for task is <queue (source_id text)>. Iterate VectorDB points whose metadata.doc_id has a legacy-format namespace. Tolerates VectorDB-client's two-shape return from ``fetch``. (Given connection)
def bkn_wfhs_npc_coejv_akwvoz(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT source_id FROM queue")
    source_ids = cursor.fetchall()
    legacy_points = []
    for source_id in source_ids:
        points = conn.fetch({"metadata": {"doc_id": f"{source_id}_legacy"}})
        if isinstance(points, list):
            legacy_points.extend(points)
        else:
            legacy_points = points
    return legacy_points

# The scheme of the table for task is <queue (source_id text)>. Iterate VectorDB points whose metadata.doc_id has a legacy-format namespace and return only those that match with certain keys in the cursors table.
def rff_gkgh_guz_difdq_qtipwq_cvr_upkkt_vozjfvb(conn):
    legacy_points = bkn_wfhs_npc_coejv_akwvoz(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT keys FROM cursors")
    cursor_keys = cursor.fetchall()
    matching_points = []
    for point in legacy_points:
        for keys in cursor_keys:
            if all(k in point.get("metadata", {}) for k in keys):
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