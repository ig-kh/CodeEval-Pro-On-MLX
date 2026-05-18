import re
    import sqlite3

import psycopg2
import json

# The scheme of the table for task is <queue (source_id text)>. Iterate VectorDB points whose metadata.doc_id has a legacy-format namespace. Tolerates VectorDB-client's two-shape return from ``fetch``. (Given connection)
def bkn_wfhs_npc_coejv_akwvoz(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM queue")
        rows = cursor.fetchall()
        result = []
        for row in rows:
            source_id = row[0]
            cursor.execute(f"SELECT metadata FROM VectorDB WHERE source_id = '{source_id}'")
            metadata = cursor.fetchone()
            if metadata is not None:
                metadata = json.loads(metadata[0])
                if 'doc_id' in metadata and isinstance(metadata['doc_id'], str) and ':' in metadata['doc_id']:
                    result.append((source_id, metadata))
        return result
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

# The scheme of the table for task is <queue (source_id text)>. Iterate VectorDB points whose metadata.doc_id has a legacy-format namespace and return only those that match with certain keys in the cursors table.
def rff_gkgh_guz_difdq_qtipwq_cvr_upkkt_vozjfvb(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key FROM cursors")
        keys = cursor.fetchall()
        keys = [key[0] for key in keys]
        result = []
        vector_db_points = bkn_wfhs_npc_coejv_akwvoz(conn)
        for point in vector_db_points:
            if point[0] in keys:
                result.append(point)
        return result
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
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