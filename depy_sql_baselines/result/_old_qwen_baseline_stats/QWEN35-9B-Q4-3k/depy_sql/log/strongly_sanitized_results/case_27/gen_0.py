# The scheme of the table for task is <queue (source_id text)>. Iterate VectorDB points whose metadata.doc_id has a legacy-format namespace. Tolerates VectorDB-client's two-shape return from ``fetch``. (Given connection)
def bkn_wfhs_npc_coejv_akwvoz(conn):
    import sqlite3

    # Connect to the database using the provided connection object
    cursor = conn.cursor()

    # SQL query to iterate VectorDB points whose metadata.doc_id has a legacy-format namespace
    query = """
    SELECT source_id, metadata->'doc_id' as doc_id
    FROM queue
    WHERE metadata->'doc_id' LIKE 'legacy%'
    """

    cursor.execute(query)
    results = cursor.execute(query)  # Tolerates two-shape return from fetch

    # Convert the fetched rows to a list of dictionaries
    columns = ['source_id', 'doc_id']
    result_dicts = [dict(zip(columns, row)) for row in results]

    return result_dicts

# The scheme of the
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