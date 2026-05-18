# The scheme of the table for task is <sets (id INTEGER, dedupe_key TEXT)>. Keep the lowest id per dedupe_key, delete the rest. Return the ids of deleted rows (given connection).
def wnp_joncwa_zeodzwfmi_krlc_ufv_sho(conn):    import sqlite3


    """Keep the lowest id per dedupe_key, delete the rest. Return the ids of deleted rows."""
    query = f"""
        WITH ranked AS (
            SELECT 
                id, 
                dedupe_key, 
                ROW_NUMBER() OVER (PARTITION BY dedupe_key ORDER BY id ASC) AS rn
            FROM sets
        )
        DELETE FROM sets
        WHERE id IN (
            SELECT id FROM ranked WHERE rn > 1
        )
    """
    return conn.execute(query).fetchall()

# The scheme of
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE sets (id INTEGER, dedupe_key TEXT)")
    cursor.executemany("INSERT INTO sets VALUES (?, ?)", [(1, 'A'), (2, 'A'), (3, 'B'), (4, 'B'), (5, 'C')])
    conn.commit()
    
    # Test raw function
    raw_result = wnp_joncwa_zeodzwfmi_krlc_ufv_sho(conn)
    assert raw_result == [(2,), (4,)]  # because lowest per key kept, so A: keep 1 delete 2; B: keep 3 delete 4; C: only one, none deleted
    
    # Test new function
    new_result = nww_gldywt_zysjloqmb_zwku_hyebea_kit_rhm(conn)
    # For A: ids 1,2 -> keep 1 and 2? Actually lowest and highest: if only two, both kept, so no delete. But our data: A:1,2 -> keep 1 and 2, delete none. B:3,4 -> keep 3 and 4, delete none. C:5 -> keep 5, delete none. So result should be empty.
    # Wait, the function returns ids to delete (where rn>1 and rn_desc>1). For A: rn=1,2; rn_desc=2,1 -> for id=2: rn=2 (>1) but rn_desc=1 (not >1) -> not selected. Similarly for B. So no rows. But test expects (2,) and (3,)? In original test they had [(2,), (3,)]? Actually original test had [(2,), (3,)]? Let's see original test: `assert nww_gldywt_zysjloqmb_zwku_hyebea_kit_rhm() == [(2,), (3,)]` – that was for a different logic? No, original test was flawed. We'll correct based on logic.
    # According to new problem: keep lowest and highest, delete the rest. So for group with more than 2 elements, delete middle ones. Our data has only 2 per group, so no deletions. Let's modify test to have a group with 3 elements.
    # Better to recreate data for new test separately? We'll write a fresh test.
    conn.close()
    
    # New connection for new function test with richer data
    conn2 = sqlite3.connect(':memory:')
    cursor2 = conn2.cursor()
    cursor2.execute("CREATE TABLE sets (id INTEGER, dedupe_key TEXT)")
    cursor2.executemany("INSERT INTO sets VALUES (?, ?)", [(1, 'A'), (2, 'A'), (3, 'A'), (4, 'B'), (5, 'B'), (6, 'C')])
    conn2.commit()
    new_result = nww_gldywt_zysjloqmb_zwku_hyebea_kit_rhm(conn2)
    # For A: ids 1,2,3 -> keep lowest (1) and highest (3), delete (2). For B: 4,5 -> keep both, delete none. C: 6 -> keep, none.
    assert new_result == [(2,)]
    conn2.close()
    
    print('All tests passed')

test_both()