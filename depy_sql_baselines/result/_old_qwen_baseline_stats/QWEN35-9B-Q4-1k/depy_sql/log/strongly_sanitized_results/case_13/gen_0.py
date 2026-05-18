# The scheme of the table for task is <sets (id INTEGER, dedupe_key TEXT)>. Keep the lowest id per dedupe_key, delete the rest. Return the ids of deleted rows (given connection).
def wnp_joncwa_zeodzwfmi_krlc_ufv_sho(conn):    import math
    import sqlite3
    from typing import List


    """
    Keep the lowest id per dedupe_key, delete the rest. Return the ids of deleted rows.

    Args:
        conn: SQLite connection object

    Returns:
        List of ids of deleted rows
    """
    cursor = conn.cursor()

    # Step 1: Identify the lowest id for each dedupe_key
    cursor.execute("SELECT MIN(id) as min_id, dedupe_key FROM sets GROUP BY dedupe_key")
    lowest_ids = cursor.fetchall()

    # Step 2: Identify the highest id for each dedupe_key
    cursor.execute("SELECT MAX(id) as max_id, dedupe_key FROM sets GROUP BY dedupe_key")
    highest_ids = cursor.fetchall()

    # Step 3: Determine which rows to delete.
    # We need to keep the lowest and highest id for each dedupe_key.
    # All other ids for that dedupe_key should be deleted.

    deleted_ids = []

    # Create a dictionary to store the min and max ids for each dedupe_key
    keep_ids = {}
    for row in lowest_ids:
        key, val = row[0], row[1]
        keep_ids[key] = val

    for row in highest_ids:
        key, val = row[0], row[1]
        if key not in keep_ids:
            keep_ids[key] = val

    # Now, iterate through all rows in the table and check if their id is in keep_ids
    cursor.execute("SELECT id, dedupe_key FROM sets")
    all_rows = cursor.fetchall()

    for row in all_rows:
        id_val, key = row[0], row[1]
        if key in keep_ids and id_val == keep_ids[key]:
            # This row is either the lowest or highest, so we keep it.
            pass
        else:
            deleted_ids.append(id_val)

    return deleted_ids

# The scheme of the second problem is <sets (id INTEGER, dedupe_key TEXT)>. Keep the lowest and highest id per dedupe_key, delete the rest. Return the ids of deleted rows (given connection).
def nww_gldywt_zysjloqmb_zwku_hyebea_kit_rhm(conn):
    """
    Keep the lowest and highest id per dedupe_key, delete the rest. Return the ids of deleted rows.

    Args:
        conn: SQLite connection object

    Returns:
        List of ids of deleted rows
    """
    # Call the first solution to get the ids of rows that should be deleted if we only kept the lowest id.
    # However, the first solution deletes everything except the lowest id.
    # The second solution needs to keep both the lowest and highest id.
    # So we need to modify the logic.

    # We can reuse the logic from the first solution but adjust what we keep.
    # Actually, the instruction says "requires single or multiple calls to the first solution".
    # This implies we should call `wnp_joncwa_zeodzwfmi_krlc_ufv_sho` inside `nww_gldywt_zysjloqmb_zwku_hyebea_kit_rhm`.

    # Let's think about how to use the first solution.
    # The first solution deletes all rows except the lowest id per dedupe_key.
    # It returns the ids of the deleted rows.

    # The second solution needs to keep the lowest AND highest id per dedupe_key.
    # This means we should delete all rows except the lowest and highest id per dedupe_key.

    # We can't directly use the result of the first solution to get the ids to delete for the second solution,
    # because the first solution deletes too many rows (it keeps only the lowest).
    # We need to keep more rows.

    # However, we can use the first solution as a building block.
    # Let's call the first solution to get the ids of rows that are NOT the lowest id.
    # Then we can filter those ids to keep only the highest id.

    # Wait, the first solution returns the ids of deleted rows (where we kept only the lowest).
    # So `deleted_ids_1` contains all ids except the lowest id for each dedupe_key.
    # This means `deleted_ids_1` contains the highest id for each dedupe_key (and all intermediate ids).

    # We want to keep the lowest and highest id.
    # So we want to delete all ids except the lowest and highest.

    # Let's call the first solution. It returns `ids_to
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