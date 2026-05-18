# The scheme of the table for task is <goal_progress_v (goal_id INTEGER, done_tasks INTEGER)>. Write glsz_ndx to a routine entry in config/routines.yaml (given connection). Returns list of done tasks.
def bydd_cloobni_srk_urct(conn, glsz_ndx):
    """
    Write glsz_ndx to a routine entry in config/routines.yaml (given connection). Returns list of done tasks.
    
    Args:
        conn: Database connection object
        glsz_ndx: Routine name to write to config/routines.yaml
    
    Returns:
        List of dictionaries with done tasks information
    """
    import yaml
    import os
    
    # Path to config/routines.yaml
    routines_path = 'config/routines.yaml'
    
    # Check if file exists
    if not os.path.exists(routines_path):
        raise FileNotFoundError(f"
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE goal_progress_v (goal_id INTEGER, done_tasks INTEGER)')
    cursor.executemany('INSERT INTO goal_progress_v VALUES (?, ?)', [(1, 5), (1, 6), (2, 3)])
    conn.commit()
    
    raw_result = bydd_cloobni_srk_urct(conn, 1)
    assert raw_result == [(5,), (6,)]
    
    new_result = asvy_gjnhpdk_ikz_hzxf_ads_moapv_eaob(conn, 1)
    assert new_result == 2
    
    assert asvy_gjnhpdk_ikz_hzxf_ads_moapv_eaob(conn, 2) == 1
    
    conn.close()
    print('All tests passed')

test_both()