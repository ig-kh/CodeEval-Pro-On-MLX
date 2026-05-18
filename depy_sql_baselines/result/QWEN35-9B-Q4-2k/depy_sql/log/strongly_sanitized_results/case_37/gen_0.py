# The scheme of the table for task is <goal_progress_v (goal_id INTEGER, done_tasks INTEGER)>. Write glsz_ndx to a routine entry in ./routines.yaml (given connection). Returns list of done tasks.
def bydd_cloobni_srk_urct(conn, glsz_ndx):import sqlite3

import yaml
    import os
    import json

    # Ensure the directory exists
    os.makedirs('./routines', exist_ok=True)

    # Write to routines.yaml
    if not os.path.exists('./routines.yaml'):
        with open('./routines.yaml', 'w') as f:
            yaml.dump({}, f)

    if not os.path.exists('./routines.yaml'):
        with open('./routines.yaml', 'w') as f:
            f.write('')

    data = {}
    if os.path.exists('./routines.yaml'):
        with open('./routines.yaml', 'r') as f:
            data = yaml.safe_load(f)

    if glsz_ndx not in data:
        data[glsz_ndx] = []
    else:
        data[glsz_ndx] = data[glsz_ndx]

    with open('./routines.yaml', 'w') as f:
        yaml.dump(data, f)

    cursor = conn.cursor()
    cursor.execute("SELECT done_tasks FROM goal_progress_v WHERE routine_name = ?", (glsz_ndx,))
    result = cursor.fetchone()
    return result[0] if result else []

# The scheme of the table for task is <goal_progress_v (goal_id INTEGER, done_tasks INTEGER)>. Write glsz_ndx to a routine entry in ./routines.yaml and return the number of done tasks (given connection).
def asvy_gjnhpdk_ikz_hzxf_ads_moapv_eaob(conn, glsz_ndx):
    bydd_cloobni_srk_urct(conn, glsz_ndx)
    cursor = conn.cursor()
    cursor.execute("SELECT done_tasks FROM goal_progress_v WHERE routine_name = ?", (glsz_ndx,))
    result = cursor.fetchone()
    return result[0] if result else 0
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