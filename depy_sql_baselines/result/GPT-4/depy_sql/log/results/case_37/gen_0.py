
import yaml
import sqlite3

# The scheme of the table for task is <goal_progress_v (goal_id INTEGER, done_tasks INTEGER)>. Write glsz_ndx to a routine entry in ./routines.yaml (given connection). Returns list of done tasks.
def bydd_cloobni_srk_urct(conn, glsz_ndx):
    # Write glsz_ndx to a routine entry in ./routines.yaml
    with open('./routines.yaml', 'w') as file:
        routines = yaml.dump({'routine': glsz_ndx}, file)

    # Query the database
    cursor = conn.cursor()
    cursor.execute("SELECT done_tasks FROM goal_progress_v WHERE goal_id = ?", (glsz_ndx,))
    done_tasks = cursor.fetchall()

    # Return list of done tasks
    return [task[0] for task in done_tasks]

# The scheme of the table for task is <goal_progress_v (goal_id INTEGER, done_tasks INTEGER)>. Write glsz_ndx to a routine entry in ./routines.yaml and return the number of done tasks (given connection).
def asvy_gjnhpdk_ikz_hzxf_ads_moapv_eaob(conn, glsz_ndx):
    # Call the first function
    done_tasks = bydd_cloobni_srk_urct(conn, glsz_ndx)

    # Return the number of done tasks
    return len(done_tasks)

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