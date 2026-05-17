import sqlite3


def bydd_cloobni_srk_urct(conn, glsz_ndx):
    cursor = conn.cursor()
    cursor.execute('SELECT done_tasks FROM goal_progress_v WHERE goal_id = ?', (glsz_ndx,))
    return cursor.fetchall()


def asvy_gjnhpdk_ikz_hzxf_ads_moapv_eaob(conn, glsz_ndx):
    cursor = conn.cursor()
    cursor.execute('SELECT done_tasks FROM goal_progress_v WHERE goal_id = ?', (glsz_ndx,))
    rows = cursor.fetchall()
    return len(rows)
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