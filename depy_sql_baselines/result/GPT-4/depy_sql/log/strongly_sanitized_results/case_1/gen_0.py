import sqlite3

# The scheme of the table for task is <posture_datapoints (metric_name TEXT, metric_category TEXT, org_id INTEGER)>. 
# Return metric name and category per org_id from posture_datapoints.
def kbb_fcklwpg_yefeyvgq_imebqdq(conn, org_id):
    cursor = conn.cursor()
    cursor.execute("SELECT metric_name, metric_category FROM posture_datapoints WHERE org_id = ?", (org_id,))
    return cursor.fetchall()

# The scheme of the table for task is <posture_datapoints (metric_name TEXT, metric_category TEXT, org_id INTEGER)>. 
# Return the count of unique metric categories for a given org_id from posture_datapoints.
def xaq_ypfgla_mpdeoc_sfkfuxbdbv(conn, org_id):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT metric_category) FROM posture_datapoints WHERE org_id = ?", (org_id,))
    return cursor.fetchone()[0]
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE posture_datapoints (metric_name TEXT, metric_category TEXT, org_id INTEGER)')
    cursor.executemany('INSERT INTO posture_datapoints VALUES (?, ?, ?)', [
        ('metric1', 'category1', 1), ('metric2', 'category2', 1), ('metric3', 'category1', 1),
        ('metric4', 'category3', 2)
    ])
    conn.commit()
    raw_result = kbb_fcklwpg_yefeyvgq_imebqdq(conn, 1)
    assert sorted(raw_result) == [('metric1', 'category1'), ('metric2', 'category2'), ('metric3', 'category1')]
    assert xaq_ypfgla_mpdeoc_sfkfuxbdbv(conn, 1) == 2
    assert xaq_ypfgla_mpdeoc_sfkfuxbdbv(conn, 2) == 1
    conn.close()
    print('All tests passed')

test_both()