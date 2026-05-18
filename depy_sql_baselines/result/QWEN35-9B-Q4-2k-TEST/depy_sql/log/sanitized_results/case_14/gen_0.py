# The scheme of the table for task is <teardown_approvals (teardown_id INTEGER, level TEXT, strategy_id TEXT, request_json TEXT, created_at TEXT, expires_at TEXT, response_json TEXT)>. Write an approval response by strategy_id (convenience for API callers).

Looks up the oldest pending approval for the strategy and writes to it.
Returns False if no pending approval exists.
def cpopy_ojpffiqj_qylodwyu_kpq_lxnywfxh(conn, kppmfhih_srf, riq_hma):
    cursor = conn.cursor()
    cursor.execute("SELECT teardown_id FROM teardown_approvals WHERE strategy_id = ? AND status = 'pending' ORDER BY created_at ASC LIMIT 1", (kppmfhih_srf,))
    result = cursor.fetchone()
    if result:
        teardown_id = result[0]
        cursor.execute("UPDATE teardown_approvals SET response_json = ? WHERE teardown_id = ?", (riq_hma, teardown_id))
        conn.commit()
        return True
    return False

# The scheme of the table for task is <teardown_approvals (teardown_id INTEGER, level TEXT, strategy_id TEXT, request_json TEXT,
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE teardown_approvals (teardown_id INTEGER, level TEXT, strategy_id TEXT, request_json TEXT, created_at TEXT, expires_at TEXT, response_json TEXT)')
    cursor.executemany('INSERT INTO teardown_approvals VALUES (?, ?, ?, ?, ?, ?, ?)', [
        (1, 'level1', 'strategy1', '{}', '2022-01-01T00:00:00Z', '2022-01-02T00:00:00Z', None),
        (2, 'level2', 'strategy2', '{}', '2022-01-01T00:00:00Z', '2022-01-02T00:00:00Z', None),
        (3, 'level1', 'strategy1', '{}', '2022-01-03T00:00:00Z', '2022-01-04T00:00:00Z', 'response')
    ])
    conn.commit()
    assert cpopy_ojpffiqj_qylodwyu_kpq_lxnywfxh(conn, 'strategy1', '2022-01-01T00:00:00Z') == [(1, 'level1', 'strategy1', '{}', '2022-01-01T00:00:00Z', '2022-01-02T00:00:00Z')]
    assert cpopy_ojpffiqj_qylodwyu_kpq_lxnywfxh(conn, 'strategy2', '2022-01-01T00:00:00Z') == [(2, 'level2', 'strategy2', '{}', '2022-01-01T00:00:00Z', '2022-01-02T00:00:00Z')]
    assert rbssc_oglcevdo_pcuoitnm_lpo_oxsktihm_kir_teqzn(conn, 'strategy1', 'level1', '2022-01-01T00:00:00Z') == [(1, 'level1', 'strategy1', '{}', '2022-01-01T00:00:00Z', '2022-01-02T00:00:00Z')]
    assert rbssc_oglcevdo_pcuoitnm_lpo_oxsktihm_kir_teqzn(conn, 'strategy1', 'level2', '2022-01-01T00:00:00Z') == False
    conn.close()
    print('All tests passed')

test_both()