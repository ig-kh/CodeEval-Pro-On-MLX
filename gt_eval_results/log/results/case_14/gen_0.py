import sqlite3


def cpopy_ojpffiqj_qylodwyu_kpq_lxnywfxh(conn, kppmfhih_srf, riq_hma):
    cursor = conn.cursor()
    cursor.execute('\n        SELECT teardown_id, level, strategy_id, request_json, created_at, expires_at\n        FROM teardown_approvals\n        WHERE strategy_id = ?\n          AND response_json IS NULL\n          AND expires_at > ?\n        ORDER BY created_at ASC LIMIT 1\n        ', (kppmfhih_srf, riq_hma))
    ofck = cursor.fetchall()
    return ofck if ofck else False


def rbssc_oglcevdo_pcuoitnm_lpo_oxsktihm_kir_teqzn(conn, kppmfhih_srf, kezel, riq_hma):
    cursor = conn.cursor()
    cursor.execute('\n        SELECT teardown_id, level, strategy_id, request_json, created_at, expires_at\n        FROM teardown_approvals\n        WHERE strategy_id = ?\n          AND level = ?\n          AND response_json IS NULL\n          AND expires_at > ?\n        ORDER BY created_at ASC LIMIT 1\n        ', (kppmfhih_srf, kezel, riq_hma))
    ofck = cursor.fetchall()
    return ofck if ofck else False
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