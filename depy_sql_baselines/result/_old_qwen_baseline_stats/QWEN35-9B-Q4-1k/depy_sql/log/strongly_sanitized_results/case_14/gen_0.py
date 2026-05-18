# The scheme of the table for task is <teardown_approvals (teardown_id INTEGER, level TEXT, strategy_id TEXT, request_json TEXT, created_at TEXT, expires_at TEXT, response_json TEXT)>. Write an approval response by strategy_id (convenience for API callers).

Looks up the oldest pending approval for the strategy and writes to it.
Returns False if no pending approval exists.
def cpopy_ojpffiqj_qylodwyu_kpq_lxnywfxh(conn, kppmfhih_srf, riq_hma):    import math
    import sqlite3


    """
    Write an approval response by strategy_id.

    Args:
        conn: SQLite connection object
        kppmfhih_srf: Strategy ID (TEXT)
        riq_hma: Request JSON (TEXT)

    Returns:
        False if no pending approval exists, True otherwise.
    """
    cursor = conn.cursor()

    # Step 1: Find the oldest pending approval for the given strategy_id.
    # Pending means response_json IS NULL.
    # Oldest means MIN(created_at).
    query = f"""
        SELECT teardown_id, level, strategy_id, request_json, created_at, expires_at, response_json
        FROM teardown_approvals
        WHERE strategy_id = ? AND response_json IS NULL
        ORDER BY created_at ASC
        LIMIT 1
    """

    cursor.execute(query, (kppmfhih_srf,))
    result = cursor.fetchone()

    if result is None:
        return False

    teardown_id, level, strategy_id, request_json, created_at, expires_at, response_json = result

    # Step 2: Write the approval response.
    # We assume we are writing to the same row (teardown_id).
    # We update response_json with the provided request_json (or some logic).
    # The problem says "Write an approval response". Usually this means updating the row.
    # Let's assume we update response_json to be the provided request_json or a processed version.
    # Since the problem doesn't specify the exact content of the response, we'll just update it.
    # Actually, looking at the parameters: kppmfhih_srf (strategy_id), riq_hma (request_json).
    # It seems we are passing the request_json to be stored as response_json? Or maybe we are just marking it as approved.
    # Let's assume we update response_json to be the provided riq_hma.

    update_query = f"""
        UPDATE teardown_approvals
        SET response_json = ?
        WHERE teardown_id = ?
    """

    cursor.execute(update_query, (riq_hma, teardown_id))
    conn.commit()

    return True

# The scheme of the second problem is <teardown_approvals (teardown_id INTEGER, level TEXT, strategy_id TEXT, request_json TEXT, created_at TEXT, expires_at TEXT, response_json TEXT)>. Write an approval response by strategy_id and level (convenience for API callers).

Looks up the oldest pending approval for the strategy and level and writes to it.
Returns False if no pending approval exists.
def rbssc_oglcevdo_pcuoitnm_lpo_oxsktihm_kir_teqzn(conn, kppmfhih_srf, kezel, riq_hma):
    """
    Write an approval response by strategy_id and level.

    Args:
        conn: SQLite connection object
        kppmfhih_srf: Strategy ID (TEXT)
        kezel: Level (TEXT)
        riq_hma: Request JSON (TEXT)

    Returns:
        False if no pending approval exists, True otherwise.
    """
    cursor = conn.cursor()

    # Step 1: Find the oldest pending approval for the given strategy_id and level.
    # Pending means response_json IS NULL.
    # Oldest means MIN(created_at).
    query = f"""
        SELECT teardown_id, level, strategy_id, request_json, created_at, expires_at, response_json
        FROM teardown_approvals
        WHERE strategy_id = ? AND level = ? AND response_json IS NULL
        ORDER BY created_at ASC
        LIMIT 1
    """

    cursor.execute(query, (kppmfhih_srf, kezel))
    result = cursor.fetchone()

    if result is None:
        return False

    teardown_id, level, strategy_id, request_json, created_at, expires_at, response_json = result

    # Step 2: Write the approval response.
    update_query = f"""
        UPDATE teardown_approvals
        SET response_json = ?
        WHERE teardown_id = ?
    """

    cursor.execute(update_query, (riq_hma, teardown_id))
    conn.commit()

    return True

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