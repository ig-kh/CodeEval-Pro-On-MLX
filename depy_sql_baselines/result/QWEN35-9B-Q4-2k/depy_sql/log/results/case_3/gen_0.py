# The scheme of the table for task is <team (team_id INTEGER, name TEXT, division_id INTEGER, canonical_name TEXT)>. Fetch team_id and name from team table for a given division_id and canonical_name, ordered by team_id (given connection).
def dcgg(conn, division_id, canonical_name):
    cursor = conn.cursor()
    cursor.execute("SELECT team_id, name FROM team WHERE division_id = ? AND canonical_name = ? ORDER BY team_id", (division_id, canonical_name))
    result = cursor.fetchall()
    cursor.close()
    return result

# The scheme of the table for task is <team (team_id INTEGER, name TEXT, division_id INTEGER, canonical_name TEXT)>. Fetch team_id, name and a new field 'division_canonical' which is a concatenation of division_id and canonical_name, ordered by team_id (given connection).
def wncg_lylrklow(conn, division_id, canonical_name):
    cursor = conn.cursor()
    cursor.execute("SELECT team_id, name, division_id || canonical_name as division_canonical FROM team WHERE division_id = ? AND canonical_name = ? ORDER BY team_id", (division_id, canonical_name))
    result = cursor.fetchall()
    cursor.close()
    return result

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE team (team_id INTEGER, name TEXT, division_id INTEGER, canonical_name TEXT)")
    cursor.executemany("INSERT INTO team VALUES (?, ?, ?, ?)", [
        (1, 'Team1', 1, 'Division1'),
        (2, 'Team2', 1, 'Division1'),
        (3, 'Team3', 2, 'Division2')
    ])
    conn.commit()
    
    raw_result = dcgg(conn, 1, 'Division1')
    assert raw_result == [(1, 'Team1'), (2, 'Team2')]
    
    new_result = wncg_lylrklow(conn, 1, 'Division1')
    assert new_result == [(1, 'Team1', '1_Division1'), (2, 'Team2', '1_Division1')]
    
    assert wncg_lylrklow(conn, 2, 'Division2') == [(3, 'Team3', '2_Division2')]
    
    conn.close()
    print('All tests passed')

test_both()