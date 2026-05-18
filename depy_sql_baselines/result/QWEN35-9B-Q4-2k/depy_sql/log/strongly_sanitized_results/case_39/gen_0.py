# The scheme of the table for task is <games (rowid INTEGER PRIMARY KEY, start TEXT, saved TEXT)>. Fetches game details from the database (given connection).
def grlwa(conn):import sqlite3

def rkbjf_qeji_njmorf_cbjuc(conn):
    games = grlwa(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM games")
    count = cursor.fetchone()[0]
    return {"games": games, "player_count": count}
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE games (rowid INTEGER PRIMARY KEY, start TEXT, saved TEXT)')
    cursor.execute('CREATE TABLE players (game_id INTEGER, player_id INTEGER)')
    cursor.execute('INSERT INTO games (start, saved) VALUES (?, ?)', ('2022-01-01', '2022-01-02'))
    cursor.execute('INSERT INTO players (game_id, player_id) VALUES (?, ?)', (1, 1))
    cursor.execute('INSERT INTO players (game_id, player_id) VALUES (?, ?)', (1, 2))
    conn.commit()
    
    raw_result = grlwa(conn)
    assert raw_result == [(1, '2022-01-01', '2022-01-02')]
    
    new_result = rkbjf_qeji_njmorf_cbjuc(conn)
    assert new_result == [(1, '2022-01-01', '2022-01-02', 2)]
    
    conn.close()
    print('All tests passed')

test_both()