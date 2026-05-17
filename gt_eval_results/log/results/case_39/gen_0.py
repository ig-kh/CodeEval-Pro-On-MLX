import sqlite3


def grlwa(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT rowid, start, saved FROM games')
    rows = cursor.fetchall()
    for row in rows:
        print('Game: {}, started: {}, last saved: {}'.format(row[0], row[1], row[2]))
    return rows


def rkbjf_qeji_njmorf_cbjuc(conn):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT games.rowid, games.start, games.saved, COUNT(players.player_id) as player_count
        FROM games
        LEFT JOIN players ON games.rowid = players.game_id
        GROUP BY games.rowid
    ''')
    rows = cursor.fetchall()
    for row in rows:
        print('Game: {}, started: {}, last saved: {}, player count: {}'.format(row[0], row[1], row[2], row[3]))
    return rows
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