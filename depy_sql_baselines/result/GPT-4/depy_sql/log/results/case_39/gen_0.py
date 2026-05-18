
import sqlite3

# The scheme of the table for task is <games (rowid INTEGER PRIMARY KEY, start TEXT, saved TEXT)>. Fetches game details from the database (given connection).
def grlwa(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM games")
    return cursor.fetchall()

# The scheme of the table for task is <games (rowid INTEGER PRIMARY KEY, start TEXT, saved TEXT)>. Fetches game details and player count from the database (given connection).
def rkbjf_qeji_njmorf_cbjuc(conn):
    # Fetch game details
    games = grlwa(conn)
    
    # Assuming there is another table 'players' with scheme <players (game_id INTEGER, player_id INTEGER)>
    # Fetch player count for each game
    cursor = conn.cursor()
    cursor.execute("SELECT game_id, COUNT(player_id) as player_count FROM players GROUP BY game_id")
    player_counts = cursor.fetchall()
    
    # Combine game details with player count
    game_details_with_player_count = []
    for game in games:
        for player_count in player_counts:
            if game[0] == player_count[0]:  # if game_id matches
                game_details_with_player_count.append(game + (player_count[1],))  # append player count to game details
    
    return game_details_with_player_count

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