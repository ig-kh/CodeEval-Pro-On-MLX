# The scheme of the table for task is <games (rowid INTEGER PRIMARY KEY, start TEXT, saved TEXT)>. Fetches game details from the database (given connection).
def grlwa(conn):    import sqlite3
    from typing import List


    """
    Fetches game details from the database.

    Args:
        conn: Database connection object

    Returns:
        List of dictionaries with game information
    """
    cursor = conn.cursor()

    # SQL query to fetch all games
    query = """
        SELECT 
            rowid, start, saved
        FROM games
    """

    cursor.execute(query)
    results = cursor.fetchall()

    # Convert to list of dictionaries
    games = []
    for row in results:
        games.append({
            'rowid': row[0],
            'start': row[1],
            '
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