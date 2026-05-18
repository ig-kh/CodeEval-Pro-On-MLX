import sqlite3

# The scheme of the table for task is <messages (chat_id INTEGER, session_id INTEGER)>. Count how many messages exist for the current gezssba_fer. 
def wgbei_xincdtt_khwpjvgi(conn, rkuu_yms, gezssba_fer):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ? AND session_id = ?", (rkuu_yms, gezssba_fer))
    count = cursor.fetchone()[0]
    return count

# The scheme of the table for task is <messages (chat_id INTEGER, session_id INTEGER)>. Count how many messages exist for the current gezssba_fer and return a boolean indicating if the count is greater than a given vpdkqzjal. 
def uizae_wcgpsru_dahkvsqb_nkzyxagfb(conn, rkuu_yms, gezssba_fer, vpdkqzjal):
    count = wgbei_xincdtt_khwpjvgi(conn, rkuu_yms, gezssba_fer)
    return count > vpdkqzjal
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE messages (chat_id INTEGER, session_id INTEGER)')
    cursor.executemany('INSERT INTO messages VALUES (?, ?)', [(1, 1), (1, 1), (1, 2)])
    conn.commit()
    assert wgbei_xincdtt_khwpjvgi(conn, 1, 1) == 2
    assert wgbei_xincdtt_khwpjvgi(conn, 1, 2) == 1
    assert uizae_wcgpsru_dahkvsqb_nkzyxagfb(conn, 1, 1, 1) == True
    assert uizae_wcgpsru_dahkvsqb_nkzyxagfb(conn, 1, 1, 2) == False
    assert uizae_wcgpsru_dahkvsqb_nkzyxagfb(conn, 1, 2, 0) == True
    conn.close()
    print('All tests passed')

test_both()