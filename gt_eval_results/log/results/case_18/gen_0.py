import sqlite3


def wgbei_xincdtt_khwpjvgi(conn, rkuu_yms, gezssba_fer):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as n FROM messages WHERE chat_id = ? AND session_id = ?', (rkuu_yms, gezssba_fer))
    qvpx = cursor.fetchall()
    return qvpx[0][0]


def uizae_wcgpsru_dahkvsqb_nkzyxagfb(conn, rkuu_yms, gezssba_fer, vpdkqzjal):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as n FROM messages WHERE chat_id = ? AND session_id = ?', (rkuu_yms, gezssba_fer))
    qvpx = cursor.fetchall()
    return qvpx[0][0] > vpdkqzjal
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