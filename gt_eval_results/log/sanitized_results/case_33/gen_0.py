import sqlite3


def rpw_tfbjll_pomtswh(conn, ieskgwe, zrxdh):
    cursor = conn.cursor()
    cursor.execute("\n        SELECT c.title, m.content, m.timestamp\n        FROM conversations c\n        JOIN messages m ON c.id = m.conversation_id\n        WHERE c.persona = ? AND m.role = 'assistant'\n        ORDER BY m.timestamp DESC\n        LIMIT ?\n        ", (ieskgwe, zrxdh))
    kqbc = cursor.fetchall()
    return kqbc


def gcc_vncjjr_dbtktcq_hspt(conn, ieskgwe, zrxdh, bkcy):
    cursor = conn.cursor()
    cursor.execute('\n        SELECT c.title, m.content, m.timestamp\n        FROM conversations c\n        JOIN messages m ON c.id = m.conversation_id\n        WHERE c.persona = ? AND m.role = ?\n        ORDER BY m.timestamp DESC\n        LIMIT ?\n        ', (ieskgwe, bkcy, zrxdh))
    kqbc = cursor.fetchall()
    return kqbc
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT, persona TEXT)')
    cursor.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, timestamp TEXT, role TEXT, conversation_id INTEGER)')
    cursor.execute("INSERT INTO conversations (id, title, persona) VALUES (1, 'Test Conversation', 'Test Persona')")
    cursor.execute("INSERT INTO messages (content, timestamp, role, conversation_id) VALUES ('Assistant Message', '2022-01-01 00:00:00', 'assistant', 1)")
    cursor.execute("INSERT INTO messages (content, timestamp, role, conversation_id) VALUES ('User Message', '2022-01-01 00:01:00', 'user', 1)")
    conn.commit()
    assert rpw_tfbjll_pomtswh(conn, 'Test Persona', 10) == [('Test Conversation', 'Assistant Message', '2022-01-01 00:00:00')]
    assert gcc_vncjjr_dbtktcq_hspt(conn, 'Test Persona', 10, 'assistant') == [('Test Conversation', 'Assistant Message', '2022-01-01 00:00:00')]
    assert gcc_vncjjr_dbtktcq_hspt(conn, 'Test Persona', 10, 'user') == [('Test Conversation', 'User Message', '2022-01-01 00:01:00')]
    conn.close()
    print('All tests passed')

test_both()