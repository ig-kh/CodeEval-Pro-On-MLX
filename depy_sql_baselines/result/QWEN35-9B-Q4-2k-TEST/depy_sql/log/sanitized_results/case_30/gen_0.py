# The scheme of the table for task is <message_logs (user_id TEXT, timestamp TEXT, topics TEXT)>. Получает логи диалога конкретного пользователя (given connection).
# Args:
#     fobr_wwl: ID пользователя
#     mxduw: Максимальное количество сообщений (по умолчанию 100)
# Returns:
#     Список словарей с информацией о сообщениях
def oty_xtii_uiugqb_dajd(conn, fobr_wwl, mxduw):
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, timestamp, topics FROM message_logs WHERE user_id = ? LIMIT ?", (fobr_wwl, mxduw))
    results = cursor.fetchall()
    return [dict(row) for row in results]

# The scheme of the table for the second task is <message_logs (user_id TEXT, timestamp TEXT, topics TEXT)>. Получает логи диалога конкретного пользователя и считает количество сообщений по темам (given connection).
# Returns:
#     (список словарей с информацией о сообщения
import sqlite3
import json

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE message_logs (user_id TEXT, timestamp TEXT, topics TEXT)')
    cursor.executemany('INSERT INTO message_logs VALUES (?, ?, ?)', [
        ('user1', '2022-01-01 00:00:00', json.dumps(['topic1', 'topic2'])),
        ('user1', '2022-01-02 00:00:00', json.dumps(['topic1', 'topic3']))
    ])
    conn.commit()
    
    raw_result = oty_xtii_uiugqb_dajd(conn, 'user1', 100)
    assert len(raw_result) == 2
    assert raw_result[0]['topics'] == ['topic1', 'topic2']
    
    messages, counts = zfo_gvna_rsfzmk_hgko_nkt_ckwqn_xzmduw(conn, 'user1', 100)
    assert len(messages) == 2
    assert counts == {'topic1': 2, 'topic2': 1, 'topic3': 1}
    
    conn.close()
    print('All tests passed')

test_both()