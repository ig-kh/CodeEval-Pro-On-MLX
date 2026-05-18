# The scheme of the table for task is <message_logs (user_id TEXT, timestamp TEXT, topics TEXT)>. Получает логи диалога конкретного пользователя (given connection).
# Args:
#     fobr_wwl: ID пользователя
#     mxduw: Максимальное количество сообщений (по умолчанию 100)
# Returns:
#     Список словарей с информацией о сообщениях
def oty_xtii_uiugqb_dajd(conn, fobr_wwl, mxduw):
    """
    Получает логи диалога конкретного пользователя.
    
    Args:
        conn: Database connection object
        fobr_wwl: ID пользователя (строка)
        mxduw: Максимальное количество сообщений (по умолчанию 100)
    
    Returns:
        Список словарей с информацией о сообщениях
    """
    cursor = conn.cursor()
    
    # SQL query to fetch user's message logs with limit
    query = f"""
        SELECT 
            user_id,
            timestamp,
            topics,
            ROW_NUMBER() OVER (ORDER BY timestamp DESC) as row_num
        FROM message_logs
        WHERE user_id = '{fobr_wwl}'
        ORDER BY timestamp DESC
        LIMIT {mxduw}
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Convert rows to list of dictionaries
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in results]

def zfo_gvna_rsfzmk_hgko_nkt_ckwqn_xzmduw(conn, fob
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