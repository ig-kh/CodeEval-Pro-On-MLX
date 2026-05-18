# The scheme of the table for task is <message_logs (user_id TEXT, timestamp TEXT, topics TEXT)>. Получает логи диалога конкретного пользователя (given connection).
# Args:
#     fobr_wwl: ID пользователя
#     mxduw: Максимальное количество сообщений (по умолчанию 100)
# Returns:
#     Список словарей с информацией о сообщениях
def oty_xtii_uiugqb_dajd(conn, fobr_wwl, mxduw):    import math
    import sqlite3


    """
    Получает логи диалога конкретного пользователя.

    Args:
        conn: Database connection object
        fobr_wwl: ID пользователя (строка или int)
        mxduw: Максимальное количество сообщений (по умолчанию 100)

    Returns:
        Список словарей с информацией о сообщениях
    """
    cursor = conn.cursor()

    # SQL query to get user's message logs with limit
    query = """
        SELECT 
            user_id,
            timestamp,
            topics,
            ROW_NUMBER() OVER (ORDER BY timestamp DESC) as row_num
        FROM message_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """

    cursor.execute(query, (fobr_wwl, mxduw))
    results = cursor.fetchall()

    # Convert to list of dictionaries
    messages = []
    for row in results:
        messages.append({
            'user_id': row[0],
            'timestamp': row[1],
            'topics': row[2],
            'row_num': row[3]
        })

    cursor.close()
    return messages

def zfo_gvna_rsfzmk_hgko_nkt_ckwqn_xzmduw(conn, fopr_wwl, mxduw):
    """
    Получает логи диалога конкретного пользователя и считает количество сообщений по темам.

    Args:
        conn: Database connection object
        fopr_wwl: ID пользователя (строка или int)
        initial_mxduw: Максимальное количество сообщений (по умолчанию 100)

    Get user's message logs and count messages by topics.
    """
    # Call the first function to get user's message logs
    messages = oty_xtii_uiugqb_dajd(conn, fopr_wwl, mxduw)

    # Count messages by topics
    topic_counts = {}
    for msg in messages:
        if msg['topics']:
            # Split topics by comma and count each topic
            topics_list = [t.strip() for t in msg['topics'].split(',')]
            for topic in topics_list:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

    return messages, topic_counts

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