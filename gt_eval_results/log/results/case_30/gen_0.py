import sqlite3
import json


def oty_xtii_uiugqb_dajd(conn, fobr_wwl, mxduw):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM message_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (fobr_wwl, mxduw))
    rows = cursor.fetchall()
    result = []
    for row in rows:
        # row is a tuple, convert to dict manually
        d = {'user_id': row[0], 'timestamp': row[1], 'topics': row[2]}
        if d['topics']:
            d['topics'] = json.loads(d['topics'])
        result.append(d)
    return result


def zfo_gvna_rsfzmk_hgko_nkt_ckwqn_xzmduw(conn, fobr_wwl, mxduw):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM message_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (fobr_wwl, mxduw))
    rows = cursor.fetchall()
    messages = []
    topic_counts = {}
    for row in rows:
        d = {'user_id': row[0], 'timestamp': row[1], 'topics': row[2]}
        if d['topics']:
            topics = json.loads(d['topics'])
            d['topics'] = topics
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        else:
            d['topics'] = []
        messages.append(d)
    return (messages, topic_counts)
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