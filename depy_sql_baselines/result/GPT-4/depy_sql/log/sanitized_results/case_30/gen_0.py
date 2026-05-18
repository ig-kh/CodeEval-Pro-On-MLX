
import sqlite3
from collections import defaultdict

# Function to get the dialog logs of a specific user
def oty_xtii_uiugqb_dajd(conn, fobr_wwl, mxduw=100):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM message_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (fobr_wwl, mxduw))
    logs = [{'user_id': row[0], 'timestamp': row[1], 'topics': row[2]} for row in cursor.fetchall()]
    return logs

# Function to get the dialog logs of a specific user and count the number of messages by topics
def zfo_gvna_rsfzmk_hgko_nkt_ckwqn_xzmduw(conn, fobr_wwl, mxduw=100):
    logs = oty_xtii_uiugqb_dajd(conn, fobr_wwl, mxduw)
    topic_counts = defaultdict(int)
    for log in logs:
        topics = log['topics'].split(',')
        for topic in topics:
            topic_counts[topic.strip()] += 1
    return logs, dict(topic_counts)

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