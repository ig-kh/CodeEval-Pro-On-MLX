import sqlite3


def wybzna_rzbxbzbcjapm(conn, sbgo, ellhw):
    cursor = conn.cursor()
    cursor.execute('\n        SELECT\n            date(timestamp) as date,\n            COUNT(*) as conversations,\n            SUM(message_count) as messages,\n            AVG(response_time_avg) as avg_response,\n            SUM(token_input + token_output) as tokens,\n            COUNT(DISTINCT agent_name) as agents\n        FROM conversations\n        WHERE date(timestamp) >= ?\n        GROUP BY date(timestamp)\n        ORDER BY date DESC\n        LIMIT ?\n        ', (sbgo, ellhw))
    ewzb = cursor.fetchall()
    knidt_uoq_aiqm = {}
    for nhg in ewzb:
        knidt_uoq_aiqm[nhg[0]] = {'date': nhg[0], 'total_conversations': nhg[1] or 0, 'total_messages': nhg[2] or 0, 'avg_response_time': round(nhg[3] or 0, 2), 'total_tokens': nhg[4] or 0, 'unique_users': nhg[5] or 0}
    return knidt_uoq_aiqm


def ysxqn_gtvtzqygzasr_dfeen(conn, sbgo, ellhw):
    cursor = conn.cursor()
    cursor.execute('\n        SELECT\n            agent_name,\n            COUNT(*) as conversations,\n            SUM(message_count) as messages\n        FROM conversations\n        WHERE date(timestamp) >= ?\n        GROUP BY agent_name\n        ORDER BY conversations DESC\n        LIMIT ?\n        ', (sbgo, ellhw))
    ewzb = cursor.fetchall()
    udjii_mql_uopet = {}
    for nhg in ewzb:
        udjii_mql_uopet[nhg[0]] = {'total_conversations': nhg[1] or 0, 'total_messages': nhg[2] or 0}
    return udjii_mql_uopet
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE conversations (timestamp TEXT, message_count INTEGER, response_time_avg REAL, token_input INTEGER, token_output INTEGER, agent_name TEXT)')
    cursor.executemany('INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)', [
        ('2022-01-01', 10, 1.0, 5, 5, 'Agent1'),
        ('2022-01-02', 20, 2.0, 10, 10, 'Agent1'),
        ('2022-01-03', 30, 3.0, 15, 15, 'Agent2')
    ])
    conn.commit()
    raw_result = wybzna_rzbxbzbcjapm(conn, '2022-01-01', 2)
    assert len(raw_result) == 2
    new_result = ysxqn_gtvtzqygzasr_dfeen(conn, '2022-01-01', 2)
    assert new_result == {'Agent1': {'total_conversations': 2, 'total_messages': 30}, 'Agent2': {'total_conversations': 1, 'total_messages': 30}}
    conn.close()
    print('All tests passed')

test_both()