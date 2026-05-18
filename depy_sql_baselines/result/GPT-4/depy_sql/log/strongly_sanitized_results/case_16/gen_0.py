
import sqlite3
import json
from collections import Counter
from operator import itemgetter

# The scheme of the table for task is <threats (reasons TEXT, is_phishing INT)>. Generate security recommendations based on analysis. Given a connection to a database with a 'threats' table (columns: reasons TEXT as JSON, is_phishing INT), return top 10 reasons by count where is_phishing=1.
def cdb_dgmvdghw_jxtalcjrwhxqvnz(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT reasons FROM threats WHERE is_phishing=1")
    rows = cursor.fetchall()

    reasons_counter = Counter()
    for row in rows:
        reasons = json.loads(row[0])
        reasons_counter.update(reasons)

    top_10_reasons = reasons_counter.most_common(10)
    return top_10_reasons

# The scheme of the table for task is <threats (reasons TEXT, is_phishing INT)>. Given a threats table with columns (source_ip, severity, is_phishing, alert_time), return top K source IPs with highest count of events where severity > 5, along with count and average severity. Sort by count descending, then by avg severity descending.
def jko_xsc_sptpqi_qmpralz(conn, k):
    cursor = conn.cursor()
    cursor.execute("SELECT source_ip, severity FROM threats WHERE severity > 5")
    rows = cursor.fetchall()

    ip_counter = Counter()
    severity_sum = {}
    for row in rows:
        ip, severity = row
        ip_counter[ip] += 1
        if ip in severity_sum:
            severity_sum[ip].append(severity)
        else:
            severity_sum[ip] = [severity]

    ip_severity_avg = {ip: sum(severity_list)/len(severity_list) for ip, severity_list in severity_sum.items()}

    ip_count_severity = [(ip, count, ip_severity_avg[ip]) for ip, count in ip_counter.items()]
    ip_count_severity.sort(key=itemgetter(1, 2), reverse=True)

    return ip_count_severity[:k]

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    # Create table for raw function
    cursor.execute('CREATE TABLE threats (reasons TEXT, is_phishing INT)')
    raw_data = [
        ('["malware"]', 1), ('["malware"]', 1), ('["phishing"]', 1), ('["malware"]', 0), ('["spam"]', 1)
    ]
    cursor.executemany('INSERT INTO threats VALUES (?, ?)', raw_data)
    
    # Create table for new function (different schema)
    cursor.execute('CREATE TABLE threats2 (source_ip TEXT, severity INT, is_phishing INT, alert_time TEXT)')
    # We'll use a separate table for new; but to avoid confusion, we'll drop and recreate? Better to have two different tables. However the new function expects table named 'threats'. So we must rename or use separate. Since test runs sequentially, we can create a new connection for new function. Simpler: test raw first, then close and open new connection for new.
    
    # Test raw
    raw_result = cdb_dgmvdghw_jxtalcjrwhxqvnz(conn)
    expected_raw = [{'reasons': ['malware'], 'count': 2}, {'reasons': ['phishing'], 'count': 1}, {'reasons': ['spam'], 'count': 1}]
    assert raw_result == expected_raw
    
    # Test new with separate connection
    conn2 = sqlite3.connect(':memory:')
    cursor2 = conn2.cursor()
    cursor2.execute('CREATE TABLE threats (source_ip TEXT, severity INT, is_phishing INT, alert_time TEXT)')
    cursor2.executemany('INSERT INTO threats VALUES (?,?,?,?)', [
        ('192.168.1.1', 8, 1, '2025-01-01'),
        ('192.168.1.1', 6, 1, '2025-01-02'),
        ('192.168.1.2', 9, 1, '2025-01-01'),
        ('192.168.1.2', 7, 0, '2025-01-03'),
        ('192.168.1.3', 4, 1, '2025-01-04'),
        ('192.168.1.1', 10, 1, '2025-01-05'),
        ('192.168.1.4', 8, 1, '2025-01-06'),
        ('192.168.1.4', 9, 1, '2025-01-07')
    ])
    conn2.commit()
    
    new_result = jko_xsc_sptpqi_qmpralz(conn2, 2)
    assert len(new_result) == 2
    assert new_result[0]['source_ip'] == '192.168.1.1'
    assert new_result[0]['count'] == 3
    assert abs(new_result[0]['avg_severity'] - 8.0) < 0.001
    assert new_result[1]['source_ip'] == '192.168.1.4'
    assert new_result[1]['count'] == 2
    assert abs(new_result[1]['avg_severity'] - 8.5) < 0.001
    
    conn2.close()
    conn.close()
    print('All tests passed')

test_both()