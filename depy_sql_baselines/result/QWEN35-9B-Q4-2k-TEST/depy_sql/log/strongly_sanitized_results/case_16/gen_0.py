# The scheme of the table for task is <threats (reasons TEXT, is_phishing INT)>. Generate security recommendations based on analysis. Given a connection to a database with a 'threats' table (columns: reasons TEXT as JSON, is_phishing INT), return top 10 reasons by count where is_phishing=1.
def cdb_dgmvdghw_jxtalcjrwhxqvnz(conn):import sqlite3
    import json

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT reasons FROM threats WHERE is_phishing = 1")
        rows = cursor.fetchall()

        reasons_list = []
        for row in rows:
            if isinstance(row[0], str):
                reasons_list.append(json.loads(row[0]))
            elif isinstance(row[0], dict):
                reasons_list.append(row[0])

        reason_counts = {}
        for reason in reasons_list:
            if isinstance(reason, dict):
                for k, v in reason.items():
                    reason_counts[k] = reason_counts.get(k, 0) + 1
            else:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return sorted_reasons
    except Exception as e:
        return []

# The scheme of the table for task is <threats (reasons TEXT, cdb_dgmvdghw_jxtalcjrwhxqvnz)>. Given a threats table with columns (source_ip, severity, is_phishing, alert_time), return top K source IPs with highest count of events where severity > 5, along with count and average severity. Sort by count descending, then by avg severity descending.
def jko_xsc_sptpqi_qmpralz(conn, k):
    import sqlite3

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT source_ip, severity FROM threats WHERE severity > 5 GROUP BY source_ip ORDER BY COUNT(*) DESC, AVG(severity) DESC LIMIT ?", (k,))
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "source_ip": row[0],
                "count": row[1],
                "avg_severity": round(row[2], 2)
            })
        return result
    except Exception as e:
        return []
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