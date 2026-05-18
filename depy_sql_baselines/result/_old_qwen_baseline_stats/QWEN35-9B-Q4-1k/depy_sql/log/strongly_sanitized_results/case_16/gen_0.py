# The scheme of the table for task is <threats (reasons TEXT, is_phishing INT)>. Generate security recommendations based on analysis. Given a connection to a database with a 'threats' table (columns: reasons TEXT as JSON, is_phishing INT), return top 10 reasons by count where is_phishing=1.
def cdb_dgmvdghw_jxtalcjrwhxqvnz(conn):    import math
    import re
    import sqlite3
    from typing import List


    """
    Generate security recommendations based on analysis.

    Args:
        conn: SQLite connection object

    Returns:
        List of tuples (reason, count) for top 10 reasons where is_phishing=1
    """
    cursor = conn.cursor()

    # Step 1: Parse the JSON reasons and count them.
    # Since SQLite doesn't have a native JSON parser in older versions, we assume it's stored as text.
    # We need to parse the JSON string into a Python dict, then count the keys.
    # However, the problem says "reasons TEXT as JSON". This implies each row has a JSON string.
    # We need to count the occurrences of each key in the JSON strings where is_phishing=1.

    # Since we cannot easily do this in SQL without extensions, we will do it in Python.
    # But the instruction says "the solution of the second problem requires single or multiple calls to the first solution".
    # This implies we should call `cdb_dgmvdghw_jxtalcjrwhxqvnz` inside `jko_xsc_sptpqi_qmpralz`.
    # But the second problem has a different table schema (<threats (source_ip, severity, is_phishing, alert_time)>).
    # The first problem uses a different schema (<threats (reasons, is_phishing)>).
    # This is a contradiction. The instruction says "the solution of the second problem requires single or multiple calls to the first solution".
    # This implies the table schema is the same for both problems, or the second problem should use the result of the first.
    # But the schemas are different.

    # Let's assume the instruction implies that we should use the logic of the first solution as a template or helper,
    # but since the schemas are different, we can't directly call it.
    # However, if we assume the table schema is dynamic or we are supposed to write a generic solution.
    # But the instruction says "Write a solution of python file to the following problems".
    # And the schemas are explicitly given in the comments.

    # If the schemas are different, we cannot call the first solution to solve the second problem.
    # Unless... the instruction implies that we should write the code such that if the schema was the same, it would work.
    # Or maybe I should ignore the schema difference and just write the code for the second problem,
    # but ensure that if I were to call the first solution (if it existed for this schema), it would be used.
    # But that doesn't make sense.

    # Let's re-read the instruction carefully.
    # "Write a solution of python file to the following problems, the solution of the second problem requires single or multiple calls to the first solution."
    # This is a constraint on the implementation of the second problem.
    # It means `jko_xsc_sptpqi_qmpralz` must call `cdb_dgmvdghw_jxtalcjrwhxqvnz`.
    # But the schemas are different.
    # This is a logical impossibility if we assume the database state is fixed.

    # However, in these types of coding challenges, sometimes the table schema is assumed to be the same for both tasks
    # even if the comments describe different columns, or the second task is a variation of the first.
    # But here the columns are completely different (reasons vs source_ip, severity).

    # Let's assume there is a mistake in the problem description or I should treat them as independent.
    # But the instruction is explicit: "requires single or multiple calls to the first solution".
    # This implies I must call `cdb_dgmvdghw_jxtalcjrwhxqvnz` inside `jko_xsc_sptpqi_qmpralz`.
    # This is impossible if the schemas are different.

    # Let's assume the table schema is actually the same for both, and the comments are just examples.
    # Or maybe I should write the code for the first problem, and then for the second problem,
    # I should call the first problem's function (which might be a generic analyzer) and then modify the result.
    # But the first problem returns (reason, count). The second problem needs (source_ip, count, avg_severity).
    # This doesn't match.

    # Let's assume the instruction is a template and I should follow the pattern:
    # Problem 1: Do X.
    # Problem 2: Do Y, which requires calling the function that does X.
    # So I will write the code for Problem 1, and then
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