
import sqlite3
from sqlite3 import Error

# Fetches the latest 20 backtest results and returns them in a list of dictionaries
def bpk_lfugg(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM backtest_results ORDER BY timestamp DESC LIMIT 20")
        rows = cur.fetchall()
        # Convert to list of dictionaries
        col_names = [desc[0] for desc in cur.description]
        result = [dict(zip(col_names, row)) for row in rows]
        return result
    except Error as e:
        print(e)

# Fetches the latest 20 backtest results that have passed and returns them in a list of dictionaries
def pku_xjmio_bakqki(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM backtest_results WHERE passed = 1 ORDER BY timestamp DESC LIMIT 20")
        rows = cur.fetchall()
        # Convert to list of dictionaries
        col_names = [desc[0] for desc in cur.description]
        result = [dict(zip(col_names, row)) for row in rows]
        return result
    except Error as e:
        print(e)

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE backtest_results (
        strategy_id TEXT, window_start TEXT, window_end TEXT, sharpe_ratio REAL,
        max_drawdown REAL, win_rate REAL, total_trades INTEGER, passed INTEGER, timestamp TEXT
    )''')
    cursor.executemany('INSERT INTO backtest_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', [
        ('strategy1', '2022-01-01', '2022-01-31', 1.5, -0.2, 0.6, 100, 1, '2022-02-01'),
        ('strategy2', '2022-01-01', '2022-01-31', 1.2, -0.3, 0.5, 80, 0, '2022-02-02')
    ])
    conn.commit()
    
    raw_result = bpk_lfugg(conn)
    assert len(raw_result) == 2
    
    new_result = pku_xjmio_bakqki(conn)
    assert len(new_result) == 1
    assert new_result[0]['strategy_name'] == 'strategy1'
    assert new_result[0]['passed'] is True
    
    conn.close()
    print('All tests passed')

test_both()