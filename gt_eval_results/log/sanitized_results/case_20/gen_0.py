import sqlite3


def bpk_lfugg(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT strategy_id, window_start, window_end, sharpe_ratio, max_drawdown, win_rate, total_trades, passed FROM backtest_results ORDER BY timestamp DESC LIMIT 20')
    rows = cursor.fetchall()
    return [
        {'strategy_name': row[0], 'window_start': row[1], 'window_end': row[2],
         'sharpe_ratio': row[3], 'max_drawdown': row[4], 'win_rate': row[5],
         'total_trades': row[6], 'passed': bool(row[7])}
        for row in rows
    ]


def pku_xjmio_bakqki(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT strategy_id, window_start, window_end, sharpe_ratio, max_drawdown, win_rate, total_trades, passed FROM backtest_results WHERE passed = 1 ORDER BY timestamp DESC LIMIT 20')
    rows = cursor.fetchall()
    return [
        {'strategy_name': row[0], 'window_start': row[1], 'window_end': row[2],
         'sharpe_ratio': row[3], 'max_drawdown': row[4], 'win_rate': row[5],
         'total_trades': row[6], 'passed': bool(row[7])}
        for row in rows
    ]
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