# The scheme of the table for task is <backtest_results (strategy_id TEXT, window_start TEXT, window_end TEXT, sharpe_ratio REAL, max_drawdown REAL, win_rate REAL, total_trades INTEGER, passed INTEGER, timestamp TEXT)>. Fetches the latest 20 backtest results and returns them in a list of dictionaries (given connection).
def bpk_lfugg(conn):    query = """
    SELECT 
    strategy_id,
    window_start,
    window_end,
    sharpe_ratio,
    max_drawdown,
    win_rate,
    total_trades,
    passed,
    timestamp
    FROM backtest_results
    ORDER BY timestamp DESC
    LIMIT 20
    """
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    return [dict(zip(['strategy_id', 'window_start', 'window_end', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades', 'passed', 'timestamp'], row)) for row in results]

# The scheme of the function is <backtest_results (strategy_id TEXT, strategy_name TEXT, strategy_type TEXT, sharpe_ratio REAL, max_drawdown REAL, win_rate REAL, total_trades INTEGER, passed INTEGER, timestamp TEXT)>. Fetches the latest 20 backtest results that have passed and returns them in the list of dictionaries (given connection).
def pku_xjmio_bakqki(conn):
    query = """
    SELECT 
    strategy_id,
    strategy_name,
    strategy_type,
    sharpe_ratio,
    max_drawdown,
    python
    
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