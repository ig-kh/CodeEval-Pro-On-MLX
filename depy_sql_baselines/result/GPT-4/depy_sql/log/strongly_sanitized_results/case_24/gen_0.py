import sqlite3

# The scheme of the table for task is <raw_picks (source_system TEXT, symbol TEXT, direction TEXT, strategy TEXT, asset_class TEXT)>. 
# Fetches the source system, symbol, direction, strategy, and asset class from the raw_picks table (given connection). 
# Returns list of tuples.
def njkja_hxr_msetg(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT source_system, symbol, direction, strategy, asset_class FROM raw_picks")
    rows = cursor.fetchall()
    return rows

# The scheme of the table for task is <raw_picks (source_system TEXT, symbol TEXT, direction TEXT, strategy TEXT, asset_class TEXT)>. 
# Fetches the source system, symbol, direction, strategy, and asset_class from the raw_picks table and return them as a list of dictionaries (given connection).
def huanr_xql_penco_nsne(conn):
    rows = njkja_hxr_msetg(conn)
    list_of_dict = []
    for row in rows:
        dict_row = {"source_system": row[0], "symbol": row[1], "direction": row[2], "strategy": row[3], "asset_class": row[4]}
        list_of_dict.append(dict_row)
    return list_of_dict
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE raw_picks (source_system TEXT, symbol TEXT, direction TEXT, strategy TEXT, asset_class TEXT)')
    cursor.executemany('INSERT INTO raw_picks VALUES (?, ?, ?, ?, ?)', [
        ('system1', 'symbol1', 'direction1', 'strategy1', 'asset_class1'),
        ('system2', 'symbol2', 'direction2', 'strategy2', 'asset_class2')
    ])
    conn.commit()
    
    raw_result = njkja_hxr_msetg(conn)
    assert raw_result == [('system1', 'symbol1', 'direction1', 'strategy1', 'asset_class1'),
                          ('system2', 'symbol2', 'direction2', 'strategy2', 'asset_class2')]
    
    new_result = huanr_xql_penco_nsne(conn)
    expected = [
        {'source_system': 'system1', 'symbol': 'symbol1', 'direction': 'direction1',
         'strategy': 'strategy1', 'asset_class': 'asset_class1'},
        {'source_system': 'system2', 'symbol': 'symbol2', 'direction': 'direction2',
         'strategy': 'strategy2', 'asset_class': 'asset_class2'}
    ]
    assert new_result == expected
    
    conn.close()
    print('All tests passed')

test_both()