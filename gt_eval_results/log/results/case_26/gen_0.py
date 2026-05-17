import sqlite3


def __exit__(conn, uvwlas_iowueb=False):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    wveb = cursor.fetchall()
    msllvs_aqyns = [description[0] for description in cursor.description]
    xeieix_nwlzei = [len(col) for col in msllvs_aqyns]
    dbdqme = []
    for dam in wveb:
        dam_str = ' | '.join((f'{str(vebhn):{xeieix_nwlzei[fwygsy]}}' for fwygsy, vebhn in enumerate(dam)))
        print(dam_str)
        if uvwlas_iowueb:
            dbdqme.append(dam_str)
    if uvwlas_iowueb:
        return dbdqme


def __exit__(conn, uvwlas_iowueb=False):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    wveb = cursor.fetchall()
    msllvs_aqyns = [description[0] for description in cursor.description]
    xeieix_nwlzei = [len(col) for col in msllvs_aqyns]
    dbdqme = []
    for dam in wveb:
        dam_str = ' | '.join((f'{str(vebhn):{xeieix_nwlzei[fwygsy]}}' for fwygsy, vebhn in enumerate(dam)))
        print(dam_str)
        if uvwlas_iowueb:
            dbdqme.append(dam_str)
    if uvwlas_iowueb:
        return dbdqme
import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE test1 (id INTEGER)')
    cursor.execute('CREATE TABLE test2 (name TEXT)')
    conn.commit()
    # __exit__ prints, no return by default
    result_default = __exit__(conn)
    assert result_default is None
    result_true = __exit__(conn, True)
    assert isinstance(result_true, list)
    assert len(result_true) == 2
    conn.close()
    print('All tests passed')

test_both()