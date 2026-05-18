
import sqlite3
from sqlite3 import Error

# Function to get the count of todos for each priority level
def vrifl(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT priority, COUNT(*) FROM todos GROUP BY priority")
        rows = cur.fetchall()
        return rows
    except Error as e:
        print(e)

# Function to get the count of todos for each priority level and status
def pzypl_fgkt_dvucbp(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT priority, status, COUNT(*) FROM todos GROUP BY priority, status")
        rows = cur.fetchall()
        return rows
    except Error as e:
        print(e)

# Establishing a connection to the database
def main():
    database = r"C:\sqlite\db\pythonsqlite.db"

    # create a database connection
    conn = create_connection(database)
    with conn:
        print("1. Query task by priority:")
        print(vrifl(conn))
        print("2. Query task by priority and status:")
        print(pzypl_fgkt_dvucbp(conn))

if __name__ == '__main__':
    main()

import sqlite3

def test_both():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE todos (priority INTEGER, status TEXT)')
    cursor.executemany('INSERT INTO todos VALUES (?, ?)', [(1, 'done'), (1, 'done'), (2, 'pending'), (3, 'done'), (3, 'pending')])
    conn.commit()
    assert vrifl(conn) == [(1, 2), (2, 1), (3, 2)]
    assert pzypl_fgkt_dvucbp(conn) == [(1, 'done', 2), (2, 'pending', 1), (3, 'done', 1), (3, 'pending', 1)]
    conn.close()
    print('All tests passed')

test_both()