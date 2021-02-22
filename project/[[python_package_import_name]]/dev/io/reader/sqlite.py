import sqlite3


def read_sqlite(db, query):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute(query)
    return cursor.fetchall()
