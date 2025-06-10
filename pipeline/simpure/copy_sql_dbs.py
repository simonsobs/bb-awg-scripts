#!/usr/bin/env python
import os
import argparse
import sqlite3
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def sqlite_db(path):
    if not os.path.isfile(path):
        print(f"{path} doesn't exist; creating new file.")

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # test if this is really sqlite file
    cur = conn.cursor()
    cur.execute('SELECT 1 from sqlite_master where type = "table"')
    try:
        data = cur.fetchone()
    except sqlite3.DatabaseError:
        msg = '%s can\'t be read as SQLite DB' % path
        raise argparse.ArgumentTypeError(msg)

    return conn


def copy_db_structure(src_db, dst_db):
    src_cur = src_db.cursor()
    dst_cur = dst_db.cursor()

    src_cur.execute('SELECT * from sqlite_master')
    src_master = src_cur.fetchall()

    src_tables = list(filter(lambda r: r['type'] == 'table', src_master))
    src_indices = list(filter(lambda r: r['type'] == 'index'
                              and r['sql'] is not None, src_master))

    for table in src_tables:
        print('Processing table:', table['name'])
        print('Delete old table in destination db, if exists')
        dst_cur.execute("DROP TABLE IF EXISTS " + table['name'])
        print('Creating table structure')
        dst_cur.execute(table['sql'])
        table_idx = list(filter(lambda r: r['tbl_name'] == table['name'],
                                src_indices))
        for idx in table_idx:
            dst_cur.execute(idx['sql'])

    src_db.close()
    dst_db.close()


parser = argparse.ArgumentParser(description='Merge data from src to dest db')
parser.add_argument('src_db', type=sqlite_db,
                    help='Source DB file path')
parser.add_argument('dst_db', type=sqlite_db,
                    help='Destination DB file path')

args = parser.parse_args()

copy_db_structure(args.src_db, args.dst_db)
