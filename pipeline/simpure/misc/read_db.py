# A helper script that reads an sqlite3 database and prints out the columns
# and the first 10 rows.

import sqlite3
import numpy as np

#db_file = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/sky_patch/satp1_iso/atomic_db_ws0.sqlite"
#db_file = "/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/sky_patch/satp1_iso/bundles_south_seed17_20250324.db"
#db_file = "/scratch/gpfs/SIMONSOBS/sat-iso/mapmaking/satp1_20250108/atomic_db.sqlite"
#db_file = "/cephfs/soukdata/user_scratch/jkania/sat-iso/v2/mapmaking/satp1_20250507/atomic_db.sqlite"
#db_file = "/cephfs/soukdata/user_scratch/jkania/sat-iso/v2/bundling/satp1_20250507/atomic_db.sqlite"
#db_file = "/cephfs/soukdata/user_scratch/jkania/sat-iso/v2/bundling/satp1_20250507/bundles_south_seed10_20250616.db"
#db_file = "/scratch/gpfs/SIMONSOBS/sat-iso/mapmaking/satp1_20250108/bundles/bundles_south_seed17_20250312.db"
db_file = "/cephfs/soukdata/user_data/kwolz/simpure/sky_patch/v1_20251006/atomic_db_ws0.sqlite"
#db_file = "/cephfs/soukdata/user_data/kwolz/simpure/sky_patch/satp3_iso/atomic_db_ws0.sqlite"

con = sqlite3.connect(db_file)
cur = con.cursor()

table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
print(db_file)
print(table_list)

for table in table_list:
    print(f"\nReading from table {table[0]}:")

    result = cur.execute(f"SELECT * from {table[0]}")
    #result = cur.execute(f"PRAGMA table_info({table[0]})")
    names = [description[0] for description in result.description]
    rows = [r for r in result]
    # print(rows)

    # import matplotlib.pyplot as plt
    # plt.hist([r[0] for r in rows], bins=100)
    # plt.savefig("ctimes_hist.png")
    # plt.clf()

    for ir, r in enumerate(rows[:1]):
        print(" ", r)
    print(len(rows), "rows")
    # print(np.unique(np.array(rows)), "unique rows")


con.close()
