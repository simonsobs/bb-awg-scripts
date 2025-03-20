import sqlite3
import numpy as np
import random

num_atomics = 1
seed = 1
bundle_db = "/scratch/gpfs/SIMONSOBS/sat-iso/bundling/satp1_20250108/bundles_seed17_01312025.db"  # noqa
atomic_db = "/scratch/gpfs/SIMONSOBS/sat-iso/mapmaking/satp1_20250108/atomic_db.sqlite"  # noqa
atomic_list_fname = f"/home/kw6905/bbdev/pwg-scripts/iso-sat-review/transfer_function/atomic_list_num{num_atomics}_seed{seed}.npz"  # noqa

# Bundle DB
conn = sqlite3.connect(bundle_db)
cursor = conn.cursor()
db_props = cursor.execute(
            "SELECT name FROM PRAGMA_TABLE_INFO('bundles')"
).fetchall()
db_props = [prop[0] for prop in db_props]
print(db_props)

query = cursor.execute("SELECT obs_id from bundles WHERE bundle_id == 0")
obs_id_list = np.asarray(query.fetchall())

# Atom DB
conn = sqlite3.connect(atomic_db)
cursor = conn.cursor()
db_props = cursor.execute(
    "SELECT name FROM PRAGMA_TABLE_INFO('atomic')"
).fetchall()
db_props = [prop[0] for prop in db_props]
print(db_props)

query = cursor.execute("SELECT obs_id, wafer, freq_channel from atomic")
atomic_list = query.fetchall()

final_list = []


random.seed(seed)
random.shuffle(atomic_list)

cnt = 0
for atomic_meta in atomic_list:
    if atomic_meta[0] in obs_id_list:
        cnt += 1
        if cnt > num_atomics:
            break
        final_list.append(tuple(atomic_meta))
        print(tuple(atomic_meta))
final_list = np.array(final_list)
print("final_list", final_list)
print("type(final_list)", type(final_list))

np.savez(atomic_list_fname, atomic_list=final_list)
