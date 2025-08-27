from sotodlib import core
from sotodlib.utils.procs_pool import get_exec_env
import sqlite3, numpy as np
import os
from pixell import enmap

def get_col_names(con, table_name):
    """Get column names from an sqlite table"""
    query = f"SELECT * FROM {table_name}"
    if isinstance(con, str):
        con = sqlite3.connect(con)
        cur = con.execute(query)
        con.close()
    else:
        cur = con.execute(query)
    desc = cur.description
    return [entry[0] for entry in desc]

def calc_science(atomic_path, rows, shape, wcs, i_ctime, col_dict):
    row = list(rows[i_ctime])
    nmissing_sub = 0
    out = []
    ctime = row[4]
    for freq in ['f090','f150']:
        for waf in ['ws0','ws1','ws2','ws3','ws4','ws5','ws6']:
            list_of_avail_splits = []
            for spl in ['det_left','det_right','det_in','det_out','det_upper','det_lower','scan_left','scan_right']:
                if os.path.isfile(atomic_path + '%s/atomic_%i_%s_%s_%s_weights.fits'%(str(ctime)[0:5],ctime,waf,freq,spl)):
                    list_of_avail_splits.append(spl)
            Nsplits = len(list_of_avail_splits)
            if Nsplits > 0:
                # I need to co-add the weights of scan_left and scan_right
                div_stack = enmap.zeros((3,)+shape,wcs=wcs)
                try:
                    for spl in ['scan_left','scan_right']:
                        div = enmap.read_map(atomic_path + '%s/atomic_%i_%s_%s_%s_weights.fits'%(str(ctime)[0:5],ctime,waf,freq,spl))
                        div = enmap.extract(div, shape, wcs)
                        div_stack += div
                except FileNotFoundError:
                    nmissing_sub += 1
                    continue
                # Now we calculate the new weight
                sub_weights = div_stack
                mean_qu = np.mean(sub_weights[1:], axis=0)
                positive = np.where(mean_qu > 0)
                sumweights = np.sum(mean_qu[positive])
                meanweights = np.mean(mean_qu[positive])
                medianweights = np.median(mean_qu[positive])

                # we prepare the list to insert. This will have all of the info for the obs, but with a random freq, wafer, split
                list_to_insert = row.copy()
                    
                list_to_insert[col_dict["freq_channel"]] = freq
                list_to_insert[col_dict["wafer"]] = waf
                list_to_insert[col_dict["split_label"]] = 'science'
                list_to_insert[col_dict["valid"]] = 0
                list_to_insert[col_dict["split_detail"]] = ','.join(map(str, list_of_avail_splits))
                list_to_insert[col_dict["prefix_path"]] = ''
                list_to_insert[col_dict["total_weight_qu"]] = sumweights
                list_to_insert[col_dict["mean_weight_qu"]] = meanweights
                list_to_insert[col_dict["median_weight_qu"]] = medianweights

                #print(list_to_insert)
                tuple_ = tuple(list_to_insert)
                out.append(tuple_)
    return out, nmissing_sub
def main(executor, as_completed_callable):
    shape, wcs = enmap.read_map_geometry("/scratch/gpfs/SIMONSOBS/so/science-readiness/footprint/v20250306/so_geometry_v20250306_sat_f090.fits")
    atomic_path = "/scratch/gpfs/SIMONSOBS/sat-iso/v2/mapmaking/satp3_20250507_bs/atomic_maps/"
    con = sqlite3.connect("/scratch/gpfs/SIMONSOBS/sat-iso/v2/mapmaking/satp3_20250507_bs/atomic_db.sqlite.updated2")
    cur = con.cursor()
    cur.execute("SELECT * FROM atomic ")
    rows = cur.fetchall()
    rows = np.array(rows)

    ctime_list = rows[:,4]
    ctime_list = np.unique(ctime_list)

    # find indices of unique ctimes
    indices = []
    for ctime in ctime_list:
        indices.append(np.where(rows[:,4] == ctime)[0][0])

    col_names = np.array(get_col_names(con, 'atomic'))
    col_dict = {}
    for ii, col_name in enumerate(col_names):
        col_dict[col_name] = ii
        
    print(len(indices))
    nmissing=0
    ncomplete = 0
    futures = [executor.submit(calc_science, atomic_path, rows, shape, wcs, i_ctime, col_dict) for i_ctime in indices]
    for future in as_completed_callable(futures):
        row_out, nmissing_sub = future.result()
        nmissing += nmissing_sub
        for tuple_ in row_out:
            #cur.executemany("INSERT INTO atomic VALUES(?)", sites)
            cur.execute(f"INSERT INTO atomic VALUES ({', '.join(['?']*len(tuple_))})", tuple_)
        ncomplete += 1
        if ncomplete % 100 == 0:
            print(f'{ncomplete+1} / {len(indices)}')
    con.commit()
    con.close()
    print(f"{nmissing} missing")

if __name__ == '__main__':
    nprocs=56
    rank, executor, as_completed_callable = get_exec_env(nprocs)
    if rank == 0:
        main(executor, as_completed_callable)
