import numpy as np
from matplotlib import pyplot as plt
import sqlite3

def get_skiplist(archive, restrict_list, wafer_slots=[f'ws{i}' for i in range(7)], freqs=['f090', 'f150']):
    """
    Get a list of which (obs wafer freq) in an archive should be skipped
    restrict_list is a list of all strings '{obs_id}_{wafer_slot}_{freq}' that you want to keep
    """
    skiplist=[]
    freqs = np.atleast_1d(freqs)
    for ioid, oid in enumerate(list(archive.keys())):
        for iws, wafer_slot in enumerate(wafer_slots):
            if wafer_slot not in archive[oid]:
                continue
            for ifreq, freq in enumerate(freqs):
                if freq not in archive[oid][wafer_slot]:
                    continue
                if restrict_list is not None:
                    tag = '_'.join([oid, wafer_slot, freq])
                    if tag not in restrict_list:
                        skiplist.append(tag)
    return skiplist

def collect(archive, keys, wafer_slots=[f'ws{i}' for i in range(7)], freqs=['f090', 'f150'], skiplist=None, return_ids=False, flatten=True, asarray=False):
    """Collect one or more entries from an archive for all atomics in that archive
    archive: Nested dictionary containing the archive as from np.load(archive_file, allow_pickle=True)[()]
    keys: List of keys. If more than one they should be compatible ie the same dimensions
    wafer_slots: Optional list of ws strings to keep
    freqs: Optional list of strings to keep
    skiplist: List of strings '{obs_id}_{wafer_slot}_{freq}' to keep
    return_ids: If True also return the id strings (as in skiplist) of collected obs
    flatten: Flatten output arrays if True
    asarray: Convert to np.array before returning
    """
    freqs = np.atleast_1d(freqs)
    wafer_slots = np.atleast_1d(wafer_slots)
    keys = np.atleast_1d(keys)
    nskipped = 0
    out = [[] for _ in keys]
    ids = []
    for ioid, oid in enumerate(list(archive.keys())):
        for iws, wafer_slot in enumerate(wafer_slots):
            if wafer_slot not in archive[oid]:
                continue
            for ifreq, freq in enumerate(freqs):
                if freq not in archive[oid][wafer_slot]:
                    continue
                if skiplist is not None:
                    if '_'.join([oid, wafer_slot, freq]) in skiplist:
                        nskipped += 1
                        continue
                for ikey, key in enumerate(keys):
                    out[ikey].append(list(np.atleast_1d(archive[oid][wafer_slot][freq][key])))
                ids.append([oid, wafer_slot, freq])
    if nskipped > 0:
        print(f"{nskipped} wafer-bands skipped")
    if flatten:
        out = [np.array(x).flatten() for x in out]
    if asarray:
        out = [np.array(x) for x in out]
    if len(out) == 1:
        out = out[0]
    if return_ids:
        return out, np.array(ids)
    else:
        return out

def _main(tel, archive_fn, atomic_db_fn, query, savename, f_sel, f_wn, bs=4, nobs=500, vtag="", freq='f090', save_plots=False, show_plots=False):

    # Load archive of extracted preprocess data
    archive = np.load(archive_fn, allow_pickle=True)[()]

    obsids = list(archive.keys())
    print(f"{len(obsids)} obs ids")
    # keys = list((list((list(archive0.values())[0]).values())[0]).keys())  # Get all keys available in the archive

    # Restrict the list of atomics by querying the atomic db
    if atomic_db_fn is not None:
        with sqlite3.connect(atomic_db_fn) as conn:
            res = np.array(conn.cursor().execute(query).fetchall())
        xid = np.array(['_'.join(row) for row in res])
    else:
        xid = None
    skiplist = get_skiplist(archive, xid, freqs=freq)

    # Collect info from the archive
    psd_means, ids = collect(archive, ['psdQ_mean', 'psdU_mean'], freqs=freq, skiplist=skiplist, flatten=False, asarray=True, return_ids=True)
    psdQ_mean, psdU_mean = psd_means

    # Get frequencies and binned freqs
    ff = np.linspace(0, 2, psdQ_mean.shape[1])  # It's an approximation but probably good enough
    sh = psdQ_mean.shape
    binned = np.mean(np.reshape(psdQ_mean, (sh[0], sh[1]//bs, bs)), axis=-1)
    binned_f = np.mean(np.reshape(ff, (sh[1]//bs, bs)), axis=-1)


    binsort = np.where(np.logical_and(f_sel[0] < binned_f, binned_f < f_sel[1]))[0]  # freq bins to sort on
    wn_bins = np.where(np.logical_and(f_wn[0] < binned_f, binned_f < f_wn[1]))[0]  # freq bins of white noise

    wn = np.mean(binned[:,wn_bins], axis=1)
    ratio = np.mean(binned[:,binsort], axis=1) / wn  # Mean of PSD in range f_sel divided by mean white nosie level

    combo = np.concatenate([ids.T, [ratio, wn]]).T
    np.savetxt(savename, combo, header="obs_id wafer_slot freq ratio wn2", fmt='%s')

    isort = np.argsort(ratio)
    sort = binned[isort]
    sort_ids = ids[isort]

    rtag = f"{str(f_sel[0]).replace('.', 'p')}-{str(f_sel[1]).replace('.', 'p')}"
    for imax in np.arange(nobs, sort.shape[0]+nobs, nobs):
        np.save(f"select_list_{vtag}_{tel}_{freq}_wnratio{rtag}_{imax-nobs}-{imax}.npy", sort_ids[imax-nobs:imax])

    ### PLOTS ###

    normed = (binned / np.mean(binned[:, wn_bins], axis=1)[:, None])[isort]
    ## PSD plot
    for imax in np.arange(nobs, sort.shape[0]+nobs, nobs):
        plt.plot(binned_f, normed.T[:,imax-nobs:imax], color='k', alpha=0.1)
        plt.title(f"{imax-nobs}-{imax}")
        plt.yscale("log")
        plt.ylim(0.8, 50)
        plt.xlim(0.0, .5)
        plt.axvline(f_sel[0])
        plt.axvline(f_sel[1])
        plt.xlabel("Freq [Hz]")
        plt.ylabel("PSD DemodQ / $N_w$")
        if save_plots:
            plt.savefig(f"psd_demodq_wnratio{rtag}_{imax-nobs}-{imax}_norm.png")
        if show_plots:
            plt.show()
        else:
            plt.close()

    ## Histogram
    plt.hist((ratio), log=False, bins=np.linspace(0.9, 2.5, 200))
    plt.xlabel(f"$\\Sigma$({f_sel[0]}-{f_sel[1]} Hz) / White noise")
    plt.ylabel("Number of atomics")
    plt.title(f"{tel} {vtag} r")
    if save_plots:
        plt.savefig(f"{tel}_{vtag}_r_histogram.png")
    if show_plots:
        plt.show()
    else:
        plt.close()

    ## CDF
    from matplotlib.ticker import MultipleLocator
    fig, ax = plt.subplots()
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    cumsum = []
    rmax = np.linspace(0.9, 2, 400)
    for rr in rmax:
        cumsum.append(np.sum(ratio < rr))
    cumsum = np.array(cumsum, dtype=np.float64)
    cumsum /= ratio.size
    plt.plot(rmax, cumsum)
    plt.grid(which='both')
    plt.xlim(0.9, 1.6)
    plt.xlabel(f"$\\Sigma$({f_sel[0]}-{f_sel[1]} Hz) / White noise")
    plt.ylabel("Cumulative fraction of atomics")
    plt.title(f"{tel} {vtag} r")
    if save_plots:
        plt.savefig(f"{tel}_{vtag}_r_cdf.png")
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    tel="satp3"
    freq='f150'
    vtag="v1"

    f_sel = (0.04, 0.14)
    f_wn = (0.6, 1.0)
    bs = 4  # Size of frequency bins in units of frequency samples

    archive_fn = f"/home/er5768/run_mapmaker/select_lists_{tel}_{vtag}_ratio/aux/psd_info_{vtag}_{tel}.npy"
    atomic_db_fn = f"/scratch/gpfs/SIMONSOBS/sat-iso/{vtag}/mapmaking/{tel}_20250108/atomic_db.sqlite"
    query = "SELECT obs_id, wafer, freq_channel FROM atomic WHERE pwv < 2 AND split_label='science' AND median_weight_qu<2e10"

    savename = f"{tel}_{freq}_{vtag}_ratio.txt"  # Save name for the txt file containing the ratios
    nobs = 500  # How many obs to group into each saved npy file

    save_plots = False
    show_plots = False

    _main(tel=tel, archive_fn=archive_fn, atomic_db_fn=atomic_db_fn, query=query, savename=savename,
          f_sel=f_sel, f_wn=f_wn, bs=bs, nobs=nobs, vtag=vtag, freq=freq, save_plots=save_plots, show_plots=show_plots)


if __name__ == '__main__':
    main()
