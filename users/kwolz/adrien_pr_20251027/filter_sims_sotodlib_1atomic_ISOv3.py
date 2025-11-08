import sys
import os
import sotodlib.preprocess.preprocess_util as pp_util
from pixell import enmap, enplot
import datetime
import time

# TODO: Replace by your local bb-awg-scrips install
sys.path.append(
    "/home/kw6905/bbdev/bb-awg-scripts/pipeline/filtering"
)
import filtering_utils as fu  # noqa


def main():
    """
    Note: make sure to checkout to sotodlib master and pip install . first!
    """
    if os.getcwd() != "/home/kw6905/bbdev/iso-sat/v3":
        print("Please do: \n"
              "cd /home/kw6905/bbdev/iso-sat/v3; "
              f"python {os.path.dirname(os.path.abspath(__file__))}/filter_sims_sotodlib_1atomic_old_master.py")
        sys.exit()

    car_map_template = "/scratch/gpfs/SIMONSOBS/so/science-readiness/footprint/v20250306/so_geometry_v20250306_sat_f090.fits"  # noqa
    freq_channel = "f090"
    obs_id = "obs_1716189238_satp1_1111101"
    wafer = "ws1"
    preprocess_config_init = "preprocessing/satp1/preprocessing_config_20250801_init.yaml"  # noqa
    preprocess_config_proc = "preprocessing/satp1/preprocessing_config_20250801_proc.yaml"  # noqa
    sim_fname = "/scratch/gpfs/SIMONSOBS/sat-iso/v3/simulations/cmb_fg_gaussian_20250925/unfiltered/sims/0000/maps_car_d9s4_f090_SATp3_1arcmin_0000.fits"  # noqa
    out_dir = "/home/kw6905/bbdev/bb-awg-scripts/users/kwolz/adrien_pr_20251027/out/old_master"
    fp_thin = 8

    # Initialize the logger
    logger = pp_util.init_logger("benchmark", verbosity=3)

    # Arguments related to pixellization
    _, wcs = enmap.read_map_geometry(car_map_template)
    nside = None
    mfmt = ".fits"

    # Load preprocessing pipeline and extract from it list of preprocessing
    # metadata (detectors, samples, etc.) corresponding to each atomic map
    configs_init, _ = pp_util.get_preprocess_context(
        preprocess_config_init
    )
    configs_proc, ctx_proc = pp_util.get_preprocess_context(
        preprocess_config_proc
    )

    # Get axis manager metadata for the given obs
    dets = {"wafer_slot": wafer, "wafer.bandpass": freq_channel}
    meta = ctx_proc.get_meta(obs_id=obs_id, dets=dets)

    # Focal plane thinning
    if fp_thin is not None:
        fp_thin = int(fp_thin)
        thinned = [
            m for im, m in enumerate(meta.dets.vals)
            if im % fp_thin == 0
        ]
        meta.restrict("dets", thinned)

    # Process data here to have t2p leakage template
    # Only need to run it once for all simulations
    # and only the pre-demodulation part.
    data_aman = pp_util.multilayer_load_and_preprocess(
        obs_id,
        configs_init,
        configs_proc,
        meta=meta,
        logger=logger,
        init_only=True,
    )
    print(data_aman)

    logger.info("Loading TOD")
    logger.debug(f"Loading CAR map: {sim_fname}")
    sim = enmap.read_map(sim_fname)

    # run filtering and preprocessing
    aman = pp_util.multilayer_load_and_preprocess_sim(
        obs_id,
        configs_init=configs_init,
        configs_proc=configs_proc,
        sim_map=sim,
        meta=meta,
        logger=logger,
        t2ptemplate_aman=data_aman
    )
    logger.info("Filtering done.")

    # Run the mapmaker
    wmap_dict, weights_dict = fu.make_map_wrapper(
        aman, ["scan_left"], "car", shape=None, wcs=wcs,
        nside=nside, logger=logger
    )
    wmap = wmap_dict["scan_left"]
    w = weights_dict["scan_left"]

    # Saving filtered atomics to disk
    atomic_fname = sim_fname.split("/")[-1].replace(
        mfmt,
        f"_{obs_id}_{wafer}_scan_left{mfmt}"
    )
    f_wmap = f"{out_dir}/{atomic_fname.replace(mfmt, '_wmap' + mfmt)}"
    f_w = f_wmap.replace('_wmap' + mfmt, '_weights' + mfmt)
    enmap.write_map(f_wmap, wmap)
    enmap.write_map(f_w, w)
    logger.info("Filtering & mapping done.")

    enplot.write(
        f_wmap.replace(".fits", ""),
        enplot.plot(wmap, colorbar=True, ticks=10)
    )


if __name__ == "__main__":
    ntries = 1
    for i in range(ntries):
        print(f"{i+1}/{ntries} at {datetime.datetime.now()}")
        try:
            main()
        except OSError as err:
            print(f"OSError at {datetime.datetime.now()}: {err}")
            print("Waiting for 2 minutes...")
            time.sleep(120)
