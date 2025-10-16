import numpy as np
import os
import argparse
import traceback
import time
from typing import Optional, Union, Callable

from sotodlib import core
from sotodlib.core.flagman import has_all_cut, has_any_cuts, count_cuts
import sotodlib.site_pipeline.util as sp_util
from sotodlib.preprocess import preprocess_util
from sotodlib.utils.procs_pool import get_exec_env


def get_parser(parser=None):
    # This collects arguments for passing in from the command line.
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('arxiv', help="Preprocessing arxiv filepath")

    parser.add_argument('configs', help="Preprocessing Configuration File")
    
    parser.add_argument(
        '--nproc',
        help="Number of parallel processes to run on.",
        type=int,
        default=16
    )

    parser.add_argument(
        '--errlog-ext',
        help="Error log file name.",
        default='iso_noise_check_err.txt'
    )

    parser.add_argument(
        '--savename',
        help="Output dictionary save name.",
        default='iso_noise_check.npy'
    )
    return parser

def get_dict_entry(base_dir, entry, config):
    # base_dir is the directory of the arxiv
    # entry is a single obs-wafer-band entry in the arxiv
    try:
        # This part just helps collect the logger to output info to
        logger = preprocess_util.init_logger('subproc_logger')
        logger.info(f'Processing entry for {entry["dataset"]}')
        
        # This part actually grabs the archive data as shown in the previous section.
        path = os.path.join( base_dir, entry['filename'])
        logger.info(f'Getting context for {entry["dataset"]}')
        configs, context = preprocess_util.get_preprocess_context(config)
        dets = {'wafer_slot':entry['dets:wafer_slot'],
                'wafer.bandpass':entry['dets:wafer.bandpass']}
        mdata = context.get_meta(entry['obs:obs_id'], dets=dets)
        del context
        del configs
        x = mdata.preprocess
        keys = []
        vals = []
        
        # The below part is general collecting key value pairs based on whatever
        # info/stats you want to gather out of the archive.
        # I've provided a good collection of useful stats here for reference
        
        # det bias and fp cuts
        m = has_all_cut(x.det_bias_flags.valid)
        keys.append('det_bias_cuts_total')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.det_bias_flags)[m]))
        keys.append('det_bias_cuts_bg')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.bg_flags)[m]))
        keys.append('det_bias_cuts_rtes')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.r_tes_flags)[m]))
        keys.append('det_bias_cuts_gt_rfrac')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.r_frac_gt_flags)[m]))
        keys.append('det_bias_cuts_lt_rfrac')
        vals.append(np.sum(has_all_cut(x.det_bias_flags.r_frac_lt_flags)[m]))
        m = has_all_cut(x.fp_flags.valid)
        keys.append('fp_cuts')
        vals.append(np.sum(has_all_cut(x.fp_flags.fp_nans)[m]))

        # trends
        m = has_all_cut(x.trends.valid)
        keys.append('trend_cuts_total')
        vals.append(np.sum(has_all_cut(x.trends.trend_flags)[m]))

        # jumps and glitches
        m = has_all_cut(x.jumps_slow.valid)
        keys.append('jumps_slow_count')
        vals.append(count_cuts(x.jumps_slow.jump_flag)[m])
        m = has_all_cut(x.jumps_2pi.valid)
        keys.append('jumps_2pi_count')
        vals.append(count_cuts(x.jumps_2pi.jump_flag)[m])
        # m = has_all_cut(x.glitches.valid)
        # keys.append('glitch_count')
        # vals.append(count_cuts(x.glitches.glitch_flags)[m])
        # keys.append('glitch_perc')
        # vals.append(100*np.asarray([np.sum(np.ptp(r.ranges(), axis=1)) for r in x.glitches.glitch_flags.ranges])[m]/mdata.samps.count)

        # noise
        # fit
        m = has_all_cut(x.noiseT.valid)
        keys.append('noise_fit_knee')
        vals.append(x.noiseT.fit[m,0])
        keys.append('noise_fit_white')
        vals.append(x.noiseT.fit[m,1])
        keys.append('noise_fit_alpha')
        vals.append(x.noiseT.fit[m,2])
        # # no fit
        # m = has_all_cut(x.white_noise_nofit.valid)
        # keys.append('noise_nofit_white')
        # vals.append(x.white_noise_nofit.white_noise[m])
        # Q
        m = has_all_cut(x.noiseQ.valid)
        keys.append('Qfknee')
        vals.append(x.noiseQ.fit[m,0])
        keys.append('Qfknee_var')
        vals.append(x.noiseQ.cov[m,0,0])
        keys.append('Qwnl')
        vals.append(x.noiseQ.fit[m,1])
        keys.append('Qwnl_var')
        vals.append(x.noiseQ.cov[m,1,1])
        keys.append('Qalpha')
        vals.append(x.noiseQ.fit[m,2])
        keys.append('Qalpha_var')
        vals.append(x.noiseQ.cov[m,2,2])

        m = has_all_cut(x.psdQ.valid)
        fmax = 2
        fselect = x.psdQ.freqs <= fmax
        meanpsd = np.mean(x.psdQ.Pxx[m][:,fselect], axis=0)
        keys.append('psdQ_mean')
        vals.append(meanpsd)
        del meanpsd        
        
        # m = has_all_cut(x.noiseQ_nofit.valid)
        # keys.append('Qwnl_nofit')
        # vals.append(x.noiseQ_nofit.white_noise[m])
        # U
        m = has_all_cut(x.noiseU.valid)
        keys.append('Ufknee')
        vals.append(x.noiseU.fit[m,0])
        keys.append('Ufknee_var')
        vals.append(x.noiseU.cov[m,0,0])
        keys.append('Uwnl')
        vals.append(x.noiseU.fit[m,1])
        keys.append('Uwnl_var')
        vals.append(x.noiseU.cov[m,1,1])
        keys.append('Ualpha')
        vals.append(x.noiseU.fit[m,2])        
        keys.append('Ualpha_var')
        vals.append(x.noiseU.cov[m,2,2])
        # m = has_all_cut(x.noiseU_nofit.valid)
        # keys.append('Uwnl_nofit')
        # vals.append(x.noiseU_nofit.white_noise[m])

        m = has_all_cut(x.psdU.valid)
        fmax = 2
        fselect = x.psdU.freqs <= fmax
        meanpsd = np.mean(x.psdU.Pxx[m][:,fselect], axis=0)
        keys.append('psdU_mean')
        vals.append(meanpsd)
        del meanpsd

        # ptp cut
        m = has_all_cut(x.ptp_flags.valid)
        keys.append('ptp_cuts')
        vals.append(np.sum(has_all_cut(x.ptp_flags.ptp_flags)[m]))

        # Total yield
        m = has_all_cut(x.tod_stats_U.valid)
        keys.append('total_yield')
        vals.append(np.sum(m))

        # TOD stats
        m = has_all_cut(x.tod_stats_Q.valid)
        keys.append("tod_stats_Q_median")
        vals.append(x.tod_stats_Q.median[m])
        keys.append("tod_stats_Q_std")
        vals.append(x.tod_stats_Q.std[m])
        keys.append("tod_stats_Q_skew")
        vals.append(x.tod_stats_Q.skew[m])
        keys.append("tod_stats_Q_kurtosis")
        vals.append(x.tod_stats_Q.kurtosis[m])
        keys.append("tod_stats_Q_ptp")
        vals.append(x.tod_stats_Q.ptp[m])

        m = has_all_cut(x.tod_stats_U.valid)
        keys.append("tod_stats_U_median")
        vals.append(x.tod_stats_U.median[m])
        keys.append("tod_stats_U_std")
        vals.append(x.tod_stats_U.std[m])
        keys.append("tod_stats_U_skew")
        vals.append(x.tod_stats_U.skew[m])
        keys.append("tod_stats_U_kurtosis")
        vals.append(x.tod_stats_U.kurtosis[m])
        keys.append("tod_stats_U_ptp")
        vals.append(x.tod_stats_U.ptp[m])
        
        return entry['obs:obs_id'], entry['dets:wafer_slot'], entry['dets:wafer.bandpass'], keys, vals
    except Exception as e:
        # Collects errors if this fails.
        logger.info(f"Error in process for {entry['dataset']}")
        errmsg = f'{type(e)}: {e}'
        tb = ''.join(traceback.format_tb(e.__traceback__))
        return None, None, None, errmsg, tb

def main(executor, as_completed_callable, arxiv, configs, nproc, errlog_ext, savename):
    # Prepares the error log file and logger to write info and errors out to
    logger = preprocess_util.init_logger('main_proc')
    base_dir = os.path.dirname(arxiv)
    errlog = os.path.join(base_dir, errlog_ext)
    logger.info('connect to database')
    logger = preprocess_util.init_logger('main_proc')
    
    # Connects to the archive database as described in the previous section
    proc = core.metadata.ManifestDb(os.path.join(base_dir, 'process_archive.sqlite'))

    # Prepares variables to collect stats into.
    outdict = {}
    noise_dict = {}
    fit_wn = []
    nofit_wn = []
    fit_knee = []
    
    ###############################
    run_list = proc.inspect()
    logger.info('run list created')
    del proc
    logger.info('deleted database connection')

    n = 0
    ntot = len(run_list)

    # Launches a parallel processing loop that'll work on either nersc or tiger
    futures = [executor.submit(get_dict_entry, base_dir=base_dir, entry=entry, config=configs) for entry in run_list]
    for future in as_completed_callable(futures):
        try:
            obsid, ws, band, keys, vals = future.result()
            logger.info(f'{n}/{ntot}: Unpacked future for {ws}, {band}')
        except Exception as e:
            logger.info('{n}/{ntot}: Future unpack error.')
            errmsg = f'{type(e)}: {e}'
            tb = ''.join(traceback.format_tb(e.__traceback__))
            f = open(errlog, 'a')
            f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
            f.close()
            continue
        futures.remove(future)
        
        if obsid is None:
            logger.info('Writing error to log.')
            f = open(errlog, 'a')
            f.write(f'\n{time.time()}, error\n{keys}\n{vals}\n')
            f.close()
        else:
            try:
                if obsid in outdict.keys():
                    if ws in outdict[obsid].keys():
                        if not band in outdict[obsid][ws].keys():
                            outdict[obsid][ws][band]={}
                    else:
                        outdict[obsid][ws]={}
                        outdict[obsid][ws][band]={}
                else:
                    outdict[obsid] = {}
                    outdict[obsid][ws] = {}
                    outdict[obsid][ws][band] = {}

                for k, v in zip(keys, vals):
                    outdict[obsid][ws][band][k] = v
                
                logger.info(f'{n}/{ntot}: Finished with {obsid} {ws} {band}.')
            except Exception as e:
                logger.info('Packaging and saving error.')
                errmsg = f'{type(e)}: {e}'
                tb = ''.join(traceback.format_tb(e.__traceback__))
                f = open(errlog, 'a')
                f.write(f'\n{time.time()}, future.result() error\n{errmsg}\n{tb}\n')
                f.close()
                continue
        n+=1
    logger.info(f'Saving to npy file.')
    np.save(savename, outdict)

if __name__ == '__main__':
    args = get_parser().parse_args()
    rank, executor, as_completed_callable = get_exec_env(args.nproc)
    args = vars(args)  # Make it dict-like
    if rank == 0:
        main(executor=executor, as_completed_callable=as_completed_callable, **args)
