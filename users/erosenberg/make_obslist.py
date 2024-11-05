import qpoint as qp
import numpy as np
import healpy as hp
from datetime import datetime

def get_radec(az, el, ctime):
    """Get RA and Dec from azel with qpoint"""
    Q = qp.QPoint()
    pitch = None 
    roll = None
    # ~SO
    lat = -(22+57/60) * np.ones_like(ctime)
    lon = -(67+47/60) * np.ones_like(ctime)

    # calculate boresight quaternions
    q_bore = Q.azel2bore(az, el, pitch, roll, lon, lat, ctime)
    q_off = Q.det_offset(0, 0, 0)

    # calculate detector pointing
    ra, dec, sin2psi, cos2psi = Q.bore2radec(q_off, ctime, q_bore)
    return ra, dec

def get_radec_ol(ol):
    """Get RA and Dec from and obslist"""
    mtime = ol['start_time'] + ol['duration']/2
    az = ol['az_center']
    el = ol['el_center']
    ra, dec = get_radec(az, el, mtime)
    return ra, dec

def get_subset_by_dist(ol, center, nobs, maxdist=None):
    """Pick up to nobs observations closest to ra_center from an obslist.
    maxdist: float in degrees, sets maximum distance from center. None for no max.
    """
    ra, dec = get_radec_ol(ol)
    dist = get_dist(ra, dec, center)
    iobs = np.argsort(dist)[:nobs]
    if maxdist is not None:
        iobs = iobs[dist[iobs] <= maxdist]
    return ol['obs_id'][iobs], iobs


def get_dist(ras_deg, decs_deg, radec_center):
    """Get angular distance in degrees from a point.
    ras_deg and decs_deg are lists of RA and Dec in degrees
    radec_center is a 2-tuple in degrees
    """
    ras_deg, decs_deg = np.atleast_1d(ras_deg, decs_deg)
    dist = np.array([hp.rotator.angdist((ras_deg[ii], decs_deg[ii]), radec_center, lonlat=True) for ii in range(len(ras_deg))])
    dist = np.squeeze(np.rad2deg(dist))
    return dist

def main():
    from sotodlib.core import Context
    ctx1 = Context('/global/homes/r/rosenber/sobs/users/rosenber/contexts/use_this_local_satp1_241031.yaml')
    ctx3 = Context('/global/homes/r/rosenber/sobs/users/rosenber/contexts/use_this_local_satp3_241031.yaml')

    ctmin1 = datetime(2024, 9, 1).timestamp()
    ctmin3 = ctmin1

    ol1 = ctx1.obsdb.query(f'subtype=="cmb" and timestamp > {ctmin1} and roll_center > -1e-3 and roll_center < 1e-3 and hwp_freq_mean > 1.9 and hwp_freq_stdev < 0.1')
    ol3 = ctx3.obsdb.query(f'subtype=="cmb" and timestamp > {ctmin3} and hwp_freq_mean > 1.9 and hwp_freq_stdev < 0.1')

    radec_center = (0, -40)
    obs_ids, iobs = get_subset_by_dist(ol1, radec_center, 30, None)
    print(obs_ids)

if __name__ == '__main__':
    main()
