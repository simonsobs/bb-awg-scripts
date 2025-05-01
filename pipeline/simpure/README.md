To run every stage do the following:

Ensure all runscripts have updated output directories, also the yaml files.
The first two steps are optional - they produce the soopercool analysis mask, which is stored under
```
/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/soopercool_inputs/masks/analysis_mask_apo10_C1.fits
```

1. Getting the observation patch
```
bash satp3/get_optimal_patch_atomics_satp3.sh
```

2. Getting the analysis mask
```
bash get_analysis_mask.sh
```

Steps 3 and 4 are optional - the result are filtered and unfiltered CMB-E, CMB-EB, and
Gaussian power-law simulations, stored under:
```
/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/cmb_sims
```
and
```
/scratch/gpfs/SIMONSOBS/users/kw6905/simpure/input_sims
```

3. Filtering and coadding TF sims
I ususally run those on tiger3 interactive `salloc -N4 -c 112 -q interactive -t 04:00:00`.
```
time bash filter_sims.sh;
bash coadd_atomics.sh
```

4. Filtering and coadding CMB sims
Change the top line in the .sh scripts from `. setup_tf_sims.sh` to `. setup_cmb_sims.sh`, and repeat step 3.

5. Compute transfer function power spectra.
I ususally run those on tiger3 interactive `salloc -N4 -c 112 -q interactive -t 04:00:00` (1 hour should suffice as well).
```
bash compute_tf_sims_power_spectra.sh
```
Then, on a login node (this is much less expensive):
```
bash compute_transfer_function.sh
```