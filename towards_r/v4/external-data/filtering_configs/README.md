Run with:

1st run (20260220): no t2p template subtraction and used bundle_id = 1 for the filtering when using ext data bundle (hm) 1
- loaded soconda evironments (soconda/3.12/v0.6.15):
    module use --append /scratch/gpfs/SIMONSOBS/modules
    module load soconda/3.12/v0.6.15
- on top of that installed: [sotodlib](https://github.com/simonsobs/sotodlib/tree/0a061ce9804f2b80da9e52137323f79ae34f7b1c)
Resulting maps: /scratch/gpfs/SIMONSOBS/external_data_e2e/v4/filtered_ext_data_20260220

2nd run (20260304): with t2p template subtraction and always used bundle_id = 0 for the filtering
- loaded latest soconda evironment (soconda/3.12/v0.6.16):
- on top of that installed: [sotodlib](https://github.com/adrien-laposta/sotodlib/tree/e23958880a15ad305afd7b3d4d8f3033a3350173)
- used bb-awg-scripts branch: https://github.com/simonsobs/bb-awg-scripts/tree/f9027962d10782c9a4477a24c8c64631bc746894
Resulting maps: /scratch/gpfs/SIMONSOBS/external_data_e2e/v4/filtered_ext_data_20260304
