#!/bin/bash

BLOCKIDX=allfmriblocks.csv
TRIALIDX=allfmritrials.csv
EDFIDX2=allfmriedfs.csv
EDFIDX=sz_edf_index.csv
SAMPCSVDIR=./outcsvs/
VIDCONDCSV=vidcondgrps.csv

FMRIEDFDIR=/mnt/coishare/data/freeviewing/data/fmri7t
OUTSIDEEDFDIR=/mnt/coishare/data/freeviewing/data/fmri7t_outside_sorted/

## Extracts trials etc. (multithreaded)
python py/01_extract_SZ_edfs.py $EDFIDX $FMRIEDFDIR $OUTSIDEEDFDIR $SAMPCSVDIR


