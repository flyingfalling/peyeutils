#!/bin/bash

BLOCKIDX=allfmriblocks.csv
TRIALIDX=allfmritrials.csv
EDFIDX=sz_edf_index.csv
SAMPCSVDIR=./outcsvs/
VIDCONDCSV=vidcondgrps.csv

FMRIEDFDIR=/mnt/coishare/data/freeviewing/data/fmri7t
OUTSIDEEDFDIR=/mnt/coishare/data/freeviewing/data/fmri7t_outside_sorted/

## Simple plotting
python plot_fmri_drift.py $EDFIDX $TRIALIDX $BLOCKIDX $VIDCONDCSV $SAMPCSVDIR

