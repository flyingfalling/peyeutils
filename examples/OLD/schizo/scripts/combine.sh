#!/bin/bash

BLOCKIDX=allfmriblocks.csv
#TRIALIDX=allfmritrials.csv
TRIALIDX=allfmriblockedtrials.csv
EDFIDX=allfmriedfs.csv; #sz_edf_index.csv
SAMPCSVDIR=./outcsvs/
VIDCONDCSV=vidcondgrps.csv

python combine_gaze_saliency.py \
       --edfcsv $EDFIDX \
       --trialcsv $TRIALIDX \
       --blockcsv $BLOCKIDX \
       --vidcondcsv $VIDCONDCSV \
       --edfcsvdir $SAMPCSVDIR
