#!/bin/bash

VIDCONDCSV=vidcondgrps.csv
EDFIDX=sz_edf_index.csv #REV: this is output.

FREEVIEWINGDATADIR=/mnt/coishare/data/freeviewing/
#FREEVIEWINGDATADIR=/home/riveale/richard_home/git/freeviewingsvn
FMRIEDFDIR=$FREEVIEWINGDATADIR/data/fmri7t
OUTSIDEEDFDIR=$FREEVIEWINGDATADIR/data/fmri7t_outside_sorted/

PATIENTIDXDIR=/mnt/coishare/data/szfmri7t/schizo_outofscanner_fromtaka/patient_info/
#PATIENTIDXDIR=/home/riveale/data/schizo_outofscanner_fromtaka/patient_info/   

SZIDX=統合失調症患者_7期_心理検査データ_20250904.xlsx
HCIDX=健常者_7期_心理検査データ_20250904_wpilots.xlsx

## Computes index from subject data and EDF dir etc.
python py/00_parse_SZ_index.py $PATIENTIDXDIR/$SZIDX $PATIENTIDXDIR/$HCIDX $FMRIEDFDIR $OUTSIDEEDFDIR


