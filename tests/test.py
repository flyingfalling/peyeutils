import peyeutils.peyeutils
import pandas as pd;

s, m, bt, b, row, error = peyeutils.peyeutils.preproc_eyelink_edf('/home/riveale/richard_home/git/freeviewingsvn/data/bigsmall/nakazawa20251003/PYFREE_nakazawa_SIZEDVA_10__endrec_start_2025-10-03-11-07-42_end_2025-10-03-11-12-52.edf', out_csv_path='outcvs')


print(s);
print(bt);
print(b);

bt.to_csv('testb.csv');
print(row);
