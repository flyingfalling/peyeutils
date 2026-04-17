#REV: searches dir for infant data.
# Gets DOB from index, searches only those dirs?
# Inside there, any level of DIR, we get the .EDF files.

import pandas as pd;
import numpy as np;

import sys;
import os;

import pyedfread


edfdir = sys.argv[1];
dobidx = sys.argv[2];


dobdf = pd.read_csv(dobidx);

filelist=list();
outdir='./infantcsvs';

print("output to {}".format(outdir));
os.makedirs(outdir, exist_ok=True);

nuniques=0;

for i, row in dobdf.iterrows():
    subj=row['subj'];

    mydir=os.path.join(edfdir, subj);
    for root, dirs, files in os.walk(mydir):
        for file in files:
            if(file.lower().endswith('.edf')):
                print("{}: {}".format(subj, os.path.join(root,file)));
                myrow = row.copy();
                myrow['edffile'] = file; #os.path.join(root,file);
                myrow['edfpath'] = root;
                #myrow['msgcsv'] = file + '.messages.csv';
                #myrow['sampcsv'] = file + '.samples.csv';
                #myrow['eventcsv'] = file + '.event.csv';
                '''
                s, e, m = pyedfread.read_edf(myrow['edffile']);
                s.to_csv(os.path.join(outdir, myrow['sampcsv']), index=False);
                e.to_csv(os.path.join(outdir, myrow['eventcsv']), index=False);
                m.to_csv(os.path.join(outdir, myrow['msgcsv']), index=False);
                '''
                filelist.append(myrow);
                nuniques+=1;
                #print(row['edffile']);
                pass;
            
            pass;
        pass;
    pass;

outdf = pd.DataFrame(filelist);

if( len( outdf[ outdf['edffile'].duplicated(keep=False) ] ) > 0 ):
    raise Exception("Overlapping");

print(outdf.columns);

outdf.to_csv('infant_idx_ONLYEDFS_20260417.csv', index=False);
