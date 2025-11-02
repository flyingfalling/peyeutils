
import peyeutils as pu;

import pandas as pd;
import sys;
import os;

from multiprocessing import Pool;






def preproc_file(fn, out_csv_path, doplot=False):
    print("Setting input EDF filename to [{}]".format(fn));
    
    row, s, m, bt, b = pu.preproc_peyefv_edf(fn, out_csv_path=out_csv_path);
    
    print(s);
    print(bt);
    print(b);
    print(row);
    
    row2 = { a:[row[a]] for a in row };
    df = pd.DataFrame(row2);
    
    if(False == row['edferror'] and doplot ):
        plotit(df.iloc[0], out_csv_path);
        pass;
    print(df);
    return df;


####### parallel func wrapper ########

def parallel_preproc( mytup ):
    fn = mytup[0];
    out_csv_path = mytup[1];
    
    return preproc_file( fn, out_csv_path );

#######   end parallel    ############




def main():
    NPROC=None; # none makes num_cpu
    
    alledfcsv=sys.argv[1];
    fmriedfdir=sys.argv[2];
    outsideedfdir=sys.argv[3];
    savecsvdir=sys.argv[4];
    
    succ = pu.utils.create_dir( savecsvdir );
    
    if( False == succ ):
        raise Exception("WTF couldn't make dir {}?".format(savecsvdir));
    
    alledf_df = pd.read_csv(alledfcsv);

    alledf_df=alledf_df.iloc[:4];
    
    ## REV: prepare to run it...
    rows = [ tuple((x[1], savecsvdir)) for x in alledf_df.iterrows() ];
    
    MULTIPROC=True;
    results=list();
    if(MULTIPROC):
        with Pool(processes=NPROC) as pool:
            results = pool.map(parallel_preproc, rows);
            pass;
        pass;
    else:
        for row in rows:
            results.append( parallel_preproc(row) );
            pass;
        pass;

    #REV: only non-error EDFs...
    alledfs = pd.concat( [ r for r in results if False==r['edferror'] ]  );
    
    alltrials=list();
    for i, row in alledfs.iterrows():
        if(False == row['edferror']):
            trialdf = pd.read_csv(os.path.join(savecsvdir, row['trials_csv']));
            alltrials.append(trialdf);
            pass;
        pass;
    alltrials = pd.concat(alltrials);
    
    bigtrialdf = pd.concat(alltrials).reset_index(drop=True);
    bigedfdf = pd.DataFrame(alledfs); #pd.concat(alledfs).reset_index(drop=True);
    
    
    print(bigtrialdf);
    print(bigedfdf);
    
    bigtrialdf.to_csv('allfmritrials.csv', index=False);
    bigedfdf.to_csv('allfmriedfs.csv', index=False); #REV: ah, I am writing over it...
    
    return 0;


if __name__=='__main__':
    exit(main());
    pass;
    
