
#REV: this will use peyeutils to read, process, and handle infant gaze,
## converted to pix_x and pix_y, after filtering saccs, etc., and downsamples, and converts to movie_ts.

import pandas as pd;
import peyeutils as pu;
import numpy as np;

import sys, os;



from multiprocessing import Pool;



def preproc_file(fn, out_csv_path, doplot=False):
    print("Setting input EDF filename to [{}]".format(fn));
    
    row, s, m, bt, b = pu.preproc_peyefv_edf(fn, out_csv_path=out_csv_path);
    
    print(s);
    print(bt);
    print(b);
    print(row);
    
    row2 = { a:[row[a]] for a in row };
    row2df = pd.DataFrame(row2);
    
    if(False == row['edferror'] and doplot ):
        plotit(row2df.iloc[0], out_csv_path);
        pass;
    #print(df);
    return row2df;


####### parallel func wrapper ########

def parallel_preproc( mytup ):
    row = mytup[0];
    fn = os.path.join(row['edfpath'], row['edffile']);
    out_csv_path = mytup[1];
    newrow = preproc_file( fn, out_csv_path );

    for c in newrow.columns:
        if( c in row ):
            if( row[c] != newrow.iloc[0][c] ):
                raise Exception("WTF column {} does not line up? Old: {} New: {}".format(c, row[c], newrow.iloc[0][c]) );
            pass;
        row[c] = newrow.iloc[0][c];
        pass;

        
    rowdf = { c:[v] for c,v in row.items() }
    print(rowdf);
    rowdf = pd.DataFrame(rowdf);
    print(rowdf);
    return rowdf;

#######   end parallel    ############




def main():
    NPROC=12; #None; #None; # none makes num_cpu
    
    alledfcsv=sys.argv[1];
    #fmriedfdir=sys.argv[2];
    #outsideedfdir=sys.argv[3];
    savecsvdir=sys.argv[2];
    
    succ = pu.utils.create_dir( savecsvdir );
    
    if( False == succ ):
        raise Exception("WTF couldn't make dir {}?".format(savecsvdir));
    
    alledf_df = pd.read_csv(alledfcsv);
    
    #alledf_df=alledf_df.iloc[:5];
    
    
    ## REV: prepare to run it...
    rows = [ tuple((x[1], savecsvdir)) for x in alledf_df.iterrows() ]; #[nrows:];
    
    #if(len(rows) != nrows ):
    #    raise Exception("REV: wtf ");
    
    print("Executing for {} rows".format(len(rows)));
    
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
    #alledfs = pd.concat( [ r for r in results if False==r.iloc[0]['edferror'] ]  );
    
    alltrials=list();
    alledfs=list();
    for rowdf in results:
        row=rowdf.iloc[0];
        if(False == row['edferror']):
            alledfs.append(rowdf);
            trialdf = pd.read_csv(os.path.join(savecsvdir, row['trials_csv']));
            alltrials.append(trialdf);
            pass;
        pass;
    
    bigtrialdf = pd.concat(alltrials).reset_index(drop=True);
    bigedfdf = pd.concat(alledfs).reset_index(drop=True);
    
    print(bigtrialdf);
    print(bigedfdf);
    
    bigtrialdf.to_csv('allinfanttrials_FULL.csv', index=False);
    bigedfdf.to_csv('allinfantedfs_FULL.csv', index=False); #REV: ah, I am writing over it...
    
    return 0;


if __name__=='__main__':
    exit(main());
    pass;
    


exit(0);












































#REV: does binoc merge too?
def process_session( s, e, m ):
    s, e, m, paramdict = pu.eyelink.preproc_EL_A_clean_samples(s, e, m);

    #REV: from EDF.
    #print(paramdict);
    #row, s, m, bt, b = pu.preproc_peyefv_edf(fn, out_csv_path=out_csv_path);
    
    DEBUG=False;
    
    
    input_tcol_units_hzsec=1e3;
    
    targ_sr_hzsec=1000;
    interptype='polynomial';
    interporder=2;
    min_sacc_dva = 0.33;
    min_isi_sec = 0.040;
    
    
    #REV: for marmoset, e.g.
    vidpxwid=640;
    viddvawid=25.48;
    dva_per_px = viddvawid/vidpxwid; #0.039241953125; #REV: assume for marmo,  = 25.48 pix/dva. Video width is "natural 640x480, i.e. width is about 25 deg wide.
    #REV: shit I don't know the calibration, except from gap haha.
    
    inxcol='gx';
    inycol='gy';
    xcol='gxcdva';
    ycol='gycdva';
    
    s[xcol] = s[inxcol] * dva_per_px;
    s[ycol] = s[inycol] * dva_per_px;
    s[xcol] -= s[xcol].mean();
    s[ycol] -= s[ycol].mean();
    
    print(s.eye.unique());
    print(s.columns);
    
    PLOT=False;
    if(PLOT):
        sns.relplot( data=s, x='gx', y='gy', kind='scatter');
        plt.show();
        pass;
    
    tname='Tsec';
    
    sl=s[s.eye=='L'].copy();
    
    #REV; extract natural SR?
    truesr = round(pu.utils.tsutils.check_samplerate(sl, tcol=tname));
    gotsr = paramdict['samplerate'];
    
    print("Got sample rate! {} (got from EL: {})".format(truesr, gotsr));
    
    truesrs = { c:gotsr for c in sl.columns  };
    
    
    ## REV: upsample to 1K
    #REV: interpolate (upsample) to 1000 Hz...
    df = pu.utils.tsutils.interpolate_df_to_samplerate(sl,
                                                       tname,
                                                       targ_sr_hzsec,
                                                       startsec=None,
                                                       endsec=None,
                                                       method=interptype,
                                                       order=interporder,
                                                       truesrs=truesrs,
                                                       tcolunit_s=1,
                                                       );
    
    
    #REV: meth who cares about events.
    '''
    sdf, ev = pu.peyeutils.preproc_and_compute_events( df=df,
                                                       tcol=tname,
                                                       xcol=xcol,
                                                       ycol=ycol,
                                                       sr_hzsec=targ_sr_hzsec,
                                                       mainseq_err_gain=1.5,
                                                       PLOT=True,
                                                      );
    '''

    
    
    
    return;


ef='EYETRACKER_SHIZUKU__start_2023-11-15-11-24-06_end_2023-11-15-11-33-32.edf.event.csv'
mf='EYETRACKER_SHIZUKU__start_2023-11-15-11-24-06_end_2023-11-15-11-33-32.edf.messages.csv'
sf='EYETRACKER_SHIZUKU__start_2023-11-15-11-24-06_end_2023-11-15-11-33-32.edf.samples.csv'
csvpath='./infantcsvs';
s = pd.read_csv(os.path.join(csvpath, sf));
e = pd.read_csv(os.path.join(csvpath, ef));
m = pd.read_csv(os.path.join(csvpath, mf));

df = process_session( s, e, m );

