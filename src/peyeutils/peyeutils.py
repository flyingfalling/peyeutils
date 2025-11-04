## Front-end (exported) functions for convenience
#REV: runs preprocessing and saves .csv files in specified dir
## Does not do anything with events other than mark blinks...
#REV: could pass other info here...
#import peyeutils.eyelink.eyelink;
#import peyeutils.preproc.preproc;

#import peyeutils.peyefv.msgutils;

#from peyeutils.utils.fsutils import *;
#import peyeutils.utils as ut;
import peyeutils as pu;
import os;
import pandas as pd;
import numpy as np;

def preproc_peyefv_edf( in_edf_path : str,
                        out_csv_path : str = None,
                       ):
    """

    Parameters
    ----------
    in_edf_path : str :  Filesystem path to edf file to read/preprocess (e.g. blah/bloop/file.edf)
        
    out_csv_path : str :  Filesystem path of directory in which to store CSV files created by this function (containing samples, messages, indices of trials/video starts/etc.).
        (Default value = None)

    Returns
    -------
    5-Tuple (row, sampdf, msgdf, trialdf, blockdf)
    row : dict : parameters and filenames of CSV files created. blocktrials_csv, blocks_csv, samples_csv, messages_csv, events_csv, edfsamples_csv, etc.

    sampdf : pandas.DataFrame : dataframe containing (preprocessed) samples

    msgdf : pandas.DataFrame : dataframe containing (preprocessed) messages from EDF
    
    trialdf : pandas.DataFrame : dataframe containing trial start/end times, video names, sizes, etc. extracted from EDF messages.

    blockdf : pandas.DataFrame : dataframe containing start/end times of blocks in the EDF file.
    
    """
    #-> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, bool):
    if( out_csv_path ):
        
        pu.utils.create_dir(out_csv_path);
        pass
    
    import pyedfread;
    
    row=dict();
    haseyetracking=True;
    error=False;
    row['edferror'] = False;
    
    #REV: expect FNAME to be UNIQUE
    fname=os.path.basename(in_edf_path);
    fdir =os.path.dirname(in_edf_path);
    
    row['edfpath'] = fdir;
    row['edffile'] = fname;
    
    print(" ++++++++ Reading [{}] ++++++++++".format(in_edf_path));
    
    try:
        s, e, m = pyedfread.read_edf(in_edf_path);
        print(s.time.min(), s.time.max());
        print(s.time);
        error=False;
        pass;
    except Exception as e:
        row['edferror'] = True;
        error=True;
        print("  -------- WARNING -- Could not read EDF file [{}], exception [{}]".format(in_edf_path, e));
        return row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    
    if( out_csv_path ):
        #mkdir(out_csv_path);
        sfn = fname + '.edfsamples.csv';
        efn = fname + '.edfevents.csv';
        mfn = fname + '.edfmessages.csv';
        
        spath=os.path.join(out_csv_path, sfn);
        epath=os.path.join(out_csv_path, efn);
        mpath=os.path.join(out_csv_path, mfn);
        
        print("Saving EDF files as CSV to [{}]  ([{}] [{}] [{}])".format(out_csv_path, sfn, efn, mfn));
        s.to_csv(spath, index=False);
        e.to_csv(epath, index=False);
        m.to_csv(mpath, index=False);
        
        row['edfsamples_csv'] = sfn;
        row['edfevents_csv'] = efn;
        row['edfmessages_csv'] = mfn;
        pass;
    
    recinfo = pu.peyefv.get_recordingsession_info(m);

    pretag='recinfo_';
    for key in recinfo:
        if( key in row ):
            print("Replacing metadata info in row for file {} (key={} original row [{}]->[{}]  (from recinfo)) -- will name everything {}PARAM".format(in_edf_path, key, row[key], recinfo[key], pretag));
            pass;
        row[pretag+key] = recinfo[key];
        pass;
    
    
    df, ev, msgs, badtrial = pu.eyelink.preproc_EL_A_clean_samples(s,e,m);
    df = pu.eyelink.preproc_EL_rawcalib_px(df, msgs);
    df = pu.peyefv.preproc_peyefreeviewing_dva_from_flatscreen(df, msgs);
    
    if( False == badtrial ):
        df = pu.preproc.preproc_SHARED_C_binoc_gaze(df, xcol='cgx_dva', ycol='cgy_dva', tcol='Tsec', exclude_thresh=2);
        pass;
    
    #df = preproc_SHARED_D_exclude_bad( df, xcol='cgx_dva', ycol='cgy_dva', badcol='bad' );
    
    if( out_csv_path ):
        #REV: preprocessed messages etc.
        sfn = fname + '.samples.csv'
        efn = fname + '.events.csv'
        mfn = fname + '.messages.csv'
        
        spath=os.path.join(out_csv_path, sfn);
        epath=os.path.join(out_csv_path, efn);
        mpath=os.path.join(out_csv_path, mfn);
        
        df.to_csv(spath, index=False);
        ev.to_csv(epath, index=False);
        msgs.to_csv(mpath, index=False);
        
        row['samples_csv'] = sfn;
        row['events_csv'] = efn;
        row['messages_csv'] = mfn;
        pass;
    
        
    
    trialdf = pu.peyefv.import_fv_trials( msgs );
    if( badtrial ):
        #trialdf['haseyetracking'] = False;
        haseyetracking=False;
        print(" BAD TRIAL (no data?)...");
        pass;
    
    
    blockdf, trialdf = pu.peyefv.import_fv_blocks(msgs, df, trialdf);
    
    trialdf['haseyetracking']=haseyetracking;
    blockdf['haseyetracking']=haseyetracking;
    row['haseyetracking'] = haseyetracking;
    
    if( out_csv_path ):
        btfname=fname+'.trials.csv';
        bfname=fname+'.blocks.csv';
        btpath=os.path.join(out_csv_path, btfname);
        bpath=os.path.join(out_csv_path, bfname);
        trialdf.to_csv(btpath, index=False);
        blockdf.to_csv(bpath, index=False);
        
        row['trials_csv'] = btfname;
        row['blocks_csv'] = bfname;
        pass;
    
    
    return row, df, msgs, trialdf, blockdf
