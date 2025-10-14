## Front-end (exported) functions for convenience
#REV: runs preprocessing and saves .csv files in specified dir
## Does not do anything with events other than mark blinks...
#REV: could pass other info here...
import os;
import peyeutils.eyelink.eyelink, peyeutils.peyefreeviewing.msgutils;
from peyeutils.utils.fsutils import *;
import pandas as pd;

def preproc_eyelink_edf( in_edf_path,
                         out_csv_path = None,
                        ):
    '''
    Returns SAMPS, MSGS, TRIALBLOCKDF, BLOCKDF, ROWDICT, ERRORBOOL
    '''

    if( out_csv_path ):
        create_dir(out_csv_path);
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), row, error;
    
    
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
    
    df, ev, msgs, badtrial = peyeutils.eyelink.eyelink.preproc_EL_A_clean_samples(s,e,m);
    

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
            
        
    
    trialdf = peyeutils.peyefreeviewing.msgutils.import_fmri_trials( msgs );
    if( badtrial ):
        #trialdf['haseyetracking'] = False;
        haseyetracking=False;
        print(" BAD TRIAL (no data?)...");
        pass;
    
        
    blockdf, trialblockdf = peyeutils.peyefreeviewing.msgutils.import_fmri_blocks(msgs, df, trialdf);
    trialdf['haseyetracking']=haseyetracking;
    trialblockdf['haseyetracking']=haseyetracking;
    blockdf['haseyetracking']=haseyetracking;
    row['haseyetracking'] = haseyetracking;
    
    
    return df, msgs, trialblockdf, blockdf, row, error;
