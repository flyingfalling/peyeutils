import pandas as pd
import numpy as np
import sys
import os;

import matplotlib.pyplot as plt
import seaborn as sns

import peyeutils.utils as ut;
import peyeutils as pu;
import peyeutils.eyemovements.saccadr as saccr;
import peyeutils.eyemovements.remodnav as rv;

from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype

def regressors_for_period( eventdf, sttime, entime,
                           smallbig_cutoff_dva = 6
                          ):
    
    evs = eventdf[ (eventdf['stsec'] >= sttime) &
                       (eventdf['ensec'] <= entime) ];
    
    saccblnkdf = evs[ (evs['label']=='BLNK') | (evs['label']=='SACC') ];
    saccdf = evs[ (evs['label']=='SACC') ];
    blnkdf = evs[ (evs['label']=='BLNK') ];
    isidf = evs[ (evs['label']=='ISI') ];
    
    o = dict();
    
    o['sess_dursec'] = entime-sttime;

    o['sb_scanpath'] = saccblnkdf['ampldva'].sum();  #REV: assume all sessions are the same length.
    o['s_scanpath'] = saccdf['ampldva'].sum();

    o['b_num'] = len(blnkdf.index); #REV: divide by duration to get blink rate per second
    o['b_rate'] = o['b_num'] / o['sess_dursec'];

    o['sb_num'] = len(saccblnkdf.index);
    o['s_num'] = len(saccdf.index);

    o['sb_rate'] = o['sb_num'] / o['sess_dursec'];
    o['s_rate'] = o['s_num'] / o['sess_dursec'];


    o['smallbig_cutoff_dva'] = smallbig_cutoff_dva;
    
    o['sb_small_scanpath'] = saccblnkdf[ saccblnkdf['ampldva'] < o['smallbig_cutoff_dva'] ]['ampldva'].sum();
    o['sb_small_num'] = len(saccblnkdf[ saccblnkdf['ampldva'] < o['smallbig_cutoff_dva'] ].index );
    o['s_small_num'] = len(saccdf[ saccdf['ampldva'] < o['smallbig_cutoff_dva'] ].index );

    o['sb_big_scanpath'] = saccblnkdf[ saccblnkdf['ampldva'] > o['smallbig_cutoff_dva'] ]['ampldva'].sum();
    o['sb_big_num'] = len(saccblnkdf[ saccblnkdf['ampldva'] > o['smallbig_cutoff_dva'] ].index );
    o['s_big_num'] = len(saccdf[ saccdf['ampldva'] > o['smallbig_cutoff_dva'] ].index );

    o['sb_small_rate'] = o['sb_small_num'] / o['sess_dursec'];
    o['s_small_rate'] = o['s_small_num'] / o['sess_dursec'];

    o['sb_big_rate'] = o['sb_big_num'] / o['sess_dursec'];
    o['s_big_rate'] = o['s_big_num'] / o['sess_dursec'];


    o['sb_med'] = saccblnkdf['ampldva'].median();
    o['s_med'] = saccdf['ampldva'].median();

    o['sb_std'] = saccblnkdf['ampldva'].std();
    o['s_std'] = saccdf['ampldva'].std();

    if( o['sb_small_num'] <= 0 or o['s_small_num'] <= 0):
        o['sb_bigsmall_ratio'] = np.nan;
        o['s_bigsmall_ratio'] = np.nan;
        pass;
    else:
        o['sb_bigsmall_ratio'] = o['sb_big_num'] / o['sb_small_num'];
        o['s_bigsmall_ratio'] = o['s_big_num'] / o['s_small_num'];
        pass;
    
    

    #REV: will be related to the thing...
    o['isi_med'] = isidf['dursec'].median();
    o['isi_std'] = isidf['dursec'].std();

    return o

def session_regressors_for_all_in_index( indexcsv, preproc_path, outfile='richard_ems_regressors.csv' ):
    indexdf=pd.read_csv(indexcsv);
    print(indexdf);
    
    outlist=list();
    
    for i, row in indexdf.iterrows():
        cond = row['Condition'];
        subj = row['Subject'];
        sess = row['Session'];
        print(cond, subj, sess);
        
        trialdf = pd.read_csv( os.path.join(preproc_path, row['trialscsv']) );
        eventdf = pd.read_csv( os.path.join(preproc_path, row['eventscsv']) );
        print(trialdf);
        print(eventdf);

        sttime = trialdf[ trialdf['TrainTest']=='Train' ]['st'].min();
        entime = trialdf[ trialdf['TrainTest']=='Train' ]['et'].max();

        cutoff=6;
        o = regressors_for_period( eventdf, sttime, entime ,
                               smallbig_cutoff_dva = cutoff );
        
        
        outlist.append(o);
        pass;

    outdf = pd.DataFrame(outlist);
    outdf['Level'] = 'Session';
    print(outdf);
    outdf.to_csv(outfile, index=False);
    return outdf;


lut1 = {
        "52-1_nonTC1_"  :  "22-1_nonTC1_",
        "52-2_nonTC1_"  :  "22-2_nonTC1_",
        "53-1_nonTC2_"  :  "23-1_nonTC2_",
        "53-2_nonTC2_"  :  "23-2_nonTC2_",
        "55-1_TC1_"     :  "25-1_TC1_",
        "55-2_TC1_"     :  "25-2_TC1_",
        "56-1_TSC1_"    :  "26-1_TSC1_",
        "52-1_rearr_nonTC1_" : "22-1rearr_nonTC1_", 
        "53-1_rep_nonTC2_"  : "23-1rep_nonTC2_",
    }
    
lut2 = { lut1[a]:a for a in lut1 }; #REV: invert dict
    

def trial_regressors_for_all_in_index( indexcsv, preproc_path, train_csv, test_csv,
                                       mergecols=['Condition', 'Subject', 'Session'],
                                      ):
    indexdf=pd.read_csv(indexcsv);
    print(indexdf);

    traindf = pd.read_csv(train_csv);

    testdf = pd.read_csv(test_csv);

    #REV: fix testdf..
    for a in lut2:
        testdf.loc[ (testdf['Condition']==a), 'Condition' ] = lut2[a];
        pass;
    
    '''
    testdf_u = testdf.groupby(mergecols, as_index=False).size().reset_index(drop=True);
    traindf_u = traindf.groupby(mergecols, as_index=False).size().reset_index(drop=True);
    print(testdf_u);
    print(traindf_u);
    print("Test unique", testdf_u.Condition.unique());
    print("Train unique", traindf_u.Condition.unique());
    '''
    
        
    # REV: merge on cond/subj/sess, every single trial should have an outcome (Corr)
    #REV: maybe broken because of rename? :(
    
    resultdf = pd.merge( left=traindf, right=testdf, left_on=mergecols, right_on=mergecols, how='left' );


    #REV: for each (COND,SUBJ,SESS), for each WORD,
    ##  Get (list of) trials in which word is uttered (label them ordinal_X)
    ##  Get trial start/end from indexdf TRIALS data (load from indexdf.trialscsv)
    ##   Get events from indexdf.eventscsv,
    ##   Compute params from start/end (using eventscsv).

    allsessions = list();
    for i, sessrow in indexdf.iterrows():
        #REV: this represents a SESSION
        trialdf = pd.read_csv( os.path.join(preproc_path, sessrow['trialscsv']) );
        eventdf = pd.read_csv( os.path.join(preproc_path, sessrow['eventscsv']) );
        print(trialdf.columns);
        #print(trialdf);
        print(eventdf.columns);
        
        
        
        mydf = sessrow.to_frame().T;

        sesslist=list();
        
        mytrials = pd.merge( left=resultdf, right=mydf, left_on=mergecols, right_on=mergecols, how='inner' );
        for word in mytrials['Word'].unique(): #REV: should groupby
            #print("Doing for {}".format(word));
            wordtrials = mytrials[ mytrials['Word']==word ]['TrialNum'].unique();

            for ordtrial, trial in enumerate(wordtrials):
                trialrow = trialdf[ (trialdf['v']==trial) & (trialdf['TrainTest']=='Train') ].iloc[0];
                
                #print(trialrow);
                sttime = trialrow['st'];
                entime = trialrow['et'];
                o = regressors_for_period( eventdf, sttime, entime );
                o['ordtrial'] = ordtrial;
                o['Word'] = word;
                o['TrialIdx'] = trial;
                sesslist.append(o);
                pass;
            #print(wordtrials['TrialNum'].unique());
            #print(wordtrials['Block'].unique());
            
            pass;

        worddf = pd.DataFrame(sesslist);
        for a in mergecols:
            worddf[a] = sessrow[a];
            pass;

        allsessions.append(worddf);
        pass;

    finalresults = pd.concat(allsessions).reset_index(drop=True);
    print(finalresults);
    
    finalresults.to_csv('bytrial.csv', index=False);

    longtrials = pd.melt(finalresults, id_vars=mergecols + ['ordtrial', 'TrialIdx', 'Word'], var_name='regressor', value_name='value');
    print(longtrials)
    longtrials.to_csv('bytrial_long.csv', index=False);
    
    
    adf = resultdf.groupby(mergecols, as_index=False).mean(numeric_only=True).reset_index(drop=True);
    print(adf.columns);
    adf = adf.dropna(subset=['Correct']);
    print(adf);
    

    foundlist=list();
    for i, row in indexdf.iterrows():
        cond = row['Condition'];
        subj = row['Subject'];
        sess = row['Session'];
        mydf = row.to_frame().T;
        mydf = pd.merge( left=resultdf, right=mydf, left_on=mergecols, right_on=mergecols, how='inner' );
        if( len(mydf.index) > 0 ):
            foundlist.append(row);
            pass;
        pass;

    founddf = pd.DataFrame(foundlist);
    print(founddf);
    exit(0);
    
    
    
    outlist=list();
    
    for i, row in indexdf.iterrows():
        cond = row['Condition'];
        subj = row['Subject'];
        sess = row['Session'];
        print(cond, subj, sess);
        
        trialdf = pd.read_csv( os.path.join(preproc_path, row['trialscsv']) );
        eventdf = pd.read_csv( os.path.join(preproc_path, row['eventscsv']) );
        print(trialdf);
        print(eventdf);

        sttime = trialdf[ trialdf['TrainTest']=='Train' ]['st'].min();
        entime = trialdf[ trialdf['TrainTest']=='Train' ]['et'].max();

        cutoff=6;
        o = regressors_for_period( eventdf, sttime, entime ,
                               smallbig_cutoff_dva = cutoff );
        
        
        outlist.append(o);
        pass;

    outdf = pd.DataFrame(outlist);
    
    return outdf;



def main():

    sessioncsv = sys.argv[1];
    csvdir = sys.argv[2];
    #outdf = session_regressors_for_all_in_index( sessioncsv, csvdir );
    
    if( len(sys.argv) > 3 ):
        train_timings_csv=sys.argv[3];
        train_outcomes = sys.argv[4];
        trial_outdf = trial_regressors_for_all_in_index( sys.argv[1], sys.argv[2], train_timings_csv, train_outcomes );
        pass;
    
    return 0;


if __name__=='__main__':
    exit( main() );
    raise Exception();
