#from peyeutils.defs import *;
#from peyeutils.peyefv.msgutils import *;

import numpy as np;
import pandas as pd;

import peyeutils as pu;

import peyeutils.peyefv as pfv;
import peyeutils.utils as ut;
import peyeutils.eyelink as el;

#from peyeutils.utils.tsutils import *;
#from peyeutils.utils.nputils import *;
#from peyeutils.utils.unitutils import *;

import peyeutils.preproc as pre;
#from peyeutils.preproc.preproc import *;



def preproc_EL_A00_add_Tsec(rawdf,
                            timecol='time',
                            timeunitsec=pu.EL_TIMEUNIT_SEC):
    """

    Parameters
    ----------
    rawdf :
        
    timecol :
         (Default value = 'time')
    timeunitsec :
         (Default value = pu.EL_TIMEUNIT_SEC)

    Returns
    -------

    """
    tvals = rawdf[timecol].to_numpy();
    rawdf = rawdf[ [c for c in rawdf.columns if c!=timecol] ]; #drop old timecol
    rawdf['EL'+timecol] = tvals;
    rawdf['Tsec'] = tvals * timeunitsec;
    MSEC=1e3;
    rawdf['Tmsec'] = rawdf['Tsec'] * MSEC;
    
    
    
    
    return rawdf;


def preproc_EL_A01_resample_time( samples,
                                  samplerate,
                                  ELtname='time',
                                  tname='Tmsec',
                                  tsecname='Tsec',
                                  timeunit=1e-3 ):
    
    if( len(samples.index) < 2 ):
        print("preproc_EL_A01_resample_time: no samples in df (you should exclude this data anyways)");
        return pd.DataFrame();
    
    print( "Will resample");
    print( samples.columns );
    
    
    ########### RESAMPLING ##########
    #samples = resample_at_rate_nearest( samples, samples[ELtname].min(), samples[ELtname].max(), ELtname, samplerate, timeunit );
        
    #REV: this does not change starting time, just resamples. (i.e. does not ZERO shift times).
    samples = ut.interpolate_df_to_samplerate( samples, tcol=ELtname, targ_srhzsec=samplerate, tcolunit_s=timeunit, );
    #samples = interpolate_df_to_samplerate( samples, tcol=tsecname, targ_srhzsec=samplerate, tcolunit_s=1 );
    print("Finished resampling");
    print(samples);
    
    
    if( timeunit != 1e-3 ):
        raise Exception("time unit for EL expected 1e-3 (msec)");
    
    #REV: add some useful other columns for time seconds/msec and zeroed to start or not.
    if( ELtname != tname ):
        if( tname in samples.columns ):
            raise Exception("Tname already in samples...");
        samples[tname] = samples[ELtname];
        samples = samples[ [c for c in samples.columns if c!=ELtname ] ];
        pass;

    #REV: add Tmsec0
    samples[tname+'0'] = samples[tname] - samples[tname].min();
    
    tsecs = samples[tname]*timeunit;
    if( tsecname in samples.columns ):
        if( False == np.all(np.isclose(tsecs, samples[tsecname])) ):
            print(tsecs, samples[tsecname]);
            raise Exception("Old tsec and new tsec are not similar?");
        pass;
    else:
        samples[tsecname] = samples[tname]*timeunit;
        pass;

    #REV: add Tsec0
    samples[tsecname+'0'] = samples[tsecname] - samples[tsecname].min();
    
    return samples;

#REV: note this sets any "NANval" EL things to NAN.
#REV: (should ovelap with "error" or not? I.e. not detected?)
def preproc_EL_A02_separate_samps_eye( samples,
                                       tname='Tmsec',
                                       eyename='eye',
                                       eyes_to_use=[pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE],
                                       unused_eyes_to_nan=False,
                                      ):
    
    llist=[];
    lold=[];
    rlist=[];
    rold=[];
    
    blist=[];
    #print(samples.columns);
    cols=list(samples.columns);
    for arg in cols:
        import re;
        res = re.match(r"(.+)_(left|right)", arg);
        if( res is not None ):
            if( res.group(2) == 'right' ):
                #print("RIGHT");
                rlist.append( res.group(1) );
                rold.append(arg);
            elif( res.group(2) == 'left' ):
                #print("LEFT");
                llist.append( res.group(1) );
                lold.append(arg);
            else:
                print("Error matched but wtf?!");
                exit(1);
                pass;
            pass;
        else:
            #print( "No match (Both)");
            blist.append( arg );
            pass;
        pass;
    
    ## (i.e. in situation where some columns were excluded etc...)
    if( sorted(llist) != sorted(rlist) ):
        print(llist);
        print(rlist);
        raise Exception("Error L/R lists not same?!");
    
    ldict = {lold[i]: llist[i] for i in range(len(lold))};
    rdict = {rold[i]: rlist[i] for i in range(len(rold))};
    
    ## llist or rlist, same columns.
    newcols = blist + llist;
    
    ldf = pd.DataFrame( columns=lold+blist );
    rdf = pd.DataFrame( columns=rold+blist );
    #print("uh", rold+blist);
    #print(samples[list(ldf.columns)]);
    
    ldf = samples[list(ldf.columns)].copy();
    rdf = samples[list(rdf.columns)].copy();
    
    ldf.rename(columns=ldict, inplace=True);
    rdf.rename(columns=rdict, inplace=True);
    
    ldf['ELeye']=pu.EL_LEFT_EYE;
    rdf['ELeye']=pu.EL_RIGHT_EYE;
    
    ldf[eyename] = pu.PEYEUTILS_LEFT_EYE;
    rdf[eyename] = pu.PEYEUTILS_RIGHT_EYE;
    
    ldf['useeye']=False;
    rdf['useeye']=False;

    if( pu.PEYEUTILS_LEFT_EYE in eyes_to_use ):
        print("USING LEFT EYE");
        ldf['useeye']=True;
        pass;
    if( pu.PEYEUTILS_RIGHT_EYE in eyes_to_use ):
        print("USING RIGHT EYE");
        rdf['useeye']=True;
        pass;
    
    #REV: just drop all unnecessary data if both eyes not used.
    #REV: better to drop everything of one eye or just keep as all NAN?
    
        
    df = pd.concat(
        [ldf, rdf]).sort_values(
            by=[tname, eyename]).reset_index(
                drop=True);
    
    if( unused_eyes_to_nan ):
        df = df[ df.useeye == True ];
        pass;
    else:
        df.loc[ (df.useeye==False), ['px','py',
                                     'gx','gy',
                                     'hx','hy'] ] = np.nan;
        pass;
        
    return df;


#REV: these are missing data eyelink.
#REV: in case any snuck by?
# MISSING_DATA -32768
# define MISSING -32768
# define INaN -32768

def preproc_EL_A03_remove_errors(df,
                                 ELcutoff=30000,
                                 ELnan_to_nan=True,
                                 errors_to_nan=True,
                                 pacol='pa',
                                 zero_pupil_to_nan=True,
                                 ):
    
    ####### REMOVE VALUES OUTSIDE CUTOFFS#########
    tonancols=list();
    if( 'gx' in df.columns ):
        df['Gnan'] = False;
        df.loc[ ((df.gx < -ELcutoff) | (df.gx > ELcutoff) | (df.gy < -ELcutoff) | (df.gy > ELcutoff)), 'Gnan' ] = True;
        tonancols += ['gx', 'gy',];
        pass;
    
    if( 'px' in df.columns ):
        df['Pnan'] = False;
        df.loc[ ((df.px < -ELcutoff) | (df.px > ELcutoff) | (df.py < -ELcutoff) | (df.py > ELcutoff)), 'Pnan' ] = True;
        #df.loc[ ((df.px < -ELcutoff) | (df.px > ELcutoff) | (df.py < -ELcutoff) | (df.py > ELcutoff)), ['px', 'py'] ] = np.nan;
        tonancols += ['px', 'py',];
        pass;
    
    if( 'hx' in df.columns ):
        df['Hnan'] = False;
        df.loc[ ((df.hx < -ELcutoff) | (df.hx > ELcutoff) | (df.hy < -ELcutoff) | (df.hy > ELcutoff)), 'Hnan' ] = True;
        #df.loc[ ((df.hx < -ELcutoff) | (df.hx > ELcutoff) | (df.hy < -ELcutoff) | (df.hy > ELcutoff)), ['hx', 'hy'] ] = np.nan;
        tonancols += ['hx', 'hy'];
        pass;
    
    if(pacol in df.columns):
        #REV: better to use "local" MAD after first rough split?
        pacutoff=ELcutoff; #30000;
        df['PAnan'] = False;
        #REV: 2025/09/15 -- PA of 0 is "nan" basically (zero area pupil?). Different than non-detection?
        #df.loc[ ((df[pacol] <= 0) | (df[pacol] > pacutoff) | (df[pacol] < -pacutoff)), pacol] = np.nan;
        if(zero_pupil_to_nan):
            df.loc[ ((df[pacol] == 0 ) |
                     (df[pacol] > pacutoff) |
                     (df[pacol] < -pacutoff)), 'PAnan'] = True;
            pass;
        else:
            df.loc[ ((df[pacol] > pacutoff) | (df[pacol] < -pacutoff)), 'PAnan'] = True;
            pass;
        tonancols += [pacol]; #REV: set pupil size to NAN too if error flag?
        pass;
    
    
    '''
    if( 'gx' in df.columns ): #and False == df['ELerror'].equals(df['Gnan']) ):
        print("Nerr={}/{}  NGnan={}   Overlap={}".format( df['ELerror'].sum(), len(df.index), df['Gnan'].sum(),
                                                          len(df[ (df.ELerror==True) & (df.Gnan==True) ].index) ));
        
        print(df.gx.max(), df.gx.min());
        #raise Exception( "Some error columns are not GX/GY nan");
    
    if( 'px' in df.columns ): # and False == df['ELerror'].equals(df['Pnan']) ):
        print("Nerr={}/{}  NPnan={}   Overlap={}".format( df['ELerror'].sum(), len(df.index), df['Pnan'].sum(),
                                                          len(df[ (df.ELerror==True) & (df.Pnan==True) ].index) ));
        print(df.px.max(), df.px.min());
        #raise Exception( "Some error columns are not PX/PY nan");

    if( 'hx' in df.columns ): # and False == df['ELerror'].equals(df['Hnan']) ):
        print("Nerr={}/{}  NHnan={}   Overlap={}".format( df['ELerror'].sum(), len(df.index), df['Hnan'].sum(),
                                                          len(df[ (df.ELerror==True) & (df.Hnan==True) ].index) ));
        print(df.hx.max(), df.hx.min());
        #raise Exception( "Some error columns are not HX/HY nan");
        
    if( pacol in df.columns ): #and False == df['ELerror'].equals(df['PAnan']) ):

        
        print("Nerr={}/{}  NPAnan={}   NPazero={}   Overlap={}   ZOverlap={}".format( df['ELerror'].sum(),
                                                                                      len(df.index),
                                                                                      df['PAnan'].sum(),
                                                                                      len( df[ df[pacol] == 0 ].index) ,
                                                                                      len(df[ (df.ELerror==True) & (df.PAnan==True) ].index) ,
                                                                                      len(df[ (df.ELerror==True) & (df[pacol]==0) ].index ) ) );
        print(df.pa.min(), df.pa.max());
        #raise Exception( "Some error columns are not PA nan");
        
        pass;
    '''

    if( errors_to_nan ):
        df.loc[ (df['errors'] != 0), tonancols ] = np.nan;
        pass;

    
    
    
    '''
    #REV: some where pa is NOT nan but GX is?
    #REV: ok seems they (PA) are all 0 (or NA)
    print(df[pacol].isna().sum(), len(df.index));
    print(len(df[ df[pacol]==0 ].index), len(df.index));
    print(len(df[ df[pacol]==0 ].index)+df[pacol].isna().sum(), len(df.index));
    print(df['gx'].isna().sum(), len(df.index));
    
    some = df[ (~df[pacol].isna()) &
               (df[pacol]>0) &
               (df['gx'].isna() ) ];
    if( len(some.index) > 0 ):
        print(some);
        raise Exception("Some pupil are positive but there is no corresponding gaze?");
    
    '''

    
    if( ELnan_to_nan ):
        if( 'gx' in df.columns ):
            df.loc[ (df['Gnan']==True), ['gx', 'gy'] ] = np.nan;
            pass;
        if( 'px' in df.columns ):
            df.loc[ (df['Pnan']==True), ['px', 'py'] ] = np.nan;
            pass;
        if( 'hx' in df.columns ):
            df.loc[ (df['Hnan']==True), ['hx', 'hy'] ] = np.nan;
            pass;
        if( pacol in df.columns ):
            df.loc[ (df['PAnan']==True), pacol ] = np.nan;
            pass;
        pass;
    
        
    
    if( 'gx' in df.columns and pu.utils.allnan( df['gx'] ) ):
        print("WARNING -> ALL GX DATA IS NAN AFTER ERRORS/CUTOFF->NAN");
        pass;
    if( 'px' in df.columns and pu.utils.allnan( df['px'] ) ):
        print("WARNING -> ALL PX DATA IS NAN AFTER ERRORS/CUTOFF->NAN");
        pass;
    if( 'hx' in df.columns and pu.utils.allnan( df['hx'] ) ):
        print("WARNING -> ALL HX DATA IS NAN AFTER ERRORS/CUTOFF->NAN");
        pass;
    if( pacol in df.columns and pu.utils.allnan( df[pacol] ) ):
        print("WARNING -> ALL PA DATA IS NAN AFTER ERRORS/CUTOFF->NAN");
        pass;
    

    #REV: just measuring pupil size not acceptable?
    if( len(tonancols) < 2 ):
        raise Exception("ERROR Neither px nor gx etc in data?!");

    #REV: clean the data...
    tormcols = ['Gnan', 'Pnan', 'PAnan', 'Hnan' ];
    df = df[ [c for c in df.columns if c not in tormcols ] ];

    '''
    import matplotlib.pyplot as plt;
    plt.plot(df[df.eye=='L'].Tmsec, df[df.eye=='L'].fgxvel);
    plt.plot(df[df.eye=='L'].Tmsec, df[df.eye=='L'].gxvel);
    plt.show();
    '''
    return df;


def preproc_EL_cull_columns(df):
    tormcols = ['samples', 'errors'];
    other=['hdata', 'htype', 'input', 'buttons']; #htype/htdata is type of head data and head data (unscaled?). Input is port input? Buttons
    # is button press etc.
    #REV: 'fgxvel_left', 'gxvel_left', rxvel, ryvel, hxvel, gxvel, etc.   rx/ry is resolution, i.e. pix/deg, but requires correct setup...
    df = df[ [c for c in df.columns if c not in tormcols ] ];
    return df;

def preproc_EL_A04_clean_events(eventdf,
                                timeunitsec=1e-3,
                                eyes_to_use=[pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE],
                                ):
    """

    Parameters
    ----------
    eventdf :
        
    timeunitsec :
         (Default value = 1e-3)
    eyes_to_use :
         (Default value = [pu.PEYEUTILS_LEFT_EYE)
    pu.PEYEUTILS_RIGHT_EYE] :
        

    Returns
    -------

    """

    #REV: this is just catching "old" versions of data (which have blink but not contains_blink).
    #REV: contains_blink means that the event itself is a saccade, but there is a blink inside of it. "blink" only is pure blink with no
    # other event type... (REV: aren't blinks "always" saccades? Could be more unnatural, e.g. occluded eye etc.).
    containblinkcol='contains_blink';
    if( containblinkcol not in eventdf.columns ):
        if( 'blink' not in eventdf.columns ):
            print("EVENTS:");
            print(eventdf);
            print("EVENT COLUMNS:");
            print(eventdf.columns);
            print("EVENT LENGTH:");
            print(len(eventdf.index));
            print("EVENT IDX:");
            print(eventdf.index);
            raise Exception("EDFEVENT did not contain a column [blink] or [contains_blinks] (after pyedfread 3.0? 2024/06)");

        #REV: this might fail if EV size is zero?
        eventdf['contains_blink'] = eventdf['blink'];
        eventdf = eventdf[ [c for c in eventdf.columns if c!='blink']]; #REV: remove "blink" as column in events (saccs)
        pass;
    

    eventdf = eventdf.rename(columns={'start':'stEL', 'end':'enEL'});
    
    
    eventdf['ELevlabel'] = eventdf['type'];
    eventdf = eventdf[ [ c for c in eventdf.columns if c != 'type' ] ]; #REV: remove confusing "type"
    eventdf['stsec'] = np.nan;
    eventdf['ensec'] = np.nan;
    eventdf['ELeye'] = eventdf['eye'];
    eventdf['eye'] = 'X'; #ERROR
    
    if( len(eventdf.index) > 0 ):
        lev = eventdf[eventdf.ELeye==pu.EL_LEFT_EYE].copy();
        rev = eventdf[eventdf.ELeye==pu.EL_RIGHT_EYE].copy();
        
        ## use same "names" as remodnav...
        lev.loc[ (lev.ELevlabel=='saccade'), 'ELevlabel' ] = 'SACC';
        lev.loc[ (lev.ELevlabel=='fixation'), 'ELevlabel' ] = 'FIXA';
        
        rev.loc[ (rev.ELevlabel=='saccade'), 'ELevlabel' ] = 'SACC';
        rev.loc[ (rev.ELevlabel=='fixation'), 'ELevlabel' ] = 'FIXA';
        
        ## RIGHT
        nrsacc = 0;
        if(len(rev.index)>0):
            #print(rev.label.unique);
            nrsacc = len(rev[rev['ELevlabel']=='SACC'].index);
            pass;
        
        ## LEFT
        nlsacc = 0;
        if(len(lev.index)>0):
            #print(lev.label.unique);
            nlsacc = len(lev[lev['ELevlabel']=='SACC'].index);
            pass;
        
        if( len(lev.index)>0 ):
            lev['eye'] = pu.PEYEUTILS_LEFT_EYE; #REV: set to "my" labels (not eyelink anymore)
            lev['stsec'] = lev['stEL'] * timeunitsec;
            lev['ensec'] = lev['enEL'] * timeunitsec;
            lev['useeye']=False;
            if( pu.PEYEUTILS_LEFT_EYE in eyes_to_use ):
                lev['useeye']=True;
                pass;
            pass;
        
        if( len(rev.index)>0 ):
            rev['eye'] = pu.PEYEUTILS_RIGHT_EYE;
            rev['stsec'] = rev['stEL'] * timeunitsec;
            rev['ensec'] = rev['enEL'] * timeunitsec;
            rev['useeye']=False;
            if( pu.PEYEUTILS_RIGHT_EYE in eyes_to_use ):
                rev['useeye']=True;
                pass;
            pass;
        
        ev = pd.concat( [lev, rev] ).sort_values(
            by=['eye','stsec']).reset_index(drop=True);
        
        #REV: drop events involving bad eye.
        ev = ev[ ev.useeye == True ];
        
        pass;
    else:
        ev = pd.DataFrame();
        pass
    
    
    
    return ev;

def preproc_EL_A06_check_nogazedata( df,
                                     colstouse ):
        
    #REV: if all false, all are all nan.
    nogazedata = all([ pu.utils.allnan( df[c] ) for c in colstouse ] );
    
    return nogazedata;







#REV: Note, that although this uses PYFV utils,
#REV: they are general for eyelink (i.e. only sample rate, tag,
#REV: etc.
def preproc_EL_A_clean_samples(rawsamps,
                               rawevents,
                               rawmessages,
                               targ_sr_hzsec=-1,
                               preblinks=False,
                               nogazecols=['gx','gy']):

    """

    Parameters
    ----------
    rawsamps :
        
    rawevents :
        
    rawmessages :
        
    preblinks :
         (Default value = False)

    Returns
    -------

    """
    
    #REV: creates DF with "tag" and "body".
    msgs = pfv.separate_EDF_msg_tags(rawmessages);
    
    #REV: this just adds Tsec, Tmsec, and moves 'time' to ELtime.
    msgs = preproc_EL_A00_add_Tsec(msgs);
    
    #  has samplerate etc., these are standard/same in ALL
    #   EDF recordings! (REV: I hope).
    elparamdict = pfv.get_elparams(msgs); 
    print(elparamdict);
    
    ELsr=elparamdict['samplerate'];
    if(targ_sr_hzsec <= 0):
        targ_sr_hzsec = ELsr;
        pass;
    
    ELeyes = [ eye for eye in elparamdict['eyes'] ];
    
    
    for eye in ELeyes:
        if eye not in [pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE]:
            raise Exception("Unrecognized eye [{}], I only recognize from: {}".format(eye, [pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE]));
        pass;
    
    print("EYELINK RECORDING FROM EYES: {} @ SR: {} Hz   (will resample to {})".format(ELeyes, ELsr, targ_sr_hzsec));
    
    #REV: resamples to time hz desired.
    #REV: could cause issues with events if they are specifically
    #REV: locked to exact index of sample?
    df = preproc_EL_A01_resample_time( rawsamps,
                                       samplerate=targ_sr_hzsec,
                                       );
    
    #REV: separates eyes and also sets ERRORS and ELNAN to nan.
    df = preproc_EL_A02_separate_samps_eye(df,
                                           eyes_to_use=ELeyes,
                                           );
    
    
    df = preproc_EL_A03_remove_errors(df);
    
    
    #REV: cleans events (i.e. renames saccade->SACC, fixa->FIXA, and sets 'stsec' and 'ensec'
    # Note, stsec and ensec will be accurate because no re-zeroing.
    # Uses "start" and "end" which are in MSEC time.
    ev = preproc_EL_A04_clean_events(rawevents,
                                     timeunitsec=pu.EL_TIMEUNIT_SEC,
                                     );
    
    
    #REV: if all bad, i.e. NAN etc., return 'badtrial=True'
    df, nogazedata = preproc_EL_A05_filter_samps_by_ELevents(df,
                                                             ev,
                                                             sr_hzsec=targ_sr_hzsec,
                                                             timeunitsec=pu.EL_TIMEUNIT_SEC,
                                                             );

    #REV: could pa also do it? Could I have pa without gaze data?
    #nogazedata = preproc_EL_A06_check_nogazedata( df, colstouse=nogazecols );
    
    elparamdict['badtrial'] = nogazedata;
    elparamdict['sr_hzsec'] = targ_sr_hzsec;
    
    if( nogazedata ):
        print("BADTRIAL (no gaze data) -- skipping pupil size analysis");
        df['bad'] = True; #should fill in all bad ...
        pass;
    else:
        df = pre.preproc_SHARED_pupilsize(df,
                                          timecol='Tsec',
                                          pacol='pa',
                                          eyecol='eye' );
        
        #REV: this adds "bad", "badEL", "badpupil", "badpupilEL"
        #REV: note EL just refers to "PRE" added guys...not necessarily EL
        preblinkcols=[];
        if(preblinks):
            preblinkcols=['elblink']; #elhasblink?
            pass;
        df = pre.preproc_SHARED_label_blinks(df,
                                             sr_hzsec=targ_sr_hzsec,
                                             blinkremoval_MAD_mult=5,
                                             blinkremoval_med_mult=1,
                                             blinkremoval_dilate_win_sec=0.030,
                                             blinkremoval_orphan_upperlimit_sec=0.010,
                                             blinkremoval_orphan_bracket_min_sec=0.050,
                                             blinkremoval_shortblink_minsize=0.100,
                                             tsecname='Tsec',
                                             eyecol='eye',
                                             valcol='px',
                                             pacol='pa',
                                             preblinkcols=preblinkcols, #REV: remove elblink detected. NOT other events...
                                             );
        pass;

    print(" !! Completed EL_A_preproc");
    return df, ev, msgs, elparamdict;
    
    

#REV: this only works because start/end are in 'msec' time, and so it is tname=Tmsec
def preproc_EL_A05_filter_samps_by_ELevents(df, ev,
                                            sr_hzsec,
                                            timeunitsec=1e-3,
                                            xname='px',
                                            yname='py',
                                            tname='Tmsec',
                                            tname0='Tmsec0',
                                            tsecname='Tsec',
                                            tsecname0='Tsec0',
                                            nan_EL_contains_blinks=False,
                                            nan_EL_pure_blinks=False,
                                            ):
    """

    Parameters
    ----------
    df :
        
    ev :
        
    sr_hzsec :
        
    timeunitsec :
         (Default value = 1e-3)
    xname :
         (Default value = 'px')
    yname :
         (Default value = 'py')
    tname :
         (Default value = 'Tmsec')
    tname0 :
         (Default value = 'Tmsec0')
    tsecname :
         (Default value = 'Tsec')
    tsecname0 :
         (Default value = 'Tsec0')
    nan_EL_contains_blinks :
         (Default value = False)
    nan_EL_pure_blinks :
         (Default value = False)

    Returns
    -------

    """

    if( tname != 'Tmsec' ):
        raise Exception("REV: Will compare Tmsec against raw EL start and end for events (i.e. units are msec), so tname must be Tmsec...");
    
    df['elblink']=False;
    df['elhasblink']=False;
    df['elsacc']=False;
    df['elfix']=False;
    df['elpurs']=False; #EL does not detect pursuits? Nope...but button, message, input etc.
    df['ELevlabel']='';   ## Name of event type (same as label in event)
    df['ELsimultevs']=0;   ## Number of different (simultaneous) events occurring in samples during each time point.
    
    if( pu.utils.allnan( df[xname] ) ):
        #raise Exception("-----------> WTF ALL NAN IN BEGINNING PUPIL PARAMS (Col: {})".format(xname));
        print("-----------> WTF ALL NAN IN BEGINNING PUPIL PARAMS (Col: {})".format(xname));
        badtrial=True;
        return df, badtrial;
    
    if( len(ev.index) > 0 ):
        #REV: all events which "contain_blink" is true.
        containsblinks = ev[(ev['contains_blink']==True)]; 

        '''
        #REV: these may be both blinks and saccades?
        if( len(containsblinks.index) > 0 ):
            print(containsblinks);
            for i, r in containsblinks.iterrows():
                print("TYPE OF THIS BLINK: {}".format(r['ELevlabel']));
                pass;
            pass;
        '''
        #REV: label i.e. "type" == blink?
        #REV: this is BLINK LABEL of "type". Assigned by pyedfread?
        #REV: all events whose label is "blink".
        #REV: these are only PURE BLINKS
        pureblinks = ev[(ev['ELevlabel']=='blink')];

        '''
        print("ENUMERATING BLINKS:");
        if( len(pureblinks.index) > 0 ):
            print(pureblinks);
            for i, r in pureblinks.iterrows():
                print(r);
                pass;
            pass;
        '''

        if( len(pureblinks.index) != len(containsblinks.index) ):
            print("Pure vs Contains blinks: ", len(pureblinks.index), len(containsblinks.index) );
            pass;
        
        #REV: this will set all events
        for rowidx, event in pureblinks.iterrows():
            eye=event.eye;
            start=event['stEL']; #start is ELstart (msec in EL raw time, not zeroed)
            end=event['enEL'];

            #REV: OK, I understand...a saccade can contain multiple blinks within it...(?).
            #REV: blinks are always shorter than a saccade etc. if contained...
            if( nan_EL_pure_blinks ):
                fixed=False;
                if( 'gx' in df.columns ):
                    #REV: include both "elblink" and "elhasblink" for "Pure" blinks.
                    df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), ['gx', 'gy', 'elblink', 'elhasblink'] ] = [np.nan, np.nan, True, True];
                    fixed=True;
                    pass;
                if( 'px' in df.columns ):
                    df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), ['px', 'py', 'elblink', 'elhasblink'] ] = [np.nan, np.nan, True, True];
                    fixed=True;
                    pass;
                if( 'hx' in df.columns ):
                    df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), ['hx', 'hy', 'elblink', 'elhasblink'] ] = [np.nan, np.nan, True, True];
                    fixed=True;
                    pass;
                if( not fixed ):
                    raise Exception("ERROR Neither px nor gx nor hx in data");
                pass;
            else:
                df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), 'elblink' ] = True; 
                pass;
            pass;
        
        
        
        #REV: this will set all events
        for rowidx, event in containsblinks.iterrows():
            eye=event.eye;
            start=event['stEL']; #start is ELstart (msec in EL raw time, not zeroed)
            end=event['enEL'];
            
            if( nan_EL_contains_blinks ):
                fixed=False;
                if( 'gx' in df.columns ):
                    df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), ['gx', 'gy', 'elhasblink'] ] = [np.nan, np.nan, True];
                    fixed=True;
                    pass;
                if( 'px' in df.columns ):
                    df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), ['px', 'py', 'elhasblink'] ] = [np.nan, np.nan, True];
                    fixed=True;
                    pass;
                if( 'hx' in df.columns ):
                    df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), ['hx', 'hy', 'elhasblink'] ] = [np.nan, np.nan, True];
                    fixed=True;
                    pass;
                if( not fixed ):
                    raise Exception("ERROR Neither px nor gx in data");
                pass;
            else:
                df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), 'elhasblink' ] = True; 
                pass;

            pass;
                
        mysaccs = ev[(ev['ELevlabel']=='saccade')]; #REV: how about blink AND SACCADE?
        for rowidx, event in mysaccs.iterrows():
            eye=event.eye;
            start=event['stEL'];
            end=event['enEL'];
            #df.loc[ df[ (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ].index, ['gx', 'gy'] ] = np.nan;
            df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), 'elsacc' ] = True;
            pass;

        myfixs = ev[(ev['ELevlabel']=='fixation')]; #REV: how about blink AND SACCADE?
        for rowidx, event in myfixs.iterrows():
            eye=event.eye;
            start=event['stEL'];
            end=event['enEL'];
            #df.loc[ df[ (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ].index, ['gx', 'gy'] ] = np.nan;
            df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), 'elfix' ] = True;
            pass;
        
        pass;
    
    
    ldf = df[df.eye==pu.PEYEUTILS_LEFT_EYE];
    rdf = df[df.eye==pu.PEYEUTILS_RIGHT_EYE];
    
    #REV: this ASSUMES that we have equal number of L/R samples!!!!! I.e. I can't drop bad eye.
    if(len(ldf.index) != len(rdf.index)):
        raise Exception("START -> Unequal LDF {}  RDF {}".format(len(ldf.index), len(rdf.index)));
    
    ldf = ldf.sort_values(by=tname).reset_index(drop=True);
    rdf = rdf.sort_values(by=tname).reset_index(drop=True);
    
    if( len(ldf[tname].diff().unique()) != 2 ): #1 for true DT, other NAN for first sample...
        raise Exception("LDF MISSING TIMEPOINTS IN EDF DATA");
    if( not np.isclose(ldf[tsecname].diff().iloc[-1], 1/sr_hzsec) ):
        print(ldf[tsecname].diff().iloc[-1], sr_hzsec);
        raise Exception("LDF Failed to match expected samplerate to DT");
    if( not np.isclose( np.nanmax(ldf[tsecname].diff()), 1/sr_hzsec ) ): #1 for true DT, other NAN for first sample...
        raise Exception("LDF NOT CONSTANT SAMPLERATE OF 1/SR tdelta");
    
    if( len(rdf[tname].diff().unique()) != 2 ): #1 for true DT, other NAN for first sample...
        raise Exception("RDF MISSING TIMEPOINTS IN EDF DATA");
    if( not np.isclose( np.nanmax(rdf[tsecname].diff()), 1/sr_hzsec ) ): #1 for true DT, other NAN for first sample...
        raise Exception("RDF NOT CONSTANT SAMPLERATE OF 1/SR tdelta");
    if( not np.isclose(rdf[tsecname].diff().iloc[-1], 1/sr_hzsec) ):
        print(rdf[tsecname].diff().iloc[-1], sr_hzsec);
        raise Exception("RDF Failed to match expected samplerate to DT");
    
    if( len(ev.index) > 0 ):
        lev = ev[ev.eye==pu.PEYEUTILS_LEFT_EYE];
        rev = ev[ev.eye==pu.PEYEUTILS_RIGHT_EYE];
        
        ####### EVENT CATEGORIZATION #########
        #REV: can events overlap?
        #REV: YES, there will be overlap of saccades and blinks at LEAST. Maybe buttons etc.? I should make a "flag"
        #REV: can a sample be part of multiple events? Need "index" of event...ugh. Put one and use NAN for not exist?
        for idx,event in lev.iterrows():
            ldf.loc[ (ldf[tsecname] >= event.stsec) & (ldf[tsecname] <= event.ensec), 'ELevlabel' ] = event.ELevlabel;
            ldf.loc[ (ldf[tsecname] >= event.stsec) & (ldf[tsecname] <= event.ensec), 'ELsimultevs' ] += 1;
            pass;
        
        for idx,event in rev.iterrows():
            rdf.loc[ (rdf[tsecname] >= event.stsec) & (rdf[tsecname] <= event.ensec), 'ELevlabel' ] = event.ELevlabel;
            rdf.loc[ (rdf[tsecname] >= event.stsec) & (rdf[tsecname] <= event.ensec), 'ELsimultevs' ] += 1;
            pass;
        
        #print("#L SACCS: {} (#timepoints with simult events: {})".format(nlsacc, len(ldf[ldf.simultevents>1].index)));
        #print("#R SACCS: {} (#timepoints with simult events: {})".format(nrsacc, len(rdf[rdf.simultevents>1].index)))
        if( ldf.ELsimultevs.max() > 1  or rdf.ELsimultevs.max() > 1 ):
            print("Largest overlaps: L: {}  R: {}".format(ldf.ELsimultevs.max(), rdf.ELsimultevs.max()));
            pass;
        
        pass;  ## END if events exist.
    
    df = pd.concat([ldf, rdf]).sort_values(by=['eye',tname]).reset_index(drop=True);
    
    badtrial=False; #REV: if badtrial was true, we would've broken/returned earlier with badtrial=True.
    # (REV would have returned true earlier if was allnan...?)
    return df, badtrial;

#REV: adds vbox space (pixels from...bottom-left?), stimulus space (if stimulus?), dvaspace (given physical).
#REV: this ASSUMES caliabration
#REV: this is only due to the "top" "bot" order of eyelink coords...FLIPY
def preproc_EL_rawcalib_px(df, msgs, FLIPY=-1):
    """

    Parameters
    ----------
    df :
        
    msgs :
        
    FLIPY :
         (Default value = -1)

    Returns
    -------

    """
    gcdict = pu.peyefv.get_gazecoords(msgs); #has l, t, r, b
    #REV: gx will be in this space.
    
    elw = gcdict['r'] - gcdict['l'];
    elh = gcdict['b'] - gcdict['t']; #REV: FLIPPED! Bottom is high.
    #REV: because I set it that way?
    elcx = gcdict['l'] + (elw/2); #This is "center of calibrated area"
    elcy = gcdict['t'] + (elh/2); #This is "center of calibrated area"
    
    
    gxpx = df.gx - elcx; #REV: will be 0 at center of calibrated area.
    gypx = df.gy - elcy;
    
    df['cgx_px'] = gxpx;
    df['cgy_px'] = FLIPY * gypx;
    
    return df;


