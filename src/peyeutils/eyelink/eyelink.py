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



def preproc_EL_A00_add_Tsec(rawdf, timecol='time', timeunitsec=pu.EL_TIMEUNIT_SEC):
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
    tvals = rawdf[timecol];
    rawdf = rawdf[ [c for c in rawdf.columns if c!=timecol] ]; #drop old timecol
    rawdf['EL'+timecol] = tvals;
    rawdf['Tsec'] = tvals * timeunitsec;
    MSEC=1e3;
    rawdf['Tmsec'] = rawdf['Tsec'] * MSEC;
    return rawdf;


def preproc_EL_A01_separate_samps_eye( samples,
                                       samplerate,
                                       ELtname='time', #REV: this is "EL" default time column
                                       tname='Tmsec',
                                       tsecname='Tsec',
                                       eyename='eye',
                                       eyes_to_use=[pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE],
                                       timeunit=1e-3,
                                       exclude_EL_blinks=False,
                                       ELcutoff=30000, #REV: their representtaion of NAN is like 100000000.. ):
    """

    Parameters
    ----------
    samples :
        
    samplerate :
        
    ELtname :
         (Default value = 'time')
    #REV: this is "EL" default time columntname :
         (Default value = 'Tmsec')
    tsecname :
         (Default value = 'Tsec')
    eyename :
         (Default value = 'eye')
    eyes_to_use :
         (Default value = [pu.PEYEUTILS_LEFT_EYE)
    pu.PEYEUTILS_RIGHT_EYE] :
        
    timeunit :
         (Default value = 1e-3)
    exclude_EL_blinks :
         (Default value = False)
    ELcutoff :
         (Default value = 30000)
    #REV: their representtaion of NAN is like 100000000.. :
        

    Returns
    -------

    """
                                       clean_errors=True,
                                       drop_bad_eyes=False,
                                      ):
    
    if( len(samples.index) < 2 ):
        print("GOOD_EL_SMAPLES_LR: no samples in df (you should exclude this data anyways)");
        return pd.DataFrame();
    
    print( "Will resample");
    print( samples.columns );
    
    
    ########### RESAMPLING ##########
    #samples = resample_at_rate_nearest( samples, samples[ELtname].min(), samples[ELtname].max(), ELtname, samplerate, timeunit );
        
    #REV: should replace NULL first?
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
    
    samples[tsecname+'0'] = samples[tsecname] - samples[tsecname].min();
    
    
        
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
    
    #REV: wtf it ignored the "pass" and obeyed only the indentation ugh.
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
    
    if( drop_bad_eyes ):
        df = df[ df.useeye == True ];
        pass;
    else:
        df.loc[ (df.useeye==False), ['px','py','gx','gy','hx','hy'] ] = np.nan;
        pass;
    
    ####### REMOVE VALUES OUTSIDE CUTOFFS#########
    fixed=False;
    if( 'gx' in df.columns ):
        df.loc[ ((df.gx < -ELcutoff) | (df.gx > ELcutoff) | (df.gy < -ELcutoff) | (df.gy > ELcutoff)), ['gx', 'gy'] ] = np.nan;
        
        if( clean_errors ):
            df.loc[ (df.errors != 0), ['gx', 'gy'] ] = np.nan;
        fixed=True;
        if( pu.utils.allnan( df['gx'] ) ):
            #raise Exception("EYEUTILS -> GOODLR -> WTF ALL NAN IN GOOD LR AFTER ELCUTOFF");
            print("WARNING -> ALL GX DATA IS NAN AFTER CUTOFF->NAN");
            pass;
        pass;
    
    if( 'px' in df.columns ):
        df.loc[ ((df.px < -ELcutoff) | (df.px > ELcutoff) | (df.py < -ELcutoff) | (df.py > ELcutoff)), ['px', 'py'] ] = np.nan;
        if( clean_errors ):
            df.loc[ (df.errors != 0), ['px', 'py'] ] = np.nan;
        fixed=True;
        if( pu.utils.allnan( df['px'] ) ):
            #raise Exception("EYEUTILS -> GOODLR -> WTF ALL NAN IN GOOD LR AFTER CUTOFF");
            print("WARNING -> ALL PX DATA IS NAN AFTER CUTOFF->NAN");
            pass;
        pass;

    if( 'hx' in df.columns ):
        df.loc[ ((df.px < -ELcutoff) | (df.px > ELcutoff) | (df.py < -ELcutoff) | (df.py > ELcutoff)), ['hx', 'hy'] ] = np.nan;
        if( clean_errors ):
            df.loc[ (df.errors != 0), ['hx', 'hy'] ] = np.nan;
        fixed=True;
        if( pu.utils.allnan( df['hx'] ) ):
            #raise Exception("EYEUTILS -> GOODLR -> WTF ALL NAN IN GOOD LR AFTER CUTOFF");
            print("WARNING -> ALL PX DATA IS NAN AFTER CUTOFF->NAN");
            pass;
        pass;
    
    #REV: better to use "local" MAD after first rough split?
    pacol='pa';
    if(pacol in df.columns):
        pacutoff=ELcutoff; #30000;
        #REV: 2025/09/15 -- PA of 0 is "nan" basically (zero area pupil?). Different than non-detection?
        df.loc[ ((df[pacol] <= 0) | (df[pacol] > pacutoff) | (df[pacol] < -pacutoff)), pacol] = np.nan;
        if( clean_errors ):
            df.loc[ (df.errors != 0), [pacol] ] = np.nan;
            pass;
        if( pu.utils.allnan( df[pacol] ) ):
            #raise Exception("EYEUTILS -> GOODLR -> WTF ALL NAN IN GOOD LR AFTER CUTOFF");
            print("WARNING -> ALL PA DATA IS NAN AFTER CUTOFF->NAN");
            pass;
        pass;
    
    if( not fixed ):
        raise Exception("ERROR Neither px nor gx in data");
    
    return df;



def preproc_EL_A02_clean_events(eventdf,
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
        eventdf = eventdf[ [c for c in eventdf.columns if c!='blink']]; #REV: remove "blink" as field of events (saccs)
        pass;
    
    
    
    
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
            lev['stsec'] = lev.start * timeunitsec;
            lev['ensec'] = lev.end * timeunitsec;
            lev['useeye']=False;
            if( pu.PEYEUTILS_LEFT_EYE in eyes_to_use ):
                lev['useeye']=True;
                pass;
            pass;
        
        if( len(rev.index)>0 ):
            rev['eye'] = pu.PEYEUTILS_RIGHT_EYE;
            rev['stsec'] = rev.start * timeunitsec;
            rev['ensec'] = rev.end * timeunitsec;
            rev['useeye']=False;
            if( pu.PEYEUTILS_RIGHT_EYE in eyes_to_use ):
                rev['useeye']=True;
                pass;
            pass;
        
        ev = pd.concat( [lev, rev] ).sort_values(by=['eye','stsec']).reset_index(drop=True);
        
        #REV: drop events involving bad eye.
        ev = ev[ ev.useeye == True ];
                
        pass;
    else:
        ev = pd.DataFrame();
        pass
    
    
    
    return ev;



def preproc_EL_A_clean_samples(rawsamps, rawevents, rawmessages,
                               preblinks=False):
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
    
    
    msgs = pfv.separate_EDF_msg_tags(rawmessages);
    
    msgs = preproc_EL_A00_add_Tsec(msgs);
    
    elparamdict = pfv.get_elparams(msgs); #rawmessages); # has samplerate etc.
    
    
    
    print(elparamdict);
    ELsr=elparamdict['samplerate'];
    ELeyes = [ eye for eye in elparamdict['eyes'] ];
    #REV: this lists "L", "R"
    for eye in ELeyes:
        if eye not in [pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE]:
            raise Exception("Unrecognized eye [{}], I only recognize from: {}".format(eye, [pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE]));
        pass;
    
    print("EYELINK RECORDING FROM EYES: {}".format(ELeyes));
    
    df = preproc_EL_A01_separate_samps_eye(rawsamps,
                                           samplerate=ELsr,
                                           eyes_to_use=ELeyes,
                                           );
    
    #REV: this removes "time"
    ev = preproc_EL_A02_clean_events(rawevents,
                                     timeunitsec=pu.EL_TIMEUNIT_SEC,
                                     );
    
    
    #REV: if all bad datapoints do what? Exit?
    df, badtrial = preproc_EL_A03_filter_samps_by_ELevents(df, ev,
                                                           sr_hzsec=ELsr,
                                                           timeunitsec=pu.EL_TIMEUNIT_SEC,
                                                           );
    if( badtrial ):
        print("BADTRIAL (no data) -- skipping pupil size analysis")
        pass;
    else:
        df = pre.preproc_SHARED_pupilsize(df,
                                          timecol='Tsec',
                                          valcol='pa',
                                          eyecol='eye' );
        
        #REV: this adds "bad", "badEL", "badpupil", "badpupilEL"
        #REV: note EL just refers to "PRE" added guys...not necessarily EL
        preblinkcols=[];
        if(preblinks):
            preblinkcols=['elblink']; #elhasblink?
            pass;
        df = pre.preproc_SHARED_label_blinks(df,
                                             sr_hzsec=ELsr,
                                             blinkremoval_MAD_mult=5,
                                             blinkremoval_med_mult=1,
                                             blinkremoval_dilate_win_sec=0.030,
                                             blinkremoval_orphan_upperlimit_sec=0.010,
                                             blinkremoval_orphan_bracket_min_sec=0.050,
                                             blinkremoval_shortblink_minsize=0.100,
                                             tsecname='Tsec',
                                             eyecol='eye',
                                             valcol='px',
                                             preblinkcols=preblinkcols, #REV: remove elblink detected. NOT other events...
                                             );
        pass;
    
    return df, ev, msgs, badtrial;
    
    
    
def preproc_EL_A03_filter_samps_by_ELevents(df, ev,
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
            start=event.start; #start is ELstart (msec in EL raw time, not zeroed)
            end=event.end;

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
            start=event.start; #start is ELstart (msec in EL raw time, not zeroed)
            end=event.end;
            
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
            start=event.start;
            end=event.end;
            #df.loc[ df[ (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ].index, ['gx', 'gy'] ] = np.nan;
            df.loc[ ( (df[tname] >= start) & (df[tname] <= end) & (df.eye == eye) ), 'elsacc' ] = True;
            pass;

        myfixs = ev[(ev['ELevlabel']=='fixation')]; #REV: how about blink AND SACCADE?
        for rowidx, event in myfixs.iterrows():
            eye=event.eye;
            start=event.start;
            end=event.end;
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

    badtrial=False;
    return df, badtrial;

#REV: adds vbox space (pixels from...bottom-left?), stimulus space (if stimulus?), dvaspace (given physical).
#REV: this ASSUMES caliabration 
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


