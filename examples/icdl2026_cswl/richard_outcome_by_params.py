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




def load_gaze_session( fn,
                       tcoltouse='RTTime', ## REV: change to use RTtime for match with outcome "onset" time... UNITS: MSEC
                       tunits_hzsec=1e3,
                       xcol='CursorX', #'XGazePosLeftEye'
                       ycol='CursorY', #'YGazePosLeftEye'
                       PLOT=True, #False,
                      ):
    
    print("Processing for FN {}".format(fn));

    if( os.path.basename(fn) == '22-1_nonTC1_100-1.gazedata'):
        print("DOING FOR {}".format('22-1_nonTC1_100-1.gazedata'));
        pass;
    else:
        print("SKIPPING");
        return;
    
        
    ### HARD-CODED CONSTANTS #####
    whratio = 4/3;
    wpx = 1024;
    hpx = 768;
    wcm = 35;
    distcm = 70;
    imgwh_px=169;
    tname='time'

    tobiisr=50; #REV; estimated?
    sr_hzsec=1000;

    interptype='polynomial';
    interporder=1;

    min_sacc_dva = 0.33;
    min_isi_sec = 0.060;
    ##############################
    
    
    
    
    
    df = pd.read_csv(fn, sep='\t');
    
    print(df.columns);
    print(df.Subject.unique(), df.Session.unique());
    
    
    
    
    df[tname] = df[tcoltouse] * 1/tunits_hzsec;
    df = df.sort_values(by=tname).reset_index(drop=True);
    
    #REV: objects 169x169, assuming "equally distant" (not true) that means about 4.62 deg visual angle wid/hei...
    # CENTERS:
    ## Assuming 112 x BL, 660 y BL  
    ## Assuming 912 x BR, 660 y BR  -- 800 pix center-center
    ## Assuming 512 x TM, 112 y TM  -- 678 pix center-center (L2 norm)

    # BL = Loc1
    # BR = Loc2
    # TC = Loc3
    
    #x1,x2,y1,y2 (x+ right, y+ DOWN)
    #blobj=[12,212,544,756]
    #brobj=[812,1012,544,756]; #I.e. assuming 28 deg width of monitor (200 pix width of objects, quite large! 1/5 i.e. 6 dva wide
    #tmobj=[412,612,12,212]
        
    DROP_BAD_DATA=True;
    if(DROP_BAD_DATA):
        df.loc[ (df['DiameterPupilRightEye'] <= 0), 'DiameterPupilRightEye'] = np.nan;
        df.loc[ (df['DiameterPupilLeftEye'] <= 0), 'DiameterPupilLeftEye'] = np.nan;
        df.loc[ (df['CursorX'] <= -0.9), ['XGazePosLeftEye', 'YGazePosLeftEye', 'XGazePosRightEye', 'YGazePosRightEye', 'CursorX', 'CursorY', 'DiameterPupilLeftEye', 'DiameterPupilRightEye'] ] = np.nan;
        df.loc[ (df['ValidityLeftEye'] != 0), ['XGazePosLeftEye', 'YGazePosLeftEye', 'DiameterPupilLeftEye'] ] = np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
        df.loc[ (df['ValidityRightEye'] != 0), ['XGazePosRightEye', 'YGazePosRightEye', 'DiameterPupilRightEye'] ] = np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
        
        df.loc[ ((df['ValidityLeftEye'] != 0) & (df['ValidityRightEye'] != 0)), ['CursorX', 'CursorY'] ] = np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
        pass;
    
    df = df.sort_values(by=tname).reset_index(drop=True); #sort time -- maybe it was not correctly sorted?
    
    #duplicate_rows = df[df.duplicated(subset=[tname], keep=False)]
    df = df.drop_duplicates(subset=[tname], keep='first');
    
    #df['XGazePosLeftEye'] =  df['XGazePosLeftEye'].rolling(50).mean(); #try LPF maybe it's just super noisey? No still looks oscillatory...
    
    df['x'] = df[xcol];
    df['y'] = df[ycol];
    
    #REV: remove unneeded columns for now...just keep 'time'
    df = df[['x','y',tname, 'Subject', 'Session', 'ID', 'TETTime',
             'RTTime', 'CursorX', 'CursorY', 'TimestampSec',
             'TimestampMicrosec', 'XGazePosLeftEye', 'YGazePosLeftEye',
             'XCameraPosLeftEye', 'YCameraPosLeftEye',
             'DiameterPupilLeftEye', 'DistanceLeftEye', 'ValidityLeftEye',
             'XGazePosRightEye', 'YGazePosRightEye', 'XCameraPosRightEye',
             'YCameraPosRightEye', 'DiameterPupilRightEye',
             'DistanceRightEye', 'ValidityRightEye', 'TrialId',
             'UserDefined_1']];
    
    tostrcols = [ 'Subject', 'Session', 'ID', 'ValidityRightEye', 'TrialId' ];
    for c in tostrcols:
        df[c] = df[c].astype(str); #REV: this will not remove None types, and they will be dropped by dropna!! So, problem...
        pass;
    
    df['ShowingSlide'] = ~( df['UserDefined_1'].isna() );
    df = df[ [ c for c in df.columns if c!='UserDefined_1'] ]; #What a hack.
    
    df['ShowingTrial'] = df['TrialId'];
    df.loc[ (~df['ShowingSlide']), 'ShowingTrial' ] = '-1';
    
    df['ShowingSlide'] = df[c].astype(str);


    #REV: TODO compute "continuous" values? E.g. pupil values, for period? (rather than events)
    #df[] = np.nanmean([df['DiameterPupilLeftEye'], df['DiameterPupilRightEye']],axis=0);
    df = df.rename(columns={'DiameterPupilLeftEye':'lpupil', 'DiameterPupilRightEye':'rpupil'});
    pupil = df[['time','lpupil','rpupil']].copy();
    #df['pupil'] = np.nanmean([df['DiameterPupilLeftEye'], df['DiameterPupilRightEye']],axis=0);
    #df['pupil'] = np.nanmean([df['DiameterPupilLeftEye'], df['DiameterPupilRightEye']],axis=0);
    #print(df.pupil);
    #plt.plot(df.time, df.pupil);
    #plt.show();
    #exit(0);
    
    
    #df.loc[ (df['UserDefined_1'].isna()), 'UserDefined_1' ] = ''; #REV: fill weird mixed with ''?;
    #REV: calculate physical dimensions of setup and derive physiological parameters (e.g. degrees of visual angle rotation)
    #REV: Min/max is +/- 15 dva, so tangent function is roughly linear, we can ignore perspective shifts in dvappx at eccentric parts
    # of screen... (~2% at 14 degrees eccentricity), y=abs(tan(x)-x) / tan(x)  at 0.26 (15 deg in radians)
    
    screenwid_dva = 2 * np.degrees( np.atan2(wcm/2, distcm) ); #REV: some simple trig
    print("Screen wid dva: {}".format(screenwid_dva));
    #dvappx = 28; #REV: assume 17inch monitor (35 cm wide) -> at 70 cm distance. 28 deg total left/right edges of monitor.
    dvappx = screenwid_dva / wpx;
    print("DVA/PX is: ", dvappx);
    
    #REV: FLIP Y
    df['y'] = hpx - df.y;
    
    df['xcdva'] = (df.x - wpx/2)*dvappx; #REV: center and convert to dva...
    df['ycdva'] = (df.y - hpx/2)*dvappx;

    ## REV: remove "outsid escreen" values ?
    #REV: set to NAN
    maxdva=20;
    df.loc[ ((df.xcdva < -maxdva) | (df.ycdva < -maxdva) | (df.xcdva > maxdva) | (df.ycdva > maxdva)) , ['xcdva', 'ycdva'] ] = np.nan;


    blbr_tm_dist=np.sqrt( (912-512)**2 + (660-112)**2 );
    bl_br_dist=np.sqrt( (912-112)**2 + 0 );

    print("Dist between two bottom objects: px={}, dva={}".format(bl_br_dist, bl_br_dist*dvappx));
    print("Dist between two bottom objects and top-mid object: px={}, dva={}".format(blbr_tm_dist, blbr_tm_dist*dvappx));
    print("Roughly 18.6 dva (diag) and 22.9 dva (bottom horiz) separation, with each image being {:3.1f} dva wide/tall".format(imgwh_px*dvappx));

    
    if(PLOT):
        plt.scatter(df['xcdva'], df['ycdva']);
        plt.show();
        plt.close();
        pass;
    
    
    
    #REV: X/Y are now in "1 dva" units.
    ##### RESET DVAPPX, CAN NOT MOVE THIS!!! #######
    dvappx=1;
    
    truesrs = { c:tobiisr for c in df.columns  };
    
    from peyeutils.utils import interpolate_df_to_samplerate;
    #df = df.groupby(tname, as_index=False).mean(numeric_only=True);
    df = df.groupby([tname], as_index=False).agg( saccr.safe_agg(df,'mean') ).reset_index(drop=True);
    print(df.dtypes);
    df = interpolate_df_to_samplerate(df, tname, sr_hzsec, startsec=None, endsec=None,
                                      method=interptype, order=interporder, truesrs=truesrs, tcolunit_s=1,
                                      #zeroTsec=self.vidtsdf['Tsec'].min(), #ensures that Tsec0 lines up with vid times too
                                      );
    
    print(df);
    print(df.columns);
    
    #exit(0);
    
    
    params1 = rv.make_default_preproc_params(samplerate_hzsec=sr_hzsec,
                                             timeunitsec=1,
                                             dva_per_px=dvappx,
                                             xname='xcdva',
                                             yname='ycdva',
                                             tname=tname);
    
    params2 = rv.make_default_params(samplerate_hzsec=sr_hzsec);
    params = params1 | params2;
    params['noiseconst'] = 8;
    params['dilate_nan_win_sec'] = 0.010;
    params['min_sac_dur_sec'] = 0.012;
    params['min_intersac_dur_sec'] = 0.020;
    params['minblinksec'] = 0.030;
    params['startvel'] = 120;
    params = rv.recompute_params(params);
    
    REMOVEBLINK=True;
    if(REMOVEBLINK):
        params['blinkcol'] = 'blink';
        df[ params['blinkcol'] ] = df['xcdva'].isna();
        pass;
    
    print(params);
    
    DROPNA=True;
    if(DROPNA):
        #ut.strsafe_interpolate( df=df, tcol=tname, method='linear');
        '''
        interpcolumns = [ colname  for colname in df.columns if (True==is_numeric_dtype(df[colname])) ]
        notinterpcolumns = [ colname  for colname in df.columns if (False==is_numeric_dtype(df[colname]))]
        print(interpcolumns);
        print(notinterpcolumns);
        strdf = df[ notinterpcolumns ];
        strdf = strdf.ffill();
        tmpdf = df[ interpcolumns ];
        tmpdf = tmpdf.interpolate(method="linear");
        df = pd.merge( left=strdf, right=tmpdf, how='inner', left_index=True, right_index=True ); #REV: should merge by default on index...
        #strdf[ interpcolumns ] = tmpdf; #REV: ghetto merge?
        #df = strdf;
        #print(df);
        df = df.dropna();
        #print(df);
        #exit(0);
        '''
        pass;
    
    print(df);
        
    
    sdf = rv.remodnav_preprocess_eyetrace2d(eyesamps=df, params=params);
    
    sparams = saccr.default_saccadr_params();
    sparams['samplerate'] = 1000;
    sparams['noiseconst'] = 4; #REV: 4 works.
    sparams['ek_vel_thresh_lambda'] = 6;
    
    sdf, ev = saccr.saccadr_sacc(sdf, sparams, tsecname='time');
    
    print(ev);
    print(ev.columns);
    
    #REV: saccades too small for noise threshold?
    #REV: We have 50 Hz, so single sample is 20 msec. Assuming we will detect via velocity, we need at least 3 points,
    #   i.e. minimum should be e.g.
    # 60 msec from start to end... We can get by with some simple smoothing and detecting simple "shift" of gaze location, but total duration
    # and total path length (actually just dx/dy, i.e. ampldva) should be related, i.e. saccades that are too "lazy" are likely noise (or some
    # kind of drift/blink). Check whether the amplitude (deg vis angle) / duration (seconds), i.e. mean velocity, is greater than some minimum
    # threshold. mvel should handle that?
    
    ## 0 ampl about 25 msec, 4 at 35, 8 at 45, 16 at 65. Add +50% "error"
    
    sixteen_val=60;
    zero_val=20;
    
    zero_over_err=40;
    sixteen_over_err=270;
    
    zero_under_err=10;
    sixteen_under_err=30;

    sixteen=16;
    zero=0;
    
    rise=sixteen_val-zero_val;
    run=sixteen-zero;
    
    over_err_slope = (sixteen_over_err-zero_over_err)/run;
    under_err_slope = (sixteen_under_err-zero_under_err)/run;
    
    def is_mainseq(myampl_dva,
                   mydur_s,
                   intercept_ms=20,
                   slope=rise/run,
                   over_err_inter=zero_over_err,
                   over_err_slope=over_err_slope,
                   under_err_inter=zero_under_err,
                   under_err_slope=under_err_slope ):
        
        mydurms = mydur_s * 1e3;
        #expected_dur = intercept_ms + myampl_dva * slope; #NOT USED

        over_err_at_ampl = over_err_inter + (myampl_dva * over_err_slope);
        over = over_err_at_ampl;
        
        under_err_at_ampl = under_err_inter + (myampl_dva * under_err_slope);
        under= under_err_at_ampl;
        
        return (mydurms <= over) & (mydurms >= under);
    
    
    
    ev = ev[ ev['ampldva'] > min_sacc_dva ];
    
    ev['ismain'] = is_mainseq( ev['ampldva'], ev['dursec'] );

    
    if(PLOT):
        xmin=0;
        xmax=25;
        
        g = sns.relplot(data=ev, x='ampldva', y='dursec', hue='ismain', kind='scatter' );
        g.ax.axhline(0);
        g.ax.set_ylim([0, 0.3]);
        g.ax.set_xlim([xmin, xmax]);
        g.ax.plot([xmin, xmax], [1e-3*zero_under_err, 1e-3*(zero_under_err+(under_err_slope*xmax))]);
        g.ax.plot([xmin, xmax], [1e-3*zero_over_err, 1e-3*(zero_over_err+(over_err_slope*xmax))]);
        g.fig.tight_layout();
        g.fig.savefig('mainseq.pdf');
        plt.show();
        
        pass;
    
    #REV: rm physiologically impossible saccades.
    ev = ev[ ev.ismain==True ].reset_index(drop=True);


    ## REV: TODO
    #REV: combine blink and saccades for purpose of ISI? Unrealistically short ISI if we allow saccade->short ISI->blink.
    #REV: Very small amplitude blink should not count as saccade? (or, it would, but blah).
    # Simpler, just remove all unrealistically short 'fixations' (i.e. < 120msec? Time to code/execute visually-based saccade?)
    
    blinkev = pu.preproc.blink_df_from_samples(sdf, dva_per_px=params['dva_per_px'], badcol='blink', tcol='time', xcol='xcdva', ycol='ycdva');
    
    #REV: should I remove blinks in which eye did not move much (< 0.5 deg ?). I.e. fixation with intermediate lbink?
    # Vision is not happening during that time and physiologically it is equivalent...and then ISI is?
    
    
    ev = pd.concat([ev, blinkev]);
    ev = ev.sort_values(by='stsec').reset_index(drop=True);
    
    
    saccblnks = ev[ (ev.label=='SACC') | (ev.label=='BLNK') ];
    saccblnks = saccblnks.sort_values(by='stsec').reset_index(drop=True);

    
    #isi = difference between end and start of next one.
    #REV: make ISI which is time between st and en
    isis = saccblnks.copy();
    
    isis['stsec'] = saccblnks.shift(1)['ensec'].copy();    #start of ISI is the "end" of the PREVIOUS one (will be null for first)
    if( len(isis.index) > 0 ):
        isis.loc[ isis.index[0], 'stsec' ] = sdf.iloc[0]['time'];
        pass;

    isis['ensec'] = saccblnks['stsec'];
    isis['dursec'] = isis['ensec'] - isis['stsec'];
    isis['label'] = 'ISI';
    
    #REV: ISI of 80 msec is about the physiological minimum (coding of eye movement takes about 60 seconds in colliculus->brainstem->
    # extraocular muscles...even if not visually based? But planned/pre-planned saccades can be faster.
    # Gap reaction time is usually <60 msec?)
    
    ## Note doing ISI between blinks/saccades (i.e. counting blinks as saccades) adds more "small" ISI,
    # removing blinks (just doing between saccs)
    ## "overestimates" length of typical ISI (since saccade-blinks will be excluded).

    ## Similarly, including blinks in saccades will underestimate length of typical ISI (will say it is too short)...
    
    
    
    isis = isis[ isis['dursec'] > min_isi_sec ];
    
    #REV: combine ISIs into EVENTS.
    ev = pd.concat([ev, isis]).reset_index(drop=True);
    ev = ev.sort_values(by='stsec').reset_index(drop=True);
    
    
    if(PLOT):
        sns.histplot(data=saccblnks, x='ampldva',
                     multiple='stack',
                     #saccblnks['ampldva'],
                     binwidth=0.5, binrange=(0, 30), hue='label' );
        plt.xlabel('Sacc Ampl (deg)');
        plt.tight_layout();
        plt.savefig('amplhist.pdf');
        plt.show();
    
        sns.histplot(isis['dursec'], binwidth=0.050, binrange=(0, 2) );
        plt.xlabel('ISI (sec)');
        plt.tight_layout();
        plt.savefig('isihist.pdf');
        plt.show();
        
        plt.close();
        
        
        print("PRINTING COLS");
        print(isis.columns);
        sns.relplot( data=isis, x='stx', y='sty', kind='scatter', size='dursec' );
        plt.title("Gaze during ISI (blinks/saccades removed)");
        plt.tight_layout();
        plt.savefig('isiXYlocs.pdf');
        plt.show();
        plt.close();
    
    
        fig = plt.figure();
        for i, row in saccblnks.iterrows():
            plt.plot( [row['stx'], row['enx']], [row['sty'], row['eny'] ] );
            pass;
        plt.xlabel("X (dva)");
        plt.ylabel("Y (dva)");
        plt.tight_layout();
        plt.savefig('saccades.pdf');
        plt.show();
    
        
        for i,fig in enumerate(
                pu.plotting.plot_gaze_chunks( df=sdf, timestamp_col='time', x_col='xcdva', y_col='ycdva', chunk_size_sec=10, events_df=ev, event_start_col='stsec', event_end_col='ensec', event_type_col='label', max_chunks_per_fig=4 )
        ):
            fig.tight_layout();
            fig.savefig('testfig_{:04d}.pdf'.format(i));
            pass;
        pass;
    
    #REV: "unroll" events into columns?
    #REV: note all times are in SECONDS (must be re-converted to msec). Easier: just convert the output
    #REV: use "Tsec"
    
    #trialevents = ut.tsutils.rle_df(sdf['TrialId'], t=sdf['time']);
    trialevents = ut.tsutils.rle_df(sdf['ShowingTrial'], t=sdf['time']);
    #print(trialevents.v.unique());
    #print(trialevents);

    #REV: now name the first one of each "Train" and second one "Test". Also figure out times of each word/etc., and where the objects
    ## are shown, and which word is being said...
    trialevents['TrainTest'] = 'ITI';
    
    for i in trialevents.v.unique():
        if int(i) >= 0:
            if( len(trialevents[ trialevents.v==i ].index) < 1 ):
                raise Exception("No trial? Impossible");
            trialevents.loc[ trialevents[(trialevents.v==i)].index[0] ,'TrainTest' ] = 'Train';
            
            if( len(trialevents[ trialevents.v==i ].index) < 2 ):
                print(trialevents[ trialevents.v==i ]);
                print("WARNING: Not two (train and test) for trial {}".format(i));
                pass;
            else:
                trialevents.loc[ trialevents[(trialevents.v==i)].index[1] ,'TrainTest' ] = 'Test';
                pass;
            
            pass;
        
        pass;
    
    
    
    return sdf, ev, trialevents, pupil;


def gaze_preproc_for_session( gazedata_path, condition, subject, session, outdir='.', skip_if_exists=False, SHAM=False ):
    #52-1_nonTC1_12-1.gazedata
    #fn = condition + str(subject) + '-' + str(session) + '.gazedata';
    fn = find_file(condition, subject, session, gazedata_path);
    fpath = os.path.join(gazedata_path, fn);
    
    if( not os.path.isfile( fpath ) ):
        raise Exception("GAZE FILE missing for [{}] [{}] [{}]".format(condition, subjection, session));
    
    sampfn = fn + '.samples.csv';
    evfn = fn + '.events.csv';
    trialfn = fn + '.trials.csv';
    pupilfn = fn + '.pupil.csv';
    outdict = dict(samplescsv=sampfn, eventscsv=evfn, trialscsv=trialfn, pupilcsv=pupilfn);

    
    
    if( skip_if_exists and
        os.path.isfile( os.path.join(outdir, evfn) ) and
        os.path.isfile( os.path.join(outdir, trialfn ) )
       ):
        print("Preprocessed CSV files existed for {}, SKIPPING PREPROC COMPUTE".format(fn));
        pass;
    else:
        if( SHAM ):
            pass;
        else:
            print("COULD NOT FIND Preprocessed CSV files for {} ({}, {}), RECOMPUTING".format(fn,
                                                                                              os.path.join(outdir, evfn),
                                                                                              os.path.join(outdir, trialfn ) ) );
            samps, ev, trialevents, pupil = load_gaze_session( fpath,
                                                        tcoltouse='RTTime', ## REV: change to use RTtime for match with outcome "onset" time... UNITS: MSEC
                                                        tunits_hzsec=1e3,
                                                        xcol='CursorX', #'XGazePosLeftEye'
                                                        ycol='CursorY', #'YGazePosLeftEye'
                                                       );
            
            os.makedirs(outdir, exist_ok=True);

            pupil.to_csv(os.path.join(outdir,pupilfn), index=False);
            samps.to_csv(os.path.join(outdir,sampfn), index=False);
            ev.to_csv(os.path.join(outdir,evfn), index=False);
            trialevents.to_csv(os.path.join(outdir,trialfn), index=False);
            pass;
        pass;
    return outdict;

def cond_fn(cond,subj,sess ):
    fn = cond + str(subj) + '-' + str(sess) + '.gazedata';
    return fn;

#REV: this is one-to-one I assume, i.e. no compression either direction.
def find_file( cond, subj, sess, path ):
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
    
    fn1 = cond_fn(cond,subj,sess);
    
    
    if( os.path.isfile( os.path.join(path, fn1) ) ):
        return fn1;
    
    if( cond in lut1 ):
        fn2 = cond_fn(lut1[cond],subj,sess);
        if( os.path.isfile( os.path.join(path, fn2) ) ):
            return fn2;
        pass;
    
    if( cond in lut2 ):
        fn3 = cond_fn(lut2[cond],subj,sess);
        if( os.path.isfile( os.path.join(path, fn3) ) ):
            return fn3;
        pass;
    
    return '';


def preproc_for_all_in_index( indexpath, gazedata_path, outdir='.', skip_if_exists=False):
    indexdf=pd.read_csv(indexpath);

    #df=indexdf.groupby(['Condition','Subject','Session']);
    df = indexdf.groupby(['Condition','Subject','Session'], as_index=False).size().reset_index(drop=True);
    print("UNIQUE SESSIONS!");
    
    df['gazedata'] = df['Condition'].astype(str) + df['Subject'].astype(str) + '-' + df['Session'].astype(str) + '.gazedata';
    #print(df.tail(50));
    
    found=0;
    total=0;
    rows=list();
    for i, row in df.iterrows():
        cond = row['Condition'];
        subj = row['Subject'];
        sess = row['Session'];
        print(cond, subj, sess);
        #fn = cond + str(subj) + '-' + str(sess) + '.gazedata';
        fn = find_file(cond, subj, sess, gazedata_path);
                    
        fpath = os.path.join(gazedata_path, fn);
        
        
        if( not os.path.isfile( fpath ) ):
            print("!! Missing file for {} {} {}".format(cond, subj, sess));
            
            pass;
        else:
            found+=1;
            print("    OK [{}]".format(fpath));
            retdict = gaze_preproc_for_session( gazedata_path, cond, subj, sess, outdir=outdir, skip_if_exists=skip_if_exists);
            #print("Finished for {}".format(retdict));
            for c in retdict:
                row[c] = retdict[c];
                pass;
            rows.append(row);
            pass;
        total+=1;
        
        pass;
    outdf = pd.DataFrame(rows);
    print(outdf);
    outdf.to_csv('session_index.csv', index=False);
    print("Total {}/{}".format(found, total));
    return;


def main():
    preproc_for_all_in_index( sys.argv[1], sys.argv[2], sys.argv[3], skip_if_exists=False );
    return 0;


if __name__=='__main__':
    exit(main());
    raise Exception();



'''
fn = sys.argv[1];
samps, ev, trialevents = load_gaze_session( fn,
                                            tcoltouse='RTTime', ## REV: change to use RTtime for match with outcome "onset" time... UNITS: MSEC
                                            tunits_hzsec=1e3,
                                            xcol='CursorX', #'XGazePosLeftEye'
                                            ycol='CursorY', #'YGazePosLeftEye'
                                           );

print(samps);
print(samps.columns);
print(ev);
print(trialevents);

samps.to_csv('samps.csv', index=False);
ev.to_csv('events.csv', index=False);
trialevents.to_csv('trials.csv', index=False);
'''
