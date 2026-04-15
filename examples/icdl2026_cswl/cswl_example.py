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


## REV: fixed parameters (provided by user, about e.g. physical
## dimensions of setup in this case).
whratio = 4/3;
wpx = 1024;
hpx = 768;
wcm = 35;
distcm = 70;
imgwh_px=169;
tname='time'

input_tcol='RTTime';

#REV: names of X and Y columns to use in this case.
inxcol='CursorX' #'XGazePosLeftEye'
inycol='CursorY' #'YGazePosLeftEye

input_tcol_units_hzsec=1e3;

tobiisr=50; #REV; estimated?
targ_sr_hzsec=1000;

interptype='polynomial';
interporder=1;

min_sacc_dva = 0.50;
min_isi_sec = 0.040;
##############################


fn='./22-1_nonTC1_100-1.gazedata';
df = pd.read_csv(fn, sep='\t');

print(df.columns);
print(df.Subject.unique(), df.Session.unique());


## Create "my" time column, and convert units to seconds.
## (Zero time is still arbitrary).
## REV: note peyeutils currently automatically convert to seconds
## For resampling etc.? and will create columns
## Tsec and Tsec0, and thus requires passing units to some functions.
df[tname] = df[input_tcol] * 1/input_tcol_units_hzsec;

#REV: ensure order is correct...(it better just be one session!)
df = df.sort_values(by=tname).reset_index(drop=True);



'''
df['bad'] = False;

df.loc[ (df['DiameterPupilRightEye'] <= 0), 'bad'] = True;
df.loc[ (df['DiameterPupilLeftEye'] <= 0), 'bad'] = True;
df.loc[ (df['CursorX'] <= -0.9), ['XGazePosLeftEye', 'YGazePosLeftEye', 'XGazePosRightEye', 'YGazePosRightEye', 'CursorX', 'CursorY', 'DiameterPupilLeftEye', 'bad'] ] = True;
df.loc[ (df['ValidityLeftEye'] != 0), 'bad' ] = True; #np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
df.loc[ (df['ValidityRightEye'] != 0), 'bad'] = True; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
df.loc[ ((df['ValidityLeftEye'] != 0) & (df['ValidityRightEye'] != 0)), 'bad' ] = True;
'''

#REV: setting missing data to NAN (rather than negative numbers it seems to be)
NAN_BAD_DATA=True;
if(NAN_BAD_DATA):
    df.loc[ (df['DiameterPupilRightEye'] <= 0), 'DiameterPupilRightEye'] = np.nan;
    df.loc[ (df['DiameterPupilLeftEye'] <= 0), 'DiameterPupilLeftEye'] = np.nan;
    df.loc[ (df['CursorX'] <= -0.9), ['XGazePosLeftEye', 'YGazePosLeftEye', 'XGazePosRightEye', 'YGazePosRightEye', 'CursorX', 'CursorY', 'DiameterPupilLeftEye', 'DiameterPupilRightEye'] ] = np.nan;
    df.loc[ (df['ValidityLeftEye'] != 0), ['XGazePosLeftEye', 'YGazePosLeftEye', 'DiameterPupilLeftEye'] ] = np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
    df.loc[ (df['ValidityRightEye'] != 0), ['XGazePosRightEye', 'YGazePosRightEye', 'DiameterPupilRightEye'] ] = np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
    
    df.loc[ ((df['ValidityLeftEye'] != 0) & (df['ValidityRightEye'] != 0)), ['CursorX', 'CursorY'] ] = np.nan; #REV: remove bad data, assume validity 0 is OK, nonzero is error.
    pass;


#REV: dropping duplicate times. Could use "mean" first, but expect no dupl.
df = df.drop_duplicates(subset=[tname], keep='first');

print("COLS", df.columns);

#REV: for simplicity, naming x and y
# Note units are still unknown (probably pixels in some screen space).
df['x'] = df[inxcol];
df['y'] = df[inycol];




#REV: dropping unnecessary columns...
#REV: remove unneeded columns for now...just keep 'time'

df = df[['x','y', tname, 'Subject', 'Session', 'ID', 'TETTime',
         'RTTime', 'CursorX', 'CursorY', 'TimestampSec',
         'TimestampMicrosec', 'XGazePosLeftEye', 'YGazePosLeftEye',
         'XCameraPosLeftEye', 'YCameraPosLeftEye',
         'DiameterPupilLeftEye', 'DistanceLeftEye', 'ValidityLeftEye',
         'XGazePosRightEye', 'YGazePosRightEye', 'XCameraPosRightEye',
         'YCameraPosRightEye', 'DiameterPupilRightEye',
         'DistanceRightEye', 'ValidityRightEye', 'TrialId',
         'UserDefined_1']];


#REV: some columns are strings which should not be imputed/interpolated!!!
## Even if they are "integer-like" (e.g. subject 100, 101)
tostrcols = [ 'Subject', 'Session', 'ID', 'ValidityRightEye', 'TrialId' ];
for c in tostrcols:
    df[c] = df[c].astype(str);
    pass;

#REV: renaming some columns.
df['ShowingSlide'] = ~( df['UserDefined_1'].isna() );
df=df.drop(columns=['UserDefined_1'])
df['ShowingTrial'] = df['TrialId'];
df.loc[ (~df['ShowingSlide']), 'ShowingTrial' ] = '-1';

#REV: as str so no interpolation
df['ShowingSlide'] = df[c].astype(str);



pupil = df[['time','DiameterPupilLeftEye','DiameterPupilRightEye']].copy();
pupil = pupil.rename(columns={'DiameterPupilLeftEye':'lpupil', 'DiameterPupilRightEye':'rpupil'});


##### Convert from "screen" pixel X/Y into degrees visual angle (DVA).
##### This is simple linear interpolation since size is so narrow
##   (linear part of tangent function, safe up to about 10deg (1% error), i.e.
## for symmetric width 20 deg, 10 left and 10 right.
## 2% at 14, 4% at 20, 10% at 30.

## peyeutils should (does) provide user utility functions to aide in this
## conversion. In this case, user passes xcdva and ycdva (x and y centered DVA),
## i.e. (0,0) is straight ahead, and x positive is right, y positive is up.
screenwid_dva = 2 * np.degrees( np.atan2(wcm/2, distcm) ); #REV: some simple trig
print("Screen wid dva: {}".format(screenwid_dva));
#dvappx = 28; #REV: assume 17inch monitor (35 cm wide) -> at 70 cm distance. 28 deg total left/right edges of monitor.
dvappx = screenwid_dva / wpx;
print("DVA/PX is: ", dvappx);

#REV: FLIP Y
df['y'] = hpx - df.y;

df['xcdva'] = (df.x - wpx/2)*dvappx; #REV: center and convert to dva...
df['ycdva'] = (df.y - hpx/2)*dvappx;


xcol='xcdva';
ycol='ycdva';

#REV: realistically, when passed into eyeutils, they should provide:

## 1) units
## 2) zero
## 3) "space" (i.e. coordinates/convention). For example, is Y-positive up?
##    is X-positive right? Is Z-positive forward?
##  REV: MAJOR QUESTION: how to handle typical "x,y" spaces versus
##   Euler angle/rotations or quarternions?
## For example, providing in "roll/pitch/yaw" (around axis)
##   is also great, and is  transferable to wearable eye-trackers,
## But it is more difficult to plot and harder to convert to screen/stimulus
## space. For example, "NWU" (north-west-up) convention, means X-positive AXIS points "north"
## (i.e. forward), meaning that due to right-hand rule, positive rotations
## will correspond to rolling clockwise.
##  Y-positive AXIS points "west", meaning that due to right-hand convention,
## positive pitch will cause DOWNWARD pitch (dive), and Z-axis points "UP",
## meaning positive rotation will yaw pointing nose LEFT.

## Converting from these to projection on sphere (i.e. just x and y "rotation")
##  may be confusing for some users...

## REV: remove "outside escreen" values ?
#REV: set to NAN
'''
NAN_OUTSIDE=False;
if( NAN_OUTSIDE ):
    maxdva=20;
    df.loc[ ((df.xcdva < -maxdva) | (df.ycdva < -maxdva) | (df.xcdva > maxdva) | (df.ycdva > maxdva)) , ['xcdva', 'ycdva'] ] = np.nan;
    pass;

df.loc[ ((df.xcdva < -maxdva) | (df.ycdva < -maxdva) | (df.xcdva > maxdva) | (df.ycdva > maxdva)) , 'bad'] = True;
'''



#REV: how many dva per x/y unit I will pass into the models
##  (a.k.a. dva/pix, dvappx).
## Since I converted already to xcdva and ycdva, they are in dva, so 1.0
xyunits_dva=1;

#REV: the true sample rates for each sensor.
## In this case all sensors are 50 Hz, but in some other cases different
## columns may be 10 Hz and some may be 100 Hz etc.
## This will determine when to not interpolate/impute (if too far from a
## real sample).
truesrs = { c:tobiisr for c in df.columns  };



from peyeutils.utils import interpolate_df_to_samplerate;


#REV: taking the mean of each timepoint (should already be fine, I already dropped duplicates above) 
df = df.groupby([tname], as_index=False).agg( saccr.safe_agg(df,'mean') ).reset_index(drop=True);

#REV: interpolate (upsample) to 1000 Hz...
df = interpolate_df_to_samplerate(df, tname, targ_sr_hzsec, startsec=None, endsec=None,
                                  method=interptype, order=interporder, truesrs=truesrs, tcolunit_s=1,
                                  );

baddatacol='mybaddata';
df[baddatacol] = False;

maxdva=20;
df.loc[ ((df.xcdva < -maxdva) | (df.ycdva < -maxdva) | (df.xcdva > maxdva) | (df.ycdva > maxdva)) , baddatacol] = True;



PUPILSIZE_BLINKS=True;
if(PUPILSIZE_BLINKS):

    #REV: separate data into left/right eye samples in "long" format.
    eyedf = pu.preproc.separate_eyes( df, regexes=['.*(Left).*', '.*(Right).*'],
                                      eyecol='eye',
                                      casesensitive=True,
                                      drop_words=['Eye']);
    
    print("FINISHED SEP");
    print(eyedf);
    print(eyedf.columns);

    #REV: compute MAD etc. of pupil size and perform some smoothing.
    eyedf = pu.preproc.preproc_SHARED_pupilsize(eyedf,
                                                timecol=tname, #e.g. 'Tsec0'
                                                pacol='DiameterPupil', #e.g. 'pa'
                                                eyecol='eye', #e.g. 'eye'
                                                characteristic_timescale_sec=0.010, #Rough characteristic timescale
                                                ## of pupil size change
                                                );

    #REV: label eye blinks in using the pupilsize information (based on velocity/acceleration deviation from surrounding noise,
    ## sudden changes in pupilsize indicate blinks (or saccades) in VOG video oculography eye tracking.
    #REV: plot pupil size too? On same graph?
    eyedf = pu.preproc.preproc_SHARED_label_blinks(eyedf,
                                                   sr_hzsec=targ_sr_hzsec, 
                                                   tsecname=tname,
                                                   eyecol='eye',
                                                   valcol='XGazePos', #REV: is pupil area NAN when no eye tracking?
                                                   badcol='ppbad',
                                                   pacol='DiameterPupil',
                                                   blinkremoval_MAD_mult=8,
                                                   blinkremoval_med_mult=1,
                                                   blinkremoval_dilate_win_sec=0.030,
                                                   blinkremoval_orphan_upperlimit_sec=0.020,
                                                   blinkremoval_orphan_bracket_min_sec=0.040,
                                                   blinkremoval_shortblink_minsize=0.070,
                                                   #patdiffcol='pa_abs_tdiff',
                                                   preblinkcols=[] );
    
    #REV: convert to an "events" dataframe (start/end of each blink etc.)
    pblinks = pu.preproc.blink_df_from_samples(eyedf,
                                               badcol='ppbad',
                                               tcol=tname,
                                               stcol='stsec',
                                               encol='ensec',
                                               stidx='stidx',
                                               enidx='enidx',
                                               xcol='XGazePos',
                                               ycol='YGazePos',
                                               dva_per_px=xyunits_dva,
                                               eyecol='eye',
                                               
                                               );

    #REV: label them
    pblinks['label'] = 'PBLNK';

    #REV: just take left as example?
    pblinks = pblinks[pblinks['eye']=='Left'];
    pass;



baddata = df[ (df[baddatacol]==True) ].copy();



#REV: remove here?
NAN_OUTSIDE=True;
if( NAN_OUTSIDE ):
    maxdva=20;
    df.loc[ ((df.xcdva < -maxdva) | (df.ycdva < -maxdva) | (df.xcdva > maxdva) | (df.ycdva > maxdva)) , ['xcdva', 'ycdva'] ] = np.nan;
    pass;








## I now have 1000 Hz clean data. I will now preprocess it to smooth via e.g.
###   Savgol filter and median filter and outlier filter, and some NAN expansion?

# These are coded in the remodnav preprocessing code.
## But I will use it even for the other saccade detection methods (saccadr)

params1 = rv.make_default_preproc_params(samplerate_hzsec=targ_sr_hzsec,
                                         timeunitsec=1,
                                         dva_per_px=xyunits_dva,
                                         xname=xcol,
                                         yname=ycol,
                                         tname=tname);


#REV: default other params for remodnav (will be ignored if not needed).
params2 = rv.make_default_params(samplerate_hzsec=targ_sr_hzsec);

#REV: combine this into single large params dict. These do not affect results very much
params = params1 | params2;

params['noiseconst'] = 8;
params['dilate_nan_win_sec'] = 0.010;
params['min_sac_dur_sec'] = 0.014;
params['min_intersac_dur_sec'] = 0.020;
params['minblinksec'] = 0.030;
params['startvel'] = 100;

#REV: this recomputes "hidden" params which are per-sample (based on per-sec params passed by user).
params = rv.recompute_params(params);


print(params);

#REV; sets 'blink' to True for NAN gaze samples, false otherwise.
REMOVEBLINK=True;
if(REMOVEBLINK):
    params['blinkcol'] = 'blink';
    df[ params['blinkcol'] ] = df['xcdva'].isna();
    pass;



#REV: this does NAN dilation etc.... it removes data, not just setting "bad" column or something? I'd rather mask...so I know
## old values for comparison.
sdf = rv.remodnav_preprocess_eyetrace2d(eyesamps=df, params=params);
    



#METHOD='saccadr'; #'remodnav';

sparams = saccr.default_saccadr_params();
sparams['samplerate'] = 1000;
sparams['noiseconst'] = 5; #REV: 4 works.
sparams['ek_vel_thresh_lambda'] = 6; # 6 works
sparams['nh_init_vel_thresh_degsec'] = 100;

print("Doing SACCR");
#sdf, sev1 = saccr.saccadr_detect_saccades(sdf, sparams, tsecname=tname);

sdf, sev1 = saccr.saccadr_detect_saccades(sdf, sparams, tsecname=tname, namedmethods=('ek',));

#print("Doing SACCR");
sdf2, sev2 = saccr.saccadr_detect_saccades(sdf, sparams, tsecname=tname, namedmethods='om,');

print("Doing SACCR");
sdf3, sev3 = saccr.saccadr_detect_saccades(sdf, sparams, tsecname=tname, namedmethods=('nh',));

print("Doing REMODNAV");
rdf, rev = rv.remodnav_classify_events(sdf, params);


sev = pd.concat( [sev1,
                  #sev2,
                  sev3] );

allsaccs = pd.concat( [ sev[sev['label']=='SACC' ],
                        rev[rev['label']=='SACC' ], ]
                      );

ev=rev;
saccs = allsaccs.reset_index(drop=True);

saccs2 = pu.eyemovements.combine.consolidate_saccades( saccs,
                                                       isi_threshold=0
                                                      );
saccs2['label'] = 'CSACC';

saccs = pd.concat([saccs, saccs2]).reset_index(drop=True);



print(ev);
print("FIRST EVENT COLUMNS: ", ev.columns); #REV: NOT HERE.

#saccs = ev[ (ev['label'] == 'SACC') ];
nonsaccs = rev[ ~(rev['label'] == 'SACC') ];
print("NON-SACCS", nonsaccs['label']);


#saccs = saccs[ saccs['ampldva'] > min_sacc_dva ];


PLOT=True;

if(PLOT):
    mainseq, mygraphics = pu.eyemovements.mainseq.mainseq_ampldur_linear_95pctl_human_chen2021_wplot( saccs['ampldva'],
                                                                                                      saccs['dursec'],
                                                                                                      error_gain=1.5,
                                                                                                     );
    xmin=0;
    xmax=25;
    mygraphics.ax.set_xlim([xmin, xmax]);
    mygraphics.savefig('mainseq.pdf');
    plt.show();
    pass;
else:
    mainseq = pu.eyemovements.mainseq.mainseq_ampldur_linear_95pctl_human_chen2021_wplot( saccs['ampldva'],
                                                                                          saccs['dursec'],
                                                                                          error_gain=1.5,
                                                                                         );
    pass;


saccs['ismain'] = mainseq; #Any way to always just make it give me first?
#saccs = saccs[ (saccs.ismain==True) ].reset_index(drop=True);


if( 'eye' in ev.columns ):
    print("In EV 0");
    pass;


blinks = pu.eyemovements.blink.compute_blinks_from_sampcol( sdf,
                                                            dva_per_px=params['dva_per_px'],
                                                            badcol=params['blinkcol'],
                                                            tcol=tname,
                                                            xcol=xcol,
                                                            ycol=ycol )

#REV: should I remove blinks in which eye did not move much (< 0.5 deg ?). I.e. fixation with intermediate lbink?
# Vision is not happening during that time and physiologically it is equivalent...and then ISI is?

print(saccs.columns);

#REV: ugh, either all computations must take into account "eye" column, or they must ignore it.
#REV: Easiest: error out if eye column is detected? Force user to do it themselves (do pipeline for each eye?).

#REV: problem we may want to "merge" at some point...

ev = pd.concat( [saccs,
                 blinks,
                 #nonsaccs, #REV: this is currently just PISI (original ISI from saccade detection...). In other cases
                 # there may also be e.g. drifts/smooth pursuits, etc.?
                 ] ).reset_index(drop=True);


if( PUPILSIZE_BLINKS ):
    ev = pd.concat( [ev, pblinks] ).reset_index(drop=True);
    pass;


ISIevents=['SACC', 'BLNK']; #REV: i.e. use blinks as saccades (gaze shifts often happen during blinks...)
isis = pu.eyemovements.isi.compute_ISIs_from_events( ev,
                                                     zerotime=sdf[tname].iloc[0], #REV: or .min()
                                                     eventstouse=ISIevents,
                                                     #stname='stsec', enname='ensec', durname='dursec', label='ISI' #defaults
                                                    );


#isis = isis[ isis['dursec'] > min_isi_sec ];

ev = pd.concat([ev, isis]);
ev = ev.sort_values(by='stsec').reset_index(drop=True);

#REV: how am I plotting double labels?


#print(ev['eye']);
DOMERGE=True;
if(DOMERGE):
    ev = pu.eyemovements.isi.eye_event_merge( ev,
                                              eyecol='eye',
                                              min_blink_dur=0.030,
                                              max_blink_amp=0.5,
                                              min_isi_dur=min_isi_sec,
                                             );
    pass;


isis = ev[ (ev['label']=='ISI') ];

saccblnks = ev[ (ev.label=='SACC') | (ev.label=='BLNK') | (ev.label=='SACCBLNK')];
#saccblnks = pd.concat( [ saccs, blinks] );
#saccblnks = saccblnks.sort_values(by='stsec').reset_index(drop=True);




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
            pu.plotting.plot_gaze_chunks_wpupil( df=sdf, timestamp_col=tname, x_col=xcol, y_col=ycol, chunk_size_sec=10,
                                          events_df=ev, event_start_col='stsec', event_end_col='ensec',
                                          event_type_col='label', max_chunks_per_fig=4,
                                          pupil_col='DiameterPupilRightEye',
                                          )
    ):
        fig.tight_layout();
        fig.savefig('testfig_{:04d}.pdf'.format(i));
        pass;
    pass;

ev.to_csv('events.csv', index=False);
print(ev);


exit(0);
