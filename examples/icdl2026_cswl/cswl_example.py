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
xcol='CursorX' #'XGazePosLeftEye'
ycol='CursorY' #'YGazePosLeftEye

input_tcol_units_hzsec=1e3;

tobiisr=50; #REV; estimated?
targ_sr_hzsec=1000;

interptype='polynomial';
interporder=1;

min_sacc_dva = 0.33;
min_isi_sec = 0.060;
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


#REV: setting missing data to NAN (rather than negative numbers it seems to be)
DROP_BAD_DATA=True;
if(DROP_BAD_DATA):
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
df['x'] = df[xcol];
df['y'] = df[ycol];




#REV: dropping unnecessary columns...
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


#REV: rename some data...
df = df.rename(columns={'DiameterPupilLeftEye':'lpupil', 'DiameterPupilRightEye':'rpupil'});
pupil = df[['time','lpupil','rpupil']].copy();


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
maxdva=20;
df.loc[ ((df.xcdva < -maxdva) | (df.ycdva < -maxdva) | (df.xcdva > maxdva) | (df.ycdva > maxdva)) , ['xcdva', 'ycdva'] ] = np.nan;




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
params['min_sac_dur_sec'] = 0.012;
params['min_intersac_dur_sec'] = 0.020;
params['minblinksec'] = 0.030;
params['startvel'] = 120;

#REV: this recomputes "hidden" params which are per-sample (based on per-sec params passed by user).
params = rv.recompute_params(params);


print(params);

#REV; sets 'blink' to True for NAN gaze samples, false otherwise.
REMOVEBLINK=True;
if(REMOVEBLINK):
    params['blinkcol'] = 'blink';
    df[ params['blinkcol'] ] = df['xcdva'].isna();
    pass;


sdf = rv.remodnav_preprocess_eyetrace2d(eyesamps=df, params=params);
    


#REV: params for saccadr. Also does not affect number of saccades etc. THAT much..
sparams = saccr.default_saccadr_params();
sparams['samplerate'] = 1000;
sparams['noiseconst'] = 4; #REV: 4 works.
sparams['ek_vel_thresh_lambda'] = 6;

sdf, ev = saccr.saccadr_sacc(sdf, sparams, tsecname=tname);

print(ev);
print(ev.columns);




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


PLOT=True;
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

exit(0);
