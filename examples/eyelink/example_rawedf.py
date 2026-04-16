## Process gaze from a raw EDF, user must provide his own physical coordinates etc...



import pandas as pd;
import numpy as np;
import sys;
import os;

import matplotlib.pyplot as plt
import seaborn as sns;

import peyeutils as pu;

import pyedfread;



DEBUG=False;


input_tcol_units_hzsec=1e3;

targ_sr_hzsec=1000;
interptype='polynomial';
interporder=2;
min_sacc_dva = 0.33;
min_isi_sec = 0.040;


edf_fn = sys.argv[1];


s, e, m = pyedfread.read_edf(edf_fn);


s, e, m, paramdict = pu.eyelink.preproc_EL_A_clean_samples(s, e, m);

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




sdf, ev = pu.peyeutils.preproc_and_compute_events( df=df,
                                                   tcol=tname,
                                                   xcol=xcol,
                                                   ycol=ycol,
                                                   sr_hzsec=targ_sr_hzsec,
                                                   mainseq_err_gain=1.5,
                                                   PLOT=True,
                                                  );



exit(0);
