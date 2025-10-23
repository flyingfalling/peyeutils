
import pandas as pd;
import numpy as np;
import math;

import peyeutils as pu;
import peyeutils.utils as ut;

def preproc_SHARED_D_exclude_bad(df, xcol, ycol, badcol='bad'):
    """

    Parameters
    ----------
    df :
        
    xcol :
        
    ycol :
        
    badcol :
         (Default value = 'bad')

    Returns
    -------

    """
    df.loc[ True==df[badcol], [xcol,ycol] ] = np.nan;
    return df;


#REV: should find a way to "keep" all the raw data but just set "Bad" flags? Fuck...
def preproc_SHARED_C_binoc_gaze(df,
                                xcol,
                                ycol,
                                tcol,
                                badcol='bad',
                                eyecol='eye',
                                exclude_thresh=-1, #Set to NAN any time points in which they are separated by this amount
                                thresh_lr_badcol='badBINOCDXY',
                                only_alleyes_binoc=False,
                                eyetags=[pu.PEYEUTILS_LEFT_EYE, pu.PEYEUTILS_RIGHT_EYE],
                                btag=pu.PEYEUTILS_BINOC_EYE,
                                ):
    """

    Parameters
    ----------
    df :
        
    xcol :
        
    ycol :
        
    tcol :
        
    badcol :
         (Default value = 'bad')
    eyecol :
         (Default value = 'eye')
    exclude_thresh :
         (Default value = -1)
    #Set to NAN any time points in which they are separated by this amountthresh_lr_badcol :
         (Default value = 'badBINOCDXY')
    only_alleyes_binoc :
         (Default value = False)
    eyetags :
         (Default value = [pu.PEYEUTILS_LEFT_EYE)
    pu.PEYEUTILS_RIGHT_EYE] :
        
    btag :
         (Default value = pu.PEYEUTILS_BINOC_EYE)

    Returns
    -------

    """
    
    if(len(eyetags) < 1):
        raise Exception("no eyetags");
    
    tmpx="__"+xcol;
    tmpy="__"+ycol;
    
    df[tmpx] = df[xcol];
    df[tmpy] = df[ycol];
    
        
    print("MEANS TMP:" , df[tmpx].mean(), df[tmpy].mean());
    
    eyedfdict = { eyetag : df[ df[eyecol]==eyetag ].sort_values(by=[tcol]).reset_index(drop=True) for eyetag in eyetags };
    #rdf = df[ df[eyecol]==rtag ].sort_values(by=[tcol]).reset_index(drop=True);
    
    for k in eyedfdict:
        if( np.all(np.isnan(eyedfdict[k][tmpx])) ):
            print("WARNING EYE {} X is all nan in binoc processing".format(k));
            pass;
        if( np.all(np.isnan(eyedfdict[k][tmpy])) ):
            print("WARNING EYE {} Y is all nan in binoc processing".format(k));
            pass;
        pass;
    
        
    eyedflist = [ eyedfdict[k] for k in eyedfdict ];
    
    lens=list();
    ts = list();
    for k in eyedfdict:
        lens.append(len(eyedfdict[k].index));
        ts.append(eyedfdict[k][tcol]);
        pass;
    lenallsame = len(np.unique(lens)) == 1;
    if( lenallsame == False ):
        raise Exception("Unequal L/R DFs...");
    
    firstts=ts[0];
    for t in ts[1:]:
        if( False == np.allclose( firstts, t ) ):
            raise Exception("Left/Right eye times diverge...");
        pass;
    
    
    
    #REV: combine "object" type columns, e.g. string "issacc" etc.
    # To combine left/right, whichever is first? Note.....
    aggrule = dict.fromkeys(df, np.nanmean);
    
    aggrule.update(dict.fromkeys(df.columns[df.dtypes.eq(object)], 'first'));

        
    bdf=df.groupby(tcol).agg(aggrule);
    #bdf[badcol] = ldf[badcol] & rdf[badcol];
    
    #bdf = pd.DataFrame( {tcol:np.mean(ldf[tcol],rdf[tcol]),
    #                     });
    #print("BDF vals", bdf[tmpx].mean(), bdf[tmpy].mean());
    
    bdf[eyecol] = btag;
    
    hasboth = ~(eyedflist[0][badcol].to_numpy());
    for x in eyedflist[1:]:
        hasboth &= ~(x[badcol].to_numpy());
        pass;
    
    
    haseither = ~(eyedflist[0][badcol].to_numpy());
    for x in eyedflist[1:]:
        haseither |= ~(x[badcol].to_numpy());
        pass;

    if(only_alleyes_binoc):
        bdf[badcol] = ~hasboth;
    else:
        bdf[badcol] = ~haseither;
        pass;

    #REV: from offset if one eye is missing, offset by mean amount from known eye to binocular. Should really use only local timepoints
    ## but meh.
    bdf['fromoffset'] = False;
    
    #print(bdf[tmpx]);
    if( (~np.isfinite(bdf[tmpx])).all() ):
        #raise Exception("BINOCULAR ALL NAN (i.e. no data in any eye)");
        print("WARNING: BINOCULAR ALL NAN (i.e. no data in any eye)");
    
    if( len(eyetags) > 1 and
        exclude_thresh >= 0 ):
        
        ldf = eyedflist[0];
        rdf = eyedflist[1];
        
        dx = ldf[tmpx] - rdf[tmpx];
        dy = ldf[tmpy] - rdf[tmpy];
        d = np.sqrt( dx**2 + dy**2 );
        
        bdf[thresh_lr_badcol] = False;
        ldf[thresh_lr_badcol] = False;
        rdf[thresh_lr_badcol] = False;
        
        #REV: fucking pandas .loc returns empty series so these error if there are none meeitng criterion
        if( (d > exclude_thresh).any() ):
            #bdf.loc[ (d > exclude_thresh), [tmpx, tmpy] ] = np.nan;
            bdf.loc[ (d > exclude_thresh), thresh_lr_badcol ] = True;
            bdf.loc[ (d > exclude_thresh), badcol ] = True;
            
            ldf.loc[ (d > exclude_thresh), thresh_lr_badcol ] = True;
            ldf.loc[ (d > exclude_thresh), badcol ] = True;
            
            rdf.loc[ (d > exclude_thresh), thresh_lr_badcol ] = True;
            rdf.loc[ (d > exclude_thresh), badcol ] = True;
            pass;
        
        #REV: doesn't exclude timepoints from L/R too?
        pass;
    
    #REV: replace with mean offset.
    if( len(eyetags) > 1 and
        not only_alleyes_binoc ):
        
        if( len(eyetags) != 2 ):
            raise Exception("Theoretically only works for 2 eyes");
        

        for eye in eyedfdict:
            eyedf=eyedfdict[eye];
            if( pu.utils.allnan( eyedf[tmpx] )  or
                pu.utils.allnan( eyedf[tmpx] ) ):
                print("WARNING: all X or Y is NAN for eye {}... (skipping for binoc)".format(eye));
                continue;

            for col in [tmpx,tmpy]:
                z = eyedf[col].to_numpy();
                zisnan = np.isnan(z);

                #REV: I should add to those that EXIST but NOT haveboth

                #REV: must nanmean because some z may not be nan...
                #bzoffset = np.nanmean(bdf[col][ hasboth ] - z[ hasboth ] );
                if( np.any( np.isnan( bdf[col][hasboth] )  ) ):
                    raise Exception("BDF should never be NAN when both exist...");
                #print(np.sum(hasboth));
                #REV: ah, if there is only one and this is it, of course it will be empty...
                if( np.sum(hasboth) > 0 ):
                    if( np.sum( (~zisnan & ~hasboth) ) < 1 ):
                        #raise Exception("None that have only one eye and are not NAN");
                        print("WARNING -- None eye {}, col {} that have only one eye and are not NAN".format(eye, col));
                        continue;
                        
                    bzoffset = np.mean(bdf[col][ ~zisnan & hasboth ] - z[ ~zisnan & hasboth ] );
                    #if( np.any( np.isnan( bzoffset ) ) ):
                    #    raise Exception("WTF bzoffset at least one is NAN");
                    if( np.isnan( bzoffset ) ):
                        raise Exception("EYE {} -- Should never happen bzoffset is NAN (by design)".format(eye));
                    
                    #REV: will this work? I'm setting binocular
                    #REV: based on "predicted" mean offset.
                    #REV: so, for missing data, I should add my
                    #REV: typical offset
                    #REV: because it would be just equal to the other
                    #REV: eye, with nanmean. So, I should add its
                    #REV: mean offset...
                    print("Mean offset eye {} col {}: {}".format(eye, col, bzoffset));
                    bdf.loc[(~zisnan & ~hasboth), [col] ] = (bdf[(~zisnan & ~hasboth)][col] + bzoffset);
                    bdf.loc[(~zisnan & ~hasboth), ['fromLRoffset'] ] = True;
                    
                    
                    pass;
                else:
                    print("WARNING: there are no datapoints with binocular data but for which eye {} is NAN".format(eye));
                    pass;
                pass;
            pass;
        pass;
    
    bdf[xcol] = bdf[tmpx];
    bdf[ycol] = bdf[tmpy];
    
    eyedflist.append(bdf);
    
    df = pd.concat(eyedflist).sort_values(by=tcol).reset_index(drop=True);
    
    #REV: drop tmp columns
    df = df[ [c for c in df.columns if c not in [tmpx, tmpy] ] ];
        
    #print("Mean L/R dist (dva): [{:5.2f}] (sd: {:5.2f})  ({:5.2f}-{:5.2f})".format(d.mean(), d.std(), d.min(), d.max() ) );
    #diffs = df.groupby('Tsec').apply()
    
    #REV: need better way of combining binoc (don't exclude NAN?). Use mean offset, detect errors, return "score"?
    #REV: need to "reverse" figure out which eyes are best at which points (tracked eye changes).
    #REV: Use the eye that has "useable" data. (Give it a score, smooth movement less than some max, no spikes, etc.).
    
    return df;




#REV: both assume "flat screen"
def preproc_SHARED_dva_from_flatscreen(df, ppm, distm, method='trig', dropraw=True, sanitythreshdva=-1):
    """

    Parameters
    ----------
    df :
        
    ppm :
        
    distm :
        
    method :
         (Default value = 'trig')
    dropraw :
         (Default value = True)
    sanitythreshdva :
         (Default value = -1)

    Returns
    -------

    """
    dva_per_m = ut.get_center_dva_per_meter( distm, ppm );
    dva_per_px = 1/ppm * dva_per_m;
    
    df['cgx_dva_linear'] = df['cgx_px'] * dva_per_px;
    df['cgy_dva_linear'] = df['cgy_px'] * dva_per_px;
    
    #REV: estimate dva in X and Y using trig.
    #REV: note basic yaw/pitch...
    #REV: just do each component separately rather than single
    #REV: rotated angle with components.
    adjpx=distm * ppm; #REV: big number
    xopppx=df['cgx_px'];
    yopppx=df['cgy_px']; # tan(th) = opp/adj. th=atan2(opp,adj)
    
    df['cgx_dva_trig'] = np.degrees(np.arctan2(xopppx, adjpx));
    df['cgy_dva_trig'] = np.degrees(np.arctan2(yopppx, adjpx));
    
    if( sanitythreshdva >= 0 ):
        xdiff = abs(df.cgx_dva_trig - df.cgx_dva_linear);
        ydiff = abs(df.cgy_dva_trig - df.cgy_dva_linear);
        
        maxx = np.nanmax( abs(df.cgx_dva_trig - df.cgx_dva_linear) );
        maxy = np.nanmax( abs(df.cgy_dva_trig - df.cgy_dva_linear) );
        if(maxx > sanitythreshdva or
           maxy > sanitythreshdva ):
            df['xdiff']=xdiff;
            df['ydiff']=ydiff;
            print(df.loc[ (df.xdiff > sanitythreshdva) | (df.ydiff > sanitythreshdva), ['cgx_px', 'cgy_px',
                                                                                        'cgx_dva_trig', 'cgx_dva_linear', 'xdiff',
                                                                                        'cgy_dva_trig', 'cgy_dva_linear', 'ydiff'] ]);
            raise Exception("Maximum divergence between linear and trig estimates of gaze DVA over threshold ({}): {} {}".format(sanitythreshdva, maxx,maxy));
        pass;
    
    if(method=='trig'):
        df['cgx_dva'] = df['cgx_dva_trig'];
        df['cgy_dva'] = df['cgy_dva_trig'];
        pass;
    elif(method=='linear'):
        df['cgx_dva'] = df['cgx_dva_linear'];
        df['cgy_dva'] = df['cgy_dva_linear'];
        pass;
    else:
        raise Exception("Uknown method");

    if( dropraw ):
        df = df[ [c for c in df.columns if
                  c not in ['cgx_dva_trig',
                            'cgy_dva_trig',
                            'cgx_dva_linear',
                            'cgy_dva_linear',]
                  ]
                ];
        pass;
    return df;




#From paper: https://link.springer.com/article/10.3758/s13428-018-1075-y
#1) Get data (done)
#2) Filtering.
#  a) Compute dilation speed for each sample tsec: (d is pupil size of this eye)
#     dprime[t] = max( abs((d[t]-d[t-1]) / (tsec[t]-tsec[t-1])), abs((d[t+1]-d[t]) / (tsec[t+1]-tsec[t]) ));
#  b) MADdprime = np.med( abs( dprime[t] - np.med( dprime ) ) ); #i.e. the median of each samples' (absolute) divergence from the median.
#  c) Thresh_dprime = median(dprime) + (CONST * MADdprime);
#   ---> any sample with dprime (dilation speed) above Thresh_dprime can be exluded.
#   ---> Then, dilate NANs, between 10msec and 30msec, due to artifacts around blinks etc. (initial pupil size change may be underestimated).
##  But, only dilate 50msec around gaps of 75msec or larger.


#3) Trend-line deviation -> create trend-line of pupil size, remove any that devaite from LPF pupil size line (same as b & c)
#4) Sparsity removal -> basically detect "chunks" of data bounded by NAN chunks larger than e.g. 40msec, and remove any resulting
#    chunk less than length e.g. 50 msec. Why not just do on/off/on/off, any chunk less than 50, separated by ANY number of NANs?
##   I guess this lets us interpolate if we have 1k hz, but have OK BAD OK BAD OK BAD etc.

#5) process valid samples by combining left/right eye, using constant offset, etc.. Smooth using e.g. 4Hz LPF.
##  -> need to specify to "exclude" gaps of 250 msec from interpolation.
def preproc_SHARED_pupilsize(sampledf,
                             timecol, #e.g. 'Tsec0'
                             pacol, #e.g. 'pa'
                             eyecol, #e.g. 'eye'
                             characteristic_timescale_sec=0.010, #Rough characteristic timescale
                             ## of pupil size change
                             ):
    """

    Parameters
    ----------
    sampledf :
        
    timecol :
        
    #e.g. 'Tsec0'valcol :
        
    #e.g. 'pa'eyecol :
        
    #e.g. 'eye'characteristic_timescale_sec :
         (Default value = 0.010)
    #Rough characteristic timescale## of pupil size change :
        

    Returns
    -------

    """
    
    lst=[];
    for eye, df in sampledf.groupby(eyecol):
        ##No, I need DIFFS. (dprime)
        df = df.sort_values(by=timecol).reset_index(drop=True);
        smoothtimename='__'+timecol;
        if(smoothtimename in df.columns):
            raise Exception("Meh, doubled smoothtimename: {}".format(smoothtimename));
        from datetime import timedelta
        df[smoothtimename] = pd.to_timedelta(df[timecol], unit='s');
        df = df.set_index(smoothtimename);
        smoothpacol=pacol+'_lpf';
        df[smoothpacol] = df[pacol].rolling(center=True,
                                              min_periods=1,
                                              window=pd.to_timedelta(characteristic_timescale_sec, unit='s'),
                                              ).mean();
        df[pacol] = df[smoothpacol];
        df = df.reset_index(drop=True);
        diffdf, newcol = ut.magnitude_change_over_time(df, valcol=pacol, timecol=timecol);
        
        print("EYE: {} (NEWCOL: {})".format(eye, newcol));
        print(diffdf.groupby(newcol, dropna=False).count().sort_values(by='Tsec'));
        print("NUM NANs in PA {}/{} (eye=={})".format(diffdf[newcol].isna().sum(), len(diffdf.index), eye));
        print(diffdf[newcol].dtype);
        print("PAdelta {}: Mean {},  Med {},  {}-{}".format(newcol, diffdf[newcol].mean(),
                                                            diffdf[newcol].median(),
                                                            diffdf[newcol].min(),
                                                            diffdf[newcol].max(),
                                                            
                                                            ));
        print("NUM ZEROS: ", len(diffdf[diffdf[newcol]==0]));
        #import matplotlib.pyplot as plt;
        #fig = plt.figure();
        #plt.hist(diffdf[newcol], bins=100);
        #diffdf = diffdf.iloc[5000:5500];
        #plt.plot(diffdf[timecol], diffdf[valcol]);
        #diffdf = diffdf
        #REV: OK issue is at high sample rates (I guess) PA does not change so zero becomes the
        ## median, which fucks up both median and MAD etc. Need a more robust way to do it.
        ## I.e. slowly downsample and identify where it becomes stable? Or take standard deviation
        ## over time (regardless of time) and use that? I.e. local entropy? Not idea...
        #fig.savefig('wtf.pdf');
        #exit(0);
        
        devdf, madval = ut.MAD_timediff(indf=diffdf, valcol=newcol);
        mad_name = newcol + '_mad';
        devdf[mad_name] = madval; #'pa_abs_tdiff_mad'  waste, whatever...
        lst.append(devdf);
        pass;
    
    #REV: should contain ALL results
    
    alldf=pd.concat(lst).reset_index(drop=True);
    
    return alldf;


#REV: to implement?
###Coe, B. C., Huang, J., Brien, D. C., White, B. J., Yep, R., & Munoz, D. P. (2024). Automated Analysis Pipeline for Extracting Saccade, Pupil, and Blink Parameters Using Video-Based Eye Tracking. Vision, 8(1), 14. https://doi.org/10.3390/vision8010014


## REV: need to smooth, median filter, and then find shifts in gaze.
## Problem is that gaze can also 'drift' and can also have periods of e.g.
## VOR, etc.
## So, we are looking for shifts on the order of less than 100~150 msec.
## And assume the eye stays where it was before/after, even if there are
## small drifts before after. However, it should detect movements in mean
## location, and optimize that, e.g. find the difference that maximally
## straddles the movement, and then crushes size until the shift starts getting
## much less (i.e. partway through shift). But what if it is weirdly shaped,
## E.g. a blink so that vertical goes down.

# "Knobs":
## MAD_mult:
## 1) A "MAD" is computed for the entire EDF file (dataframe). This is dpa_MAD. (based on pupil area)
#  2) We define a "threshold" as:  median of absolute change in pupil size (over time, e.g. change between samples divided by change in time between samples). Absvalue. Note "PA" is arbitrary units.
# Then, threshold is: median * MEDCONST (blinkremoval_med_mult) +
#             dpa_MAD * blinkremoval_MAD_mult (NCONST).
#  
#        dpa_MAD = eyedf.pa_abs_tdiff_mad.unique()[0];
#        thresh = ( (np.nanmedian(eyedf.pa_abs_tdiff) * MEDCONST) +
#                   (dpa_MAD  *  NCONST)
#                  );

## REV: todo, make a "trend line" and remove guys which deviate from that?
# REV: note in MRI during EPI we have "foundational function" oscillating at some rate. We can do a fourier transform and identify
# that weird oscillation and filter it out? Will depend on the FMRI. Low pass filter of e.g. 4Hz seems good...but that is for pupil size?
# But that will remove SACCADES? That's fine. We are not filtering GAZE POSITION...just pupil size. We need to filter gaze too... Gaze
## position pupil size stayed relatively constant, just shifted up/down left/right...OK. Does pupil size change too? We have "bad tracking".
def preproc_SHARED_label_blinks(df,
                                sr_hzsec,
                                blinkremoval_MAD_mult=5,
                                blinkremoval_med_mult=1,
                                blinkremoval_dilate_win_sec=0.030,
                                blinkremoval_orphan_upperlimit_sec=0.020,
                                blinkremoval_orphan_bracket_min_sec=0.040,
                                blinkremoval_shortblink_minsize=0.070,
                                tsecname='Tsec',
                                eyecol='eye',
                                valcol='px', #REV: is pupil area NAN when no eye tracking?
                                badcol='bad',
                                pacol='pa',
                                #patdiffcol='pa_abs_tdiff',
                                preblinkcols=[] ): #'elhasblink']):
    """

    Parameters
    ----------
    df :
        
    sr_hzsec :
        
    blinkremoval_MAD_mult :
         (Default value = 5)
    blinkremoval_med_mult :
         (Default value = 1)
    blinkremoval_dilate_win_sec :
         (Default value = 0.050)
    blinkremoval_orphan_upperlimit_sec :
         (Default value = 0.020)
    blinkremoval_orphan_bracket_min_sec :
         (Default value = 0.040)
    blinkremoval_shortblink_minsize :
         (Default value = 0.100)
    tsecname :
         (Default value = 'Tsec')
    eyecol :
         (Default value = 'eye')
    valcol :
         (Default value = 'pa')
    badcol :
         (Default value = 'bad')
    preblinkcols :
         (Default value = [] ): #'elhasblink'])

    Returns
    -------

    """
    newdflist = list();

    if( badcol in df.columns ):
        raise Exception("WTF [bad] column ({}) already in df".format(badcol));
    
    for eye, eyedf in df.groupby(eyecol):
        
        eyedf = eyedf.sort_values(by=tsecname).reset_index(drop=True);
        neweyedf=eyedf.copy();
        
        #################################### EYELINK BLINK/BAD DATA DETECTION #####################################
        
        finites = np.array(~np.isfinite( eyedf[valcol] ));
        
        #REV: preblinks are e.g. detected previously by EL etc.
        preblinks = np.full(len(eyedf.index), False); 
        for pb in preblinkcols:
            preblinks = preblinks | eyedf[pb];
            pass;
        
        #REV: add event due to "bad detection"? Better to handle as "events" or "dense data"?
        
        #REV: should I expand ALL of them with NAN-dilation (before combining?). In case? Should work same way?
        
        #print( len(blinks), len(blinks[blinks==True]));
        #print( len(finites), len(finites[finites==True]));
        
        badeye = len(finites) == len(finites[finites==True]);
        if( badeye ):
            #REV: will not 'readd' with correct "bad data"? Just do at end.
            neweyedf['badpupilPRE'] = True; #combined EL/pupilarea, no expansion or orphans.
            neweyedf['badpupil'] = True; #just pupil > thresh
            neweyedf['badPRE'] = True; #just gaze (pupil center detection) is NAN, or pre-detected as blink by e.g. eyelink
            neweyedf['bad'] = True; #combined, and expanded NANs, re-added orphans.
            newdflist.append(neweyedf);
            continue;
        
        #print(eye, eyedf.pa.min(), eyedf.pa.max(), eyedf.pa.std());
        #print(eye, eyedf.pa_abs_tdiff.min(), eyedf.pa_abs_tdiff.max());
        patdiffcol = pacol + '_abs_tdiff';
        madname=patdiffcol + '_mad';
        
        if( len( eyedf[madname].unique() ) != 1 ):
            print(eyedf[madname]);
            raise Exception("WTF pupil tidff mad not one");
        
        
        
        
        
        baddata = ( finites | preblinks );
        pevdf = ut.cond_rle_df( baddata, val=True, t=eyedf[tsecname] ); #REV: creates a DF of "events" TRUE where 
        finiteblink_evdf = pevdf.copy();
        finiteblink_sdf = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx); #REV: should be same as  baddata
        print(finiteblink_sdf);
        
        #REV: choose some percentile? Not median? Base on distribution?
        #REV: If I base on "percentile" I will specify about how much I will exclude...e.g. 95% means I will exclude those above 95%
        #REV: I can't necessarily predict beforehand how many points need to be removed per trial...it will depend on the trial.
        #REV: trials with large variance for example? Use STD? STD is just e.g. 67% 
        
        if( len(eyedf[madname].unique()) != 1 ):
            raise Exception( "More than one MAD: {}".format(eyedf[madname].unique()));
        
        dpa_MAD = eyedf[madname].unique()[0];
        thresh = ( (np.nanmedian(eyedf[patdiffcol]) * blinkremoval_med_mult) +
                   (dpa_MAD  *  blinkremoval_MAD_mult)
                  );
        
        pevdf = ut.events_over_thresh(eyedf[patdiffcol], thresh=thresh, t=np.array(eyedf[tsecname]) );
        dpupil_evdf = pevdf.copy();
        dpupil_sdf = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx);
        
        
        
        badx = (dpupil_sdf | baddata);
        pevdf = ut.cond_rle_df( badx, val=True, t=eyedf[tsecname] );
        dpupil_finiteblink_pevdf = pevdf;
        dpupil_finiteblink_sdf = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx);
        
        DILATE_SEC_WIN=blinkremoval_dilate_win_sec; #0.055; #20 msec...?
        winsamp = math.ceil(sr_hzsec * DILATE_SEC_WIN); #e.g. 1e3 * 2e-2 = 2e1 = 2 * 10 = 20
        badxdilated = ut.dilate_val(badx, val=True, winsamp=winsamp);
        
        
        pevdf = ut.cond_rle_df( badxdilated, val=True, t=eyedf[tsecname] );
        dilated_pevdf = pevdf;
        dilated_sdf = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx);
                
        GOODTOOSMALL=blinkremoval_orphan_upperlimit_sec; #=0.055;
        BADBIGENOUGH=blinkremoval_orphan_bracket_min_sec; #0.040; #REV: *all* bads must NECESARILY be here since dilated?
        
        for i in range(len(pevdf.index)):
            idx=pevdf.index[i]; #iloc[i];
            row=pevdf.iloc[i];
            if( (row.v == False) and
                (row.lent < GOODTOOSMALL) and
                ((i==0) or (i==len(pevdf.index)-1) or ((i>0) and
                                                       (i<len(pevdf.index)-1) and
                                                       (pevdf.iloc[i-1].lent>=BADBIGENOUGH) and
                                                       (pevdf.iloc[i+1].lent>=BADBIGENOUGH)) )
               ):
                pevdf.loc[idx, 'v'] = True; #REV: swap it.
                pass;
            
            pass;
        
        badxdilated = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx);
        
        #REV: dumb, but combines true/true etc. I guess.
        pevdf = ut.cond_rle_df( badxdilated, val=True, t=eyedf[tsecname] );
        noorphan_pevdf = pevdf;
        noorphan_sdf = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx);
        
        
        #REV: this captures other stuff too...better to just capture clean saccades having
        # low pupil size variance naturally...
        REMOVE_SHORT_BLINKS=False;
        if( REMOVE_SHORT_BLINKS ):
            MIN_BLINK_SEC=blinkremoval_shortblink_minsize; #0.100;
            for i in range(len(pevdf.index)):
                idx=pevdf.index[i]; #iloc[i];
                row=pevdf.iloc[i];
                if( (row.v == True) and
                    (row.lent < MIN_BLINK_SEC) ):
                    pevdf.loc[idx, 'v'] = False; #REV: swap it.
                    pass;
                
                pass;
            
            badxlongblinks = ut.inverse_rle(pevdf.v, pevdf.sidx, pevdf.lidx);
            
            #REV: dumb, but combines true/true etc. I guess.
            pevdf = ut.cond_rle_df( badxlongblinks, val=True, t=eyedf[tsecname] );
            pass;
        
        
        neweyedf['badpupilPRE'] = dpupil_finiteblink_sdf; #combined EL/pupilarea (MAD)
        neweyedf['badpupil'] = dpupil_sdf; #just pupil > thresh
        neweyedf['badPRE'] = finiteblink_sdf; #just gaze target was NAN, or was blink in preblink
        neweyedf['bad'] = noorphan_sdf; #combined, and expanded NANs, re-added orphans.
        
        newdflist.append(neweyedf);
        
        pass;
    
    df = pd.concat(newdflist).sort_values(by=[eyecol, tsecname]).reset_index(drop=True);
    
    return df;






def blink_df_from_samples(df,
                           badcol='bad',
                           tcol='Tsec0',
                           stcol='stsec',
                           encol='ensec'
                           ):
    ev = pu.utils.cond_rle_df( df[badcol], val=True, t=df[tcol] );
    ev = ev[ ev['v'] == True ].reset_index(drop=True);
    ev[stcol] = ev['st'];
    ev[encol] = ev['et'];
    ev['label'] = 'BLNK';
    ev = ev[ [ c for c in ev if c in [stcol, encol, 'label'] ] ];
    return ev;



