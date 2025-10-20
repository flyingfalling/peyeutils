
#REV: roll won't work because it re-introduces on other side...
def shift_elements(arr, num, fill_value):
    result = np.empty_like(arr)
    num = int(num); #REV: ceiling?
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
        pass;
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
        pass;
    else:
        result[:] = arr
        pass;
    return result


def merge_from_prefix( sampdf, prefixes, col ):
    cols=[ p+col for p in prefixes ];
    sampdf[ col ] = np.nanmean( sampdf[ cols ], axis=1 );
    return; #REV: sampdf in place?


#REV: they implement this in CPP
#REV: this is already couched in terms of dt
#REV: (which will be SECONDS, or e.g. DEG/SECOND).
#REV: and X (will be DEGREES)
def compute_velocity_ek(x, tw_samp, dt):
    max_hs = tw_samp//2; #REV: half-span
    isfinite = np.isfinite(x);
    vel = np.zeros(len(x), dtype=x.dtype);
    
    #REV: naive impl...
    for imid in range(1, len(x)): #len(x)-1
        tdiff=0;
        twei=0;
        
        hs=max_hs;
        while( hs>0 ):
            ist = imid - hs;
            ien = imid + hs;
            if( ist >= 0 and
                ien < len(x) and
                isfinite[ist] and
                isfinite[ien]
               ):
                tdiff+= x[ien]-x[ist];
                twei += 2*hs;
                pass;
            hs-=1;
            pass;
        
        if(twei > 0):
            vel[imid] = tdiff/(twei*dt);
            pass;
        pass;
    
    return vel;



#REV: they use 6 sample moving average, and 250 hz eyetracker, so 6 samples is...0.024 seconds i.e. 24 msec
#REV: saccadr uses 20 msec default.
def diff_ek(x, y, params):
    sr = params['samplerate'];
    tw_sec = params['ek_vel_window_sec'];
    dt = 1/sr;
    tw_samp = int(tw_sec * sr);
    if( (tw_samp % 2)==0 ):
        tw_samp+=1;
        pass;
    if( tw_samp < 3 ):
        tw_samp=3;
        pass;
    
    # diff
    xvel = compute_velocity_ek(x, tw_samp, dt); # * sr; vel_ek includes /dt
    yvel = compute_velocity_ek(y, tw_samp, dt); # * sr;
    ampl = np.sqrt(xvel**2 + yvel**2);
    
    return xvel, yvel, ampl;


def dilate_nans( df, cols, params ):
    sr=params['samplerate'];
    dilate_nan_win_samp = math.ceil(params['dilate_nan_win_sec'] * sr);
    #min_blink_samp = int(params['min_blink_sec'] * sr);
    mask = np.full( len(df.index), False );
    for col in cols:
        mask = mask | (get_dilated_nan_mask( df[col],
                                             dilate_nan_win_samp ) );
        pass;
    
    for col in cols:
        df[col][mask] = np.nan;
        pass;
    
    return df;


def dilate_val( arr, val, winsamp ):
    #min_blink_samp = int(params['min_blink_sec'] * sr);
    mask = np.full( len(arr), False );
    mask = mask | (get_dilated_mask( arr,
                                     winsamp,
                                     val ) );
    arr[mask] = val;
        
    return arr;

def dilate_xy_nans( df, params ):
    sr=params['samplerate'];
    dilate_nan_win_samp = math.ceil(params['dilate_nan_win_sec'] * sr);
    min_blink_samp = math.ceil(params['min_blink_sec'] * sr);
    xname=params['xname'];
    yname=params['yname'];
    mask = get_dilated_nan_mask( df[xname],
                                 dilate_nan_win_samp,
                                 min_blink_samp );
    df[xname][mask] = np.nan;
    df[yname][mask] = np.nan;
    return df;


def diff_nh(x, y, params):
    sg_order = params['nh_savgol_order'];
    sg_win_sec = params['nh_savgol_window_sec'];
    sr = params['samplerate'];
    dt = 1/sr;
    sg_win_samp = sg_window_sec * sr;
    
    vx = np.diff(x) / dt;
    vy = np.diff(y) / dt;
    v = np.sqrt( vx**2 + vy**2 );
    
    fvx = savgol_filter(vx, savgol_win_samp, sg_order );
    fvy = savgol_filter(vy, savgol_win_samp, sg_order );
    fv = savgol_filter(v, savgol_win_samp, sg_order );
    
    return fvx, fvy, fv;


#REV: only include finites...
def sd_via_median_estimator(x):
    sd = np.sqrt( np.nanmedian(x**2) - np.nanmedian(x)**2 );
    SMALL_EPSIL=0.00000001;
    if( sd < SMALL_EPSIL ):
        sd = np.sqrt( np.nanmean(x**2) - np.nanmean(x)**2 );
        pass;
    if( sd < SMALL_EPSIL ):
        print("ERROR, median too small...");
        exit(1);
        pass;
    
    return sd;

# Engbert and Kliegl (2003) \doi{10.1016/S0042-6989(03)00084-1}
#REV: original method uses milliseconds (and degrees?)
def method_ek(df, params, eyepfix):
    sr = params['samplerate'];
    
    #velth_degsec = params['ek_vel_thresh_degsec']; 
        
    #REV: they use "6". Then, they multiply vel_thresh times sigma_xy
    #REV: sd of velocity will be standard deviation (in deg/sec) of velocity.
    #REV: they simply check locations where normalized velocity > 1
    #REV: normalized velocity is euclid dist (L2 norm) of both velocity components,
    #REV: with each component normalized by threshold
    #REV: where threshold is the standard deviation of that velocity component multiplied by 6...

    #REV: OH, they require "right and left eye overlap" for saccade to be defined! Cool ;)
    
    lambda_velth = params['ek_vel_thresh_lambda']; #REV: multiple of SD (where SD is median estimator...) 
    
    sd_funct = params['ek_sd_funct'] if 'ek_sd_funct' in params else sd_via_median_estimator;
    
    min_dur_sec = params['ek_min_dur_sec'];
    min_dur_samp = math.ceil(min_dur_sec*sr);
    
    #REV: not needed (checked after...).
    #They used to have check  that it is superthresh surrounded by subthresh, etc., but commented out.
    
    #min_sep_sec = params['ek_min_sep_sec'];
    #min_sep_samp=math.ceil(min_sep_sec*sr);

    
    xvel = df[eyepfix+'xvel'];
    yvel = df[eyepfix+'yvel'];

    sd_x = sd_funct( xvel );
    sd_y = sd_funct( yvel );
    
    #print(sd_x, sd_y);
    
    sd_x_th = sd_x * lambda_velth;
    sd_y_th = sd_y * lambda_velth;
    
    vel_norm = np.sqrt( (xvel / sd_x_th)**2 + (yvel / sd_y_th)**2 );
    
    #print(vel_norm);
    
    vals, sts, lens = rle( vel_norm > 1.0 );
    
    vals = (vals == True) & (lens > min_dur_samp);
    
    saccsamps = inverse_rle( vals, sts, lens );
    
    return saccsamps;


#REV: method_om_adaptive

#REV: shit this will not filter blinks, because blinks may have larger peak velocity than non-blinks?

#REV: same as method_om, but adaptively only handles saccades (from a large group of data) adaptively, starting from the largest
#     saccades...and going smaller for both peak value and threshold value. (velocity).
#     Note blinks will have weird after-acceleration/velocity, but may still have higher peak velocity than normal saccades, so
#     selection criterion will not work (for microsaccads, they used avg pvel because it is assumed to be higher than surrounding noise).
#REV; But for blinks/artifacts, they will by definition have higher peaks...just different shapes?
#REV: I could do "interactively", i.e. show each "cluster" (show velocity traces) and have user specify which is the treal group?
#REV: they should be (roughly) symmetric...although pre time likely shorter than post time.
#REV: also, they should have a specific DURATION!!!! (I should include in PCA). Relationship of duration versus peak velocity.
#REV: Select the groupings which produce the best fit to known data (hand labelled?).
def method_om_adaptive(df, params, eyepfix):
    
    return;


def method_om(df, params, eyepfix,
              tsecname='Tsec'):
    min_inter_peak_sec = params['om_min_inter_peak_sec'];
    max_peaks_per_sec = params['om_max_peaks_per_sec'];
    vel_thresh_degsec = params['om_vel_thresh_degsec']; #REV: slow microsaccs with noisy system (zuber 1965)
    peak_detect_shift_sec = params['om_vel_peak_detect_shift_sec'];
    sr = params['samplerate'];
    peak_detect_shift_samp = int( peak_detect_shift_sec * sr );
    #pca_var_thresh_degsec = params['om_pca_var_thresh_degsec']; #REV: not used now (would normally only used PCA components > this thresh of
    #explanation)
    
    vel = df[eyepfix+'vel']; #REV: amplitude of velocity, i.e. length in vector direction. (otherwise it would be N-component)
    
    acc = df[eyepfix+'acc']; #REV: this is MAGNITUDE of acceleration (or decel), i.e. abs(acc)
    #BIGNUM=1e10;
    #REV: just fill with repeat...
    if( len(vel) < 1 ):
        print("Vel 0 len");
        exit(1);
        pass;
    
    
    #REV: fine, because guarantee large amounts between peaks...
    #REV: but, too large and all points in sacc will be a peak...
    #REV: fine though as I go from largest to smallest...
    rshift = shift_elements( vel, peak_detect_shift_samp, vel[0] ); #REV: fill empty with bignum so that it will never detect peak at ends...
    lshift = shift_elements( vel, -peak_detect_shift_samp, vel[len(vel)-1] );

    #REV: this is not a good peak detector...
    #REV: will overselect... with >= and <=!!!
    #REV: but, will quickly abort because distance from it to
    #REV: that guy will be small (and it will already have selected all
    #REV: those points anyways...)
    peaks = (vel >= rshift) & (vel >= lshift) & (vel > vel_thresh_degsec);
    
    #REV: we know "irow"
    peaks = np.array(np.where( True == peaks )[0], dtype=int); #REV: 0 because it returns tuple for some reason fuck you.
    
    pdf = pd.DataFrame.from_dict( dict( pidx=peaks, vel=vel[peaks] ) );
    
    pdf = pdf.sort_values( by='vel', ascending=False ).reset_index(drop=True);
    
    pdf = pdf[ ~pdf.vel.isna() ];
    
    #REV: select peaks. pidx is "location" (i.e. index in velocity array) of peak, vel is value.
    #REV: we will make a list of "accepted" peaks (actually, their indicies in the velocity array
    #REV: based on ordering.
    
    #REV: note we are going from highest peak velocity to lowest.
    issel = [ pdf.iloc[0].pidx ];
    
    for idx, row in pdf.iloc[1:].iterrows():
        ip = int(row.pidx); #REV: index in vel array
        timetosel = df.iloc[issel][tsecname] - df.iloc[ip][tsecname]; #REV: distance from peak in question to all other peaks
        #print(timetosel);
        if( np.nanmin(np.abs(timetosel)) < min_inter_peak_sec ): #REV: if minimum of all distances is less than min_inter_peak, throw away (skip)
            continue;
        
        #REV: if abs time_to_sel less than 0.5...
        #REV: that is "one"
        halfsec=0.5;
        peaks_in_1sec = np.nansum(np.abs(timetosel) < halfsec);   #REV: total number of saccades within 0.5 sec of me (i.e. including me, sum will be
        #print(peaks_in_1sec);
        # number within 1 second window)
        #REV: if past free parameter max peaks per sec, throw away (skip)
        if( peaks_in_1sec >= max_peaks_per_sec ):
            continue;
        
        #: otherwise, append (keep this guy)
        issel.append( ip );
        pass;
    
    #REV: drop peaks we didnt keep. (only keep those selected...)
    pdf = pdf[ pdf.pidx.isin(issel) ]; #REV: these are pidx, offsets in vel array.

    #print(pdf);
    
    #REV: select "chunks" around each peak above vel thresh.
    ispeak = np.full(len(vel), False);
    
    #forwards
    for ip in pdf.pidx:
        idx = ip;
        while( (idx < len(vel)) and
               np.isfinite(vel[idx]) and
               (vel[idx] > vel_thresh_degsec)
               #REV: check same trial...
               ):
            ispeak[idx] = True;
            idx += 1;
            pass;
        pass;
    
    #backwards
    for ip in pdf.pidx:
        idx = ip - 1;
        while( idx >= 0 and
               np.isfinite(vel[idx]) and
               (False == ispeak[idx]) and
               (vel[idx] > vel_thresh_degsec)
               #REV: check same trial...
               ):
            ispeak[idx] = True;
            idx -= 1;
            pass;
        pass;

    #REV: at this point I no longer need "issel" because
    #     I've amalgamated (merged) them all together in sample space.
    
    vals, sts, lens = rle( ispeak );
    #REV: he defines "onset" and "offset" in "peak times" (just concats peaks only?)
    #REV: not he includes both YES and NO for peaks...


    #REV: remove those that have a NAN immediately before or after them?
    #REV: do it after... ("return from blink")
    #REV: can also do with "dilate nans" but will vary.
    
    
    sacclist=[];
    for v, s, l in zip(vals, sts, lens):
        e=s+l; #REV: is is NOT INCLUDED (0+1 is #1)
        
        #REV: don't bother with non-peaks
        if (False==v):
            continue;
        
        peakvel = np.nanmax( vel[s:e] );
        
        #REV: could be zero accel phase
        # if it suddenly jumps up from below threshold to above, and that is the peak, then decreases but still above thresh.
        
        #REV: similar for decel phase.
        
        #REV: take first or middle of a long run of equal values?
        ipeakvel = np.where( vel[s:e] >= peakvel )[0]; #REV: tuple, then first.
        ipeakvel = ipeakvel[ len(ipeakvel)//2 ]; #REV: take middle if there are many in a row.
        #print(ipeakvel);
        
        true_ipeakvel = s+ipeakvel;
        #print(vel[s:e]);
        #REV: wait, if < small val it will get fucked!!!!
        
        #REV: wtf some are only 1 sample long...
        
        #print(true_ipeakvel-s);
        #print(e-true_ipeakvel);
        
        #REV: +1 because end of slice is PAST end of actual slice.
        peakaccel = np.nanmax( acc[s:(true_ipeakvel+1)] );
        peakdecel = np.nanmax( acc[(true_ipeakvel):e] ); #REV: peak should never be peak accel or decel...its velocity switch so should be 0.
    
        #print(peakvel, peakaccel, peakdecel);
        
        if( not np.isfinite(peakvel) or
            not np.isfinite(peakaccel) or
            not np.isfinite(peakdecel)
            ):
            print( peakvel, peakaccel, peakdecel );
            print("Peak had NAN velocity/accel/decel. Skipping");
            continue;
        
        logpvel=np.log( peakvel );
        logpaccel = np.log( peakaccel );
        
        #REV: accel must be ABS ?!?!?! (this is euclid i.e. sqrt( xvel**2, yvel**2 ), so it will be positive. OK.
        #REV: do not change to min!!!!
        logpdecel = np.log( peakdecel );

        if( (~np.isfinite(logpvel)) or
            (~np.isfinite(logpaccel)) or
            (~np.isfinite(logpdecel))
            ):
            print( logpvel, logpaccel, logpdecel );
            print("LOG Peak had NAN velocity/accel/decel. Skipping");
            continue;
        
        #REV: why do they not have a minimum length?
        sacclist.append( dict( stidx=s,
                               enidx=e,
                               duridx=l,
                               val=v,
                               zlogpvel=logpvel,
                               zlogpaccel=logpaccel,
                               zlogpdecel=logpdecel
                              )
                        );
        pass;
    
    #REV: this correctly makes columns based on values of each dict etc.
    sdf = pd.DataFrame(sacclist);
    
    #REV: keep only saccades
    #sdf = sdf[ sdf.val == True ];
    
    propcols = ['zlogpvel', 'zlogpaccel', 'zlogpdecel'];
    
    #REV: default skip nan of pandas std() is to skip all NAN (don't include)
    #REV: normalize and center (z score) of each of the columns (logpvel, logpaccel, etc.)
    for col in propcols:
        sdf[col] = (sdf[col] - sdf[col].mean()) / sdf[col].std();
        pass;
    
    #REV: column length is 
    #REV: drop where col is NA?
    sdf = sdf.dropna();
    
    if( len(sdf.index) < 1 ):
        print("Method OM: No saccades.");
        return np.full(len(vel), False);
    else:
        print("Method OM: Probably {} saccades.".format(len(sdf.index)));
        #print(sdf);
        pass;
    
    #REV: now convert to 2d space from properties space (in this case, 3d because we use paccel, pvel, and pdecel)
    #REV: via PCA
    
    from sklearn.decomposition import PCA
    saccpca = PCA(n_components=2); #REV: just get the first 2 components...
    saccpca.fit( sdf[ propcols ].to_numpy() );
    
    #pcacomps = saccpca.components_;
    #pcaexpvar = saccpca.explained_variance_;
    
    #REV: they re-rotate the data into those dimensions coordinates
    #REV: that is X! And only take first 2 new coordinates i.e. 2d
    #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.transform
    
    invs = saccpca.transform( sdf[ propcols ].to_numpy() );
    #REV: row is "event", col is "component" in new 2d space...
    
    sdf['pc1'] = invs[:,0];
    sdf['pc2'] = invs[:,1];
    
    #REV: new coordinates for each saccade (now in a 2d space).
    #REV: now, cluster within that space to determine saccades versus noise...
    
    clustercols=['pc1', 'pc2'];
    
    kmeans=dict();
    scores=dict();
    maxcut=min(len(sdf.index), 4) + 1;
    if( len(sdf.index) < 3 ):
        print("WTF only a single event (sdf in saccadr)?!?!");
        bestk=0;
        kmeans[0] = np.zeros( len(sdf.index), dtype=int);
        pass;
    else:
        #REV: 2, maxcut? Separate into "noise" or "not"?
        for groups_n in range(2,maxcut): # 2, 3, 4
            #REV: split into N groups of equal size i.e. levels (2, 3, 4) based on ordering of values.
            velcut = pd.cut( sdf.zlogpvel, groups_n );
            #print(velcut);
            #REV: changed velcut to string, since I don't actually use values...
            sdf['velcut'] = velcut.astype(str); #REV: assign indices of the equal cut groups. These are just use to initialize group centers for k-means clustering.
            #print(sdf.columns);
            #print(sdf);
            #print(sdf.head());
            
            #REV: what is the point of this? Just a starting point for Kmeans? OK.
            #grpavg = sdf.groupby('velcut', as_index=False).mean().reset_index(drop=True);
            grpavg = sdf.groupby(['velcut'], as_index=False).agg( safe_agg(sdf,'mean') ).reset_index(drop=True);
            #print("For groups: {}".format(groups_n));
            #print(grpavg);
            #print(sdf);  #REV: velcut is a tuple wtf. OK, problem is that it divides into even parts, but no items may fall into some of those
            #cuts...fuck.
            
            from sklearn.metrics import silhouette_samples, silhouette_score
            from sklearn.cluster import KMeans
            
            #REV: they do vals, and cluster centers...
            #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            #REV: no init will be slower...but whatever.
            kmeans[groups_n] = KMeans(n_clusters=groups_n,
                                      #init=grpavg[clustercols].to_numpy(),
                                      n_init="auto").fit( X=sdf[clustercols].to_numpy() );
            
            
            # Compute width of each (silhouette) cluster
            scores[groups_n] = silhouette_score( X=sdf[clustercols].to_numpy(), labels=kmeans[groups_n].labels_, metric='euclidean');
                        
            #REV: they give cluster, center, silwidth
            pass;
        
        kdf = pd.DataFrame.from_dict( data=scores,
                                      orient='index',
                                      columns=['score'] ).reset_index(
                                          names='k');
        
        bestk = kdf.sort_values(by='score', ascending=False).iloc[0].k;
        pass;
    
    #REV: assign to grpidx column the group assignments for the best-scored number of groups for clustering (i.e. best k).
    sdf['grpidx'] = kmeans[bestk].labels_;
    
    before=len(sdf.index);
    
    #REV: then take only the saccades of the cluster with highest mean pvel.
    #bestgrp = sdf.groupby('grpidx', as_index=False).mean().reset_index(drop=True).sort_values(by='zlogpvel', ascending=False).iloc[0].grpidx;
    bestgrp = sdf.groupby(['grpidx'], as_index=False).agg(safe_agg(sdf,'mean')).reset_index(drop=True).sort_values(by='zlogpvel', ascending=False).iloc[0].grpidx;
    
    #REV: drop saccades outside of that group (only keep saccades/peaks which were classified in that group during clustering).
    sdf = sdf[ sdf.grpidx==bestgrp ];
    after=len(sdf.index);
    print("Dropped {}->{} saccades".format(before,after));

    print(sdf);
    
    #REV: now use start/end values to actually mark it out in time series (samples)
    saccresult = np.full(len(vel), False);
    
    for idx, row in sdf.iterrows():
        saccresult[ row.stidx:row.enidx ] = True; #REV: the +1 because it only selects for each...
        pass;
    
    print("Marked {}/{} as saccs".format(np.nansum(saccresult==True), len(saccresult)));
    
    return saccresult;


#REV: I should adapt for REMoDNaV (remodnav)
#REV: they change way that pt is calculated to better match stats
# of vel distributions.

#REV: they median filter the raw data points though... i.e. x, y


#REV: just simple spike filter (i.e. down-up-down becomes down up up or
# down down up)
def stampe_filter(df, params):
    for n in ('xname', 'yname'):
        df[params[n]] = filter_spikes( df[params[n]] );
        pass;
    return df;


#REV: this preprocesses for saccadr. Just spike removal (stampe) filter and NAN window.
#REV: also, removes values too long in a row of exact same value (more than N seconds)

#REV: could filter with:
#  1) median filter (remove jumpy noise in x/y signal)
#  2) savgol filter (large saccs).

def preproc_saccadr(df, params):
    if( 'tstartsec' in params and 'tlensec' in params ):
        df = resample_at_rate_nearest( df, params['tstartsec'], params['tstartsec']+params['tlensec'], params['tname'], params['samplerate'], params['timeunit'] );
        #REV: start time at tstart
        pass;
    else:
        df = resample_at_rate_nearest( df, df[params['tname']].min(), df[params['tname']].max(), params['tname'], params['samplerate'], params['timeunit'] );
        pass;
    df = remove_suspicious_repeats(df, params);
    #df = dilate_xy_nans(df, params);
    df = stampe_filter(df, params); #REV: this prevenst OM from working LOL. (uses down-up-down for detecting peaks wtf?)
    return df;

#REV: this was basis for redmonav
def method_nh(df, params, eyepfix):
    max_vel_degsec = params['nh_max_vel_degsec']; #1k
    max_acc_degsecsec = params['nh_max_acc_degsecsec']; #100k
    init_vel_thresh_degsec = params['nh_init_vel_thresh_degsec']; #100
    
    sr=params['samplerate'];
    dt = 1/sr;

    vel = df[eyepfix+'vel'];
    acc = df[eyepfix+'acc'];
    

    badpeak = (vel>max_vel_degsec) | (acc>max_acc_degsecsec);
    isbad = badpeak | np.isnan(vel);
    badpeaks = np.where( True==badpeak )[0];
    
    
    medv = np.nanmedian( vel );
    
    for peak in badpeaks:
        idx=peak+1; #index in sample array
        while( (idx < len(vel)) and
               np.isfinite(vel[idx]) and
               (False == isbad[idx]) and
               (vel[idx] > medv)
              ):
            isbad[idx] = True;
            idx += 1;
            pass;
        pass;
    
    
    for peak in badpeaks[::-1]:
        idx = peak-1;
        while( idx >= 0 and
               np.isfinite(vel[idx]) and
               (False == isbad[idx]) and
               (vel[idx] > medv)
               ):
            isbad[idx] = True;
            idx -= 1;
            pass;
        pass;
    
    isgood = ~isbad;
        
    ibelow = isgood & (vel < init_vel_thresh_degsec);
    if( np.count_nonzero(ibelow) < 1 ):
        print("ERROR, one super-long saccade? Not below-thresh");
        return ibelow;


    #REV: iterate
    newPT = init_vel_thresh_degsec;
    pt = newPT*2;
    while( abs(newPT - pt) > 1 ):
        pt = newPT;
        ibelow = isgood & (vel<pt);
        mu = np.nanmean(vel[ibelow]);
        sig = np.nanstd(vel[ibelow]);
        noiseconst = 6;
        newPT = mu + noiseconst * sig; #REV: noise
        pass;
    
    onset_thresh = mu + 3*sig;
    
    issacc = isgood & (vel > pt);
    peaks = np.where(True == issacc)[0];
    
    # REV: foward
    for peak in peaks:
        idx = peak+1;
        while( (idx < len(vel)) and
               (False == issacc[idx]) and
               np.isfinite( vel[idx] ) and
               (vel[idx] > onset_thresh)
               ):
            issacc[idx] = True;
            idx += 1;
            pass;
        pass;


    #REV: backwards
    for peak in peaks[::-1]:
        idx = peak-1;
        while( idx >= 0 and
               (False == issacc[idx]) and
               np.isfinite( vel[idx] ) and
               (vel[idx] > onset_thresh)
              ):
            issacc[idx] = True;
            idx -= 1;
            pass;
        pass;
    
    return issacc;







#REV: *requires* equally spaced samples at a sample rate.
#REV: *requires* that parameters and x, y locations be couched in terms of (visual/ocular) deg/sec.
#REV: requires x to be named xcdva and y to be ycdva. If both eyes,
#     must have l*cdva and r*cdva (c implies "centered", with (0,0) straight ahead)
#     although this is not necessary for saccade extraction to work.
#     Only works with 1 or 2 eyes (not more, although theoretically possible...).

#REV: theoretically should work epr "trial". However, some clustering methods (e.g. method_om) work best
#REV: when there are multiple trials (within a subject/stimulus class), i.e. it can extract more accurate
#REV: clusters based on peak velocity, peak acceleration and peak deceleration.
#REV: however, forward/backwards tracking is still within trial.

#REV: original samples are labelled by "trial" variable (column), which they groupby.
#REV: they extract info for each trial (peaks, etc.), then send back to single flat table.

#REV: seems dplyr (tidyverse?) gives access to .data inside. .data is the argument of tidyverse functions...
#REV: i.e. inside the "pipe"

#REV: i think select (-c("NAME")) selects all columns BUT name. (-c("NAME", "BOO")) would select all except those two...


#REV: yea...need to add "trialidx" (pk?) to dframe...
#REV: for groupby(df['Date']).transform('sum') wtf?
#REV: better: apply
#https://stackoverflow.com/questions/30244952/how-do-i-create-a-new-column-from-the-output-of-pandas-groupby-sum

'''
(pd
     .DataFrame({
        'Date': ['2021-03-11','2021-03-12','2021-03-13','2021-03-11','2021-03-12','2021-03-13',
                 '2021-03-11','2021-03-12','2021-03-13','2021-03-11','2021-03-12','2021-03-13'], 
        'Product': ['shirt','shirt','shirt','shoes','shoes','shoes',
                    'shirt','shirt','shirt','shoes','shoes','shoes'], 
        'Color': ['yellow','yellow','yellow','yellow','yellow','yellow',
                  'blue','blue','blue','blue','blue','blue'], # new!
        'ItemsSold': [300, 400, 234, 80, 10, 120,
                      123, 84, 923, 0, 220, 94],
        })
    .groupby(['Product', 'Color']) # We group by 2 fields now
    .apply(lambda gdf: (gdf
        .sort_values('Date')
        .assign(CumulativeItemsSold=lambda df: df['ItemsSold'].cumsum())))
    .droplevel([0,1]) # We drop 2 levels now

 '''

#REV: note with pipe passes whole df, apply only each subgroup.
#https://stackoverflow.com/questions/47226407/pandas-groupby-pipe-vs-apply


def default_saccadr_params():
    d = dict(nh_max_vel_degsec=1e3,
             nh_max_acc_degsecsec=1e5,
             nh_init_vel_thresh_degsec=100, #REV: 300 in remodnav??!
             nh_savgol_order=2, #not used unless diff_nh
             nh_savgol_window_sec=0.019, # not used unless diff_nh
             om_min_inter_peak_sec=0.030,
             om_max_peaks_per_sec=5,
             om_vel_thresh_degsec=5, #REV: was 3 wtf?
             om_vel_peak_detect_shift_sec=0.0075,
             #om_pca_var_thresh_degsec=0.05, #not used (currently)
             ek_vel_thresh_lambda=6,
             ek_min_dur_sec=0.012,
             #ek_min_sep_sec=0.012, #REV: not currently used
             ek_vel_window_sec=0.024, #REV: 24 msec in original paper
             dilate_nan_win_sec=0.015,
             min_blink_sec=0.010, #REV: only dilate nans longer
             saccadr_min_sep_sec=0.040, #0.012,
             saccadr_min_dur_sec=0.012,
             blink_vel_thresh_degsec=2500,
             );
    return d;

def filter_nans_beforeafter( votes, x ):
    #REV: must be bool array.
    if( votes.dtype != bool ):
        print("WTF votes filter nans beforeafter not boolean...");
        exit(1);
        pass;
    
    vals, sts, lens = rle(votes);
    for v, s, l in zip(vals, sts, lens):
        e = s+l;
        if( (True==v) and
            (
                ((s>0)          and (not np.isfinite(x[s-1]))) #1 before is NAN
                or
                ((e<(len(x)-1)) and (not np.isfinite(x[e]  ))) #1 after is NAN
            )
           ):
            votes[s:e] = False;  #Set all to FALSE (remove the possible saccade...)
            pass;
        pass;
    
    return votes;


def saccadr_detect_saccs( df,
                          methods=(method_ek, method_om, method_nh),
                          velocity_function=diff_ek,
                          binocular="merge",
                          tcol='Tsec',
                         ):
    

    return df, evdf;




#REV: sampdf is modified "in place"?
def saccadr_sacc( sampdf,
                  params,
                  methods=(method_ek, method_om, method_nh),
                  velocity_function=diff_ek,
                  binocular="merge",
                  tsecname='Tsec',
                 ):
    
    #sampdf = sampdf.copy();
    
    #REV: sort by time point (note assumes it must be resampled at regular rate, check that and that diff roughly matches
    #     parameter 1/samplerate);
    sr=params['samplerate'];
    blink_vel_thresh_degsec = params['blink_vel_thresh_degsec'];
    
    vote_thresh=0.99*(len(methods)-1)/len(methods);
    
    isreg = is_regular_samples( sampdf, sr, tname=tsecname );
    if( False == isreg ):
        raise Exception("Error, saccadr_sacc expects regularly resampled (dense) times");
        
    
    sampdf = sampdf.sort_values(by=tsecname).reset_index(drop=True);
    
    isbinoc=False;
    prefixes=('',);
    if( 'xcdva' not in sampdf.columns and 'lxcdva' in sampdf.columns and 'rxcdva' in columns ):
        print("BINOCULAR!")
        isbinoc=True;
        prefixes=('l', 'r');
        pass;
    
    #REV: basically take mean of both eyes.
    if( isbinoc and binocular == "cyclopean" ):
        print("CYCLOPEAN!")
        sampdf['xcdva'] = np.nanmean( [sampdf.lxcdva, sampdf.rxcdva], axis=0 );
        sampdf['ycdva'] = np.nanmean( [sampdf.lycdva, sampdf.rycdva], axis=0 );
        prefixes=('',);
        pass;
    
    #REV: now I will make a vote for each location.
    #REV: note velocity is always DEG/SEC
    #REV: acc is DEG/SEC/SEC (and is absolute value, i.e. euclid dist
    #     of xvel and yval!!!!)
    
    for pfix in prefixes:
        print("Doing for prefix [{}] (empty=no eyes specified)".format(pfix));
        votecols=[];
        
        xvel, yvel, vel =  velocity_function( sampdf[pfix+'xcdva'],
                                              sampdf[pfix+'ycdva'],
                                              params
                                             );

        #REV: optionally, remove biologically unrealistic values and re-dilate?
        #REV: removing those over some maximum velocity (likely blinks or biologically unrealistic data...)
        vel[ vel > blink_vel_thresh_degsec ] = np.nan;

        sampdf[pfix+'xvel'] = xvel;
        sampdf[pfix+'yvel'] = yvel;
        sampdf[pfix+'vel'] = vel;
        
        cols = ['xcdva', 'ycdva', 'vel', 'xvel', 'yvel'];
        cols = [pfix+s for s in cols];
        
        sampdf = dilate_nans(sampdf, cols, params);
                
        #REV: could run median filter if I want...
        sampdf[pfix+'medvel'] = vel;
        
        
        xacc, yacc, acc = velocity_function( sampdf[pfix+'xvel'],
                                             sampdf[pfix+'yvel'],
                                             params
                                            );
        sampdf[pfix+'xacc'] = xacc;
        sampdf[pfix+'yacc'] = yacc;
        sampdf[pfix+'acc'] = acc;
        
        
        for i, m in enumerate(methods):
            
            #################################
            ####### WORK IS HERE!! ##########
            #################################
            
            #REV: note this just returns VOTES!!! No info about
            #REV: velocity etc... I have to go re-extract it.
            methodvotecol = pfix+'vote_'+str(i)+"_{}".format(m.__name__);
            votecols.append(methodvotecol);
            sampvotes = methods[i]( sampdf, params, eyepfix=pfix );
            sampvotes = filter_nans_beforeafter( sampvotes, sampdf[pfix+'vel'] );
            sampdf[methodvotecol] = sampvotes;
            
            pass;
        
        
        
        #REV: average by-row (between merged);
        normvotecol=pfix+'normvotes';
        sampdf[normvotecol] = np.nanmean( sampdf[ votecols ], axis=1 );
        
        print( sampdf[ sampdf[normvotecol] > 0 ][ votecols + [normvotecol] ] );
        
        #print( sampdf[pfix+'normvotes'] );
        pass;
    
    #REV: FUCK it returns both X and Y acceleration and "ampl"????
    
    #REV: sampdf now has "lnormvotes", "rnormvotes" etc, etc. but also "normvotes" which is avg of them.
    if( isbinoc and binocular=="merge" ):
        merge_from_prefix( sampdf, prefixes, 'normvotes' );
        merge_from_prefix( sampdf, prefixes, 'xcdva' );
        merge_from_prefix( sampdf, prefixes, 'ycdva' );
        merge_from_prefix( sampdf, prefixes, 'xvel' );
        merge_from_prefix( sampdf, prefixes, 'yvel' );
        merge_from_prefix( sampdf, prefixes, 'vel' );
        merge_from_prefix( sampdf, prefixes, 'xacc' );
        merge_from_prefix( sampdf, prefixes, 'yacc' );
        merge_from_prefix( sampdf, prefixes, 'acc' );
        pass;
    
    
    min_separation_sec = params['saccadr_min_sep_sec'];
    min_duration_sec = params['saccadr_min_dur_sec'];
    
    delta_t_sec=1/sr;
    min_sep_samples=math.ceil(min_separation_sec * sr);
    min_dur_samples=math.ceil(min_duration_sec * sr);
    
    evdfs=[];
    for pfix in prefixes:
        vals, starts, lens = rle( sampdf[pfix+'normvotes'] > vote_thresh );
        #REV: only doe for where vals==True for saccs
        
        #REV: first, need to check...
        # if true, or (false but surrounded by true on both sides and with length < min_separation between saccades in samples)
        #REV: then, keep it as a "saccade"?
        vals = (vals == True) | (  (vals == False)
                                   & (shift_elements(vals, 1, False)==True)
                                   & (shift_elements(vals, -1, False)==True)
                                   & (lens < min_sep_samples)
                                 );
        
        remarked_samps = inverse_rle(vals, starts, lens);
        vals, starts, lens = rle(remarked_samps);
        vals = (vals==True) & (lens >= min_dur_samples);

        #REV: include both saccs and ISI...
        #vals = np.where( True==vals )[0];
        #starts = starts[vals];
        #lens = lens[vals];
                
        if( len(vals) == 1 and vals[0] == False ):
            print("SACCADR: No saccades detected by votes!");
            pass;
        
        dictlist=[];
        #REV: assume sroted by TSEC
        for v, s, l in zip(vals, starts, lens):
            if( l == 1 ):
                #REV: huh...should never be of length L lol
                continue;
            e=s+l-1; #REV: oh shit do I have an off-by-one error here?
            #print("Event: s={}  e={}  (l={})   {}".format(s, e, l, v));
            #REV: aligning names with removdnav
            ev = dict(stsec=sampdf[tsecname][s],
                      ensec=sampdf[tsecname][e],
                      eye=pfix,
                      stx=sampdf[pfix+'xcdva'][s],
                      sty=sampdf[pfix+'ycdva'][s],
                      enx=sampdf[pfix+'xcdva'][e],
                      eny=sampdf[pfix+'ycdva'][e],
                      pvel=np.nanmax(sampdf[ (sampdf[tsecname]>=sampdf[tsecname][s]) & (sampdf[tsecname]<sampdf[tsecname][e]) ][pfix+'vel'] ),
                      medvel=np.nanmedian(sampdf[ (sampdf[tsecname]>=sampdf[tsecname][s]) & (sampdf[tsecname]<sampdf[tsecname][e]) ][pfix+'vel'] ),
                      avgvel=np.nanmean(sampdf[ (sampdf[tsecname]>=sampdf[tsecname][s]) & (sampdf[tsecname]<sampdf[tsecname][e]) ][pfix+'vel'] ),
                      label='SACC' if (True==v) else 'ISI',
                      );

            dictlist.append(ev);
            pass;

        if( len(dictlist) > 0 ):
            evdfs.append( pd.DataFrame(dictlist) );
            pass;
        
        pass;

    if( len(evdfs)>0):
        evdf = pd.concat( evdfs );
        
        #REV: compute other parameters...
        evdf['dydva'] = evdf.eny - evdf.sty;
        evdf['dxdva'] = evdf.enx - evdf.stx;
        evdf['dursec'] = evdf.ensec - evdf.stsec;
        evdf['angle'] = np.arctan2( evdf.dydva, evdf.dxdva );
        evdf['ampl'] = np.sqrt( evdf.dydva**2 + evdf.dxdva**2 );

        pass;
    else:
        evdf = pd.DataFrame();
        pass;
    
        
    
    #REV: DONE, but first I need to check parameters etc. because it is probably all fucked up.
    #REV: also, remove blinks
    #REV: also, combine this stuff into left/right eye.
    
    #REV: compute "final" with sacc info, e.g. accel, start/end, times.
    
    #REV: note I want to also return "votes" for start/end of saccades for each separate method. Just do methods one at a time I guess...
    
    return sampdf, evdf; #REV: has velocities of L/R eye etc.
