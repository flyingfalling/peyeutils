from peyeutils.utils import filter_spikes;


import numpy as np;
import pandas as pd;
import math;


from statsmodels.robust.scale import mad;

from scipy import signal
#from scipy import ndimage

from scipy.signal import savgol_filter
from scipy.ndimage import median_filter



## This finds peaks in data (note, shit, it just removes NAN data wtf?)
def find_peaks(vels, threshold):
    def _get_vels(start, end):
        v = vels[start:end]
        v = v[~np.isnan(v)]
        return v;
    
    sacs = list();
    sac_on = None;
    for i, v in enumerate(vels):
        ## If not in sacc and v goes above threshold, set the index of the saccade onset.
        if sac_on is None and v > threshold:
            # start of a saccade
            sac_on = i;
            pass;

        ## Else, if I'm in a saccade and I'm below some threshold, add me as a saccade, appending tuple of
        ## [ previdx, nowidx, velarray ] where vellarray is just the list of velocities with NANs elided...
        ## And turn off sacc
        elif sac_on is not None and v < threshold:
            sacs.append([ sac_on,
                          i,
                          _get_vels( sac_on,
                                     min(len(vels), i + 1))
                         ]);
            sac_on = None;
            pass;
        pass;
    
    if sac_on:
        # end of data, but velocities still high
        sacs.append([ sac_on,
                      len(vels) - 1,
                      _get_vels(sac_on, len(vels))])
        pass;
    
    return sacs;


def get_adaptive_saccade_velocity_velthresh( vels, params ):
    cur_thresh = params['startvel'];
    noiseconst = params['noiseconst'];

    #REV: all velocities BELOW the cut (the threshold)
    #REV: iterate until I am not moving down very much each time.
    def _get_thresh(cut):
        # helper function
        vel_uthr = vels[vels < cut]
        med = np.median(vel_uthr)
        scale = mad(vel_uthr); #median absolute deviation
        return ( med+(2*noiseconst*scale), med, scale );
    
    # re-compute threshold until value converges
    count = 0;
    diff = 2;
    MAXITER=30;
    while( diff > 1 and count < MAXITER):  # less than 1deg/s difference
        old_thresh = cur_thresh;
        cur_thresh, med, scale = _get_thresh(old_thresh);
        if( not cur_thresh ):
            # safe-guard in case threshold runs to zero in
            # case of really clean and sparse data
            cur_thresh = old_thresh;
            break;
        diff = abs(old_thresh - cur_thresh)
        count += 1;
        pass;
    
    return cur_thresh, (med + noiseconst * scale);

#REV: shit these are sensitive to timescale (and local noise?)
def find_movement_onsetidx( vels, start_idx, sac_onset_velthresh ):
    idx = start_idx;

    ## While I'm still bigger than the onset threshold OR I'm bigger than the one before me,
    ##    step to the one before me.
    while idx > 0  and (vels[idx] > sac_onset_velthresh or
                        vels[idx] > vels[idx - 1]):
        
        #REV: he means local MAXIMUM?!?!?!
        # find first local minimum after vel drops below onset threshold
        # going backwards in time
        
        # we used to do this, but it could mean detecting very long
        # saccades that consist of (mostly) missing data
        #REV: ah, but what about "blinks"? If saccade includes a blink...hard to compute properly properties...
        #         or np.isnan(vels[sacc_start])):
        idx -= 1
        pass;
    return idx;


def find_movement_offsetidx( vels, start_idx, off_velthresh ):
    idx = start_idx;
    # shift saccade end index to the first element that is below the
    # velocity threshold

    ## While we didn't run out of things and
    ## I'm still greater than offset thresh OR still bigger than the next one, step to the  next one.
    while ((idx < len(vels) - 1) and
           (vels[idx] > off_velthresh or
            (vels[idx] > vels[idx + 1]))
           ):
            # we used to do this, but it could mean detecting very long
            # saccades that consist of (mostly) missing data
            #    or np.isnan(vels[idx])):
        idx += 1
        pass;
    return idx


def find_psoend( velocities, sac_velthresh, sac_peak_velthresh ):

    #REV: this is finding very high peaks of saccades
    pso_peaks = find_peaks(velocities, sac_peak_velthresh)

    ## "high" peak saccade onsets?
    if pso_peaks:
        pso_label = 'HPSO';
        pass;
    else: ## "low" peak saccade onsets (maybe drifts?)
        pso_peaks = find_peaks(velocities, sac_velthresh);
        if pso_peaks:
            pso_label = 'LPSO';
            pass;
        pass;
    if not pso_peaks:
        # no PSO
        return;
    
    # find minimum after the offset of the last reported peak
    pso_end = find_movement_offsetidx( velocities, pso_peaks[-1][1], sac_velthresh );
    
    if np.isnan(velocities[:pso_end]).sum():
        # we do not tolerate NaNs in PSO itervals
        return;
    if pso_end > len(velocities):
        # velocities did not go down within the given window
        return;
    
    return pso_label, pso_end;

def make_event(mydata, idx, lab, stidx, enidx, params={}):
    #print(mydata);
    #print(idx, stidx, enidx);
    #REV: mydata better be sorted...
    xname=params['xname'];
    yname=params['yname'];
    event = {"idx":idx,
             "label":lab,
             "stidx":stidx,
             "enidx":enidx-1,
             "stx":mydata.loc[stidx][xname],
             "sty":mydata.loc[stidx][yname],
             "enx":mydata.loc[enidx-1][xname],
             "eny":mydata.loc[enidx-1][yname]
             };
    
    if( len(mydata[ ~np.isnan(mydata.vel) ].index) > 0 ):
        event["pvel"] = mydata.vel.max();
        if 'dva_per_px' not in params:
            print("lol wtf");
            exit(1);
            pass;
        event["dxdva"] = params['dva_per_px'] * (event["enx"] - event["stx"]);
        event["dydva"] = params['dva_per_px'] * (event["eny"] - event["sty"]);
        event["medvel"] = np.median( mydata.vel );
        event["avgvel"] = np.nanmean( mydata.vel );
        event["angle"] = math.degrees( math.atan2( event["dxdva"], event["dydva"] ) );
        event["ampl"] = math.sqrt( (event["dxdva"])**2 + (event["dydva"])**2  ); #REV: assumes these are pitch/yaw angles.
        pass;
    else:
        event["pvel"] = np.nan;
        event["dxdva"] = np.nan;
        event["dydva"] = np.nan;
        event["medvel"] = np.nan;
        event["avgvel"] = np.nan;
        
        event["angle"] = np.nan;
        event["ampl"] = np.nan;
        pass;
    
    return event;
    
#REV: again, candidate locs is in terms of indices, not time column...
def detect_saccades( candidate_locs, eyesamps, start=None, end=None, winlen=None, params={}):
    #winlen, min_intersaccade_duration=0.04, min_sac_dur, min_intersac_dur , max_sac_freq ):
    saccade_events = list();
    
    if start is None and end is None:
        start = 0;
        end = len(eyesamps.index);
        pass;
    
    
    if( winlen is None ):
        sac_peak_velthresh, sac_onset_velthresh = get_adaptive_saccade_velocity_velthresh( eyesamps.loc[start:end]['vel'],
                                                                                           params=params);
        if( candidate_locs is None ):
            candidate_locs = [(e[0] + start, e[1] + start, e[2])
                              for e in find_peaks( eyesamps.loc[start:end]['vel'],
                                                   sac_peak_velthresh)];
            pass;
        pass;
    
    status = np.zeros((len(eyesamps.index),), dtype=int);
    
    
    
    # loop over all peaks sorted by the sum of their velocities
    # i.e. longer and faster goes first
    #print(candidate_locs);
    for i, props in enumerate(sorted(
            candidate_locs, key=lambda x: x[2].sum(), reverse=True)
                              ):
        
        sacc_start, sacc_end, peakvels = props;
        
        # extract velocity data in the vicinity of the peak to
        # calibrate threshold
        if( winlen ):
            win_start = max(
                start,
                sacc_start - int(winlen / 2));
            
            win_end = min(
                end,
                sacc_end + winlen - (sacc_start - win_start));

            #print("DET SAC", params);
            sac_peak_velthresh, sac_onset_velthresh = get_adaptive_saccade_velocity_velthresh( eyesamps.iloc[int(win_start):int(win_end)].vel, params=params);
            pass;
        
        # move backwards in time to find the saccade onset
        sacc_start = find_movement_onsetidx( eyesamps['vel'], sacc_start, sac_onset_velthresh);

        # move forward in time to find the saccade offset
        sacc_end = find_movement_offsetidx( eyesamps['vel'], sacc_end, sac_onset_velthresh);
        
        mydata = eyesamps.iloc[sacc_start:sacc_end]; #REV: this needs to be indexed as DF...

        #print(params);
        if(sacc_end - sacc_start < params['min_sac_dur']):
            continue;
        elif(np.nansum(np.isnan(mydata[params['xname']]))):  # pragma: no cover
            # should not happen
            #lgr.debug('Skip saccade candidate, missing data')
            continue;
        elif(status[max(0, (sacc_start - params['min_intersac_dur'])):min(len(eyesamps.index), (sacc_end + params['min_intersac_dur']))].sum()):
            #lgr.debug('Skip saccade candidate, too close to another event')
            continue;
        
        #REV: exclude saccades that start or end with the trial, and saccades whose immediate before/after is NAN (or inf).
        elif( (sacc_start == 0)
              or
              (sacc_end == (len(eyesamps.vel)-1))
              or
              ( (sacc_start > 0) and (not np.isfinite(eyesamps.vel[sacc_start-1])) )
              or
              ( (sacc_end < (len(eyesamps.vel)-1) ) and (not np.isfinite(eyesamps.vel[sacc_end+1])) ) #REV: assume for this end is included...
             ):
            continue;
        
        
        #REV: if all NAN return NAN for everything.
        event = make_event( mydata, i, "SACC", sacc_start, sacc_end, params=params); #xname, yname );
        
        #REV: this returns it, then when this is accessed next it starts here...
        yield event.copy();
        
        saccade_events.append(event);
        
        # mark as a saccade
        status[sacc_start:sacc_end] = 1;
        
        pso = find_psoend( mydata.loc[sacc_end:(sacc_end + params['max_pso_dur'])].vel,
                           sac_onset_velthresh,
                           sac_peak_velthresh);
        
        if(pso):
            pso_label, pso_end = pso;
            psoevent = make_event( mydata, i, pso_label, sacc_end, sacc_end+pso_end, params=params); #xname, yname );
            if psoevent['ampl'] < saccade_events[-1]['ampl']:
                # discard PSO with amplitudes larger than their
                # anchor saccades
                yield psoevent.copy()
                # mark as a saccade part
                status[sacc_end:(sacc_end + pso_end)] = 1;
                pass;
            else:
                #REV: pso is larger than sacc...
                pass;
            pass;
        
        if(params['max_sac_freq'] and float(len(saccade_events)) / len(eyesamps.index) > params['max_sac_freq']):
            #REV: detecting too many saccs per unit time...just stop.
            print("BREAKING B/C DETECTING TOO MANY SAC/TIME ({}>{})".format(float(len(saccade_events)) / len(eyesamps.index), params['max_sac_freq']));
            break;
        pass;
    pass;



def fix_or_pursuit( eyesamps, start, end, params={}): #min_fix_dur, pursuit_velthresh, min_purs_dur , dva_per_px, samplerate ):
    if(end - start < params['min_fix_dur']):
        return;
    
    # we have at least enough data for a really short fixation
    win_data = eyesamps.iloc[start:end].copy();
    
    # heavy smoothing of the time series, whatever this non-saccade
    # interval is, the key info should be in its low-freq components
    def _butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter( order,
                              normal_cutoff,
                              btype='low',
                              analog=False)
        return b, a;

    b, a = _butter_lowpass(params['lp_cutoff_freq'], params['samplerate_hzsec']);
    xname=params['xname'];
    yname=params['yname'];
    if( xname == yname ):
        print("Error, xname==yname");
        exit(1);
    win_data[xname] = signal.filtfilt(b, a, win_data[xname], method='gust')
    win_data[yname] = signal.filtfilt(b, a, win_data[yname], method='gust')
    # no entry for first datapoint!
    
    velspx = np.sqrt( np.diff(win_data[xname])**2 + np.diff(win_data[yname])**2 );
    win_vels = velspx * params['dva_per_px'] * params['samplerate_hzsec'];
    
    pursuit_peaks = find_peaks(win_vels, params['pursuit_velthresh']);
    
    # detect rest is very similar in logic to _detect_saccades()
    
    # status map indicating which event class any timepoint has been
    # assigned to so far, zero is fixation
    pursuit_tps = np.zeros((len(win_vels),), dtype=int)

    # loop over all peaks sorted by the sum of their velocities
    # i.e. longer and faster goes first
    for i, props in enumerate(sorted(
            pursuit_peaks, key=lambda x: x[2].sum(), reverse=True)):
        pursuit_start, pursuit_end, peakvels = props
        
        # move backwards in time to find the pursuit onset
        pursuit_start = find_movement_onsetidx(
            win_vels, pursuit_start, params['pursuit_velthresh']);
        
        # move forward in time to find the pursuit offset
        pursuit_end = find_movement_offsetidx(
            win_vels, pursuit_end, params['pursuit_velthresh'])

        if pursuit_end - pursuit_start < params['min_purs_dur']:
            #pursuit candidate too short.
            continue
        
        # mark as a pursuit event
        pursuit_tps[pursuit_start:pursuit_end] = 1;
        pass;
    
    evs = []
    for i, tp in enumerate(pursuit_tps):
        if not evs:
            # first event info
            evs.append([tp, i, i])
        elif evs[-1][0] == tp:
            # more of the same type of event, extend existing record
            evs[-1][-1] = i;
        else:
            evs.append([tp, i, i]);
    # take out all the evs that are too short
    evs = [ev for ev in evs
           if ev[2] - ev[1] >= {
               1: params['min_purs_dur'],
               0: params['min_fix_dur']
           }[ev[0]]];
    
    merged_evs = []
    for i, ev in enumerate(evs):
        if i == len(evs) - 1:
            merged_evs.append(ev)
            break
        if ev[0] == evs[i + 1][0]:
            # same type as coming event, merge and ignore this one
            evs[i + 1][1] = ev[1]
            continue
        else:
            # make boundary in the middle
            boundary = ev[2] + int( (evs[i + 1][1] - ev[2]) / 2 ); #REV: will this be affected by NAN?
            ev[2] = boundary
            evs[i + 1][1] = boundary
            merged_evs.append(ev)
    if not merged_evs:
        # if we found nothing, this is all a fixation
        merged_evs.append([0, 0, len(win_data)])
    else:
        # compensate for tiny snips at start and end
        merged_evs[0][1] = 0
        merged_evs[-1][2] = len(win_data)
        pass;
    
    # submit
    for ev in merged_evs:
        label = 'PURS' if ev[0] else 'FIXA'
        # +1 to compensate for the shift in the velocity
        # vector index
        estart = start + ev[1];
        eend = start + ev[-1];
        
        # change of events or end
        yield make_event( eyesamps, None, label, estart, eend, params=params);
        pass;
    pass;





def classify_intersaccade_period_helper( eyesamps,
                                         start,
                                         end,
                                         saccade_detection,
                                         params={}                                 #       min_sac_dur,     max_pso_dur,   min_intersac_dur
                                        ):
    # no NaN values in data at this point!
    label_remap = {
        'SACC': 'ISAC',
        'HPSO': 'IHPS',
        'LPSO': 'ILPS',
    }
    
    length = end - start;
    
    # detect saccades, if the there is enough space to maintain minimal
    # distance to other saccades
    if length > (
            2 * params['min_intersac_dur']) \
            + params['min_sac_dur'] + params['max_pso_dur']:
        saccades = detect_saccades( None,
                                    eyesamps,
                                    start,
                                    end,
                                    winlen=None,
                                    params=params);
        saccade_events = []
        if saccades is not None:
            kill_pso = False;
            for s in saccades:
                if kill_pso:
                    kill_pso = False
                    if s['label'].endswith('PSO'):
                        continue
                if s['stidx'] - start < params['min_intersac_dur'] or \
                        end - s['enidx'] < params['min_intersac_dur']:
                    # too close to another saccade
                    kill_pso = True;
                    continue;
                s['label'] = label_remap.get(s['label'], s['label'])
                # need to make a copy of the dict to not have outside
                # modification interfere with further inside processing
                yield s.copy();
                saccade_events.append(s);
                pass;
            pass;
        if saccade_events:
            # and now process the intervals between the saccades
            for e in classify_intersaccade_periods( eyesamps,
                                                    start,
                                                    end,
                                                    sorted(saccade_events,
                                                           key=lambda x: x['stidx']),
                                                    saccade_detection=False,
                                                    params=params):
                yield e;
            return;
        pass;

    # what is this time between two saccades?
    for e in fix_or_pursuit(eyesamps, start, end, params=params):
        yield e;
        pass;
    pass;




            
def classify_intersaccade_period( eyesamps,
                                  start,
                                  end,
                                  saccade_detection,
                                  params={}):
    # split the ISP up into its non-NaN pieces:
    win_start = None;
    for idx in range(start, end + 1):
        if win_start is None and \
           idx < len(eyesamps.index) and not np.isnan(eyesamps.iloc[idx][params['xname']]):
            win_start = idx;
        elif win_start is not None and \
             ((idx == end) or np.isnan(eyesamps.iloc[idx][params['xname']])):
            for e in classify_intersaccade_period_helper( eyesamps,
                                                          win_start,
                                                          idx,
                                                          saccade_detection,
                                                          params=params):
                yield e;
                pass;
                # reset non-NaN window start
            win_start = None;
            pass;
        pass;
    pass;




def classify_intersaccade_periods(  eyesamps,
                                    start,
                                    end,
                                    saccade_events,
                                    saccade_detection,
                                    params={}
                                  ):
    prev_sacc = None;
    prev_pso = None;
    
    for ev in saccade_events:
        if prev_sacc is None:
            if 'SAC' not in ev['label']:
                continue;
        elif prev_pso is None and 'PS' in ev['label']:
            prev_pso = ev;
            continue;
        elif 'SAC' not in ev['label']:
            continue
        
        # at this point we have a previous saccade (and possibly its PSO)
        # on record, and we have just found the next saccade
        # -> inter-saccade window is determined
        if prev_sacc is None:
            win_start = start;
        else:
            if prev_pso is not None:
                win_start = prev_pso['enidx'];
                pass;
            else:
                win_start = prev_sacc['enidx']
                pass;
            pass;
        
        # enforce dtype for indexing
        win_end = ev['stidx'];
        if win_start == win_end:
            prev_sacc = ev;
            prev_pso = None;
            continue;
        
        for e in classify_intersaccade_period( eyesamps,
                                               win_start,
                                               win_end,
                                               saccade_detection=saccade_detection,
                                               params=params):
            yield e;
            pass;
        
        # lastly, the current saccade becomes the previous one
        prev_sacc = ev;
        prev_pso = None;

    if prev_sacc is not None and prev_sacc['enidx'] == end:
        return;
    
    # and for everything beyond the last saccade (if there was any)
    for e in classify_intersaccade_period( eyesamps,
                                           start if prev_sacc is None
                                           else prev_sacc['enidx'] if prev_pso is None
                                           else prev_pso['enidx'],
                                           end,
                                           saccade_detection=saccade_detection,
                                           params=params):
        yield e;
        pass;
    pass;






#REV: R package based on old algo (huh??? This is not remodnav):
# https://github.com/tmalsburg/saccades

#REV: classify events using remodnav
def remodnav_classify_events(eyesamps, params): #sac_window_sec=1.0):
    samplerate=params['samplerate_hzsec'];
    
    # find threshold velocities
    #print("CLASS EVENTS", params);
    sac_peak_med_velthresh, sac_onset_med_velthresh = get_adaptive_saccade_velocity_velthresh( eyesamps['medvel'],
                                                                                               params=params );
    
    #REV: this is simply left/right is below curr.
    saccade_locs = find_peaks( eyesamps['medvel'],
                               sac_peak_med_velthresh );
    
    #REV: now I have rough locations, which is list of:
    #[ startidx, endidx, [list of non-nan velocities (note no longer represents true times since it dropped NANs?)]
    
    events = []
    saccade_events = []
    sac_window_samples = samplerate * params['sac_window_sec'];
    
    for e in detect_saccades( candidate_locs=saccade_locs,
                              eyesamps=eyesamps,
                              start=0,
                              end=len(eyesamps.index),
                              winlen=sac_window_samples,
                              params=params ):
        
        saccade_events.append(e.copy());
        events.append(e);
        pass;
    
    #REV: inter saccade periods (fixations or slow phases etc.?)
    events.extend( classify_intersaccade_periods( eyesamps=eyesamps,
                                                  start=0,
                                                  end=len(eyesamps.index),
                                                  # needs to be in order of appearance
                                                  saccade_events=sorted(saccade_events, key=lambda x: x['stidx']),
                                                  saccade_detection=True,
                                                  params=params )
                  );
    
    df = pd.DataFrame( events );
    
    #print(df);
    if( len(df.index) > 0 ):
        df['stsec'] = df['stidx'] / samplerate;
        df['ensec'] = df['enidx'] / samplerate;
        
        df.sort_values( inplace=True, by='stidx' );
        df.reset_index( inplace=True, drop=True );
        pass;
    
    return df;







def make_default_params(): #samplerate_hzsec, dva_per_px): # samplerate, dva_per_px ):
    #REV: these are represented in seconds. Need to mult by sample rate...

    #REV: wtf, PK 377 at 4.100, PK 333 at 3.200 sec
    #REV: microsaccs not detected properly...

    #REV: integrate path integral, and time passed?

    #REV: how can I remove "weird shit"? It does not even detect fucking pursuits...

    #REV: what is "biologically unrealistic". Doesnt' work, especially if we have patient population, which often has different kinematics...
    #REV: so, can't really rely on, specific eye movmeent types, main sequence, etc. necessarily?

    #REV: option 1) remove ANY suspicious data -- we will be left with nothing.
    #REV: option 2) ignore it and hope for best -- but looks like there is a lot of suspricious data...
    
    #params={ 'min_intersac_dur_sec':0.04, 'min_sac_dur_sec':0.01, 'min_fix_dur_sec':0.04, 'min_purs_dur_sec':0.04, 'max_pso_dur_sec':0.04, 'max_sac_freq_sec':1.0, 'lp_cutoff_freq':4.0, 'startvel':300.0, 'noiseconst':3.0, 'sac_window_sec':1.0, 'pursuit_velthresh':2.0 };
    params=dict( min_intersac_dur_sec=0.020,
                 min_sac_dur_sec=0.012,
                 min_fix_dur_sec=0.040,
                 min_purs_dur_sec=0.020,
                 max_pso_dur_sec=0.040, #REV: don't both with PSO...
                 max_sac_freq_sec=2.0, #2
                 lp_cutoff_freq=8.0, #8.0,
                 startvel=100.0, #REV: whoops? #was 300
                 noiseconst=8.0,
                 sac_window_sec=1.0, #1
                 pursuit_velthresh=2.0,
                 filterspikes=True );
    
    newdict={};
    for p in params:
        if 'dur_sec' in p:
            newdict[ p[:-4] ] = int(params[p] * samplerate_hzsec);
            pass;
        elif 'freq_sec' in p:
            newdict[ p[:-4] ] = int(params[p] / samplerate_hzsec);
            pass;
        else:
            pass;
        pass;
    
    params = {**params, **newdict};
    
    #params['samplerate_hzsec'] = samplerate;
    #params['dva_per_px'] = dva_per_px;
    
    return params;




def make_default_preproc_params(samplerate_hzsec, dva_per_px, xname, yname, tname): # samplerate, dva_per_px ):
    #REV: these are represented in seconds. Need to mult by sample rate...

    #REV: actually, it is the SLOWER velocities that are due to after-blink...as pupil size changes ;(
    #REV: onset of blink is probably saccade like? Or drft? Usually up or down, but dpeends on eye position when starts...
    
    #params={ 'min_intersac_dur_sec':0.04, 'min_sac_dur_sec':0.01, 'min_fix_dur_sec':0.04, 'min_purs_dur_sec':0.04, 'max_pso_dur_sec':0.04, 'max_sac_freq_sec':1.0, 'lp_cutoff_freq':4.0, 'startvel':300.0, 'noiseconst':3.0, 'sac_window_sec':1.0, 'pursuit_velthresh':2.0 };
    params=dict(dilate_nan_win_sec=0.010, #0.010 #REV: based on method_OM
                minblinksec=0.020,
                medianfiltlensec=0.050,
                savgollensec=0.019,
                savgolorder=2,
                maxveldegsec=2000,
                #maxaccdegsecsec=# let's say it goes from 300 to 500 in one sample. That is 200 deg/sec in 2.5 msec, i.e. 80k deg/sec/sec
                samplerate_hzsec=samplerate_hzsec,
                dva_per_px=dva_per_px,
                blink_vel_thresh_degsec=5000,
                #240 means it goes 600 deg/sec in one sample at 400hz.
                blink_acc_thresh_degsecsec=800000, #300000, #REV: average over window? No, acc will go up, down, up?! Fuck...
                #REV: shorter time implies higher acceleration.
                #blink_acc_lpf_window_sec=0.010,
                xname=xname,
                yname=yname,
                tname=tname,
                );
    
    return params;








## I think this is for remodnav?

#REV: samples must be from single eye etc., i.e. can't be interlaced.
#REV: note eye samps must be "dense", i.e. every sample must be 1/samplerate_hz_sec seconds apart.
#REV: these values *SHOULD* represent linear pitch (x) and yaw (y). If not, you must convert to them.
#REV: time must be in SECONDS...
### TIME MUST BE MONOTONIC INCREASING, WILL SORT TO MAKE SURE (will resample to regular sampling rate with NANs)

#REV: note, this does NOT DETECT BLINKS (assumes they are already detected as NANs).
def remodnav_preprocess_eyetrace2d(eyesamps : pd.DataFrame,
                                   params : dict):
    
    samplerate=params['samplerate_hzsec'];
    dva_per_px=params['dva_per_px'];
    xname=params['xname'];
    yname=params['yname'];
    tname=params['tname'];
    dilate_nan_window = int(params['dilate_nan_win_sec'] * samplerate);
    min_blink_window = int(params['minblinksec'] * samplerate);
    savgol_window = int(params['savgollensec'] * samplerate);
    median_window = int(params['medianfiltlensec'] * samplerate );
    savgolorder = params['savgolorder'];
    
    
    if( 'vel' in eyesamps.columns or
        'medvel' in eyesamps.columns or
        'acc' in eyesamps.columns ):
        raise Exception("Error, vel or medvel or acc is already in the eyesamps DF?");
    
    #REV: just to keep medvels and acc even if ALL NAN.
    eyesamps['vel'] = np.nan;
    eyesamps['medvel'] = np.nan;
    eyesamps['acc'] = np.nan;
    
    if(allnan(eyesamps[xname]) or allnan(eyesamps[yname])):
        print("EYEUTILS -> PREPROC FIRST LINE -> ALL NAN");
        return eyesamps;
    
    #print('beginning', len(eyesamps));
    
    #REV: resample first?

    '''
    if( 'tstartsec' in params and 'tlensec' in params ):
        eyesamps = resample_at_rate_nearest( eyesamps, params['tstartsec'], params['tstartsec']+params['tlensec'], tname, params['samplerate_hzsec'], params['timeunit'] );
        #REV: start time at tstart
        pass;
    else:
        eyesamps = resample_at_rate_nearest( eyesamps, eyesamps[tname].min(), eyesamps[tname].max(), tname, params['samplerate_hzsec'], params['timeunit'] );
        pass;
    '''
    
    #print('after resamp', len(eyesamps));
    
    
    #REV: PYTHON logical precedence not > and > or...
    if( savgol_window > 0 ): #params['savgollensec'] != 0 ) ):
        if( (savgol_window % 2 != 1) or (savgol_window < params['savgolorder'] ) ):
            raise Exception("Error preproc, savgol filter needs to be odd length (in samples), and window must be >= order (unless window is 0) {} {} {}".format(savgol_window, samplerate, params['savgollensec']));
        pass;
    
        
    eyesamps.sort_values( by=tname, inplace=True );
    
    #REV: handle improper size of data based on sample rate?
    #print(len(eyesamps.index));
    #print("EYESAMP TIME DIFF", eyesamps.time.max() - eyesamps.time.min());
    dur = (eyesamps[tname].max() - eyesamps[tname].min()) * (1/params['timeunit']); # e.g. 1/0.001 is 1000.
    #print("Time sacc diff: {}".format(dur));
    GRACE_SEC=0.0005*dur; #REV: drift...
    expecteddur=((len(eyesamps.index)-1) / samplerate); #500 samples / 500 samp/sec = 1 sec. #2 samples makes 2 msec...so 1 sample is...0.
    
    if( abs(expecteddur - dur) > GRACE_SEC ):
        raise Exception("ERROR, missing samples (not dense)  expected from #samples {}   observed {}".format(expecteddur, dur));
        
    
    if(allnan(eyesamps[xname]) or allnan(eyesamps[yname])):
        print("EYEUTILS, PREPROC BEGINNING BEFORE SPIKEFILTER -> ALL NAN");
        return eyesamps;
        

    if( params.filterspikes ):
        eyesamps[xname] = filter_spikes(eyesamps[xname]);
        eyesamps[yname] = filter_spikes(eyesamps[yname]);
        pass;
    
    #print('after spikes', len(eyesamps));
    
    if(allnan(eyesamps[xname]) or allnan(eyesamps[yname])):
        print("EYEUTILS, PREPROC AFTER FILTER SPIKES -> ALL NAN");
        return eyesamps;
        
    
    
    #REV: up-down-up-down oscillation...is what? 
    
    
    if( savgol_window > 0 ):
        eyesamps[xname] = savgol_filter(eyesamps[xname], savgol_window, savgolorder);
        eyesamps[yname] = savgol_filter(eyesamps[yname], savgol_window, savgolorder);
        pass;
    
    if(allnan(eyesamps[xname]) or allnan(eyesamps[yname])):
        print("EYEUTILS -> PREPROC AFTER SAVGOL -> ALL NAN");
        return eyesamps;


    #print('after savgol', len(eyesamps));
    
    ############# REMOVE INSANE VALUES #####################
    
    
    blink_vel_thresh_degsec = params['blink_vel_thresh_degsec'];
    blink_acc_thresh_degsecsec = params['blink_acc_thresh_degsecsec'];
    
    velspx = np.sqrt( np.diff(eyesamps[xname])**2 + np.diff(eyesamps[yname])**2 ); #REV: this is PER SAMPLE. Shit...is this radians?

    #REV: if time is in msec, then what? I don't use that info...
    velsdva = velspx * dva_per_px * samplerate / params['timeunit'];   #REV: per sec. Note missing first value?
    velsdva = np.append([0], velsdva);
    
    eyesamps['vel'] = velsdva;
    
    accs = np.zeros(len(velsdva));
    accs[1:] = (velsdva[1:] - velsdva[:-1]) * samplerate; #REV: just a ghetto diff thing (diff between self shifted by one);
    
    #velsdva[ velsdva > blink_vel_thresh_degsec ] = np.nan;
    #accs[ accs > blink_acc_thresh_degsecsec ] = np.nan;
    
    #print('after vel', len(eyesamps));
    
    eyesamps['acc'] = accs;
    
    cols = [xname, yname, 'vel', 'acc']; #REV: remove actual samples too I guess lol...
    es2 = eyesamps[ (eyesamps.acc >= blink_acc_thresh_degsecsec) | (eyesamps.vel >= blink_vel_thresh_degsec) ]; #[cols] = np.nan;
    print("Detected {} timepoints outside of velocity or acceleration allowance.".format(len(es2.index)));
    
    eyesamps.loc[ ((eyesamps.acc >= blink_acc_thresh_degsecsec) | (eyesamps.vel >= blink_vel_thresh_degsec)), cols ] = np.nan;
    
    pren = eyesamps[ eyesamps[cols].isna().any(axis=1) ]; #  len(eyesamps[ eyesamps[cols].isna() ].index);
    
    eyesamps.loc[ eyesamps[cols].isna().any(axis=1), cols ] = np.nan;
    
    eyesamps = dilate_nans(eyesamps, cols, params);

    #print('after nans', len(eyesamps));
    
    if(allnan(eyesamps[xname]) or allnan(eyesamps[yname])):
        print("EYEUTILS -> PREPROC AFTER DILATE NANS -> ALL NAN");
        return eyesamps;
    
    postn = eyesamps[ eyesamps[cols].isna().any(axis=1) ]; #  len(eyesamps[ eyesamps[cols].isna() ].index);
    print("Dilated from {}->{} NAN".format(len(pren.index), len(postn.index)));
    
    
    ############# END REMOVE INSANE VALUES ################
    
    if( xname == yname ):
        raise Exception("WARNING/ERROR, xname==yname");
        
    
    if( dva_per_px != 1 ):
        raise Exception("DVAPPX NOT 1");
    
    #print('before median', len(eyesamps));
    med_vels = eyesamps.vel;
    
    if( median_window > 0 ):
        med_vels = np.zeros((len(eyesamps.index),), eyesamps.vel.dtype); #REV: zeros of original size
        
        med_vels[1:] =  np.sqrt( np.diff(median_filter(eyesamps[xname],
                                                       size=median_window)) ** 2 +
                                 np.diff(median_filter(eyesamps[yname],
                                                       size=median_window)) ** 2
                                );
        med_vels *= dva_per_px * samplerate;
        #med_vels[ get_dilated_nan_mask(med_vels, dilate_nan_window, 0) ] = np.nan;
        pass;

    #print('after median', len(eyesamps));
    
    eyesamps['medvel'] = med_vels;
    
    #REV: only dilate medvels? :( Second time...
    cols = ['medvel'];
    eyesamps = dilate_nans(eyesamps, cols, params);

    #print('after med nans', len(eyesamps));
    
    #print("Filtering out >1k deg/s");
    finalvels = [];
    for vel in eyesamps.vel:
        if( vel  >  params['maxveldegsec'] ):
            print("REV: Detected too-high velocity {}? Maybe bad filter params?".format(vel));
            vel = finalvels[-1]; #REV: just replace it with the previous velocity? Fine with high sample rates...
            pass;
        finalvels.append(vel);
        pass;

    #print('after finalvels', len(eyesamps));
    
    eyesamps.vel = np.array(finalvels);
    
    if( median_window <= 0 ):
        eyesamps.medvel = eyesamps.vel;
        pass;
    
    #REV: difference of velocity in deg/sec.
    #REV: change in velocity between two samples should then represent
    #REV: acceleration... 25 deg/sec - 26 deg/sec represents
    #REV: change of 1 deg/sec in one sample time, i.e. 4 msec,
    #REV: so to get deg/sec/sec, multiply by samplerate. OK.
    
    if(allnan(eyesamps[xname]) or allnan(eyesamps[yname])):
        print("EYEUTILS -> PREPROC END OF FUNCT -> ALL NAN");
        return eyesamps;
    
    return eyesamps;
