## Front-end (exported) functions for convenience
#REV: runs preprocessing and saves .csv files in specified dir
## Does not do anything with events other than mark blinks...
#REV: could pass other info here...
#import peyeutils.eyelink.eyelink;
#import peyeutils.preproc.preproc;

#import peyeutils.peyefv.msgutils;

#from peyeutils.utils.fsutils import *;
#import peyeutils.utils as ut;
import peyeutils as pu;
import os;
import pandas as pd;
import numpy as np;


#REV: remove 'weird' oscillations from data (e.g. 50 Hz etc. from fRMRI?). Do fourier, and filter out? However, what about 'direction'...?

#REV: remove "unrealistic" periods of zero eye movements (i.e. nostril detected as pupil?).



def prepare_data( df,
                  tcol,
                  targ_sr_hzsec,
                  truesrs : dict, #REV: true samplerates of each sensor (column cluster).
                  method='polynomial',
                  order=2,
                  tcolunit_s=1,
                  startsec=None,
                  endsec=None,
                  ):
    
    
    return;
                  
#REV: user must:

## Rescale all time to seconds (and pass name of seconds time column)
## Rescale all X/Y to dva (and pass name of columns, xname and yname?). Center irrelevant, but we want to keep correspondence...
## Separate eyes (if they want) and pass "eye" column separately.
## Specify pupil-size? Column... (and normalize to mean etc.).

## Pass other params
## badcol represents unreliable/missing data points (xcol, ycol should be set to NAN?)
## blinkcol represents locations of known blinks?

## REV: user must "upsample" their data first, and ensure "level" (label/id) columns are stored as "string" to avoid
## imputation.

def preproc_and_compute_events(df,
                               tcol,
                               xcol,
                               ycol,
                               sr_hzsec,
                               mainseq_err_gain, #REV: 1.5 for eyeXSL, 3? for tobii g3...
                               badcol='',
                               nablinks=True,
                               blinkcol='blink',
                               eyecol='eye',
                               PLOT=False,
                               DEBUG=False,
                               
                               ):

    import pandas as pd;
    import numpy as np;
    
    min_sacc_dva = 0.33;
    min_isi_sec = 0.060; #REV: true minimum time to code saccade?
    
    blinksacc_merge_envelop_sec=0.040;
    
    ##############################
    xyunits_dva=1;

    sr = pu.utils.tsutils.check_samplerate(df, tcol=tcol );

    if( not np.isclose( sr, sr_hzsec ) ):
        raise Exception("Samplerate not good, expected {}, got {}".format(sr_hzsec, sr));

    ########### PREPROCESSING (smooth/savgol filter/median filter/dilate NANs  ################
    params1 = pu.eyemovements.remodnav.make_default_preproc_params(samplerate_hzsec=sr_hzsec,
                                                                   timeunitsec=1,
                                                                   dva_per_px=xyunits_dva,
                                                                   xname=xcol,
                                                                   yname=ycol,
                                                                   tname=tcol);
    
    #REV: default other params for remodnav (will be ignored if not needed).
    params2 = pu.eyemovements.remodnav.make_default_params(samplerate_hzsec=sr_hzsec);

    #REV: combine this into single large params dict. These do not affect results very much
    params = params1 | params2;
    
    
    params['noiseconst'] = 8;
    params['dilate_nan_win_sec'] = 0.010;
    params['min_sac_dur_sec'] = 0.010;
    params['min_intersac_dur_sec'] = min_isi_sec; #0.020; #REV: problem with this is situations with VOR fast phases and "real" saccades.
    params['minblinksec'] = 0.030;
    params['startvel'] = 100;

    #REV: this recomputes "hidden" params which are per-sample (based on per-sec params passed by user).
    params = pu.eyemovements.remodnav.recompute_params(params);
    
    #REV; sets 'blink' to True for NAN gaze samples, false otherwise.
    
    params['blinkcol'] = blinkcol;
    if( nablinks ):
        df[ params['blinkcol'] ] = df[xcol].isna();
        pass;

    #REV: this does NAN dilation etc.... it removes data, not just setting "bad" column or something? 
    sdf = pu.eyemovements.remodnav.remodnav_preprocess_eyetrace2d(eyesamps=df, params=params);



    ################ SACCADE DETECTION #######################
    
    sparams = pu.eyemovements.saccadr.default_saccadr_params();
    sparams['samplerate'] = sr_hzsec;
    sparams['noiseconst'] = 4; #REV: 4 works.
    sparams['ek_vel_thresh_lambda'] = 6; # 6 works
    sparams['ek_min_dur_sec']=0.012;
    sparams['nh_init_vel_thresh_degsec'] = 100;
    sparams['om_max_peaks_per_sec']=10;
    sparams['om_vel_thresh_degsec']=25; #REV: was 3 or 5 wtf? #REV: won't work for saccade -> pursuit? REV: relative to "surround"?
    sparams['om_vel_peak_detect_shift_sec']=0.0075;
    sparams['om_usepca']=False;
    
    sparams['saccadr_min_sep_sec']=min_isi_sec; #0.050;
    sparams['saccadr_min_dur_sec']=0.010;
    
       
    
    sdf, sev = pu.eyemovements.saccadr.saccadr_detect_saccades(sdf, sparams, tsecname=tcol, xname=xcol, yname=ycol);
    rdf, rev = pu.eyemovements.remodnav.remodnav_classify_events(sdf, params); #REV: ah, x/y names are stored in "params"
    
    ev=rev;
    
    allsaccs = pd.concat( [ sev[sev['label']=='SACC' ],
                            rev[rev['label']=='SACC' ],
                           ]
                          );
    
    
    saccs = allsaccs.reset_index(drop=True);
    
    if(PLOT):
        import matplotlib.pyplot as plt;
        import seaborn as sns;
        
        mainseq, mygraphics = pu.eyemovements.mainseq.mainseq_ampldur_linear_95pctl_human_chen2021_wplot( saccs['ampldva'],
                                                                                                          saccs['dursec'],
                                                                                                          error_gain=mainseq_err_gain,
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
    saccs = saccs[ (saccs.ismain==True) ].reset_index(drop=True); #REV: main seq, very small ones would be bad too?

    saccs = saccs[ saccs['ampldva'] > min_sacc_dva ];


    #REV: remove "impossible" ones before that.
    saccs = pu.eyemovements.combine.intersection_saccades( saccs,
                                                          );
    if(DEBUG):
        saccs['label'] = 'OSACC';
        saccs2['label'] = 'SACC';
        saccs = pd.concat([saccs, saccs2]).reset_index(drop=True);
        pass;
    
    nonsaccs = rev[ ~(rev['label'] == 'SACC') ];
    
    blinks = pu.eyemovements.blink.compute_blinks_from_sampcol( sdf,
                                                                dva_per_px=params['dva_per_px'],
                                                                badcol=params['blinkcol'],
                                                                tcol=tcol,
                                                                xcol=xcol,
                                                                ycol=ycol )
    
    #REV: should I remove blinks in which eye did not move much (< 0.5 deg ?). I.e. fixation with intermediate lbink?
    # Vision is not happening during that time and physiologically it is equivalent...and then ISI is?
    
    ev = pd.concat( [saccs,
                     blinks,
                     nonsaccs, #REV: this is currently just PISI (original ISI from saccade detection...). In other cases
                     # there may also be e.g. drifts/smooth pursuits, etc.?
                     ] ).reset_index(drop=True);

    ev = ev.sort_values(by='stsec').reset_index(drop=True);
    
    ev = pu.eyemovements.isi.eye_event_merge( ev,
                                              eyecol=eyecol,
                                              min_isi_dur=blinksacc_merge_envelop_sec,
                                             );
    
    ISIevents=['SACC', 'BLNK']; #REV: i.e. use blinks as saccades (gaze shifts often happen during blinks...)
    isis = pu.eyemovements.isi.compute_ISIs_from_events( ev,
                                                         zerotime=sdf[tcol].iloc[0], #REV: or .min()
                                                         eventstouse=ISIevents,
                                                        );
    ev = pd.concat([ev, isis]);
    ev = ev.sort_values(by='stsec').reset_index(drop=True);
    #isis = isis[ isis['dursec'] > min_isi_sec ];
    
    isis = ev[ (ev['label']=='ISI') ];
    
    saccblnks = ev[ (ev.label=='SACC') | (ev.label=='BLNK') | (ev.label=='SACCBLNK')];
    
    if(PLOT):
        
        sns.histplot(data=saccblnks,
                     x='ampldva',
                     multiple='stack',
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
        
        
        
        sns.relplot( data=isis, x='stx', y='sty', kind='scatter', size='dursec' );
        plt.title("Gaze during ISI (blinks/saccades removed)");
        plt.tight_layout();
        plt.savefig('isiXYlocs.pdf');
        plt.show();
        
        
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
                pu.plotting.plot_gaze_chunks_wpupil( df=sdf, timestamp_col=tcol, x_col=xcol, y_col=ycol, chunk_size_sec=10,
                                              events_df=ev, event_start_col='stsec', event_end_col='ensec',
                                              event_type_col='label', max_chunks_per_fig=4,
                                              pupil_col='DiameterPupilRightEye',
                                              )
        ):
            fig.savefig('testfig_{:04d}.pdf'.format(i));
            pass;
        
        pass;
        
    return sdf, ev;








def preproc_peyefv_edf( in_edf_path : str,
                        out_csv_path : str = None,
                       ):
    """

    Parameters
    ----------
    in_edf_path : str :  Filesystem path to edf file to read/preprocess (e.g. blah/bloop/file.edf)
        
    out_csv_path : str :  Filesystem path of directory in which to store CSV files created by this function (containing samples, messages, indices of trials/video starts/etc.).
        (Default value = None)

    Returns
    -------
    5-Tuple (row, sampdf, msgdf, trialdf, blockdf)
    row : dict : parameters and filenames of CSV files created. blocktrials_csv, blocks_csv, samples_csv, messages_csv, events_csv, edfsamples_csv, etc.

    sampdf : pandas.DataFrame : dataframe containing (preprocessed) samples

    msgdf : pandas.DataFrame : dataframe containing (preprocessed) messages from EDF
    
    trialdf : pandas.DataFrame : dataframe containing trial start/end times, video names, sizes, etc. extracted from EDF messages.

    blockdf : pandas.DataFrame : dataframe containing start/end times of blocks in the EDF file.
    
    """
    #-> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, bool):
    if( out_csv_path ):
        
        pu.utils.create_dir(out_csv_path);
        pass
    try:
        import pyedfread;
        hasPYEDF=True;
        print("Found PYEDFREAD");
        pass;
    except:
        hasPYEDF=False;
        print("Could not find PYEDFREAD, will not be able to handle EDF files");
        pass;
    
    row=dict();
    haseyetracking=True;
    error=False;
    row['edferror'] = False;
    
    #REV: expect FNAME to be UNIQUE
    fname=os.path.basename(in_edf_path);
    fdir =os.path.dirname(in_edf_path);
    
    row['edfpath'] = fdir;
    row['edffile'] = fname;
    
    print(" ++++++++ Reading [{}] ++++++++++".format(in_edf_path));
    
    try:
        s, e, m = pyedfread.read_edf(in_edf_path);
        print(s.time.min(), s.time.max());
        print(s.time);
        error=False;
        pass;
    except Exception as e:
        row['edferror'] = True;
        error=True;
        print("  -------- WARNING -- Could not read EDF file [{}], exception [{}]".format(in_edf_path, e));
        return row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    
    if( out_csv_path ):
        #mkdir(out_csv_path);
        sfn = fname + '.edfsamples.csv';
        efn = fname + '.edfevents.csv';
        mfn = fname + '.edfmessages.csv';
        
        spath=os.path.join(out_csv_path, sfn);
        epath=os.path.join(out_csv_path, efn);
        mpath=os.path.join(out_csv_path, mfn);
        
        print("Saving EDF files as CSV to [{}]  ([{}] [{}] [{}])".format(out_csv_path, sfn, efn, mfn));
        s.to_csv(spath, index=False);
        e.to_csv(epath, index=False);
        m.to_csv(mpath, index=False);
        
        row['edfsamples_csv'] = sfn;
        row['edfevents_csv'] = efn;
        row['edfmessages_csv'] = mfn;
        pass;
    
    recinfo = pu.peyefv.get_recordingsession_info(m);

    pretag='recinfo_';
    for key in recinfo:
        if( key in row ):
            print("Replacing metadata info in row for file {} (key={} original row [{}]->[{}]  (from recinfo)) -- will name everything {}PARAM".format(in_edf_path, key, row[key], recinfo[key], pretag));
            pass;
        row[pretag+key] = recinfo[key];
        pass;
    
    
    df, ev, msgs, badtrial = pu.eyelink.preproc_EL_A_clean_samples(s,e,m);
    df = pu.eyelink.preproc_EL_rawcalib_px(df, msgs); #REV: this *ASSUMES* that viewbox etc. is true.
    df = pu.peyefv.preproc_peyefreeviewing_dva_from_flatscreen(df, msgs);
    
    if( False == badtrial ):
        df = pu.preproc.preproc_SHARED_C_binoc_gaze(df, xcol='cgx_dva', ycol='cgy_dva', tcol='Tsec', exclude_thresh=2);
        pass;
    
    #df = preproc_SHARED_D_exclude_bad( df, xcol='cgx_dva', ycol='cgy_dva', badcol='bad' );
    
    if( out_csv_path ):
        #REV: preprocessed messages etc.
        sfn = fname + '.samples.csv'
        efn = fname + '.events.csv'
        mfn = fname + '.messages.csv'
        
        spath=os.path.join(out_csv_path, sfn);
        epath=os.path.join(out_csv_path, efn);
        mpath=os.path.join(out_csv_path, mfn);
        
        df.to_csv(spath, index=False);
        ev.to_csv(epath, index=False);
        msgs.to_csv(mpath, index=False);
        
        row['samples_csv'] = sfn;
        row['events_csv'] = efn;
        row['messages_csv'] = mfn;
        pass;
    
        
    
    trialdf = pu.peyefv.import_fv_trials( msgs );
    if( badtrial ):
        #trialdf['haseyetracking'] = False;
        haseyetracking=False;
        print(" BAD TRIAL (no data?)...");
        pass;
    
    
    blockdf, trialdf = pu.peyefv.import_fv_blocks(msgs, df, trialdf);
    
    trialdf['haseyetracking']=haseyetracking;
    blockdf['haseyetracking']=haseyetracking;
    row['haseyetracking'] = haseyetracking;
    
    if( out_csv_path ):
        btfname=fname+'.trials.csv';
        bfname=fname+'.blocks.csv';
        btpath=os.path.join(out_csv_path, btfname);
        bpath=os.path.join(out_csv_path, bfname);
        trialdf.to_csv(btpath, index=False);
        blockdf.to_csv(bpath, index=False);
        
        row['trials_csv'] = btfname;
        row['blocks_csv'] = bfname;
        pass;
    
    
    return row, df, msgs, trialdf, blockdf
