
## REV: easier to make a "class" which takes a row (with specific names) and then it can access those with helper functions.
## E.g. based on what kind of file it is, get samples etc.? Can I add additional names?
##      So easiest thing is just to load the DF and it will create a chunk of classes?

import pandas as pd;
import sys;
import os;
import peyeutils as pu;

def process_events(rowdic):
    row=rowdic['row'];
    csvdir=rowdic['csvdir'];

    samppath = os.path.join(csvdir, row['samples_csv']);
    df = pd.read_csv(samppath);
    df = df[ df.eye=='B' ];

    #REV these "times" will be correct because they are just rle (run-length encoding) of samples.
    blinkev = pu.preproc.blink_df_from_samples(df);
    blinkev['method'] = 'blink';
    
    import peyeutils.eyemovements.remodnav as rv;

    sr = row['recinfo_samplerate'];
        
    params1 = rv.make_default_preproc_params(samplerate_hzsec=sr,
                                             timeunitsec=1,
                                             dva_per_px=1, xname='cgx_dva',
                                             yname='cgy_dva',
                                             tname='Tsec0');
    #REV: TSEC this will be offset from beginning of EDF (not block...fuck).
    
    params2 = rv.make_default_params(samplerate_hzsec=sr);
    params = params1 | params2;
    
    
    print("remodnav: preproc eyetrace");
    rdf = rv.remodnav_preprocess_eyetrace2d(eyesamps=df, params=params);
    rdf['method'] = 'remodnav';
    
    print("remodnav: classify");
    ev = rv.remodnav_classify_events(rdf, params);
    
    ev = pd.concat( [ev, blinkev] );
    
    evfname = row['edffile'] + '.events2.csv';
    row['events2_csv'] = evfname;
    
    evpath = os.path.join( csvdir, evfname );
    ev.to_csv(evpath, index=False);
    
    return row; #Oh, this will not be a 1-row DF...


def plotrow(rowdic):
    row=rowdic['row'];
    csvdir=rowdic['csvdir'];
    
    samppath = os.path.join(csvdir, row['samples_csv']);
    evpath = os.path.join(csvdir, row['events2_csv']);
    trialspath = os.path.join(csvdir, row['trials_csv']);

    samps = pd.read_csv(samppath);
    ev = pd.read_csv(evpath);

    print(ev);
    print(ev[ev['label']=='SACC']);
    trials = pd.read_csv(trialspath);
    print(samps.columns);
    print(ev.columns);
    print(trials.columns);
    for i,fig in enumerate(
            pu.plotting.plot_gaze_chunks( df=samps, timestamp_col='Tsec',
                                          x_col='cgx_dva', y_col='cgy_dva',
                                          chunk_size_sec=5,
                                          events_df=ev,
                                          event_start_col='stsec',
                                          event_end_col='ensec',
                                          event_type_col='label',
                                          stimulus_df=trials,
                                          stim_start_col='start_s',
                                          stim_end_col='end_s',
                                          stim_name_col='video',
                                          max_chunks_per_fig=5 )
            ):
        figbase=os.path.join( csvdir, row['edffile'] );
        fn = figbase + '_timeplot_{:04d}.pdf'.format(i)
        print("Saving [{}]".format(fn));
        fig.savefig(fn);
        pass;
        
    return;

def main():
    rowcsv = sys.argv[1];
    csvdir = sys.argv[2];
    
    rowdf = pd.read_csv(rowcsv);
    
    rowdf = rowdf[:2];
    results = list();
    for i, row in rowdf.iterrows():
        results.append( process_events( dict(row=row, csvdir=csvdir) ) );
        pass;
    
    rowdf = pd.DataFrame(results);
    print(rowdf);

    for i, row in rowdf.iterrows():
        plotrow(dict(row=row, csvdir=csvdir) );
        pass;
    
    return 0;

if __name__=='__main__':
    exit(main());

