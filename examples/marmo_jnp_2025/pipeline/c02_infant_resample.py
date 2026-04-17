import peyeutils as pu;

import pandas as pd;

import numpy as np;

import sys, os;

from multiprocessing import Pool;


# --- 4. The Extraction Function ---
def get_windowed_mean(target_t, source_df, hw, s, tcol='movie_ts'):
    # Find data strictly inside this specific time window
    lower_bound = target_t - hw
    upper_bound = target_t + hw
    
    # Filter the 500 Hz dataframe for this window
    window_data = source_df[(source_df['movie_ts'] >= lower_bound) & (source_df['movie_ts'] < upper_bound)]
    
    # Calculate the mean. 
    # NOTE: .mean() automatically drops NaNs before doing the math!
    # If the window is entirely NaNs or empty, it gracefully returns NaN.
    return window_data[s].mean()


def main():
    #alltrialscsv=sys.argv[1];
    alledfscsv=sys.argv[1];
    csvdir=sys.argv[2];

    #REV: compute age at time of session, and merge the two.
    #REV: convert gaze into "pix_x" and "pix_y" which are centered pixels, but assuming "normal" vid_width of 640 and height of 480
    # Note each video is different...

    #REV: merge binocular based on dist > 2 dva...just take nanmean...
    import peyeutils as pu;
    biglist = list();
    df = pd.read_csv(alledfscsv);
    print(df.columns);
    for i, row in df.iterrows():
        trialcsv = row['trials_csv'];
        mytrials = pd.read_csv( os.path.join(csvdir, trialcsv) );
        
        print(mytrials);
        if( len(mytrials.index) > 0 ):
            mytrials = mytrials.drop(columns=['haseyetracking']);
            for c in mytrials.columns:
                if( c in row.to_dict() ):
                    raise Exception("WTF FAIL {} double".format(c));
                pass;
            mytrials = mytrials.assign( **row.to_dict() );
            
            
            #print(mytrials);
            #exit(0);
            biglist.append(mytrials);
            pass;
        #print(trials);
        pass;

    fulldf = pd.concat(biglist);

    fulldf['dob'] = pd.to_datetime(fulldf['dob']);
    fulldf['recinfo_SESSION_DATE'] = pd.to_datetime(fulldf['recinfo_SESSION_DATE'], format='%Y-%m-%d-%H-%M-%S');
    fulldf['age'] = fulldf['recinfo_SESSION_DATE'] - fulldf['dob'];
    fulldf['vidscale'] = 640 / fulldf['vidw_px']; #REV: e.g. if 2x the size, we should mult this number by every val to get pix_x and pix_y.
    
    
    print(fulldf['age']);
    import seaborn as sns;
    secperday = 60*60*24;
    fulldf['agedays'] = fulldf['age'].dt.total_seconds() / secperday;
    sns.displot(data=fulldf, x='agedays', kind='hist', hue='subj');
    import matplotlib.pyplot as plt;
    plt.show();
    fulldf.to_csv('all_infant_trialsubjdob.csv', index=False);
    
    #REV: "cut" into ages (in days)
    
    agecuts = [ (150, 215), #4-7 mo
                (235, 305), #8-10 mo
                (325, 395), #11 to 13
                (415, 485), #14 to 16
                (505, 575),  # 17 to 20
                (600, 750), # up to 2yrs
                (1400, 3600), # over 4
                ];
    
    tidx=0;

    finallist=list();

    for edf, edfdf in fulldf.groupby('samples_csv', as_index=False):
        fn=os.path.join(csvdir, edf);
        edfsamps = pd.read_csv( fn );
        print( " FOR {}".format(edf));
        
        for i, row in edfdf.iterrows():
            samps = edfsamps[ (edfsamps['Tsec'] >= row.start_s) & (edfsamps['Tsec'] < row.end_s) ].reset_index(drop=True);
            #    for i, row in fulldf.iterrows():
            
            
            samps = samps[ (samps['Tsec'] >= row.start_s) & (samps['Tsec'] < row.end_s) ].reset_index(drop=True);
            samps['movie_ts'] = samps['Tsec'] - samps['Tsec'].min();
            
            
            
            
            #REV: problem it seems that FLIPY is -1 (by default WTF?), so it is inverting Y, but I think I used bottom-left in
            # peyeutils as the 0? Oh wait no, maybe I am using top-left. Because order is top/bot/left/right.
            #print(samps.columns);
            print(row['video']);
            samps['pix_x'] = samps['cgx_px'] * row['vidscale'];
            samps['pix_y'] = samps['cgy_px'] * row['vidscale'];
            MAXX=340;
            samps.loc[(samps.pix_x > MAXX) | (samps.pix_x < -MAXX) |
                      (samps.pix_y > MAXX) | (samps.pix_y < -MAXX), ['pix_x', 'pix_y'] ] = np.nan;
            
            
            samps.loc[ (samps['bad']==True), ['pix_x', 'pix_y', 'cgx_dva', 'cgy_dva'] ] = np.nan;
            
            #sns.relplot(data=samps, x='pix_x', y='pix_y', kind='scatter');
            #sns.jointplot(data=samps, x='pix_x', y='pix_y');
            #sns.jointplot(data=samps, x='cgx_dva', y='cgy_dva');
            #plt.show();
            
            binoc = samps.groupby('movie_ts', as_index=False).mean(numeric_only=True);
            
            window_duration = 1 / 30 
            half_window = window_duration / 2 
            window_rows = int(round(500 / 30)) # 17 rows
            # --- 2. Generate exactly rounded 30 Hz timestamps ---
            # Create an array of 30Hz steps, then round to 3 decimal places
            target_times = np.round(np.arange(0, samps['movie_ts'].max(), 1/30), 3)
            
            # Create a DataFrame for our new 30 Hz data
            df30hz = pd.DataFrame({'movie_ts': target_times})
            sensor_cols = ['pix_x', 'pix_y' ]; #, 'cgx_dva', 'cgy_dva']

            for s in sensor_cols:
                df30hz[s] = [
                    get_windowed_mean(t, samps, half_window, s) for t in df30hz['movie_ts']
                ];
                pass;
            
            
            print(df30hz);
            df30hz['trialidx'] = str(tidx) + 'i';
            df30hz['vid'] = row['video'];
            df30hz['subj'] = row['subj'];
            df30hz['agedays'] = row['agedays'];

            if( (np.isfinite(df30hz['pix_x']).sum() / len(df30hz.index)) > 0.10 ): #10 pct?
                finallist.append(df30hz);
                tidx+=1;
                pass;
            pass;
        pass;
    
    finaldf=pd.concat(finallist);
    finaldf.to_csv('infant_gaze_finaldf.csv', index=False);
    
    
    return 0;

#REV: for a given file, loads samples (and trial), for each trial, gets data for that trial, binocularizes, resamples (removes blinks etc.)
## and normalizes to "full" pixel space of original video?

if __name__=='__main__':
    exit(main());
    pass;
