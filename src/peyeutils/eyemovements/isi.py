
import pandas as pd;
import numpy as np;

def compute_ISIs_from_events( ev,
                              zerotime,
                              eventstouse=['SACC', 'BLNK'],
                              label='ISI',
                              stname='stsec',
                              enname='ensec',
                              durname='dursec',
                              ):
    saccblnks = ev[ ev.label.isin(eventstouse) ];
    saccblnks = saccblnks.sort_values(by=stname).reset_index(drop=True);
    
    isis = saccblnks.copy();
    
    isis[stname] = saccblnks.shift(1)[enname].copy();    #start of ISI is the "end" of the PREVIOUS one (will be null for first)
    print(isis);
    if( len(isis.index) > 0 ):
        isis.loc[ isis.index[0], stname ] = zerotime;
        pass;

    isis[enname] = saccblnks[stname];
    isis[durname] = isis[enname] - isis[stname];
    isis['label'] = label;
    
    return isis;



def add_ISIs_to_events( ev,
                        zerotime,
                        eventstouse=['SACC', 'BLNK'],
                        label='ISI',
                        stname='stsec',
                        enname='ensec',
                        durname='dursec',
                       ):
    
    isis = compute_ISIs_from_events( ev,
                                     zerotime,
                                     eventstouse=eventstouse,
                                     label=label,
                                     stname=stname,
                                     enname=enname,
                                     durname=dursec,
                                    );

    ev = pd.concat( [ev, isis] ).reset_index(drop=True);
    
    return ev;



import pandas as pd
import numpy as np


def generalized_eye_event_merge(df, min_blink_dur=0.060, max_blink_amp=2.0, min_isi_dur=0.020):
    # Ensure chronological order and valid indices
    df = df.sort_values('stsec').reset_index(drop=True)

    if( 'eye' in df.columns and len(df['eye'].unique()) != 1 ):
        raise Exception("More than one eye level in df for merge_eye_events... {}".format(df['eye'].unique()));
    
    # --- 1. Noise Imputation ---
    # Convert "glitch" blinks into ISIs so they can be merged into surrounding events
    noise_mask = (df['label'] == 'BLNK') & \
                 (df['dursec'] < min_blink_dur) & \
                 (df['ampldva'] < max_blink_amp)
    df.loc[noise_mask, 'label'] = 'ISI'

    # --- 2. Bridge Identification ---
    # We only want ISIs to act as the "glue". 
    # Actual events (SACC, PURSUIT, etc.) should not be bridges themselves.
    is_bridge = (df['label'] == 'ISI') & (df['dursec'] < min_isi_dur)
    
    # A row is part of a merge group if it is a bridge OR adjacent to one
    merge_mask = is_bridge | is_bridge.shift(1).fillna(False) | is_bridge.shift(-1).fillna(False)

    # --- 3. Grouping Logic ---
    # Increment group_id only when we are NOT in a continuous merge sequence
    df['group_id'] = (~(merge_mask & merge_mask.shift(1).fillna(False))).cumsum()

    # --- 4. The Priority-Based Aggregator ---
    def aggregate_group(group):
        unique_labels = set(group['label'].unique())
        
        # Priority Logic for the new label:
        # 1. If it has a Saccade AND a Blink -> SACBLNK
        # 2. If it has any 'real' event, keep that label (SACC, PURSUIT, DRIFT)
        # 3. Default to the most frequent non-ISI label, or ISI if all are ISI
        
        if 'BLNK' in unique_labels and 'SACC' in unique_labels:
            final_label = 'SACBLNK'
        elif 'BLNK' in unique_labels:
            final_label = 'BLNK'
        elif 'SACC' in unique_labels:
            final_label = 'SACC'
        elif len(unique_labels - {'ISI'}) > 0:
            # Pick the most "significant" label that isn't ISI
            # (e.g., if it merged a PURSUIT and an ISI, it's a PURSUIT)
            remaining = list(unique_labels - {'ISI'})
            final_label = remaining[0] 
        else:
            final_label = 'ISI'

        # Spatial calculations (Global Start to Global End)
        stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
        enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
        
        # Distance formula: $ \Delta = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} $
        dist = np.sqrt((enx - stx)**2 + (eny - sty)**2)

        return pd.Series({
            'stsec': group['stsec'].min(),
            'ensec': group['ensec'].max(),
            'stx': stx, 'sty': sty,
            'enx': enx, 'eny': eny,
            'label': final_label,
            'ampldva': dist,
            'pvel': group['pvel'].max(),
            'dursec': group['ensec'].max() - group['stsec'].min(),
            'eye': group['eye'].iloc[0]
        })

    return df.groupby('group_id').apply(aggregate_group).reset_index(drop=True)



def merge_eye_events(df, min_blink_dur=0.060, max_blink_amp=2.0, min_isi_dur=0.020):
    """
    Cleans and merges eye events for a single eye's dataframe.
    """

    if( 'eye' in df.columns and len(df['eye'].unique()) != 1 ):
        raise Exception("More than one eye level in df for merge_eye_events... {}".format(df['eye'].unique()));
    # Ensure chronological order within the eye
    df = df.sort_values('stsec').reset_index(drop=True)

    # --- 1. Impute noise blinks as ISI ---
    noise_mask = (df['label'] == 'BLNK') & \
                 (df['dursec'] < min_blink_dur) & \
                 (df['ampldva'] < max_blink_amp)
    df.loc[noise_mask, 'label'] = 'ISI'

    # --- 2. Identify merge candidates ---
    # An event should merge if it IS a short ISI or if the PREVIOUS event was a short ISI
    is_short_isi = (df['label'] == 'ISI') & (df['dursec'] < min_isi_dur)
    
    # This mask marks every row that is part of a "merge cluster"
    merge_mask = is_short_isi | is_short_isi.shift(1).fillna(False) | is_short_isi.shift(-1).fillna(False)

    # --- 3. Create Group IDs ---
    # We want a new ID every time we hit a row that is NOT part of a merge sequence
    # or the start of a new merge sequence.
    df['new_group'] = ~(merge_mask & merge_mask.shift(1).fillna(False))
    df['group_id'] = df['new_group'].cumsum()

    # --- 4. Define Aggregation ---
    def aggregate_group(group):
        labels = group['label'].unique()
        
        # Determine the new label
        if 'BLNK' in labels and 'SACC' in labels:
            final_label = 'SACBLNK'
        elif 'BLNK' in labels:
            final_label = 'BLNK'
        elif 'SACC' in labels:
            final_label = 'SACC'
        else:
            final_label = 'ISI'

        # Calculate spatial displacement
        stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
        enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
        
        # Assuming ampldva is Euclidean distance in DVA
        # If your data is in pixels, you'd apply your px->dva conversion here
        dist = np.sqrt((enx - stx)**2 + (eny - sty)**2)

        return pd.Series({
            'stsec': group['stsec'].min(),
            'ensec': group['ensec'].max(),
            'stx': stx,
            'sty': sty,
            'enx': enx,
            'eny': eny,
            'label': final_label,
            'ampldva': dist,
            'pvel': group['pvel'].max(), # Peak velocity in the cluster
            'dursec': group['ensec'].max() - group['stsec'].min(),
            'eye': group['eye'].iloc[0]
        })

    return df.groupby('group_id').apply(aggregate_group).reset_index(drop=True)

'''
# --- EXECUTION ON MULTI-EYE DF ---
# This isolates the logic per eye level
processed_df = df.groupby('eye', group_keys=False).apply(
    lambda x: merge_eye_events(x, min_blink_dur=0.060, min_isi_dur=0.020)
)
'''


#REV: combines samples/event, to compute: #samples in ISI available (as pct total possible?)
## Also, mean X/Y, stdev X/Y, mean (abs) velocity, etc..
def compute_isi_params(ev, samps, ):
    return ev;
