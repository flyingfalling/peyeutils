
import pandas as pd;
import numpy as np;

def compute_ISIs_from_events( ev,
                              zerotime,
                              eventstouse=['SACC', 'BLNK'],
                              label='ISI',
                              stname='stsec',
                              enname='ensec',
                              durname='dursec',
                              eyecol='eye',
                              ):

    if eyecol not in ev:
        print("Adding eyecol {} to ev".format(eyecol));
        ev[eyecol]='';
        pass;

    isilist = list();
    for eye, eyedf in ev.groupby(eyecol, as_index=False):
        saccblnks = eyedf[ eyedf.label.isin(eventstouse) ];
        saccblnks = saccblnks.sort_values(by=stname).reset_index(drop=True);
        
        isis = saccblnks.copy();
        
        isis[stname] = saccblnks.shift(1)[enname].copy();    #start of ISI is the "end" of the PREVIOUS one (will be null for first)
        
        if( len(isis.index) > 0 ):
            isis.loc[ isis.index[0], stname ] = zerotime;
            pass;
        
        isis[enname] = saccblnks[stname];
        isis[durname] = isis[enname] - isis[stname];
        isis['label'] = label;

        isilist.append(isis);
        pass;

    isis = pd.concat(isilist).reset_index(drop=True);
    
    return isis;



def add_ISIs_to_events( ev,
                        zerotime,
                        eventstouse=['SACC', 'BLNK'],
                        label='ISI',
                        stname='stsec',
                        enname='ensec',
                        durname='dursec',
                        eyecol='eye',
                       ):
    
    isis = compute_ISIs_from_events( ev,
                                     zerotime,
                                     eventstouse=eventstouse,
                                     label=label,
                                     stname=stname,
                                     enname=enname,
                                     durname=dursec,
                                     eyecol=eyecol,
                                    );

    ev = pd.concat( [ev, isis] ).reset_index(drop=True);
    
    return ev;











def eye_event_merge( df,
                     min_isi_dur=0.040,
                     eyecol='eye',
                     ):
    if( eyecol not in df.columns ):
        print("Adding missing eyecol {} to df with empty string geneye event merge".format(eyecol));
        df[eyecol]='';
        pass;

    evlist=list();
    for eye, eyedf in df.groupby(eyecol, as_index=False):
        #ev2 = _eye_event_merge_final(eyedf,
        #                            min_blink_dur=min_blink_dur,
        #                            max_blink_amp=max_blink_amp,
        #                            min_isi_dur=min_isi_dur,
        #                            );
        ev2 = absorb_blink_artifacts(eyedf, margin_sec=min_isi_dur); #REV: this could have been done by just expanding (dilating) nn
        ev2[eyecol] = eye;
        evlist.append(ev2);
        pass;
    ev = pd.concat(evlist).reset_index(drop=True);
    return ev;







import pandas as pd
import numpy as np

def absorb_blink_artifacts(df, margin_sec=0.050):
    """
    Absorbs artifactual saccades surrounding a blink by creating a 'Blink Envelope'.
    Preserves all metadata using the First-Row Copy pattern.
    """
    if df.empty: return df

    # 1. Isolate the relevant events
    # We ignore ISIs/Fixations so they don't accidentally get swallowed
    mask = df['label'].str.upper().isin(['BLNK', 'SACC', 'SACBLNK'])
    work_df = df[mask].copy()
    others = df[~mask].copy()
    
    if work_df.empty: return df
    work_df = work_df.sort_values('stsec').reset_index(drop=True)

    # 2. Build the Envelope (Expand Blinks, keep Saccades normal)
    work_df['match_start'] = np.where(
        work_df['label'].str.upper() == 'BLNK', 
        work_df['stsec'] - margin_sec, 
        work_df['stsec']
    )
    
    work_df['match_end'] = np.where(
        work_df['label'].str.upper() == 'BLNK', 
        work_df['ensec'] + margin_sec, 
        work_df['ensec']
    )

    # 3. Grouping by Envelope Intersection
    run_max = work_df['match_end'].cummax().shift(1)
    work_df['group_id'] = ((run_max.isna()) | (work_df['match_start'] > run_max)).cumsum()

    merged_list = []
    
    # 4. The Metadata-Safe Aggregator
    for _, group in work_df.groupby('group_id'):
        res = group.iloc[0].copy()
        
        if len(group) > 1:
            # We ONLY update if a blink actually swallowed something
            # Use actual times, not the artificial match envelope times
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            res['dursec'] = res['ensec'] - res['stsec']
            
            if 'stidx' in group.columns: res['stidx'] = group['stidx'].min()
            if 'enidx' in group.columns: res['enidx'] = group['enidx'].max()
            if 'idx' in group.columns: res['idx'] = group['stidx'].min()
            
            # Spatial Coordinates
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            
            # Geometry
            res['dxdva'] = enx - stx
            res['dydva'] = eny - sty
            res['ampldva'] = np.sqrt(res['dxdva']**2 + res['dydva']**2)
            res['angle'] = np.degrees(np.arctan2(res['dydva'], res['dxdva']))
            
            # Kinematics
            res['pvel'] = group['pvel'].max()
            res['avgvel'] = group['avgvel'].mean()
            if 'medvel' in group.columns: res['medvel'] = group['medvel'].mean()

            # Override: If this group swallowed anything, the whole thing is a Blink.
            res['label'] = 'BLNK'

        # Clean up temp columns
        for col in ['group_id', 'match_start', 'match_end']:
            if col in res.index: 
                res = res.drop(col)
            
        merged_list.append(res)

    # 5. Recombine
    merged_df = pd.DataFrame(merged_list)
    final_df = pd.concat([merged_df, others], ignore_index=True, join='outer')
    
    return final_df.sort_values('stsec').reset_index(drop=True)



'''
def _eye_event_merge(df,
                     min_blink_dur=0.060,
                     max_blink_amp=2.0,
                     min_isi_dur=0.020
                     ):
    # Ensure chronological order and valid indices
    df = df.sort_values('stsec').reset_index(drop=True)
    
    
    #if( 'eye' in df.columns and len(df['eye'].unique()) != 1 ):
    #    raise Exception("More than one eye level in df for merge_eye_events... {}".format(df['eye'].unique()));
    
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
        # 1. If it has a Saccade AND a Blink -> SACCBLNK
        # 2. If it has any 'real' event, keep that label (SACC, PURSUIT, DRIFT)
        # 3. Default to the most frequent non-ISI label, or ISI if all are ISI
        
        if 'BLNK' in unique_labels and 'SACC' in unique_labels:
            final_label = 'SACCBLNK'
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
        })

    return df.groupby('group_id').apply(aggregate_group).reset_index(drop=True)
'''





def _eye_event_merge(df,
                     min_blink_dur,
                     max_blink_amp,
                     min_isi_dur,):
    """
    Generalized eye event merger (Blinks, Saccades, ISIs).
    Fixes FutureWarnings for Pandas 3.0 compatibility.
    """
    # Ensure chronological order
    df = df.sort_values('stsec').reset_index(drop=True)

    # 1. Noise Imputation
    noise_mask = (df['label'] == 'BLNK') & \
                 (df['dursec'] < min_blink_dur) & \
                 (df['ampldva'] < max_blink_amp)
    df.loc[noise_mask, 'label'] = 'ISI'

    # 2. Bridge Identification (using shift with fill_value to avoid downcasting warning)
    is_bridge = (df['label'] == 'ISI') & (df['dursec'] < min_isi_dur)
    
    # FIX: shift(..., fill_value=False) prevents boolean promotion to object/float
    merge_mask = (
        is_bridge | 
        is_bridge.shift(1, fill_value=False) | 
        is_bridge.shift(-1, fill_value=False)
    )

    # 3. Grouping Logic
    # FIX: Again, shift(fill_value=False) used here to avoid warnings
    df['group_id'] = (~(merge_mask & merge_mask.shift(1, fill_value=False))).cumsum()

    # 4. Aggregation
    def aggregate_group(group):
        labels = set(group['label'].unique())
        
        # Priority Logic: Saccade + Blink = SACCBLNK
        if 'BLNK' in labels and 'SACC' in labels:
            final_label = 'SACCBLNK'
        elif 'BLNK' in labels:
            final_label = 'BLNK'
        elif 'SACC' in labels:
            final_label = 'SACC'
        elif len(labels - {'ISI'}) > 0:
            final_label = list(labels - {'ISI'})[0]
        else:
            final_label = 'ISI'

        # Spatial calculations
        stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
        enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
        
        # Euclidean distance in DVA
        dist = np.sqrt((enx - stx)**2 + (eny - sty)**2)

        return pd.Series({
            'stsec': group['stsec'].min(),
            'ensec': group['ensec'].max(),
            'stx': stx, 'sty': sty,
            'enx': enx, 'eny': eny,
            'label': final_label,
            'ampldva': dist,
            'pvel': group['pvel'].max(),
            'dursec': group['ensec'].max() - group['stsec'].min()
        })

    # FIX: include_groups=False silences the grouping column warning
    return (
        df.groupby('group_id')
        .apply(aggregate_group, include_groups=False)
        .sort_values(by='stsec')
        .reset_index(drop=True)
    )





def _eye_event_merge_slow_old(df,
                           min_blink_dur,
                           max_blink_amp,
                           min_isi_dur,):
    """
    Robustly merges events (Blinks, Saccades, ISIs) using the manual-loop pattern
    to prevent silent data loss.
    """
    if df.empty:
        return df

    # 1. Noise Imputation (Small Blinks -> ISI)
    # We use a copy to ensure we aren't working on a slice
    df = df.copy()
    noise_mask = (df['label'].str.upper() == 'BLNK') & \
                 (df['dursec'] < min_blink_dur) & \
                 (df['ampldva'] < max_blink_amp)
    df.loc[noise_mask, 'label'] = 'ISI'

    # 2. Identify Bridge Groups (using short ISIs)
    is_bridge = (df['label'].str.upper() == 'ISI') & (df['dursec'] < min_isi_dur)
    
    # Logic: If a row is a bridge, or next to a bridge, it belongs in a group
    merge_mask = (
        is_bridge | 
        is_bridge.shift(1, fill_value=False) | 
        is_bridge.shift(-1, fill_value=False)
    )

    # Every continuous sequence of 'True' merges; every 'False' stays isolated
    df['group_id'] = (~(merge_mask & merge_mask.shift(1, fill_value=False))).cumsum()

    merged_list = []
    # 3. Manual Loop - This is the "Safety Rail"
    for _, group in df.groupby('group_id'):
        # ALWAYS start with a full copy of the first row's metadata
        res = group.iloc[0].copy()
        
        if len(group) > 1:
            # Metadata update
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            res['dursec'] = res['ensec'] - res['stsec']
            
            # Spatial update: First valid Start -> Last valid End
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            
            res['ampldva'] = np.sqrt((enx - stx)**2 + (eny - sty)**2)
            res['pvel'] = group['pvel'].max()
            
            # Label Priority
            labels = set(group['label'].str.upper())
            if 'BLNK' in labels and 'SACC' in labels:
                res['label'] = 'SACBLNK'
            elif 'BLNK' in labels:
                res['label'] = 'BLNK'
            elif 'SACC' in labels:
                res['label'] = 'SACC'
            elif len(labels - {'ISI'}) > 0:
                res['label'] = list(labels - {'ISI'})[0]
            else:
                res['label'] = 'ISI'

        # Clean up the temp ID and add to our safe list
        if 'group_id' in res.index:
            res = res.drop('group_id')
        merged_list.append(res)

    # 4. Final Reconstruction
    final_df = pd.DataFrame(merged_list)
    return final_df.sort_values('stsec').reset_index(drop=True)





def _eye_event_merge_slow2(df, min_blink_dur=0.060, max_blink_amp=2.0, min_isi_dur=0.020):
    if df.empty: return df

    # 1. Noise Imputation (Small Blinks -> ISI)
    df = df.copy()
    noise_mask = (df['label'].str.upper() == 'BLNK') & \
                 (df['dursec'] < min_blink_dur) & \
                 (df['ampldva'] < max_blink_amp)
    df.loc[noise_mask, 'label'] = 'ISI'

    # 2. Identify Bridge Groups
    is_bridge = (df['label'].str.upper() == 'ISI') & (df['dursec'] < min_isi_dur)
    merge_mask = (is_bridge | is_bridge.shift(1, fill_value=False) | 
                  is_bridge.shift(-1, fill_value=False))
    
    # Generate Group IDs
    df['group_id'] = (~(merge_mask & merge_mask.shift(1, fill_value=False))).cumsum()

    merged_list = []
    for _, group in df.groupby('group_id'):
        # START WITH A COPY (Preserves: angle, eye, ismain, avgvel, etc.)
        res = group.iloc[0].copy()
        
        if len(group) > 1:
            # --- Update Timestamps & Indices ---
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            res['dursec'] = res['ensec'] - res['stsec']
            
            # Preserve pointers to raw data array
            if 'stidx' in group.columns: res['stidx'] = group['stidx'].min()
            if 'enidx' in group.columns: res['enidx'] = group['enidx'].max()
            
            # --- Update Spatial Coordinates ---
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            
            # --- Recalculate Vector Components ---
            # These must be updated to match the new displacement
            res['dxdva'] = enx - stx
            res['dydva'] = eny - sty
            res['ampldva'] = np.sqrt(res['dxdva']**2 + res['dydva']**2)
            res['angle'] = np.degrees(np.arctan2(res['dydva'], res['dxdva']))
            
            # --- Update Velocity ---
            res['pvel'] = group['pvel'].max()
            # Note: avgvel is harder to recalculate perfectly without raw data,
            # but max() or mean() of the components is a better proxy than just the first row.
            res['avgvel'] = group['avgvel'].mean() 
            
            # --- Resolve Label ---
            labels = set(group['label'].str.upper())
            if 'BLNK' in labels and 'SACC' in labels:
                res['label'] = 'SACCBLNK'
            elif 'BLNK' in labels:
                res['label'] = 'BLNK'
            elif 'SACC' in labels:
                res['label'] = 'SACC'

        # Clean up and append
        if 'group_id' in res.index: res = res.drop('group_id')
        merged_list.append(res)

    return pd.DataFrame(merged_list).sort_values('stsec').reset_index(drop=True)





def _eye_event_merge_final(df,
                           min_blink_dur,
                           max_blink_amp,
                           min_isi_dur,
                           ):
    
    if df.empty: return df

    # 1. Noise Imputation (Small Blinks -> ISI)
    df = df.copy()
    noise_mask = (df['label'].str.upper() == 'BLNK') & \
                 (df['dursec'] < min_blink_dur) & \
                 (df['ampldva'] < max_blink_amp)
    df.loc[noise_mask, 'label'] = 'ISI'

    # 2. Identify Bridge Groups (ISIs smaller than threshold)
    is_bridge = (df['label'].str.upper() == 'ISI') & (df['dursec'] < min_isi_dur)
    merge_mask = (is_bridge | is_bridge.shift(1, fill_value=False) | 
                  is_bridge.shift(-1, fill_value=False))
    
    # Generate Group IDs
    df['group_id'] = (~(merge_mask & merge_mask.shift(1, fill_value=False))).cumsum()

    merged_list = []
    for _, group in df.groupby('group_id'):
        # 3. START WITH A COPY (Preserves eye, ismain, etc.)
        res = group.iloc[0].copy()
        
        if len(group) > 1:
            # --- Update Timestamps & Indices ---
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            res['dursec'] = res['ensec'] - res['stsec']
            
            if 'stidx' in group.columns: res['stidx'] = group['stidx'].min()
            if 'enidx' in group.columns: res['enidx'] = group['enidx'].max()
            if 'idx' in group.columns: res['idx'] = group['stidx'].min() 

            # --- Update Spatial Coordinates ---
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            
            # --- Recalculate Vector Components ---
            res['dxdva'] = enx - stx
            res['dydva'] = eny - sty
            res['ampldva'] = np.sqrt(res['dxdva']**2 + res['dydva']**2)
            res['angle'] = np.degrees(np.arctan2(res['dydva'], res['dxdva']))
            
            # --- Update Velocity ---
            res['pvel'] = group['pvel'].max()
            res['avgvel'] = group['avgvel'].mean() 
            if 'medvel' in group.columns:
                res['medvel'] = group['medvel'].mean()
            
            # --- Resolve Label ---
            labels = set(group['label'].str.upper())
            if 'BLNK' in labels and 'SACC' in labels:
                res['label'] = 'SACCBLNK'
            elif 'BLNK' in labels:
                res['label'] = 'BLNK'
            elif 'SACC' in labels:
                res['label'] = 'SACC'
            elif len(labels - {'ISI'}) > 0:
                # Keep other events like PURSUIT if they exist
                res['label'] = list(labels - {'ISI'})[0]
            else:
                res['label'] = 'ISI'

        # Clean up and append
        if 'group_id' in res.index: res = res.drop('group_id')
        merged_list.append(res)
        pass;

    return pd.DataFrame(merged_list).sort_values('stsec').reset_index(drop=True)
