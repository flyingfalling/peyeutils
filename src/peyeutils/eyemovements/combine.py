import pandas as pd
import numpy as np


def consolidate_saccades(df,
                         eyecol='eye',
                         isi_threshold=0.010,):
    if( len((df['label'].unique() )) != 1 ):
        raise Exception("consolidate_saccades, got more than 1 unique labels (should only have SACC) : {}".format(df['label'].unique()));
    if( df['label'].unique()[0] != 'SACC' ):
        raise Exception("Consolidate saccades, label is not SACC (should only have SACC) : {}".format(df['label'].unique()));
    
    if eyecol not in df.columns:
        df[eyecol] = '';
        print("consolidate saccades, adding eyecol {} to df as empty string".format(eyecol));
        pass;
    eyelist=list();
    for eye, eyedf in df.groupby(eyecol, as_index=False):
        #myev=_consolidate_saccades_slow(df=df, isi_threshold=isi_threshold);
        myev= consolidate_consensus_saccades(df=df, isi_threshold=isi_threshold);
        myev[eyecol] = eye;
        eyelist.append(myev);
        pass;
    
    import pandas as pd;
    ev = pd.concat(eyelist).reset_index(drop=True);
    return ev;
    
def intersection_saccades(df,
                         eyecol='eye',
                         isi_threshold=0.010,):
    if( len((df['label'].unique() )) != 1 ):
        raise Exception("consolidate_saccades, got more than 1 unique labels (should only have SACC) : {}".format(df['label'].unique()));
    if( df['label'].unique()[0] != 'SACC' ):
        raise Exception("Consolidate saccades, label is not SACC (should only have SACC) : {}".format(df['label'].unique()));
    
    if eyecol not in df.columns:
        df[eyecol] = '';
        print("consolidate saccades, adding eyecol {} to df as empty string".format(eyecol));
        pass;
    eyelist=list();
    for eye, eyedf in df.groupby(eyecol, as_index=False):
        #myev=_consolidate_saccades_slow(df=df, isi_threshold=isi_threshold);
        myev= consolidate_saccades_strict(df=df);
        myev[eyecol] = eye;
        eyelist.append(myev);
        pass;
    
    import pandas as pd;
    ev = pd.concat(eyelist).reset_index(drop=True);
    return ev;











def consolidate_consensus_saccades(df, isi_threshold=0.0, group_cols=None):
    """
    Consolidates saccades into a single consensus envelope.
    Explicitly handles gaps and overlapping detections.
    """
    if df.empty:
        return df

    # 1. Sort by Group and Time
    sort_order = (group_cols if group_cols else []) + ['stsec']
    df = df.sort_values(sort_order).reset_index(drop=True)

    def process_group(group):
        if len(group) == 0:
            return pd.DataFrame()
        
        # Sort internal to group just in case
        group = group.sort_values('stsec')
        
        merged = []
        # Initialize current consensus with the first row
        curr = group.iloc[0].to_dict()
        
        for i in range(1, len(group)):
            nxt = group.iloc[i]
            
            # CONSENSUS LOGIC: Does the next detection overlap with 
            # our current consensus (plus the allowed gap)?
            if nxt['stsec'] <= (curr['ensec'] + isi_threshold):
                # We merge: Expand the end time to the maximum seen so far
                if nxt['ensec'] > curr['ensec']:
                    curr['ensec'] = nxt['ensec']
                    curr['enx'] = nxt['enx'] # Update end coords to latest
                    curr['eny'] = nxt['eny']
                
                # Take the highest peak velocity recorded in this window
                if 'pvel' in nxt:
                    curr['pvel'] = max(curr.get('pvel', 0), nxt['pvel'])
            else:
                # We hit a gap: Push the consensus and start a new one
                merged.append(curr)
                curr = nxt.to_dict()
        
        # Don't forget the last one (prevents dropping the final saccade!)
        merged.append(curr)
        return pd.DataFrame(merged)

    # 2. Apply the greedy logic per eye/trial
    if group_cols:
        consensus = df.groupby(group_cols, group_keys=False).apply(process_group).reset_index(drop=True)
    else:
        consensus = process_group(df)

    # 3. RECALCULATION STEP
    # This fixes the "frankensaccade" metrics using the new consensus timestamps/coords
    consensus['dursec'] = consensus['ensec'] - consensus['stsec']
    
    # Calculate Amplitude (Euclidean distance)
    # $$Amplitude = \sqrt{(enx - stx)^2 + (eny - sty)^2}$$
    dx = consensus['enx'] - consensus['stx']
    dy = consensus['eny'] - consensus['sty']
    consensus['ampldva'] = np.sqrt(dx**2 + dy**2)
    
    # Calculate Average Velocity
    # $$AvgVel = \frac{Amplitude}{Duration}$$
    consensus['avgvel'] = np.where(
        consensus['dursec'] > 0, 
        consensus['ampldva'] / consensus['dursec'], 
        0
    )

    return consensus



import pandas as pd
import numpy as np

def _consolidate_saccades_slow(df,
                               isi_threshold ):
    # Use the same mask logic from the debug version
    is_sacc_mask = df['label'].str.upper().str.startswith('SACC').fillna(False)
    sacc_df = df[is_sacc_mask].copy()
    
    if sacc_df.empty:
        return df

    # Sort and reset index exactly as in the debug version
    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)
    
    # Grouping logic
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    sacc_df['group_id'] = ((running_max_end.isna()) | (sacc_df['stsec'] > (running_max_end + isi_threshold))).cumsum()

    merged_list = []
    # Use the same iteration pattern
    for g_id, group in sacc_df.groupby('group_id'):
        # Start with the copy of the first row
        res = group.iloc[0].copy()
        
        if len(group) > 1:
            # Update values only if merging
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            
            res['ampldva'] = np.sqrt((enx - stx)**2 + (eny - sty)**2)
            res['pvel'] = group['pvel'].max()
            res['label'] = 'SACCBLNK' if any(group['label'].str.upper() == 'SACCBLNK') else 'SACC'

        # Force drop exactly as in the debug version
        merged_list.append(res.drop('group_id'))

    # Reconstruct exactly as in the debug version
    merged_saccs = pd.DataFrame(merged_list)
    others = df[~is_sacc_mask].copy()
    
    final_df = pd.concat([merged_saccs, others], ignore_index=True)
    
    return final_df.sort_values('stsec').reset_index(drop=True)













def consolidate_saccades_strict(df):
    """
    Consolidates overlapping saccades using Strict Consensus (Intersection).
    If a large master saccade envelopes disjoint smaller ones, the master 
    is dropped and the smaller disjoint saccades are preserved.
    """
    if df.empty: return df

    # 1. Filter and Sort
    is_sacc_mask = df['label'].str.upper().str.startswith('SACC').fillna(False)
    sacc_df = df[is_sacc_mask].copy()
    others = df[~is_sacc_mask].copy()
    
    if sacc_df.empty: return df
    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)

    # 2. Strict Overlap Grouping (Must physically overlap)
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    sacc_df['group_id'] = ((running_max_end.isna()) | (sacc_df['stsec'] >= running_max_end)).cumsum()

    # --- 3. Recursive Resolution Logic ---
    def resolve_overlap_group(sub_group):
        # Base Case: Isolated saccade
        if len(sub_group) == 1:
            res = sub_group.iloc[0].copy()
            return [res]

        new_stsec = sub_group['stsec'].max() # Latest start
        new_ensec = sub_group['ensec'].min() # Earliest end

        # SUCCESS: All detections mutually overlap
        if new_ensec > new_stsec:
            res = sub_group.iloc[0].copy()
            res['stsec'] = new_stsec
            res['ensec'] = new_ensec
            res['dursec'] = new_ensec - new_stsec
            
            # Bind Spatial Coordinates to the winning timestamps
            row_max_st = sub_group.loc[sub_group['stsec'].idxmax()]
            row_min_en = sub_group.loc[sub_group['ensec'].idxmin()]
            
            res['stx'], res['sty'] = row_max_st['stx'], row_max_st['sty']
            res['enx'], res['eny'] = row_min_en['enx'], row_min_en['eny']
            
            if 'stidx' in sub_group.columns: res['stidx'] = row_max_st['stidx']
            if 'enidx' in sub_group.columns: res['enidx'] = row_min_en['enidx']
            
            # Re-Compute Geometry
            res['dxdva'] = res['enx'] - res['stx']
            res['dydva'] = res['eny'] - res['sty']
            res['ampldva'] = np.sqrt(res['dxdva']**2 + res['dydva']**2)
            res['angle'] = np.degrees(np.arctan2(res['dydva'], res['dxdva']))
            
            # Kinematics
            res['pvel'] = sub_group['pvel'].max()
            res['avgvel'] = sub_group['avgvel'].mean()
            if 'medvel' in sub_group.columns: res['medvel'] = sub_group['medvel'].mean()
            
            return [res]
            
        # CONFLICT: Empty Intersection (e.g., Master enveloping disjoint shorts)
        else:
            # Identify the longest saccade ("master") and drop it
            idx_to_drop = (sub_group['ensec'] - sub_group['stsec']).idxmax()
            remaining = sub_group.drop(index=idx_to_drop).copy()
            remaining = remaining.sort_values('stsec').reset_index(drop=True)
            
            # Re-evaluate the remaining saccades to see if they form new disjoint groups
            run_max = remaining['ensec'].cummax().shift(1)
            remaining['sub_g'] = ((run_max.isna()) | (remaining['stsec'] >= run_max)).cumsum()
            
            # Recursively resolve the newly split groups
            results = []
            for _, sg in remaining.groupby('sub_g'):
                results.extend(resolve_overlap_group(sg))
            return results

    # 4. Execute Resolution
    merged_list = []
    for _, group in sacc_df.groupby('group_id'):
        resolved_rows = resolve_overlap_group(group)
        for row in resolved_rows:
            # Clean up temp identifiers
            if 'group_id' in row.index: row = row.drop('group_id')
            if 'sub_g' in row.index: row = row.drop('sub_g')
            merged_list.append(row)

    # 5. Recombine
    merged_df = pd.DataFrame(merged_list)
    final_df = pd.concat([merged_df, others], ignore_index=True, join='outer')
    return final_df.sort_values('stsec').reset_index(drop=True)

















'''


#EREV: slow for for loop...
def _consolidate_saccades_slow(df, isi_threshold=0.010):
    """
    Stable version based on the working debug logic.
    Uniformly handles merged and isolated saccades as Series objects.
    """
    if df.empty:
        return df

    # 1. Filter and Sort
    # Even if all rows are SACC, this ensures we have a clean copy to work on
    is_sacc_mask = df['label'].str.upper().str.startswith('SACC').fillna(False)
    sacc_df = df[is_sacc_mask].copy()
    others = df[~is_sacc_mask].copy()
    
    if sacc_df.empty:
        return df

    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)

    # 2. Grouping
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    # A new group starts if it's the first row or start > (previous_max_end + threshold)
    sacc_df['group_id'] = ((running_max_end.isna()) | 
                           (sacc_df['stsec'] > (running_max_end + isi_threshold))).cumsum()

    # 3. Uniform Aggregation (The 'Debug' Flow)
    merged_list = []
    for _, group in sacc_df.groupby('group_id'):
        # ALWAYS extract the Series first (this is what made the debug version work)
        res = group.iloc[0].copy()
        
        if len(group) > 1:
            # Only perform merge math if there's actually a cluster
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            res['dursec'] = res['ensec'] - res['stsec']
            
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            
            res['ampldva'] = np.sqrt((enx - stx)**2 + (eny - sty)**2)
            res['pvel'] = group['pvel'].max()

            #REV: this should never happen, we always detect SACCBLNK after sacc...
            # Label priority: SACBLNK beats SACC
            if any(group['label'].str.upper() == 'SACCBLNK'):
                res['label'] = 'SACCBLNK'
            else:
                res['label'] = 'SACC'

        # Uniformly drop the temp ID from the Series
        if 'group_id' in res.index:
            res = res.drop('group_id')
            
        merged_list.append(res)

    # 4. Reconstruct
    merged_df = pd.DataFrame(merged_list)
    
    # join='outer' and ignore_index=True ensure absolute metadata safety
    final_df = pd.concat([merged_df, others], ignore_index=True, join='outer')
    
    return final_df.sort_values('stsec').reset_index(drop=True)






def _consolidate_saccades_slow(df, isi_threshold=0.010):
    # 1. Identify Saccades (Robust to 'SACC', 'SACBLNK', etc.)
    # Use a copy to avoid SettingWithCopy warnings
    is_sacc_mask = df['label'].str.upper().str.startswith('SACC').fillna(False)
    sacc_df = df[is_sacc_mask].copy()
    
    if sacc_df.empty:
        return df

    # Sort strictly by time
    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)

    # 2. Grouping without using 'apply'
    # We find the groups first
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    new_group_mask = (running_max_end.isna()) | (sacc_df['stsec'] > (running_max_end + isi_threshold))
    sacc_df['group_id'] = new_group_mask.cumsum()

    # 3. Manual Aggregation (Accountable and safe)
    merged_list = []
    for _, group in sacc_df.groupby('group_id'):
        if len(group) == 1:
            # Drop the group_id before adding back
            merged_list.append(group.drop(columns=['group_id']).iloc[0])
            continue
            
        # Create the merged record based on the first row's metadata
        res = group.iloc[0].copy()
        res['stsec'] = group['stsec'].min()
        res['ensec'] = group['ensec'].max()
        res['dursec'] = res['ensec'] - res['stsec']
        
        # Spatial: First Start -> Last End
        stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
        enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
        res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
        
        res['ampldva'] = np.sqrt((enx - stx)**2 + (eny - sty)**2)
        res['pvel'] = group['pvel'].max()
        
        # If the group contained a SACBLNK, keep that more descriptive label
        if any(group['label'].str.upper() == 'SACBLNK'):
            res['label'] = 'SACBLNK'
        else:
            res['label'] = 'SACC'

        merged_list.append(res.drop('group_id'))

    # Convert list of Series back to DataFrame
    merged_saccs = pd.DataFrame(merged_list)

    # 4. Re-Combine with absolute column safety
    others = df[~is_sacc_mask].copy()
    
    # Use join='outer' to ensure no columns are dropped if there's a mismatch
    final_df = pd.concat([merged_saccs, others], ignore_index=True, join='outer')
    
    return final_df.sort_values('stsec').reset_index(drop=True)



def _consolidate_saccades(df, isi_threshold=0.010):
    """
    Merges SACC events that are overlapping or separated by less than isi_threshold.
    Assumes df contains only the events for a single eye.
    """
    # 1. Filter for saccades and sort by start time
    # (We keep others out of the merge logic to avoid swallowing blinks/fixations)
    sacc_df = df[df['label'] == 'SACC'].copy()
    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)

    if sacc_df.empty:
        return sacc_df

    # 2. Identify Groups
    # A new group starts if the current start time is greater than 
    # the max end time seen so far + our allowable gap (isi_threshold)
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    
    # Logic: if stsec > (previous_max_end + buffer), it's a brand new saccade.
    new_group_mask = (running_max_end.isna()) | (sacc_df['stsec'] > (running_max_end + isi_threshold))
    sacc_df['group_id'] = new_group_mask.cumsum()

    # 3. Aggregate Saccade Groups
    def aggregate_saccades(group):
        # Coordinates: Start of first detection, end of last detection
        stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
        enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
        
        # Calculate resulting spatial displacement
        dist = np.sqrt((enx - stx)**2 + (eny - sty)**2)

        return pd.Series({
            'stsec': group['stsec'].min(),
            'ensec': group['ensec'].max(),
            'stx': stx, 'sty': sty,
            'enx': enx, 'eny': eny,
            'label': 'SACC',
            'ampldva': dist,
            'pvel': group['pvel'].max(), # Conservative peak velocity
            'dursec': group['ensec'].max() - group['stsec'].min(),
            'count': len(group) # Useful to see how many detections were merged
        })

    merged_saccs = sacc_df.groupby('group_id').apply(aggregate_saccades,
                                                     #include_groups=False
                                                     ).reset_index(drop=True)
    
    return merged_saccs;


def _consolidate_saccades_debug(df, isi_threshold=0.010, target_time=None):
    # --- STAGE 1: FILTERING ---
    is_sacc_mask = df['label'].str.upper().str.startswith('SACC').fillna(False)
    sacc_df = df[is_sacc_mask].copy()
    
    if target_time:
        found_in_raw = not df[(df['stsec'] > target_time - 0.1) & (df['stsec'] < target_time + 0.1)].empty
        found_in_sacc = not sacc_df[(sacc_df['stsec'] > target_time - 0.1) & (sacc_df['stsec'] < target_time + 0.1)].empty
        print(f"DEBUG: Found in raw DF? {found_in_raw}")
        print(f"DEBUG: Found in sacc_df (after filter)? {found_in_sacc}")

    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    sacc_df['group_id'] = ((running_max_end.isna()) | (sacc_df['stsec'] > (running_max_end + isi_threshold))).cumsum()

    # --- STAGE 2: LOOPING ---
    merged_list = []
    for g_id, group in sacc_df.groupby('group_id'):
        is_target_group = False
        if target_time and any((group['stsec'] > target_time - 0.1) & (group['stsec'] < target_time + 0.1)):
            is_target_group = True
            print(f"DEBUG: Processing target group {g_id}, size: {len(group)}")

        res = group.iloc[0].copy()
        if len(group) > 1:
            res['stsec'] = group['stsec'].min()
            res['ensec'] = group['ensec'].max()
            stx, sty = group['stx'].iloc[0], group['sty'].iloc[0]
            enx, eny = group['enx'].iloc[-1], group['eny'].iloc[-1]
            res['stx'], res['sty'], res['enx'], res['eny'] = stx, sty, enx, eny
            res['ampldva'] = np.sqrt((enx - stx)**2 + (eny - sty)**2)
            res['pvel'] = group['pvel'].max()
            res['label'] = 'SACBLNK' if any(group['label'].str.upper() == 'SACBLNK') else 'SACC'

        # Remove group_id before appending
        merged_list.append(res.drop('group_id'))
        
        if is_target_group:
            print(f"DEBUG: Successfully appended target to merged_list.")

    # --- STAGE 3: RECOMBINATION ---
    merged_saccs = pd.DataFrame(merged_list)
    others = df[~is_sacc_mask].copy()
    
    final_df = pd.concat([merged_saccs, others], ignore_index=True)
    
    if target_time:
        found_in_final = not final_df[(final_df['stsec'] > target_time - 0.1) & (final_df['stsec'] < target_time + 0.1)].empty
        print(f"DEBUG: Found in final_df? {found_in_final}")

    return final_df.sort_values('stsec').reset_index(drop=True)





def _diagnose_vanishing_saccade(df, target_start=32.6, target_end=33.0):
    print("--- STEP 1: RAW DATA CHECK ---")
    raw_window = df[(df['stsec'] >= target_start) & (df['stsec'] <= target_end)]
    print(f"Rows found in raw window:\n{raw_window[['stsec', 'ensec', 'label']]}\n")

    # --- STEP 2: MASKING ---
    # We use fillna(False) because NaNs in a boolean mask can drop rows during ~mask
    is_sacc_mask = df['label'].str.upper().str.startswith('SACC').fillna(False)
    sacc_df = df[is_sacc_mask].copy()
    
    sacc_window = sacc_df[(sacc_df['stsec'] >= target_start) & (sacc_df['stsec'] <= target_end)]
    print("--- STEP 2: SACCADE FILTER CHECK ---")
    print(f"Saccades found after filtering:\n{sacc_window[['stsec', 'ensec', 'label']]}\n")

    if sacc_window.empty:
        print("!!! ERROR: Saccade lost during initial filtering. Check labels/whitespaces.\n")
        return

    # --- STEP 3: GROUPING ---
    sacc_df = sacc_df.sort_values('stsec').reset_index(drop=True)
    # Using 10ms threshold as discussed
    running_max_end = sacc_df['ensec'].cummax().shift(1)
    sacc_df['group_id'] = ((running_max_end.isna()) | (sacc_df['stsec'] > (running_max_end + 0.010))).cumsum()
    
    target_group_id = sacc_df.loc[sacc_window.index[0], 'group_id']
    group_data = sacc_df[sacc_df['group_id'] == target_group_id]
    
    print("--- STEP 3: GROUPING CHECK ---")
    print(f"Target Group ID: {target_group_id}")
    print(f"Full content of that group:\n{group_data[['stsec', 'ensec', 'label', 'group_id']]}\n")

    # --- STEP 4: AGGREGATION ---
    merged_list = []
    for g_id, group in sacc_df.groupby('group_id'):
        res = group.iloc[0].copy()
        if len(group) > 1:
            res['stsec'], res['ensec'] = group['stsec'].min(), group['ensec'].max()
            res['label'] = 'SACBLNK' if any(group['label'] == 'SACBLNK') else 'SACC'
        merged_list.append(res.drop('group_id', errors='ignore'))

    merged_df = pd.DataFrame(merged_list)
    merged_window = merged_df[(merged_df['stsec'] >= target_start) & (merged_df['stsec'] <= target_end)]
    print("--- STEP 4: MERGED LIST CHECK ---")
    print(f"Target found in merged list:\n{merged_window[['stsec', 'ensec', 'label']]}\n")

    # --- STEP 5: FINAL RECOMBINATION ---
    others = df[~is_sacc_mask].copy()
    final_df = pd.concat([merged_df, others], ignore_index=True).sort_values('stsec').reset_index(drop=True)
    
    final_window = final_df[(final_df['stsec'] >= target_start) & (final_df['stsec'] <= target_end)]
    print("--- STEP 5: FINAL DATAFRAME CHECK ---")
    print(f"Target found in final DF:\n{final_window[['stsec', 'ensec', 'label']]}")
    
    return final_df

'''
