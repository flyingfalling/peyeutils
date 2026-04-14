import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import for legend patches
import math


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Import for advanced subplots
import matplotlib.patches as mpatches


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

def plot_gaze_chunks(
    df, timestamp_col, x_col, y_col, chunk_size_sec=10,
    events_df=None, event_start_col=None, event_end_col=None, event_type_col=None,
    stimulus_df=None, stim_start_col=None, stim_end_col=None, stim_name_col=None,
    max_points_per_sec=None,
    max_chunks_per_fig=10,
    #baddata=pd.DataFrame(),
    ylim=None,
    proplist=list(),
):
    """
    Plots gaze data in paginated chunks using ABSOLUTE timestamps.
    No time normalization is performed.

    Args:
        df (pd.DataFrame): DataFrame with gaze data.
        timestamp_col, x_col, y_col (str): Column names for gaze data.
        chunk_size_sec (int): Duration of each chunk.
        
        events_df (pd.DataFrame): DataFrame with discrete events.
        event_start_col, event_end_col, event_type_col (str): Event columns.
        
        stimulus_df (pd.DataFrame): DataFrame with stimulus presentation times.
        stim_start_col, stim_end_col, stim_name_col (str): Stimulus columns.

        max_points_per_sec (int, optional): Max sample rate to plot (downsamples).
        max_chunks_per_fig (int): Max chunks per figure for pagination.
    """
    if( ylim is None ):
        ylim = np.max( [abs(df[x_col]).max(), abs(df[y_col]).max()] );
        print(df[[x_col, y_col]]);
        print("SETTING YLIM TO: {}".format(ylim));
        pass;
    
    # --- 1. Prepare Gaze Data ---
    data = df.copy()
    if not pd.api.types.is_numeric_dtype(data[timestamp_col]):
        raise TypeError(f"Timestamp column '{timestamp_col}' must be numeric (e.g., seconds).")
    
    # --- 1b. Downsample Gaze Data (if requested) ---
    if max_points_per_sec is not None and max_points_per_sec > 0:
        try:
            # Create a 0-based time column *only* for resampling
            time_offset = data[timestamp_col].min()
            data['time_sec_norm'] = data[timestamp_col] - time_offset
            data['timestamp'] = pd.to_timedelta(data['time_sec_norm'], unit='s')
            data = data.set_index('timestamp')
            
            rule_ms = 1000 / max_points_per_sec
            rule = f"{rule_ms:.0f}ms"
            
            # Resample. .mean() acts as a low-pass filter.
            data = data.resample(rule).mean().reset_index()
            
            # Re-create the absolute timestamp column from the new 0-based time
            data['time_sec_norm'] = data['timestamp'].dt.total_seconds()
            data[timestamp_col] = data['time_sec_norm'] + time_offset
            
            # Clean up temporary column
            data = data.drop(columns=['time_sec_norm', 'timestamp'])
            print(f"Resampled gaze data to {rule} ({max_points_per_sec} Hz)")
        except Exception as e:
            print(f"Warning: Could not resample data: {e}. Plotting all points.")
            # Revert if it fails
            data = df.copy()

    # --- 2. Prepare Event & Stimulus Colors ---
    event_color_map = {}
    event_types_list = []
    event_args_provided = all([events_df is not None, event_start_col, event_end_col, event_type_col])
    
    if event_args_provided:
        try:
            event_types_list = sorted(events_df[event_type_col].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(event_types_list)))
            event_color_map = {etype: color for etype, color in zip(event_types_list, colors)}
        except Exception as e:
            print(f"Warning: Error processing events_df: {e}. Skipping event plotting.")
            event_args_provided = False

    stim_color_map = {}
    stim_args_provided = all([stimulus_df is not None, stim_start_col, stim_end_col, stim_name_col])
    
    if stim_args_provided:
        try:
            stim_types_list = sorted(stimulus_df[stim_name_col].unique())
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(stim_types_list)))
            stim_color_map = {stype: color for stype, color in zip(stim_types_list, colors)}
        except Exception as e:
            print(f"Warning: Error processing stimulus_df: {e}. Skipping stimulus plotting.")
            stim_args_provided = False

    # --- 3. Determine Overall Layout (using absolute times) ---
    global_min_time = data[timestamp_col].min()
    global_max_time = data[timestamp_col].max()
    total_duration = global_max_time - global_min_time
    
    total_num_chunks = int(np.ceil(total_duration / chunk_size_sec))
    if total_num_chunks == 0: 
        print("Error: No time duration in gaze data.")
        return
        
    total_num_figures = int(np.ceil(total_num_chunks / max_chunks_per_fig))
    
    plot_width = 16
    plot_height_per_chunk = 3 

    # --- 4. Loop Over Each FIGURE (Page) ---
    for fig_idx in range(total_num_figures):
        
        start_chunk_idx = fig_idx * max_chunks_per_fig
        end_chunk_idx = min((fig_idx + 1) * max_chunks_per_fig, total_num_chunks)
        num_chunks_this_fig = end_chunk_idx - start_chunk_idx
        if num_chunks_this_fig <= 0: continue

        fig = plt.figure(figsize=(plot_width, plot_height_per_chunk * num_chunks_this_fig)); #, constrained_layout=True)
        outer_gs = gridspec.GridSpec(nrows=num_chunks_this_fig, ncols=1, 
                                     figure=fig, hspace=0.4)
        all_line_handles = [] 

        # --- 5. Loop and Plot Each CHUNK for THIS figure ---
        for i in range(num_chunks_this_fig):
            global_chunk_idx = start_chunk_idx + i
            
            # --- FIX: Chunk boundaries are absolute ---
            start_time = global_min_time + (global_chunk_idx * chunk_size_sec)
            end_time = global_min_time + ((global_chunk_idx + 1) * chunk_size_sec)

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[i], 
                height_ratios=[4, 1], hspace=0.05
            )
            gaze_ax = fig.add_subplot(inner_gs[0])
            event_ax = fig.add_subplot(inner_gs[1], sharex=gaze_ax)
            
            # --- 5b. Plot Stimulus Spans (Absolute) ---
            if stim_args_provided:
                # Filter using absolute times
                relevant_stim = stimulus_df[
                    (stimulus_df[stim_start_col] < end_time) &
                    (stimulus_df[stim_end_col] > start_time)
                ]
                for _, stim in relevant_stim.iterrows():
                    stim_name = stim[stim_name_col]
                    color = stim_color_map.get(stim_name, 'gray')
                    
                    # Get absolute event times
                    stim_start_abs = stim[stim_start_col]
                    stim_end_abs = stim[stim_end_col]
                    
                    # Clip to chunk boundaries
                    span_start = max(stim_start_abs, start_time)
                    span_end = min(stim_end_abs, end_time)
                    
                    if span_start < span_end:
                        gaze_ax.axvspan(span_start, span_end, color=color, 
                                        alpha=0.4, zorder=0, ec='none')
                        text_x = (span_start + span_end) / 2
                        gaze_ax.text(text_x, 0.97, stim_name, 
                                     color=color*0.8, alpha=1.0, ha='center', 
                                     va='top', fontsize=9, weight='bold',
                                     transform=gaze_ax.get_xaxis_transform())

            # --- 5c. Plot Gaze Traces (Absolute) ---
            chunk_data = data[
                (data[timestamp_col] >= start_time) & 
                (data[timestamp_col] < end_time)
            ]
            if not chunk_data.empty:
                h1, = gaze_ax.plot(chunk_data[timestamp_col], chunk_data[x_col], label=f'{x_col} (X)')
                h2, = gaze_ax.plot(chunk_data[timestamp_col], chunk_data[y_col], label=f'{y_col} (Y)')
                if not all_line_handles: 
                    all_line_handles = [h1, h2]
                pass;

            '''
            if( len(baddata.index) > 0 ):
                badchunk_data = baddata[ (baddata[timestamp_col] >= start_time) & 
                                         (baddata[timestamp_col] < end_time)
                                         ];
                if not badchunk_data.empty:
                    h1, = gaze_ax.plot(badchunk_data[timestamp_col], badchunk_data[x_col], color='grey');
                    h2, = gaze_ax.plot(badchunk_data[timestamp_col], badchunk_data[y_col], color='grey');
                    pass;
                pass;
            '''
            
            # --- 5d. Plot Events (Absolute) ---
            if event_args_provided and len(event_types_list) > 0:
                # Filter using absolute times
                relevant_events = events_df[
                    (events_df[event_start_col] < end_time) &
                    (events_df[event_end_col] > start_time)
                ]
                for event_idx, event in relevant_events.iterrows():
                    event_type = event[event_type_col]
                    if event_type not in event_types_list: continue
                    
                    y_level = event_types_list.index(event_type)
                    color = event_color_map[event_type]
                    
                    # Get absolute event times
                    event_start_abs = event[event_start_col]
                    event_end_abs = event[event_end_col]
                    
                    # Clip to chunk boundaries
                    span_start = max(event_start_abs, start_time)
                    span_end = min(event_end_abs, end_time)
                    
                    if span_start < span_end:
                        event_ax.hlines(y=y_level, xmin=span_start, xmax=span_end, 
                                        color=color, linewidth=6)
                        text_x = span_start + (span_end-span_start)/2
                        event_ax.text(text_x, y_level, str(event_idx), ha='center', 
                                      va='center', fontsize=7, color='white', 
                                      weight='medium')

                for y_level, etype in enumerate(event_types_list):
                    event_ax.hlines(y=y_level, xmin=start_time, xmax=end_time,
                                    color=event_color_map[etype], 
                                    linewidth=1, alpha=0.3)

            # --- 5e. Format Axes ---
            # Set X-axis limits to the absolute chunk times
            gaze_ax.set_xlim(start_time, end_time)
            gaze_ax.set_ylim(-ylim, ylim);
            gaze_ax.grid(True, linestyle=':', alpha=0.7)
            # Title now reflects absolute time
            gaze_ax.set_title(f'Time: {start_time:.1f}s – {end_time:.1f}s', loc='left')
            
            if i == num_chunks_this_fig // 2: 
                gaze_ax.set_ylabel('Gaze Displacement')
            
            plt.setp(gaze_ax.get_xticklabels(), visible=False)

            if event_args_provided and len(event_types_list) > 0:
                event_ax.set_ylim(len(event_types_list) - 0.5, -0.5) 
                event_ax.set_yticks(range(len(event_types_list)))
                event_ax.set_yticklabels(event_types_list, fontsize=9)
                event_ax.tick_params(axis='y', length=0) 
            else:
                event_ax.set_yticks([]) 
                
            event_ax.set_frame_on(False) 
            event_ax.tick_params(axis='x', length=0) 
            
            if i == num_chunks_this_fig - 1: 
                event_ax.set_xlabel('Time (seconds)')
            else:
                plt.setp(event_ax.get_xticklabels(), visible=False)

        # --- 6. Final Figure Formatting (per figure) ---
        if all_line_handles:
            fig.legend(handles=all_line_handles, loc='upper right', 
                       bbox_to_anchor=(0.99, 0.99))
            pass;
        title = '';
        line='';
        print("PROP LIST", proplist);
        proplist = sorted(proplist, key=len);
        for i in proplist:
            line = line+'{} '.format(i);
            if( len(line) > 30 ):
                title+=line+'\n';
                line='';
                pass;
            pass;
        if(len(line) > 0 ):
            title+=line+'\n';
            pass;
        
        #if total_num_figures > 1:
        title += f'(Page {fig_idx + 1} of {total_num_figures})'
        #    pass;
        fig.suptitle(title, fontsize=16); #, y=1.02)
        #fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)
        
        yield fig;
        pass;
    return; 








def plot_gaze_chunks(
    df, timestamp_col, x_col, y_col, chunk_size_sec=10,
    pupil_col=None, eye_col=None, eyes_to_plot=None,
    events_df=None, event_start_col=None, event_end_col=None, event_type_col=None,
    stimulus_df=None, stim_start_col=None, stim_end_col=None, stim_name_col=None,
    max_points_per_sec=None,
    max_chunks_per_fig=10
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    
    # --- 1. Data Prep ---
    data = df.copy()
    
    # Identify groups to plot
    if eye_col and eye_col in data.columns:
        all_eyes = data[eye_col].unique().tolist()
        plot_groups = eyes_to_plot if eyes_to_plot else all_eyes
        data = data[data[eye_col].isin(plot_groups)]
    else:
        plot_groups = [None] 

    # --- 1b. Downsampling (Absolute Time) ---
    if max_points_per_sec is not None and max_points_per_sec > 0:
        rule = f"{1000 / max_points_per_sec:.0f}ms"
        
        def resample_group(group_df):
            t_min = group_df[timestamp_col].min()
            group_df['_delta'] = pd.to_timedelta(group_df[timestamp_col] - t_min, unit='s')
            resampled = group_df.set_index('_delta').resample(rule).mean().reset_index()
            resampled[timestamp_col] = resampled['_delta'].dt.total_seconds() + t_min
            return resampled.drop(columns=['_delta'])

        if eye_col and eye_col in data.columns:
            data = data.groupby(eye_col, group_keys=False).apply(resample_group).reset_index(drop=True)
        else:
            data = resample_group(data)

    # --- 2. Setup Colors ---
    # Gaze colors (tab10)
    base_palette = plt.cm.tab10.colors
    eye_colors = {eye: base_palette[i % len(base_palette)] for i, eye in enumerate(plot_groups)}

    # --- NEW: Pupil shades (Gradient of greys) ---
    # We create a list of grey levels from 0.3 (dark) to 0.7 (light)
    grey_levels = np.linspace(0.3, 0.7, len(plot_groups))
    pupil_shades = {eye: (g, g, g) for eye, g in zip(plot_groups, grey_levels)}

    has_events = all([events_df is not None, event_start_col, event_end_col, event_type_col])
    has_stim = all([stimulus_df is not None, stim_start_col, stim_end_col, stim_name_col])

    event_types = sorted(events_df[event_type_col].unique()) if has_events else []
    ev_colors = dict(zip(event_types, plt.cm.Set1(np.linspace(0, 1, len(event_types))))) if has_events else {}

    # --- 3. Time Windows ---
    global_start = data[timestamp_col].min()
    global_end = data[timestamp_col].max()
    total_chunks = int(np.ceil((global_end - global_start) / chunk_size_sec))
    num_figs = int(np.ceil(total_chunks / max_chunks_per_fig))

    # --- 4. Plotting Loop ---
    for f_idx in range(num_figs):
        c_start = f_idx * max_chunks_per_fig
        c_end = min((f_idx + 1) * max_chunks_per_fig, total_chunks)
        chunks_here = c_end - c_start
        
        fig = plt.figure(figsize=(16, max(4 * chunks_here, 8)), constrained_layout=True)
        gs_outer = gridspec.GridSpec(chunks_here, 1, figure=fig, hspace=0.4)

        for i in range(chunks_here):
            chunk_idx = c_start + i
            t0 = global_start + (chunk_idx * chunk_size_sec)
            t1 = t0 + chunk_size_sec

            gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[i], 
                                                        height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(gs_inner[0])
            ev_ax = fig.add_subplot(gs_inner[1], sharex=ax)
            p_ax = ax.twinx() if pupil_col else None

            # --- Plot Gaze & Pupil Per Eye ---
            for group_id in plot_groups:
                if group_id is not None:
                    chunk = data[(data[eye_col] == group_id) & (data[timestamp_col] >= t0) & (data[timestamp_col] < t1)]
                    lbl = f"{group_id} "
                    color = eye_colors[group_id]
                    p_color = pupil_shades[group_id] # --- NEW: Use specific shade ---
                else:
                    chunk = data[(data[timestamp_col] >= t0) & (data[timestamp_col] < t1)]
                    lbl = ""
                    color = base_palette[0]
                    p_color = (0.5, 0.5, 0.5)

                if chunk.empty: continue
                
                # Plot Gaze
                ax.plot(chunk[timestamp_col], chunk[x_col], color=color, label=f'{lbl}X')
                ax.plot(chunk[timestamp_col], chunk[y_col], color=color, linestyle='--', label=f'{lbl}Y', alpha=0.7)
                
                # --- NEW: Plot Pupil with unique shade and label ---
                if p_ax and pupil_col in chunk:
                    p_ax.plot(chunk[timestamp_col], chunk[pupil_col], color=p_color, alpha=0.5, 
                              linewidth=1.2, label=f'{lbl}Pupil')

            # --- Plot Stimuli ---
            if has_stim:
                rel_stim = stimulus_df[(stimulus_df[stim_start_col] < t1) & (stimulus_df[stim_end_col] > t0)]
                for _, s in rel_stim.iterrows():
                    ss, se = max(s[stim_start_col], t0), min(s[stim_end_col], t1)
                    ax.axvspan(ss, se, color='teal', alpha=0.08, zorder=0)
                    ax.text((ss+se)/2, 0.96, s[stim_name_col], transform=ax.get_xaxis_transform(), 
                            ha='center', va='top', fontweight='bold', alpha=0.4, color='teal', fontsize=9)

            # --- Plot Events ---
            if has_events:
                rel_ev = events_df[(events_df[event_start_col] < t1) & (events_df[event_end_col] > t0)]
                for idx, ev in rel_ev.iterrows():
                    etype = ev[event_type_col]
                    y_pos = event_types.index(etype)
                    ec = ev_colors.get(etype, 'black')
                    es, ee = max(ev[event_start_col], t0), min(ev[event_end_col], t1)
                    ev_ax.hlines(y_pos, es, ee, color=ec, linewidth=10)
                    ev_ax.text((es+ee)/2, y_pos, str(idx), color='white', ha='center', va='center', fontsize=8)
                
                ev_ax.set_yticks(range(len(event_types)))
                ev_ax.set_yticklabels(event_types, fontsize=9)
                ev_ax.set_ylim(-0.5, len(event_types)-0.5)
            
            # Formatting
            ax.set_xlim(t0, t1)
            ax.set_title(f"Time: {t0:.1f}s - {t1:.1f}s", loc='left', fontsize=10)
            ax.grid(True, alpha=0.1)
            plt.setp(ax.get_xticklabels(), visible=False)
            if i == chunks_here - 1: ev_ax.set_xlabel("Time (s)")
            if p_ax: 
                p_ax.set_ylabel("Pupil Size", color=(0.4, 0.4, 0.4), fontsize=9)
                p_ax.tick_params(axis='y', colors=(0.4, 0.4, 0.4), labelsize=8)

        # Legend Cleanup (Deduplicate labels from Gaze and Pupil axes)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = p_ax.get_legend_handles_labels() if p_ax else ([], [])
        unique = dict(zip(l1 + l2, h1 + h2))
        fig.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=9)
        fig.suptitle(f"Gaze Timeplot - Page {f_idx + 1}", fontsize=14)
        yield fig;
