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
    ylim=None,
    propdict={},
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
        ylim = np.max( abs(df[x_col]).max(), abs(df[y_col]).max() );
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

        fig = plt.figure(figsize=(plot_width, plot_height_per_chunk * num_chunks_this_fig), constrained_layout=True)
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
            
        title = 'Gaze: {}\n'.format(propdict); #Gaze Trace Timecourse
        #if total_num_figures > 1:
        title += f'(Page {fig_idx + 1} of {total_num_figures})'
        #    pass;
        fig.suptitle(title, fontsize=16); #, y=1.02)
        #fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)
        
        yield fig;
        pass;
    return; 

