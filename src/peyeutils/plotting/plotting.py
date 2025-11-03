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

def plot_gaze_chunks(
    df, timestamp_col, x_col, y_col, chunk_size_sec=10,
    events_df=None, event_start_col=None, event_end_col=None, event_type_col=None,
    stimulus_df=None, stim_start_col=None, stim_end_col=None, stim_name_col=None,
    max_points_per_sec=None,
    max_chunks_per_fig=10
):
    """
    Plots gaze data in paginated, stacked chunks with separate event timelines.

    Args:
        df (pd.DataFrame): DataFrame with gaze data.
        timestamp_col, x_col, y_col (str): Column names for gaze data.
        chunk_size_sec (int): Duration of each chunk.
        
        events_df (pd.DataFrame): DataFrame with discrete events (e.g., blinks).
        event_start_col, event_end_col, event_type_col (str): Event columns.
        
        stimulus_df (pd.DataFrame): DataFrame with stimulus presentation times.
        stim_start_col, stim_end_col, stim_name_col (str): Stimulus columns.

        max_points_per_sec (int, optional): Max sample rate to plot.
                                          Data is downsampled (e.g., to 50 Hz)
                                          to improve performance.
        max_chunks_per_fig (int): Max chunks per figure for pagination.
    """
    
    # --- 1. Prepare Gaze Data ---
    data = df.copy()
    if not pd.api.types.is_numeric_dtype(data[timestamp_col]):
        raise TypeError(f"Timestamp column '{timestamp_col}' must be numeric (e.g., seconds).")
    
    # Normalize time to start at 0
    data['time_sec'] = data[timestamp_col] - data[timestamp_col].min()

    # --- 1b. Downsample Gaze Data (New) ---
    if max_points_per_sec is not None and max_points_per_sec > 0:
        try:
            # Convert to a TimedeltaIndex to use resample
            data['timestamp'] = pd.to_timedelta(data['time_sec'], unit='s')
            data = data.set_index('timestamp')
            
            # Calculate rule, e.g., 50 Hz -> 20ms
            rule_ms = 1000 / max_points_per_sec
            rule = f"{rule_ms:.0f}ms"
            
            # Resample. .mean() acts as a low-pass filter.
            data = data.resample(rule).mean().reset_index()
            
            # Convert time back to seconds
            data['time_sec'] = data['timestamp'].dt.total_seconds()
            print(f"Resampled gaze data to {rule} ({max_points_per_sec} Hz)")
        except Exception as e:
            print(f"Warning: Could not resample data: {e}. Plotting all points.")
            # Revert if it fails
            data = df.copy()
            data['time_sec'] = data[timestamp_col] - data[timestamp_col].min()

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
            # Use a different, more subtle colormap for backgrounds
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(stim_types_list)))
            stim_color_map = {stype: color for stype, color in zip(stim_types_list, colors)}
        except Exception as e:
            print(f"Warning: Error processing stimulus_df: {e}. Skipping stimulus plotting.")
            stim_args_provided = False

    # --- 3. Determine Overall Layout ---
    max_time = data['time_sec'].max()
    total_num_chunks = int(np.ceil(max_time / chunk_size_sec))
    if total_num_chunks == 0: return
    total_num_figures = int(np.ceil(total_num_chunks / max_chunks_per_fig))
    
    plot_width = 16
    plot_height_per_chunk = 3 # Increased height to make room for event plot

    # --- 4. Loop Over Each FIGURE (Page) ---
    for fig_idx in range(total_num_figures):
        
        start_chunk_idx = fig_idx * max_chunks_per_fig
        end_chunk_idx = min((fig_idx + 1) * max_chunks_per_fig, total_num_chunks)
        num_chunks_this_fig = end_chunk_idx - start_chunk_idx
        if num_chunks_this_fig <= 0: continue

        # Create the figure
        fig = plt.figure(figsize=(plot_width, plot_height_per_chunk * num_chunks_this_fig))
        
        # Create an outer grid for the chunks
        outer_gs = gridspec.GridSpec(nrows=num_chunks_this_fig, ncols=1, 
                                     figure=fig, hspace=0.4)
        
        all_line_handles = [] # For the main legend

        # --- 5. Loop and Plot Each CHUNK for THIS figure ---
        for i in range(num_chunks_this_fig):
            global_chunk_idx = start_chunk_idx + i
            start_time = global_chunk_idx * chunk_size_sec
            end_time = (global_chunk_idx + 1) * chunk_size_sec

            # Create an inner grid: 2 rows (gaze, events)
            # height_ratios=[4, 1] means gaze plot is 4x taller than event plot
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[i], 
                height_ratios=[4, 1], hspace=0.05
            )
            
            # --- 5a. Create Gaze and Event Axes ---
            gaze_ax = fig.add_subplot(inner_gs[0])
            event_ax = fig.add_subplot(inner_gs[1], sharex=gaze_ax) # Share X-axis
            
            # --- 5b. Plot Stimulus Spans (on gaze_ax) ---
            if stim_args_provided:
                relevant_stim = stimulus_df[
                    (stimulus_df[stim_start_col] < end_time) &
                    (stimulus_df[stim_end_col] > start_time)
                ]
                for _, stim in relevant_stim.iterrows():
                    stim_name = stim[stim_name_col]
                    color = stim_color_map.get(stim_name, 'gray')
                    span_start = max(stim[stim_start_col], start_time)
                    span_end = min(stim[stim_end_col], end_time)
                    
                    if span_start < span_end:
                        gaze_ax.axvspan(span_start, span_end, color=color, 
                                        alpha=0.4, zorder=0, ec='none')
                        
                        # Add text annotation
                        text_x = (span_start + span_end) / 2
                        # Use data-coords for X, axes-coords (0-1) for Y
                        gaze_ax.text(text_x, 0.97, stim_name, 
                                     color=color*0.8, # Darker text
                                     alpha=1.0, ha='center', va='top', 
                                     fontsize=9, weight='bold',
                                     transform=gaze_ax.get_xaxis_transform())

            # --- 5c. Plot Gaze Traces (on gaze_ax) ---
            chunk_data = data[
                (data['time_sec'] >= start_time) & (data['time_sec'] < end_time)
            ]
            if not chunk_data.empty:
                h1, = gaze_ax.plot(chunk_data['time_sec'], chunk_data[x_col], label=f'{x_col} (X)')
                h2, = gaze_ax.plot(chunk_data['time_sec'], chunk_data[y_col], label=f'{y_col} (Y)')
                if not all_line_handles: # Store handles for legend
                    all_line_handles = [h1, h2]
            
            # --- 5d. Plot Events (on event_ax) ---
            if event_args_provided and len(event_types_list) > 0:
                relevant_events = events_df[
                    (events_df[event_start_col] < end_time) &
                    (events_df[event_end_col] > start_time)
                ]
                for event_idx, event in relevant_events.iterrows():
                    event_type = event[event_type_col]
                    if event_type not in event_types_list: continue
                    
                    y_level = event_types_list.index(event_type)
                    color = event_color_map[event_type]
                    
                    span_start = max(event[event_start_col], start_time)
                    span_end = min(event[event_end_col], end_time)
                    
                    if span_start < span_end:
                        # Draw thick "on" line
                        event_ax.hlines(y=y_level, xmin=span_start, xmax=span_end, 
                                        color=color, linewidth=6)
                        
                        # Write event index
                        text_x = span_start + (span_end-span_start)/2
                        event_ax.text(text_x, y_level, str(event_idx), ha='center', 
                                      va='center', fontsize=7, color='white', 
                                      weight='medium')

                # Draw thin baseline for all event types
                for y_level, etype in enumerate(event_types_list):
                    event_ax.hlines(y=y_level, xmin=start_time, xmax=end_time,
                                    color=event_color_map[etype], 
                                    linewidth=1, alpha=0.3)

            # --- 5e. Format Axes ---
            gaze_ax.set_xlim(start_time, end_time)
            gaze_ax.grid(True, linestyle=':', alpha=0.7)
            gaze_ax.set_title(f'Time: {start_time:.1f}s – {end_time:.1f}s', loc='left')
            if i == num_chunks_this_fig // 2: # Middle plot gets Y label
                gaze_ax.set_ylabel('Gaze Displacement')
            
            # Hide gaze_ax x-labels (they are shared with event_ax)
            plt.setp(gaze_ax.get_xticklabels(), visible=False)

            # Format event_ax
            if event_args_provided and len(event_types_list) > 0:
                event_ax.set_ylim(len(event_types_list) - 0.5, -0.5) # Inverted
                event_ax.set_yticks(range(len(event_types_list)))
                event_ax.set_yticklabels(event_types_list, fontsize=9)
                event_ax.tick_params(axis='y', length=0) # Hide Y ticks, keep labels
            else:
                event_ax.set_yticks([]) # Hide if no events
                
            event_ax.set_frame_on(False) # Remove box
            event_ax.tick_params(axis='x', length=0) # Hide X ticks
            
            if i == num_chunks_this_fig - 1: # Last plot gets X label
                event_ax.set_xlabel('Time (seconds)')
            else:
                plt.setp(event_ax.get_xticklabels(), visible=False)


        # --- 6. Final Figure Formatting (per figure) ---
        if all_line_handles:
            fig.legend(handles=all_line_handles, loc='upper right', 
                       bbox_to_anchor=(0.99, 0.99))
            
        title = 'Gaze Trace Timecourse'
        if total_num_figures > 1:
            title += f' (Page {fig_idx + 1} of {total_num_figures})'
        
        fig.suptitle(title, fontsize=16, y=1.02)
        
        # Use constrained_layout to prevent overlap
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)
        
        #plt.show()
        yield fig;
        pass;
    return;







def plot_gaze_chunksOLD(
    df, timestamp_col, x_col, y_col, chunk_size_sec=10,
    events_df=None, event_start_col=None, event_end_col=None, event_type_col=None,
    max_chunks_per_fig=1,
):
    """
    Plots gaze data in stacked chunks, paginated into multiple figures
    if the total number of chunks is large.

    Args:
        df (pd.DataFrame): DataFrame with gaze data.
        timestamp_col (str): Column name for timestamps (in seconds).
        x_col (str): Column name for horizontal gaze.
        y_col (str): Column name for vertical gaze.
        chunk_size_sec (int, optional): Duration of each chunk. Defaults to 10.
        
        events_df (pd.DataFrame, optional): DataFrame with event data.
        event_start_col (str, optional): Column name for event start times.
        event_end_col (str, optional): Column name for event end times.
        event_type_col (str, optional): Column name for the event category.
        
        max_chunks_per_fig (int, optional): The maximum number of chunks (subplots)
                                          to plot per figure. Defaults to 10.
    """
    
    # --- 1. Prepare Data ---
    data = df.copy()
    if not pd.api.types.is_numeric_dtype(data[timestamp_col]):
        raise TypeError(f"Timestamp column '{timestamp_col}' must be numeric (e.g., seconds).")
    data['time_sec'] = data[timestamp_col] - data[timestamp_col].min()
    
    # --- 2. Prepare Event Colors ---
    color_map = {}
    event_args_provided = all([events_df is not None, event_start_col, event_end_col, event_type_col])
    
    if event_args_provided:
        try:
            unique_types = events_df[event_type_col].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            color_map = {etype: color for etype, color in zip(unique_types, colors)}
        except Exception as e:
            print(f"Warning: Error processing events_df: {e}. Skipping event plotting.")
            event_args_provided = False

    # --- 3. Determine Overall Layout ---
    max_time = data['time_sec'].max()
    if max_time == 0:
        print("Error: No time duration in data.")
        return
        
    total_num_chunks = int(np.ceil(max_time / chunk_size_sec))
    if total_num_chunks == 0:
        total_num_chunks = 1
        
    # Calculate number of figures needed
    total_num_figures = int(np.ceil(total_num_chunks / max_chunks_per_fig))
    
    plot_width = 20  # Base width for each plot
    plot_height = 5  # Base height for each plot

    # --- 4. Loop Over Each FIGURE (Page) ---
    for fig_idx in range(total_num_figures):
        
        # --- 4a. Determine chunks for THIS figure ---
        start_chunk_idx = fig_idx * max_chunks_per_fig
        end_chunk_idx = min((fig_idx + 1) * max_chunks_per_fig, total_num_chunks)
        num_chunks_this_fig = end_chunk_idx - start_chunk_idx
        
        if num_chunks_this_fig <= 0:
            continue
        
        # --- 4b. Create Subplots for THIS figure ---
        plt.close();
        fig, axes = plt.subplots(
            nrows=num_chunks_this_fig, 
            ncols=1, 
            figsize=(plot_width, plot_height * num_chunks_this_fig),
            sharey=True,
            constrained_layout=True # Better at preventing overlap
        )
        if num_chunks_this_fig == 1:
            axes = [axes] # Make it iterable

        # --- 5. Loop and Plot Each CHUNK for THIS figure ---
        for i in range(num_chunks_this_fig):
            global_chunk_idx = start_chunk_idx + i
            ax = axes[i] # Local axis index (0 to num_chunks_this_fig-1)
            
            # Time range based on the GLOBAL chunk index
            start_time = global_chunk_idx * chunk_size_sec
            end_time = (global_chunk_idx + 1) * chunk_size_sec

            # --- 5a. Plot Event Spans ---
            if event_args_provided:
                relevant_events = events_df[
                    (events_df[event_start_col] < end_time) &
                    (events_df[event_end_col] > start_time)
                ]
                for _, event in relevant_events.iterrows():
                    event_type = event[event_type_col]
                    color = color_map.get(event_type, 'gray')
                    span_start = max(event[event_start_col], start_time)
                    span_end = min(event[event_end_col], end_time)
                    if span_start < span_end:
                        ax.axvspan(span_start, span_end, color=color, alpha=0.7, zorder=0)

            # --- 5b. Plot Gaze Traces ---
            chunk_data = data[
                (data['time_sec'] >= start_time) & 
                (data['time_sec'] < end_time)
            ]
            if not chunk_data.empty:
                ax.plot(chunk_data['time_sec'], chunk_data[x_col], label=f'{x_col} (X)')
                ax.plot(chunk_data['time_sec'], chunk_data[y_col], label=f'{y_col} (Y)')
            else:
                ax.text(0.5, 0.5, 'No data in this interval', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color='gray')

            # --- 5c. Format Each Subplot ---
            ax.set_xlim(start_time, end_time)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_title(f'Time: {start_time:.1f}s – {end_time:.1f}s', loc='left')

            # Only label axes on the last/middle plot OF THIS FIGURE
            if i == num_chunks_this_fig - 1:
                ax.set_xlabel('Time (seconds)')
            if i == num_chunks_this_fig // 2:
                ax.set_ylabel('Gaze Displacement')

        # --- 6. Final Figure Formatting (per figure) ---
        all_line_handles = []
        for ax in axes:
            if ax.get_lines():
                all_line_handles, _ = ax.get_legend_handles_labels()
                break
        
        event_patches = []
        if event_args_provided:
            for event_type, color in color_map.items():
                patch = mpatches.Patch(color=color, label=event_type, alpha=0.4)
                event_patches.append(patch)

        all_handles = all_line_handles + event_patches
        if all_handles:
            fig.legend(handles=all_handles, loc='upper right', bbox_to_anchor=(1.02, 0.99))
            
        # Add pagination to the title
        title = 'Gaze Trace Timecourse by Chunk (with Events)'
        if total_num_figures > 1:
            title += f' (Page {fig_idx + 1} of {total_num_figures})'
        
        fig.suptitle(title, fontsize=16, y=1.03)
        
        # Show the completed figure
        #plt.show()
        #return fig;
        yield fig;
        pass;
    return;
    
# --- --- --- --- --- --- --- --- ---
# EXAMPLE USAGE (LONGER DATA)
# --- --- --- --- --- --- --- --- ---

# 1. Create Sample Gaze Data (150 seconds long)
sample_rate = 100
duration_sec = 150 # <-- Increased duration
num_samples = sample_rate * duration_sec
timestamps = np.linspace(0, duration_sec, num_samples)

x_gaze = (0.8 * np.sin(timestamps * 0.2) + 0.2 * np.random.randn(num_samples))
y_gaze = (0.5 * np.cos(timestamps * 0.3) + 0.2 * np.random.randn(num_samples))

# Add a few "saccades"
for t in [22, 68, 110]:
    s_start = int(t * sample_rate)
    s_end = int((t + 0.1) * sample_rate)
    s_steps = s_end - s_start
    if s_steps > 0:
        x_gaze[s_start:s_end] = np.linspace(x_gaze[s_start], np.random.uniform(-2, 2), s_steps)
        y_gaze[s_start:s_end] = np.linspace(y_gaze[s_start], np.random.uniform(-2, 2), s_steps)

df_gaze = pd.DataFrame({
    'timestamp_sec': timestamps,
    'gaze_pos_x': x_gaze,
    'gaze_pos_y': y_gaze
})

# 2. Create Sample Event Data (spanning the longer duration)
event_data = {
    'start_time': [2.0,  9.5,  12.0, 21.8, 25.0,     38.0, 70.0, 85.0, 105.0, 110.1, 140.0],
    'end_time':   [5.0,  10.5, 12.5, 22.2, 50.0,     42.0, 90.0, 85.5, 115.0, 110.3, 148.0],
    'event_name': ['Fixation', 'Blink', 'Fixation', 'Saccade', 'Task Block 1', 'Blink', 'Task Block 2', 'Blink', 'Fixation', 'Saccade', 'Task Block 1']
}
df_events = pd.DataFrame(event_data)


# 3. Call the plotting function
# With 150s of data and 10s chunks, we expect 15 chunks.
# With max_chunks_per_fig=10, this will produce 2 figures.
# Figure 1: 10 chunks (0-100 seconds)
# Figure 2: 5 chunks (100-150 seconds)
plot_gaze_chunks(
    df=df_gaze,
    timestamp_col='timestamp_sec',
    x_col='gaze_pos_x',
    y_col='gaze_pos_y',
    chunk_size_sec=10,
    
    events_df=df_events,
    event_start_col='start_time',
    event_end_col='end_time',
    event_type_col='event_name',
    
    max_chunks_per_fig=10 # <-- New argument controls pagination
)

# Example: If you wanted fewer, taller pages, you could use:
# max_chunks_per_fig=5 
# This would produce 3 figures (5 chunks, 5 chunks, 5 chunks)
