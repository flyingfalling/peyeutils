import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Import for legend patches

def plot_gaze_chunks(
    df, timestamp_col, x_col, y_col, chunk_size_sec=10,
    events_df=None, event_start_col=None, event_end_col=None, event_type_col=None
):
    """
    Plots gaze data (X and Y) in stacked, long-form horizontal chunks,
    with optional shaded backgrounds for events.

    Args:
        df (pd.DataFrame): DataFrame with gaze data.
        timestamp_col (str): Column name for timestamps (in seconds).
        x_col (str): Column name for horizontal gaze.
        y_col (str): Column name for vertical gaze.
        chunk_size_sec (int, optional): Duration of each chunk in seconds.
        
        events_df (pd.DataFrame, optional): DataFrame with event data.
        event_start_col (str, optional): Column name for event start times.
        event_end_col (str, optional): Column name for event end times.
        event_type_col (str, optional): Column name for the event category/type.
    """
    
    # --- 1. Prepare Data ---
    data = df.copy()
    if not pd.api.types.is_numeric_dtype(data[timestamp_col]):
        raise TypeError(f"Timestamp column '{timestamp_col}' must be numeric (e.g., seconds).")
    data['time_sec'] = data[timestamp_col] - data[timestamp_col].min()
    
    # --- 2. Prepare Event Colors (New) ---
    color_map = {}
    event_args_provided = all([events_df is not None, event_start_col, event_end_col, event_type_col])
    
    if event_args_provided:
        try:
            unique_types = events_df[event_type_col].unique()
            # Get colors from a distinct colormap
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            color_map = {etype: color for etype, color in zip(unique_types, colors)}
        except KeyError:
             print(f"Warning: Could not find event columns. Check column names. Skipping event plotting.")
             event_args_provided = False
        except Exception as e:
            print(f"Warning: Error processing events_df: {e}. Skipping event plotting.")
            event_args_provided = False

    # --- 3. Determine Plot Layout ---
    max_time = data['time_sec'].max()
    if max_time == 0:
        print("Error: No time duration in data.")
        return
    num_chunks = int(np.ceil(max_time / chunk_size_sec))
    if num_chunks == 0:
        num_chunks = 1

    # --- 4. Create Subplots ---
    plot_width = 15
    plot_height = 2
    fig, axes = plt.subplots(
        nrows=num_chunks, 
        ncols=1, 
        figsize=(plot_width, plot_height * num_chunks),
        sharey=True,
        constrained_layout=True
    )
    if num_chunks == 1:
        axes = [axes]

    # --- 5. Loop and Plot Each Chunk ---
    for i in range(num_chunks):
        ax = axes[i]
        
        start_time = i * chunk_size_sec
        end_time = (i + 1) * chunk_size_sec

        # --- 5a. Plot Event Spans (New) ---
        if event_args_provided:
            # Filter events that *overlap* with this chunk
            relevant_events = events_df[
                (events_df[event_start_col] < end_time) &
                (events_df[event_end_col] > start_time)
            ]
            
            for _, event in relevant_events.iterrows():
                event_type = event[event_type_col]
                color = color_map.get(event_type, 'gray') # Default to gray
                
                # Clip the span to the chunk's boundaries
                span_start = max(event[event_start_col], start_time)
                span_end = min(event[event_end_col], end_time)
                
                if span_start < span_end:
                    ax.axvspan(span_start, span_end, color=color, alpha=0.3, zorder=0)

        # --- 5b. Plot Gaze Traces ---
        chunk_data = data[
            (data['time_sec'] >= start_time) & 
            (data['time_sec'] < end_time)
        ]

        line_handles = []
        if not chunk_data.empty:
            line1, = ax.plot(chunk_data['time_sec'], chunk_data[x_col], label=f'{x_col} (X)')
            line2, = ax.plot(chunk_data['time_sec'], chunk_data[y_col], label=f'{y_col} (Y)')
            line_handles = [line1, line2]
        else:
            ax.text(0.5, 0.5, 'No data in this interval', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='gray')

        # --- 5c. Format Each Subplot ---
        ax.set_xlim(start_time, end_time)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_title(f'Time: {start_time:.1f}s – {end_time:.1f}s', loc='left')

        if i == num_chunks - 1:
            ax.set_xlabel('Time (seconds)')
        if i == num_chunks // 2:
            ax.set_ylabel('Gaze Displacement')

    # --- 6. Final Figure Formatting ---
    # Get handles for gaze lines from the first non-empty plot
    all_line_handles = []
    for ax in axes:
        if ax.get_lines():
            all_line_handles, _ = ax.get_legend_handles_labels()
            break
            
    # Create proxy patches for the event legend
    event_patches = []
    if event_args_provided:
        for event_type, color in color_map.items():
            patch = mpatches.Patch(color=color, label=event_type, alpha=0.4)
            event_patches.append(patch)

    # Combine all handles for a single legend
    all_handles = all_line_handles + event_patches
    if all_handles:
        fig.legend(handles=all_handles, loc='upper right', bbox_to_anchor=(1.02, 0.99))
        
    fig.suptitle('Gaze Trace Timecourse by Chunk (with Events)', fontsize=16, y=1.03)
        
    return fig, axes;


def __test():

    # --- --- --- --- --- --- --- --- ---
    # EXAMPLE USAGE (Corrected)
    # --- --- --- --- --- --- --- --- ---

    # 1. Create Sample Gaze Data
    sample_rate = 100
    duration_sec = 45
    num_samples = sample_rate * duration_sec
    timestamps = np.linspace(0, duration_sec, num_samples)

    x_gaze = (0.8 * np.sin(timestamps * 0.2) + 0.2 * np.random.randn(num_samples))
    y_gaze = (0.5 * np.cos(timestamps * 0.3) + 0.2 * np.random.randn(num_samples))

    saccade_start = int(22 * sample_rate)
    saccade_end = int(22.1 * sample_rate)
    num_saccade_steps = saccade_end - saccade_start 

    if num_saccade_steps > 0:
        x_gaze[saccade_start:saccade_end] = np.linspace(x_gaze[saccade_start], -1.5, num_saccade_steps)
        y_gaze[saccade_start:saccade_end] = np.linspace(y_gaze[saccade_start], 1.0, num_saccade_steps)

    df_gaze = pd.DataFrame({
        'timestamp_sec': timestamps,
        'gaze_pos_x': x_gaze,
        'gaze_pos_y': y_gaze
    })

    print("Sample Gaze DataFrame head:")
    print(df_gaze.head())

    # --- --- --- --- --- --- --- --- ---
    # 2. Create Sample Event Data (New)
    # --- --- --- --- --- --- --- --- ---
    event_data = {
        'start_time': [2.0,  9.5,  12.0, 21.8, 25.0,     38.0],
        'end_time':   [5.0,  10.5, 12.5, 22.2, 40.0,     42.0],
        'event_name': ['Fixation', 'Blink', 'Fixation', 'Saccade', 'Task Block', 'Blink']
    }
    df_events = pd.DataFrame(event_data)

    print("\nSample Events DataFrame:")
    print(df_events)


    # 3. Call the plotting function
    plot_gaze_chunks(
        df=df_gaze,
        timestamp_col='timestamp_sec',
        x_col='gaze_pos_x',
        y_col='gaze_pos_y',
        chunk_size_sec=10,

        # --- Pass event data ---
        events_df=df_events,
        event_start_col='start_time',
        event_end_col='end_time',
        event_type_col='event_name'
    )
    return;
