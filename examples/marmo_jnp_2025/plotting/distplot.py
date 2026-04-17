import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np;
import sys
from matplotlib.lines import Line2D

from distplot import *; #REV: import self?


#REV: todo:
## plot "distribution" of each species overall (to show it's not just
##  "wider looking" -> it's DIFFERENT locations).

## NULL MODEL -> shuffle time points, what is distance ABOVE that?
## REV: do "within" each video?
## REV: either use all data


#REV: I *should* only compare WITHIN same video (for each species),
#    because otherwise bias of which videos were watched more will have a large effect!
##  E.g. marmosets (new ones) watched all MTV clips...

# Also bias in which time points were attended in which vids.
## E.g. babies / kids love watching mario, hate watching news shit. So it's more of a "content" problem...

## REV; should do: per-video!


if __name__=='__main__':
    
    incsv = sys.argv[1];
    df = pd.read_csv(incsv);
    tag='';
    if( len(sys.argv) > 2 ):
        tag='_' + sys.argv[2];
        pass;
    
    # 1. Broadest overview (1 line per species)
    fig = plot_distributions(df, grouping_level='species', within_species_only=True);
    fig.savefig('species_diffs{}.pdf'.format(tag));
    plt.close();
    
    fig = plot_distributions_with_variance(df, grouping_level='species', within_species_only=True);
    fig.savefig('species_diffs_wvar{}.pdf'.format(tag));
    plt.close();
    
    # 2. Medium detail (1 line per subject, averaging all their interactions)
    fig = plot_distributions(df, grouping_level='subject', within_species_only=True);
    fig.savefig('subject_diffs{}.pdf'.format(tag));
    plt.close();
    
    fig = plot_distributions_with_variance(df, grouping_level='subject', within_species_only=True);
    fig.savefig('subject_diffs_wvar{}.pdf'.format(tag));
    plt.close();
    # 3. Highest detail (1 line per unique pair of subjects)
    #REV: can't see shit.
    #plot_distributions(df, grouping_level='pair' ); #, vid_filter='vid_001')
    '''
    for vid, vdf in df.groupby('vid'):
        print("Doing for {}".format(vid));
        fig = plot_distributions(vdf, grouping_level='species', within_species_only=True);
        fig.savefig('species_diffs_{}{}.pdf'.format(vid,tag));
        plt.close();
        pass;
    '''
    pass;




def plot_distributions_with_variance(df, grouping_level='species', within_species_only=False, central_tendency='median'):
    """
    Plots mean distributions with standard deviation bands across videos.
    grouping_level: 'species', 'subject', or 'pair'
    central_tendency: 'median', 'mean', or None (draws a dashed axvline for the group)
    """
    df_plot = df.copy()
    
    # 1. Apply Filters
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    df_plot['pair'] = df_plot['subj1'] + " & " + df_plot['subj2']
    
    # 2. Setup Grouping & Line Thickness
    if grouping_level == 'species':
        hue_col = 'spec1'
        linewidth = 2.5
    elif grouping_level == 'subject':
        hue_col = 'subj1'
        linewidth = 1.5
    elif grouping_level == 'pair':
        hue_col = 'pair'
        linewidth = 1.0
    else:
        raise ValueError("grouping_level must be 'species', 'subject', or 'pair'")

    # 3. Dynamic Color Logic
    palette = {}
    unique_species = df_plot['spec1'].unique()
    
    for sp in unique_species:
        if grouping_level == 'species':
            items = [sp]
        elif grouping_level == 'subject':
            items = df_plot[df_plot['spec1'] == sp]['subj1'].unique()
        elif grouping_level == 'pair':
            items = df_plot[df_plot['spec1'] == sp]['pair'].unique()
            
        n_items = len(items)
        if sp == 'human':
            intensities = [0.55] if n_items == 1 else np.linspace(0.35, 0.65, n_items)
        else:
            intensities = [0.7] if n_items == 1 else np.linspace(0.4, 0.95, n_items)
            
        for i, item in enumerate(items):
            if sp == 'marmo': palette[item] = plt.cm.Blues(intensities[i])
            elif sp == 'human': palette[item] = plt.cm.YlOrBr(intensities[i])
            elif sp == 'macaq': palette[item] = plt.cm.Reds(intensities[i])
            elif sp == 'infant': palette[item] = plt.cm.Greens(intensities[i])
            else: palette[item] = plt.cm.Greys(intensities[i])

    # 4. Plotting & Math Setup
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    max_dist = df_plot['dist_px'].max()
    
    # Shared grids to allow for matrix averaging 
    n_bins = 50
    x_grid_hist = np.linspace(0, max_dist, n_bins + 1)
    x_centers_hist = (x_grid_hist[:-1] + x_grid_hist[1:]) / 2
    x_grid_ecdf = np.linspace(0, max_dist, 200)

    legend_elements = []

    # 5. Calculation & Plotting Loop
    for group_val in df_plot[hue_col].unique():
        group_df = df_plot[df_plot[hue_col] == group_val]
        color = palette[group_val]
        
        hist_densities = []
        ecdf_curves = []
        
        # Calculate curves for each video independently
        for vid in group_df['vid'].unique():
            vid_dist = group_df[group_df['vid'] == vid]['dist_px'].values
            
            if len(vid_dist) < 2:
                continue
                
            # Density=True normalizes THIS specific video array to integrate to 1.0 probability
            counts, _ = np.histogram(vid_dist, bins=x_grid_hist, density=True)
            hist_densities.append(counts)
            
            vid_dist_sorted = np.sort(vid_dist)
            y_ecdf = np.searchsorted(vid_dist_sorted, x_grid_ecdf, side='right') / len(vid_dist_sorted)
            ecdf_curves.append(y_ecdf)
            
        if not hist_densities:
            continue

        # Convert to arrays to calculate cross-video Mean and Std Dev
        hist_densities = np.array(hist_densities)
        mean_hist = np.mean(hist_densities, axis=0)
        std_hist = np.std(hist_densities, axis=0)
        
        ecdf_curves = np.array(ecdf_curves)
        mean_ecdf = np.mean(ecdf_curves, axis=0)
        std_ecdf = np.std(ecdf_curves, axis=0)
        
        # Plot Histogram
        axes[0].step(x_centers_hist, mean_hist, where='mid', color=color, linewidth=linewidth)
        axes[0].fill_between(x_centers_hist, 
                             np.clip(mean_hist - std_hist, 0, None), 
                             mean_hist + std_hist, 
                             step='mid', color=color, alpha=0.2)
                             
        # Plot ECDF
        axes[1].plot(x_grid_ecdf, mean_ecdf, color=color, linewidth=linewidth)
        axes[1].fill_between(x_grid_ecdf, 
                             np.clip(mean_ecdf - std_ecdf, 0, 1), 
                             np.clip(mean_ecdf + std_ecdf, 0, 1), 
                             color=color, alpha=0.2)
                             
        # --- NEW: Central Tendency Line ---
        if central_tendency == 'mean':
            c_val = group_df['dist_px'].mean() # Overall mean for this entire group
            axes[0].axvline(c_val, color=color, linestyle='--', linewidth=linewidth, alpha=0.8)
            axes[1].axvline(c_val, color=color, linestyle='--', linewidth=linewidth, alpha=0.8)
        elif central_tendency == 'median':
            c_val = group_df['dist_px'].median() # Overall median for this entire group
            axes[0].axvline(c_val, color=color, linestyle='--', linewidth=linewidth, alpha=0.8)
            axes[1].axvline(c_val, color=color, linestyle='--', linewidth=linewidth, alpha=0.8)

        # Store for the custom legend
        legend_elements.append(Line2D([0], [0], color=color, lw=linewidth, label=group_val))

    # 6. Formatting & Legend
    # Add a dummy marker so the legend explains what the dashed line is
    if central_tendency in ['mean', 'median']:
        legend_elements.append(Line2D([0], [0], color='black', lw=1.5, linestyle='--', label=f'Group {central_tendency.capitalize()}'))

    filter_txt = " (Within-Species Only)" if within_species_only else " (All Data)"
    
    axes[0].set_title(f"Mean Distance Histogram ± 1 SD{filter_txt}")
    axes[0].set_xlabel("Distance (px)")
    axes[0].set_ylabel("Density (Average Across Videos)")

    axes[1].set_title(f"Mean Cumulative Distribution ± 1 SD{filter_txt}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion (Average Across Videos)")
    
    n_categories = len(legend_elements)
    ncol = 3 if n_categories > 15 else 2 if n_categories > 8 else 1
    
    axes[1].legend(
        handles=legend_elements, 
        loc="lower right", 
        title=hue_col.capitalize(), 
        ncol=ncol, 
        framealpha=0.9
    )

    XMAX=600;
    YMAX=0.010;
    axes[0].set_ylim([0, YMAX]);
    
    axes[0].set_xlim([0, XMAX]);
    axes[1].set_xlim([0, XMAX]);
    
    plt.tight_layout()
    return fig;




def plot_distributions_with_variance_OLD(df, grouping_level='species', within_species_only=False):
    """
    Plots distributions with variance bands representing the standard deviation across videos.
    grouping_level: 'species', 'subject', or 'pair'
    """
    df_plot = df.copy()
    
    # 1. Apply Filters
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    df_plot['pair'] = df_plot['subj1'] + " & " + df_plot['subj2']
    
    # 2. Setup Grouping & Line Thickness
    if grouping_level == 'species':
        hue_col = 'spec1'
        linewidth = 2.5
    elif grouping_level == 'subject':
        hue_col = 'subj1'
        linewidth = 1.5
    elif grouping_level == 'pair':
        hue_col = 'pair'
        linewidth = 1.0
    else:
        raise ValueError("grouping_level must be 'species', 'subject', or 'pair'")

    # 3. Dynamic Color Logic
    palette = {}
    unique_species = df_plot['spec1'].unique()
    
    for sp in unique_species:
        if grouping_level == 'species':
            items = [sp]
        elif grouping_level == 'subject':
            items = df_plot[df_plot['spec1'] == sp]['subj1'].unique()
        elif grouping_level == 'pair':
            items = df_plot[df_plot['spec1'] == sp]['pair'].unique()
            
        n_items = len(items)
        if sp == 'human':
            intensities = [0.55] if n_items == 1 else np.linspace(0.35, 0.65, n_items)
        else:
            intensities = [0.7] if n_items == 1 else np.linspace(0.4, 0.95, n_items)
            
        for i, item in enumerate(items):
            if sp == 'marmo': palette[item] = plt.cm.Blues(intensities[i])
            elif sp == 'human': palette[item] = plt.cm.YlOrBr(intensities[i])
            elif sp == 'macaq': palette[item] = plt.cm.Reds(intensities[i])
            elif sp == 'infant': palette[item] = plt.cm.Greens(intensities[i])
            else: palette[item] = plt.cm.Greys(intensities[i])

    # 4. Plotting & Math Setup
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create shared X-axis grids so we can average the video arrays together
    max_dist = df_plot['dist_px'].max()
    
    # For Histogram: 50 bins across the whole range
    n_bins = 50
    x_grid_hist = np.linspace(0, max_dist, n_bins + 1)
    x_centers_hist = (x_grid_hist[:-1] + x_grid_hist[1:]) / 2
    
    # For ECDF: 200 smooth points across the whole range
    x_grid_ecdf = np.linspace(0, max_dist, 200)

    legend_elements = []

    # 5. Calculation & Plotting Loop
    for group_val in df_plot[hue_col].unique():
        group_df = df_plot[df_plot[hue_col] == group_val]
        color = palette[group_val]
        
        hist_densities = []
        ecdf_curves = []
        
        # Calculate curves for each video independently
        for vid in group_df['vid'].unique():
            vid_dist = group_df[group_df['vid'] == vid]['dist_px'].values
            
            # Skip videos that don't have enough data points for this specific pair/subject
            if len(vid_dist) < 2:
                continue
                
            # Compute Histogram Density (Area = 1) for this video
            counts, _ = np.histogram(vid_dist, bins=x_grid_hist, density=True)
            hist_densities.append(counts)
            
            # Compute ECDF for this video
            vid_dist_sorted = np.sort(vid_dist)
            y_ecdf = np.searchsorted(vid_dist_sorted, x_grid_ecdf, side='right') / len(vid_dist_sorted)
            ecdf_curves.append(y_ecdf)
            
        # If no valid videos existed for this group, skip it
        if not hist_densities:
            continue

        # Convert to arrays to calculate cross-video Mean and Std Dev
        hist_densities = np.array(hist_densities)
        mean_hist = np.mean(hist_densities, axis=0)
        std_hist = np.std(hist_densities, axis=0)
        
        ecdf_curves = np.array(ecdf_curves)
        mean_ecdf = np.mean(ecdf_curves, axis=0)
        std_ecdf = np.std(ecdf_curves, axis=0)
        
        # Plot Histogram: Mean (step line) + Variance (shaded area)
        axes[0].step(x_centers_hist, mean_hist, where='mid', color=color, linewidth=linewidth)
        axes[0].fill_between(x_centers_hist, 
                             np.clip(mean_hist - std_hist, 0, None), # Density can't go below 0
                             mean_hist + std_hist, 
                             step='mid', color=color, alpha=0.2)
                             
        # Plot ECDF: Mean (smooth line) + Variance (shaded area)
        axes[1].plot(x_grid_ecdf, mean_ecdf, color=color, linewidth=linewidth)
        axes[1].fill_between(x_grid_ecdf, 
                             np.clip(mean_ecdf - std_ecdf, 0, 1), # ECDF bounded between 0 and 1
                             np.clip(mean_ecdf + std_ecdf, 0, 1), 
                             color=color, alpha=0.2)
                             
        # Store for the custom legend
        legend_elements.append(Line2D([0], [0], color=color, lw=linewidth, label=group_val))

    # 6. Formatting
    filter_txt = " (Within-Species Only)" if within_species_only else " (All Data)"
    
    axes[0].set_title(f"Mean Distance Histogram ± 1 SD{filter_txt}")
    axes[0].set_xlabel("Distance (px)")
    axes[0].set_ylabel("Density (Average Across Videos)")

    axes[1].set_title(f"Mean Cumulative Distribution ± 1 SD{filter_txt}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion (Average Across Videos)")
    
    # Auto-column Legend
    n_categories = len(legend_elements)
    ncol = 3 if n_categories > 15 else 2 if n_categories > 8 else 1
    
    axes[1].legend(
        handles=legend_elements, 
        loc="lower right", 
        title=hue_col.capitalize(), 
        ncol=ncol, 
        framealpha=0.9
    )
    
    plt.tight_layout()
    return fig;



def plot_distributions(df, grouping_level='pair', within_species_only=False, vid_filter=None):
    """
    grouping_level: 'species', 'subject', or 'pair'
    """
    df_plot = df.copy()
    
    # 1. Apply Filters
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if vid_filter is not None:
        df_plot = df_plot[df_plot['vid'] == vid_filter]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    # Create the pair identifier just in case it's needed
    df_plot['pair'] = df_plot['subj1'] + " & " + df_plot['subj2']
    
    # 2. Setup Data Grouping
    if grouping_level == 'species':
        hue_col = 'spec1'
        title_prefix = "Species-Level"
        linewidth = 2.5
    elif grouping_level == 'subject':
        hue_col = 'subj1'
        title_prefix = "Subject-Level"
        linewidth = 1.5
    elif grouping_level == 'pair':
        hue_col = 'pair'
        title_prefix = "Pair-Level"
        linewidth = 1.0
    else:
        raise ValueError("grouping_level must be 'species', 'subject', or 'pair'")

    # 3. Dynamic Color Logic
    palette = {}
    unique_species = df_plot['spec1'].unique()
    
    for sp in unique_species:
        # Determine what items we are coloring for this species
        if grouping_level == 'species':
            items = [sp]
        elif grouping_level == 'subject':
            items = df_plot[df_plot['spec1'] == sp]['subj1'].unique()
        elif grouping_level == 'pair':
            items = df_plot[df_plot['spec1'] == sp]['pair'].unique()
            
        n_items = len(items)
        
        # Calculate intensities
        if sp == 'human':
            intensities = [0.55] if n_items == 1 else np.linspace(0.35, 0.65, n_items)
        else:
            intensities = [0.7] if n_items == 1 else np.linspace(0.4, 0.95, n_items)
            
        # Assign colors
        for i, item in enumerate(items):
            if sp == 'marmo':
                palette[item] = plt.cm.Blues(intensities[i])
            elif sp == 'human':
                palette[item] = plt.cm.YlOrBr(intensities[i])
            elif sp == 'macaq':
                palette[item] = plt.cm.Reds(intensities[i])
            elif sp == 'infant':
                palette[item] = plt.cm.Greens(intensities[i])
            else:
                palette[item] = plt.cm.Greys(intensities[i])

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    filter_txt = "Within-Species" if within_species_only else "All Data"
    vid_txt = f" | Vid: {vid_filter}" if vid_filter else ""
    title_suffix = f" ({filter_txt}{vid_txt})"

    # Histogram
    sns.histplot(
        data=df_plot, x='dist_px', hue=hue_col, palette=palette,
        element='step', fill=False, stat='density', common_norm=False, 
        linewidth=linewidth, ax=axes[0], legend=False # Turn off first legend
    )
    axes[0].set_title(f"{title_prefix} Distance Histogram{title_suffix}")
    axes[0].set_xlabel("Distance (px)")

    # Cumulative
    sns.ecdfplot(
        data=df_plot, x='dist_px', hue=hue_col, palette=palette,
        linewidth=linewidth, ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix} Cumulative Distribution{title_suffix}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion")
    
    # --- SMART LEGEND FORMATTING ---
    n_categories = len(df_plot[hue_col].unique())
    
    # Auto-calculate columns based on how many lines there are
    ncol = 1
    if n_categories > 15:
        ncol = 3
    elif n_categories > 8:
        ncol = 2
        
    # Place inside the bottom right of the ECDF plot (usually empty space)
    sns.move_legend(
        axes[1], 
        "lower right", 
        title=hue_col.capitalize(), 
        ncol=ncol, 
        framealpha=0.9, # Mostly solid background
        fontsize='small' # Shrink font slightly if needed
    )
    
    plt.tight_layout()
    return fig;


'''

def plot_distance_distributions(df, within_species_only=False, vid_filter=None):
    df_plot = df.copy()
    
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if vid_filter is not None:
        df_plot = df_plot[df_plot['vid'] == vid_filter]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    df_plot['pair'] = df_plot['subj1'] + " & " + df_plot['subj2']
    
    # --- COLOR LOGIC ---
    unique_species = df_plot['spec1'].unique()
    pair_colors = {}
    
    for sp in unique_species:
        sp_pairs = df_plot[df_plot['spec1'] == sp]['pair'].unique()
        n_items = len(sp_pairs)
            
        for i, pair in enumerate(sp_pairs):
            if sp == 'marmo':
                val = 0.7 if n_items == 1 else np.linspace(0.4, 0.95, n_items)[i]
                pair_colors[pair] = plt.cm.Blues(val)
            elif sp == 'human':
                # Capped at 0.65 to strictly avoid the brown spectrum of YlOrBr
                val = 0.55 if n_items == 1 else np.linspace(0.35, 0.65, n_items)[i]
                pair_colors[pair] = plt.cm.YlOrBr(val) 
            elif sp == 'macaq':
                val = 0.7 if n_items == 1 else np.linspace(0.4, 0.95, n_items)[i]
                pair_colors[pair] = plt.cm.Reds(val)
            else:
                val = 0.7 if n_items == 1 else np.linspace(0.4, 0.95, n_items)[i]
                pair_colors[pair] = plt.cm.Greys(val)

    # -----------------------

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" | Filter: {'Within-Species' if within_species_only else 'All'} | Vid: {vid_filter if vid_filter else 'All'}"

    sns.histplot(
        data=df_plot, x='dist_px', hue='pair', palette=pair_colors,
        element='step', fill=False, stat='density', common_norm=False,  
        ax=axes[0], legend=False
    )
    axes[0].set_title(f"Distance Histogram{title_suffix}")
    axes[0].set_xlabel("Distance (px)")

    sns.ecdfplot(
        data=df_plot, x='dist_px', hue='pair', palette=pair_colors,
        ax=axes[1]
    )
    axes[1].set_title(f"Cumulative Distribution{title_suffix}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion")
    
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1), title="Subj1 & Subj2 Pair")
    
    plt.tight_layout()
    plt.show()
    return;



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_aggregated_distributions(df, aggregate_by='species', within_species_only=False):
    df_plot = df.copy()
    
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- COLOR LOGIC ---
    if aggregate_by == 'species':
        hue_col = 'spec1'
        title_prefix = "Species-Level"
        
        palette = {
            'marmo': plt.cm.Blues(0.7),
            'human': plt.cm.YlOrBr(0.55),  # 0.55 is a solid orange, no brown
            'macaq': plt.cm.Reds(0.7)
        }
        linewidth = 2.5 
        
    elif aggregate_by == 'subject':
        hue_col = 'subj1'
        title_prefix = "Subject-Level"
        
        palette = {}
        unique_species = df_plot['spec1'].unique()
        
        for sp in unique_species:
            sp_subjects = df_plot[df_plot['spec1'] == sp]['subj1'].unique()
            n_items = len(sp_subjects)
            
            for i, subj in enumerate(sp_subjects):
                if sp == 'marmo':
                    val = 0.7 if n_items == 1 else np.linspace(0.4, 0.95, n_items)[i]
                    palette[subj] = plt.cm.Blues(val)
                elif sp == 'human':
                    # Capped at 0.65 to strictly avoid browns
                    val = 0.55 if n_items == 1 else np.linspace(0.35, 0.65, n_items)[i]
                    palette[subj] = plt.cm.YlOrBr(val)
                elif sp == 'macaq':
                    val = 0.7 if n_items == 1 else np.linspace(0.4, 0.95, n_items)[i]
                    palette[subj] = plt.cm.Reds(val)
                else:
                    val = 0.7 if n_items == 1 else np.linspace(0.4, 0.95, n_items)[i]
                    palette[subj] = plt.cm.Greys(val)
                    
        linewidth = 1.5
    else:
        raise ValueError("aggregate_by must be either 'species' or 'subject'")
    # -----------------------

    title_suffix = f" (Within-Species Only)" if within_species_only else " (All Data)"

    sns.histplot(
        data=df_plot, x='dist_px', hue=hue_col, palette=palette,
        element='step', fill=False, stat='density', common_norm=False, 
        linewidth=linewidth, ax=axes[0]
    )
    axes[0].set_title(f"{title_prefix} Distance Histogram{title_suffix}")
    axes[0].set_xlabel("Distance (px)")

    sns.ecdfplot(
        data=df_plot, x='dist_px', hue=hue_col, palette=palette,
        linewidth=linewidth, ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix} Cumulative Distribution{title_suffix}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion")
    
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1), title=hue_col.capitalize())
    
    if axes[0].get_legend() is not None:
         axes[0].get_legend().remove()
    
    plt.tight_layout()
    plt.show()
    return;

'''


'''

def plot_distance_distributions(df, within_species_only=True, vid_filter=None):
    # Work on a copy to avoid modifying the original dataframe
    df_plot = df.copy()
    
    # 1. Apply user requested filters
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if vid_filter is not None:
        df_plot = df_plot[df_plot['vid'] == vid_filter]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    # 2. Create a unique identifier for the pair
    df_plot['pair'] = df_plot['subj1'] + " & " + df_plot['subj2']
    
    # 3. Setup the color palettes
    unique_species = df_plot['spec1'].unique()
    
    # Define base palettes for your species. 
    # We slice [3:] to drop the lightest, hard-to-see shades in the palette.
    palette_map = {
        'marmo': sns.color_palette("Blues", n_colors=15)[3:],   
        'human': sns.color_palette("YlOrBr", n_colors=15)[3:], # Yellow/Orange/Browns
        'macaq': sns.color_palette("Reds", n_colors=15)[3:]     
    }
    
    # Map each specific pair to a distinct hue of their species' color
    pair_colors = {}
    for sp in unique_species:
        # Get all unique pairs where species1 is this species
        sp_pairs = df_plot[df_plot['spec1'] == sp]['pair'].unique()
        
        # Get the color palette for this species (default to Greens if species isn't mapped)
        sp_colors = palette_map.get(sp, sns.color_palette("Greens", n_colors=15)[3:])
        
        # Distribute the hues among the subject pairs
        for i, pair in enumerate(sp_pairs):
            pair_colors[pair] = sp_colors[i % len(sp_colors)]

    # 4. Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" | Filter: {'Within-Spec' if within_species_only else 'All'} | Vid: {vid_filter if vid_filter else 'All'}"

    # Plot A: Standard Histogram 
    # stat='density' and common_norm=False normalizes each pair to a total area of 1
    sns.histplot(
        data=df_plot, 
        x='dist_px', 
        hue='pair', 
        palette=pair_colors,
        element='step',     # Outline steps rather than filled bars for readability
        fill=False, 
        stat='density', 
        common_norm=False,  
        ax=axes[0],
        legend=False
    )
    axes[0].set_title(f"Distance Histogram{title_suffix}")
    axes[0].set_xlabel("Distance (px)")

    # Plot B: Cumulative Histogram
    # ecdfplot automatically normalizes data to 1.0 (100%)
    sns.ecdfplot(
        data=df_plot, 
        x='dist_px', 
        hue='pair', 
        palette=pair_colors,
        ax=axes[1]
    )
    axes[1].set_title(f"Cumulative Distribution{title_suffix}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion")
    
    # Move the legend outside the plot so it doesn't cover your data
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1), title="Subj1 & Subj2 Pair")
    
    plt.tight_layout()
    plt.show()

    return;


def plot_aggregated_distributions(df, aggregate_by='species', within_species_only=False):
    """
    Plots aggregated data distributions.
    aggregate_by : str
        Either 'species' (1 line per species) or 'subject' (1 line per subject)
    """
    df_plot = df.copy()
    
    # Optional filter for within-species only
    if within_species_only:
        df_plot = df_plot[df_plot['spec1'] == df_plot['spec2']]
        
    if df_plot.empty:
        print("No data left after filtering!")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Setup Data Grouping & Colors based on user choice
    if aggregate_by == 'species':
        hue_col = 'spec1'
        title_prefix = "Species-Level"
        
        # Give each species 1 distinct, solid, dark color from their palette
        palette = {
            'marmo': sns.color_palette("Blues")[5],
            'human': sns.color_palette("YlOrBr")[5],
            'macaq': sns.color_palette("Reds")[5]
        }
        linewidth = 2.5 # Make lines thicker since there are fewer of them
        
    elif aggregate_by == 'subject':
        hue_col = 'subj1'
        title_prefix = "Subject-Level"
        
        # Base palettes mapping
        palette_map = {
            'marmo': sns.color_palette("Blues", n_colors=10)[3:],   
            'human': sns.color_palette("YlOrBr", n_colors=10)[3:], 
            'macaq': sns.color_palette("Reds", n_colors=10)[3:]     
        }
        
        # Map each individual subject to a slightly different shade of their species' color
        palette = {}
        unique_species = df_plot['spec1'].unique()
        
        for sp in unique_species:
            # Find all unique subjects belonging to this species
            sp_subjects = df_plot[df_plot['spec1'] == sp]['subj1'].unique()
            sp_colors = palette_map.get(sp, sns.color_palette("Greens", n_colors=10)[3:])
            
            for i, subj in enumerate(sp_subjects):
                palette[subj] = sp_colors[i % len(sp_colors)]
                
        linewidth = 1.5
    else:
        raise ValueError("aggregate_by must be either 'species' or 'subject'")

    title_suffix = f" (Within-Species Only)" if within_species_only else " (All Data)"

    # Plot A: Standard Histogram (Pooled Distribution)
    sns.histplot(
        data=df_plot,
        x='dist_px',
        hue=hue_col,
        palette=palette,
        element='step',
        fill=False,
        stat='density',
        common_norm=False, 
        linewidth=linewidth,
        ax=axes[0],
        
    )
    axes[0].set_title(f"{title_prefix} Distance Histogram{title_suffix}")
    axes[0].set_xlabel("Distance (px)")

    # Plot B: Cumulative Histogram (Pooled Distribution)
    sns.ecdfplot(
        data=df_plot,
        x='dist_px',
        hue=hue_col,
        palette=palette,
        linewidth=linewidth,
        ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix} Cumulative Distribution{title_suffix}")
    axes[1].set_xlabel("Distance (px)")
    axes[1].set_ylabel("Cumulative Proportion")
    
    # Fix the Legend
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1), title=hue_col.capitalize())
    
    # For histograms, sometimes seaborn duplicates the legend, this removes the extra one:
    if axes[0].get_legend() is not None:
         axes[0].get_legend().remove()
    
    plt.tight_layout()
    plt.show()
    return;
'''


#plot_aggregated_distributions(df, aggregate_by='species', within_species_only=True);

#plot_aggregated_distributions(df, aggregate_by='species', within_species_only=False);

#plot_distance_distributions(df, within_species_only=True);
