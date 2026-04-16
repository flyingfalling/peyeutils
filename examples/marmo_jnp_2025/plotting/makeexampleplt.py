import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# --- Configuration & Style Helper ---

# Define the subjects and their groups
subjects = [
    'Marmoset A', 'Marmoset B', 'Marmoset C', 'Marmo P', 'Marmo J',          # Marmosets
    'Macaque O', 'Macaque U',                                                # Macaques
    'Human C', 'Human D', 'Human S1', 'Human S2',                            # Original Humans
    'Human X1', 'Human X2', 'Human X3', 'Human X4', 'Human X5', 'Human X6'   # New Humans
]

# Define colors based on the image provided
# Marmosets: Blues/Cyans
# Macaques: Red/Browns
# Humans: Yellows/Oranges
colors = [
    '#0044cc', '#2266ee', '#66aaee', '#0088aa', '#00aabb', # Marmo colors (5)
    '#660000', '#ff0000',                                  # Macaque colors (2)
    '#bb8800', '#eebb00', '#ffaa66', '#ffcc88',            # Original Human colors (4)
    '#ffdd99', '#ffeeaa', '#ffbb77', '#ffaa55', '#ee9944', '#cc7733' # New Human colors (6)
]

def generate_distribution(median, spread=0.05, n=100):
    """Generates a random normal distribution centered roughly on the median."""
    # We generate normal data and shift it so the median matches exactly what we want
    data = np.random.normal(loc=median, scale=spread, size=n)
    current_median = np.median(data)
    diff = median - current_median
    return data + diff

def style_boxplot(ax, bplot, colors):
    """Applies the specific visual style from the uploaded image to a boxplot object."""
    # Matplotlib boxplots return a dictionary of lines. We iterate and color them.
    # The dictionary keys are 'boxes', 'whiskers', 'medians', 'caps', 'fliers'
    
    # We have N subjects, so we iterate through indices
    for i in range(len(bplot['boxes'])):
        color = colors[i]
        
        # Box (Outline only)
        bplot['boxes'][i].set_color(color)
        bplot['boxes'][i].set_linewidth(2)
        bplot['boxes'][i].set_facecolor('none') # Transparent fill
        
        # Medians
        bplot['medians'][i].set_color(color)
        bplot['medians'][i].set_linewidth(2)
        
        # Whiskers (2 per box)
        bplot['whiskers'][i*2].set_color(color)
        bplot['whiskers'][i*2+1].set_color(color)
        
        # Caps (2 per box)
        bplot['caps'][i*2].set_color(color)
        bplot['caps'][i*2+1].set_color(color)
        
        # Fliers (Outliers) - styled as '+'
        bplot['fliers'][i].set_markeredgecolor(color)
        bplot['fliers'][i].set_marker('+')
        bplot['fliers'][i].set_markersize(6)

def finalize_plot(ax, title, y_label="AUROC"):
    """Applies the axis spines, labels, and reference lines."""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel(y_label, fontsize=14)
    
    # Reference line at 0.5
    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=2)
    
    # X-axis formatting
    ax.set_xticks(range(1, len(subjects) + 1))
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=12)
    
    # Y-axis formatting
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='y', labelsize=12, width=2, length=6)
    
    # Despine (remove top and right borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

# --- 1. Generate the Main "AUROC for all videos" Plot ---

# Estimated medians from the original image + the user's specific new values
# Original estimates: A~0.61, B~0.60, C~0.59, MacO~0.64, MacU~0.63, HumC~0.63, HumD~0.61, S1~0.63, S2~0.64
# New Marmo estimates: P~0.59, J~0.605
# New Human estimates: Similar to existing humans (approx 0.61 - 0.64)
main_medians = [
    0.61, 0.60, 0.58, 0.59, 0.605,            # Marmosets
    0.64, 0.63,                               # Macaques
    0.63, 0.61, 0.63, 0.64,                   # Original Humans
    0.62, 0.635, 0.615, 0.64, 0.625, 0.63     # New Humans
]

# Create dataset
main_data = [generate_distribution(m, spread=0.06) for m in main_medians]

fig_main, ax_main = plt.subplots(figsize=(12, 6)) # Increased width for more subjects
bplot_main = ax_main.boxplot(main_data, patch_artist=True, widths=0.6, showfliers=True)

style_boxplot(ax_main, bplot_main, colors)
finalize_plot(ax_main, "AUROC for all videos")
plt.tight_layout()
plt.savefig('plot_main_modified.png', dpi=300)
plt.close() # Close to free memory

# --- 2. Generate the 5 Feature Plots ---

features = ["Motion", "Luminance", "Orientation", "Color", "Flicker"]

for feature in features:
    feature_data = []
    
    for i, subject in enumerate(subjects):
        # Base random generation: between 0.52 and 0.57
        # We'll treat this as a uniform range for the median, then build a distribution
        base_median = np.random.uniform(0.52, 0.57)
        
        # Apply Logic modifiers
        is_marmoset = "Marmo" in subject
        is_human_or_macaque = ("Human" in subject) or ("Macaque" in subject)
        
        final_median = base_median

        # Generate the distribution for this subject/feature
        # Using slightly tighter spread for features to make differences obvious
        dist = generate_distribution(final_median, spread=0.04)
        feature_data.append(dist)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6)) # Increased width for more subjects
    bplot = ax.boxplot(feature_data, patch_artist=True, widths=0.6, showfliers=True)
    
    style_boxplot(ax, bplot, colors)
    finalize_plot(ax, feature)
    
    plt.tight_layout()
    plt.savefig(f'plot_{feature.lower()}.png', dpi=300)
    plt.close()

print("All plots generated successfully with new subjects.")
