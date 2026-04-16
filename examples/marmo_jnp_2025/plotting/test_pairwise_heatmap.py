import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

import sys;

# 1. Load the data
# (Using the csv data you provided as a string for this example)
df = pd.read_csv(sys.argv[1]);


# 2. Preprocessing
# Create a combined label "Species: Subject" for clearer plotting
df['id1'] = df['spec1'] + ": " + df['subj1']
df['id2'] = df['spec2'] + ": " + df['subj2']

# Sort the pairs within each row to ensure (A, B) is treated same as (B, A)
# This handles the directionality so we can aggregate all runs between two individuals
sorted_ids = np.sort(df[['id1', 'id2']].values, axis=1)
df['row_id'] = sorted_ids[:, 0]
df['col_id'] = sorted_ids[:, 1]

# 3. Aggregation
# Calculate Mean and Std Dev for the 'xycc' correlation column
agg_df = df.groupby(['row_id', 'col_id'])['xycc'].agg(['mean', 'std']).reset_index()

# Handle cases where there is only 1 run (std will be NaN), replace with 0 for display
agg_df['std'] = agg_df['std'].fillna(0)

# 4. Matrix Creation (Pivot)
# Create matrices for Mean (for color) and Annotation (for text)
matrix_mean = agg_df.pivot(index='row_id', columns='col_id', values='mean')
matrix_std = agg_df.pivot(index='row_id', columns='col_id', values='std')

# Re-sort index and columns to ensure they match and look orderly
all_ids = sorted(list(set(df['row_id']).union(set(df['col_id']))))
matrix_mean = matrix_mean.reindex(index=all_ids, columns=all_ids)
matrix_std = matrix_std.reindex(index=all_ids, columns=all_ids)

# Create a mask for the lower triangle (keep upper half + diagonal)
# np.tril creates a mask for the lower triangle. 
# We set k=-1 to keep the diagonal visible, or k=0 to hide diagonal. 
# Usually self-correlation (diagonal) is useful, so we hide strictly below diagonal.
mask = np.tril(np.ones_like(matrix_mean, dtype=bool), k=-1)

# Create custom labels: "Mean \n +/- SD"
# We format them to 2 decimal places
annot_labels = (matrix_mean.round(2).astype(str) + 
                "\n" + r"$\pm$" + 
                matrix_std.round(2).astype(str))

# Handle NaN in labels (where no data exists for that specific pair in the pivot)
annot_labels = annot_labels.replace('nan\n$\pm$nan', '')

# 5. Plotting
plt.figure(figsize=(10, 8))

# Use a diverging colormap (coolwarm) centered at 0
sns.heatmap(matrix_mean, 
            mask=mask, 
            annot=annot_labels, 
            fmt='', # interpreting labels as raw strings
            cmap='coolwarm', 
            center=0, 
            vmin=-1, vmax=1,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5, "label": "Mean Correlation (xycc)"})

plt.title('Mean Correlations between Individuals (by Species)\n(Values shown: Mean $\pm$ SD)')
plt.xlabel('') # Hide x label as it's redundant with y
plt.ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
