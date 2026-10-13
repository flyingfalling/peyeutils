import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('bigsmall_bigsampledf.csv');

print(df.groupby('xy')['xyvalue'].quantile([0.05, 0.25, 0.50, 0.75, 0.95]))
print(df.columns);

#REV: set all values > 1.25 of video size to nan.
CUTOFF=1.25;
df.loc[ abs(df['xyvalue']) > CUTOFF, 'xyvalue' ] = np.nan

#REV: make displot of cumulative distr function
g = sns.displot(data=df, x='xyvalue', col='xy', kind='ecdf', hue='vidsize')
g = g.add_legend()

g.fig.savefig('/mnt/coishare/data/bigsmalldistr.pdf')
