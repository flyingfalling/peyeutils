import pandas as pd
import seaborn as sns;
import numpy as np
import matplotlib.pyplot as plt;

import sys

incsv = sys.argv[1];
df = pd.read_csv(incsv);
#df = df.dropna(subset=['pix_x']); #REV: is this causing it?
df = df[['pix_x', 'pix_y', 'species', 'subj', 'vid', 'movie_ts', 'trialidx']];
df = pd.melt( df, id_vars=['species', 'subj', 'vid', 'movie_ts', 'trialidx'],
              value_vars=['pix_x', 'pix_y'], var_name='axis', value_name='value' );
sns.displot( data=df, x='value', kind='hist', hue='species', col='axis', element='step', stat='probability', fill=False, common_norm=False );

plt.savefig('spatial_density.pdf');

