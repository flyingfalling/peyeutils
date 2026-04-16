## Filler, see pymarmostim on server.


## This will actually just check some stats


import pandas as pd
import sys
import os

wajd_idx = sys.argv[1];
wajdmuscimol_idx = sys.argv[2];


df = pd.read_csv( wajd_idx );
mdf = pd.read_csv( wajdmuscimol_idx );

df['date'] = pd.to_datetime(df.date);
mdf['date'] = pd.to_datetime(mdf.date);
#print(df.date.dtype);

print(df.columns);
print(mdf.columns);

df = pd.merge(left=df, left_on=['date', 'trialcsv', 'subj'], right=mdf, right_on=['date', 'trialcsv', 'subj'], how='outer' ); #REV: both outer, inner, and left should be same
#print(df);
print(df.columns);



df = df[ (df.yesOpto == False) & (df.muscimol==False) ];
df['vid'] = df.vid.str[:-4];
print(df.vid);
result = df.groupby(['subj', 'vid']).size();

pd.set_option('display.max_rows', None)
print(result);


## Wow, wajd marmo saw each video like 20 times!!!!
###  I could do "within-subject" -- problem is that I should do
###  "per-video" but only on videos they have seen minimum...3x? 2x?

## With cetain number of "good" timepoints?

## if it is only one biased video, and that video is different between species
## then what is the point?
