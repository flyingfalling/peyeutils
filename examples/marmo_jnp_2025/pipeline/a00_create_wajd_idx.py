## Filler, see pymarmostim on server.


## This will actually just check some stats


import pandas as pd
import sys
import os

#REV
df = pd.read_csv( sys.argv[1] );

df = df[ df.yesOpto == False ];
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
