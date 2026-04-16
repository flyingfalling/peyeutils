import pandas as pd
import sys;


fn = sys.argv[1];

df = pd.read_csv(fn);
print(df);
print(df.columns);
print(df.date);
print(df.date.dtype);
df['date'] = pd.to_datetime(df['date']);
df = df.sort_values(by=['subj', 'date']).reset_index(drop=True);
print(df[['subj', 'date']]);
#for i, row in df.iterrows():
#    print(row

df[['subj', 'date', 'trialcsv']].to_csv('wajd_dates.csv', sep=' ', index=False);
