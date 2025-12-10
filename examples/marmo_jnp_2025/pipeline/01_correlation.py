

import pandas as pd
import numpy as np;
import sys
import os;
import seaborn as sns;
import matplotlib.pyplot as plt;


def main():
    wajd_idx=sys.argv[1];
    wajd_gaze=sys.argv[2];
    orig_idx=sys.argv[3];
    orig_gaze=sys.argv[4];


    widxdf=pd.read_csv(wajd_idx);
    widxdf['wotype']='w';
    widxdf['trialidx'] = widxdf['trialidx'].astype(str) + 'w'
    widxdf['vid'] = widxdf['vid'].str[:-4];
    print(widxdf.vid.unique());
    
    oidxdf=pd.read_csv(orig_idx);
    oidxdf['wotype']='o';
    oidxdf['trialidx'] = oidxdf['trialidx'].astype(str) + 'o'
    print(oidxdf.vid.unique());
    
    wgazedf=pd.read_csv(wajd_gaze);
    wgazedf['wotype']='w';
    wgazedf['trialidx']= wgazedf['trialidx'].astype(str) + 'w';
    
    ogazedf=pd.read_csv(orig_gaze);
    ogazedf['wotype']='o';
    ogazedf['trialidx']= ogazedf['trialidx'].astype(str) + 'o';
    
    idxdf = pd.concat([widxdf, oidxdf]);
    gazedf = pd.concat([wgazedf, ogazedf]);
    print("BEFORE DROP: ", gazedf.shape);
    gazedf = gazedf.dropna(subset=['timems']);
    gazedf = gazedf.dropna(subset=['movie_ts']);
    gazedf = gazedf[ gazedf.movie_ts >= 0 ];
    print("AFTER DROP: ", gazedf.shape);
    #gazedf['trialidx'] = gazedf['trialidx'].astype('category');
    #gazedf['wotype'] = gazedf['wotype'].astype('category');
    idxdf['vid'] = idxdf['vid'].astype(str);
    print(idxdf[ idxdf['vid']=='nan' ]);
    print(idxdf.columns);
    print(gazedf.columns);
    print(len(idxdf.vid.unique()));
    group_col = ['movie_ts', 'trialidx'];
    print("GDF",gazedf.columns);
    agg_columns = gazedf.columns.difference(group_col)
    numeric_cols = gazedf[agg_columns].select_dtypes(include=[np.number]).columns; #.drop(group_col, errors='raise').tolist()
    non_numeric_cols = gazedf[agg_columns].select_dtypes(exclude=[np.number]).columns; #.drop(group_col, errors='raise').tolist()
    agg_dict = {}
    # Add numeric columns to the dictionary with 'mean' function
    for col in numeric_cols:
        agg_dict[col] = 'mean'
        pass;
    # Add non-numeric columns to the dictionary with 'first' function
    for col in non_numeric_cols:
        agg_dict[col] = 'first'
        pass;
    print(agg_dict);
    
    gazedf = gazedf.groupby(group_col).agg(agg_dict).reset_index(); #mean(numeric_only=True);
    gazedf = gazedf.reset_index(drop=True);
    print("AFTER COMBINE");
    print(gazedf.columns);
    
    for v, vdf in idxdf.groupby('vid'):
        print("DOING for [{}]".format(v));
        trials = vdf['trialidx'].to_numpy();
        #print(vdf);
        print("Legal trials: {}".format(trials));
        
        #print(gazedf.trialidx);
        #print(gazedf['trialidx'].isin(trials) )
        vgazedf = gazedf.loc[ gazedf['trialidx'].isin(trials) ];
        #print(vgazedf);
        
        for tidx, tdf in vgazedf.groupby('trialidx'):
            
            print("Trial [{}] has {} timepoints ({}-{}) ({}-{})".format(tidx, len(tdf.index), tdf['timems'].min(), tdf['timems'].max(), tdf['movie_ts'].min(), tdf['movie_ts'].max()) );

            print(tdf.movie_ts.diff().unique());
            if( np.any(tdf.movie_ts.duplicated()) ):
                print("DUPLICATES");
                print(tdf[ tdf.movie_ts.duplicated(keep=False) ][['timems', 'movie_ts', 'pix_x', 'pix_y', 'eyelink_ts']]);
                pass;
            pass;
        
        pass;
    
    
    print(len(gazedf.index));
    bigdf = pd.merge(left=gazedf, right=idxdf, on='trialidx');
    print(len(bigdf.index));
    g = sns.relplot( data=bigdf, row='vid', col='species', hue='subj', x='movie_ts', y='pix_x', kind="line", linewidth=0.5, marker=None)
    #plt.show();
    g.figure.savefig('myfig.pdf');
    
    
    return 0;


if __name__=='__main__':
    exit(main());
