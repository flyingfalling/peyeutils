

import pandas as pd
import numpy as np;
import sys
import os;


def main():
    wajd_idx=sys.argv[1];
    wajd_gaze=sys.argv[2];
    orig_idx=sys.argv[3];
    orig_gaze=sys.argv[4];


    widxdf=pd.read_csv(wajd_idx);
    widxdf['wotype']='w';
    widxdf['trialidx'] = widxdf['trialidx'].astype(str) + 'w'
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
    #gazedf = gazedf[ gazedf.movie_ts >= 0 ];
    print("AFTER DROP: ", gazedf.shape);
    #gazedf['trialidx'] = gazedf['trialidx'].astype('category');
    #gazedf['wotype'] = gazedf['wotype'].astype('category');
    idxdf['vid'] = idxdf['vid'].astype(str);
    print(idxdf[ idxdf['vid']=='nan' ]);
    print(idxdf.columns);
    print(gazedf.columns);
    print(len(idxdf.vid.unique()));
    
    for v, vdf in idxdf.groupby('vid'):
        print("DOING for [{}]".format(v));
        trials = vdf['trialidx'].to_numpy();
        print(vdf);
        print("Legal trials: {}".format(trials));
        
        print(gazedf.trialidx);
        print(gazedf['trialidx'].isin(trials) )
        vgazedf = gazedf.loc[ gazedf['trialidx'].isin(trials) ];
        print(vgazedf);
        
        for tidx, tdf in vgazedf.groupby('trialidx'):
            print("Trial [{}] has {} timepoints ({}-{}) ({}-{})".format(tidx, len(tdf.index), tdf['timems'].min(), tdf['timems'].max(), tdf['movie_ts'].min(), tdf['movie_ts'].max()) );
            pass;
        
        pass;
    
    return 0;


if __name__=='__main__':
    exit(main());
