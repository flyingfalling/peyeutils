

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
    widxdf['species']='marmo';
    print(widxdf.vid.unique());
    print(widxdf.species.unique());
    
    oidxdf=pd.read_csv(orig_idx);
    oidxdf['wotype']='o';
    oidxdf['trialidx'] = oidxdf['trialidx'].astype(str) + 'o'
    print(oidxdf.vid.unique());
    print(oidxdf.species.unique());
    
    wgazedf=pd.read_csv(wajd_gaze);
    wgazedf['wotype']='w';
    wgazedf['trialidx']= wgazedf['trialidx'].astype(str) + 'w';
    
    ogazedf=pd.read_csv(orig_gaze);
    ogazedf['wotype']='o';
    ogazedf['trialidx']= ogazedf['trialidx'].astype(str) + 'o';
    
    idxdf = pd.concat([widxdf, oidxdf]);

    idxdf['species']=idxdf['species'].astype(str);
    print(idxdf[idxdf['species']=='nan']);
    
    
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

            #print(tdf.movie_ts.diff().unique());
            if( np.any(tdf.movie_ts.duplicated()) ):
                print("DUPLICATES");
                print(tdf[ tdf.movie_ts.duplicated(keep=False) ][['timems', 'movie_ts', 'pix_x', 'pix_y', 'eyelink_ts']]);
                pass;
            pass;
        
        pass;

    ################# CORRELATIONS ##################
    #REV: compute pairwise comparisons of the upper triangle. For now just take evenly spaced and interpolate.

    #REV: null model is correlation with OTHER videos?!
    ## With random shuffle from prior?
    
    
    
    
    bigdf = pd.merge(left=gazedf, right=idxdf, on='trialidx');
    print(len(bigdf.index));
    bigdf = bigdf.sort_values(by=['trialidx', 'movie_ts']).reset_index(drop=True);
    
    ##REV: todo "plot" and show CC?
    
    nrow=len( idxdf.vid.unique() );
    rowhei=4;
    rowwid=8;
    
    ncol=len( idxdf.species.unique() );
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(rowwid*ncol, rowhei*nrow), sharey=True, sharex=True);
    ax=0;
    
    print("Unique species: {}".format( idxdf.species.unique()));
    
    
    
    
    
    for v, vdf in idxdf.groupby('vid'):
        
        print("DOING for [{}]".format(v));
        trials = vdf['trialidx'].to_numpy();
        #print(vdf);
        print("Legal trials: {}".format(trials));
        
        #print(gazedf.trialidx);
        #print(gazedf['trialidx'].isin(trials) )
        #vgazedf = gazedf.loc[ gazedf['trialidx'].isin(trials) ];
        vgazedf = bigdf.loc[ bigdf['trialidx'].isin(trials) ];
        #print(vgazedf);

        
        for i, (spec, specdf) in enumerate(vgazedf.groupby('species')):
            print("Species: {}".format(spec));
            #REV: ah no error estimator -> Fast
            sns.lineplot(data=specdf, x='movie_ts', y='pix_x', hue='subj', style='trialidx', lw=0.2, ax=axs[ax][i]);
            pass;
        axs[ax][i].set_ylim([-400,400]);
        
        
        ax+=1;
        
            
        dictlist=list();
        for tidx1, tdf1 in vgazedf.groupby('trialidx'):
            subj1=idxdf[ idxdf.trialidx==tidx1 ].iloc[0].subj;
            spec1=idxdf[ idxdf.trialidx==tidx1 ].iloc[0].species;
                    
            for tidx2, tdf2 in vgazedf.groupby('trialidx'):
                if(tidx1 > tidx2):
                    subj2=idxdf[ idxdf.trialidx==tidx2 ].iloc[0].subj;
                    spec2=idxdf[ idxdf.trialidx==tidx2 ].iloc[0].species;
                    
                    mylen=np.min([len(tdf1.index), len(tdf2.index)]);
                    tdf1=tdf1.iloc[:mylen].reset_index(drop=True);
                    tdf2=tdf2.iloc[:mylen].reset_index(drop=True); #pd corr uses index?
                    
                    
                    x1=tdf1.pix_x;
                    x2=tdf2.pix_x;
                    xcc = x1.corr( x2 ); #REV: these values are LOW. Anyways, plot them to see...
                    
                    y1=tdf1.pix_y;
                    y2=tdf2.pix_y;
                    ycc = y1.corr( y2 );
                    
                    #xcc = np.corrcoef(x1, x2)[0][0];
                    #ycc = np.corrcoef(y1, y2)[0][0];
                    xycc = (xcc+ycc)/2;
                    dictlist.append( dict(npts=mylen, vid=v, t1=tidx1, t2=tidx2, xcc=xcc, ycc=ycc, xycc=xycc, subj1=subj1, subj2=subj2, spec1=spec1, spec2=spec2) );
                    print("CC: {} {} {}".format(xcc,ycc,xycc));
                    pass;
                pass;
            pass;
        pass;

    fig.savefig('manualplot.pdf');
    
    corrdf = pd.DataFrame( dictlist );
    corrdf.to_csv('gazecoors.csv', index=False);
    
    ################# SALMAP AUROC ##################
    
    
    
    #################################################
    
    '''
    print(len(gazedf.index));
    g = sns.relplot( data=bigdf, row='vid', col='species', hue='subj', x='movie_ts', y='pix_x', kind='line', linewidth=0.2, style='trialidx'); #kind="scatter", s=0.2)
    #plt.show();
    g.figure.savefig('myfig.png');
    '''
    
    return 0;


if __name__=='__main__':
    exit(main());
