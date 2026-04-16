

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
    
    #REV: remove wajd trials with muscimol or opto stim...
    widxdf = widxdf[ (widxdf['yesOpto'] == False) &
                     (widxdf['muscimol'] == False)
                     ];
        
    widxdf['wotype']='w';
    widxdf['trialidx'] = widxdf['trialidx'].astype(str) + 'w'
    widxdf['vid'] = widxdf['vid'].str[:-4];
    widxdf['species']='marmo';
    print(widxdf.vid.unique());
    print(widxdf.species.unique());
    
    oidxdf=pd.read_csv(orig_idx);
    oidxdf['wotype']='o';
    oidxdf['trialidx'] = oidxdf['trialidx'].astype(str) + 'o'

    '''
    print("WAJD");
    print(widxdf.iloc[0]);
    
    print("CHEN");
    print(oidxdf.iloc[0]);

    #REV: should rename...
    exit(0);
    '''
    
    
    
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
                raise Exception("Duplicates, should never happen!");
            
            pass;
        
        pass;
    
    
        
    ################# CORRELATIONS ##################
    #REV: compute pairwise comparisons of the upper triangle. For now just take evenly spaced and interpolate.
    
    #REV: null model is correlation with OTHER videos?!
    ## With random shuffle from prior?
    
    
    
    
    bigdf = pd.merge(left=gazedf, right=idxdf, on=['trialidx', 'wotype'], how='left');
    print(len(bigdf.index));
    bigdf = bigdf.sort_values(by=['trialidx', 'movie_ts']).reset_index(drop=True);
    print(bigdf.subj.unique());
    #bigdf.groupby(['subj']).count().to_csv('wtf.csv');
    
    #REV: clean data?
    bigdf.loc[ ( (bigdf.pix_x > 400) | (bigdf.pix_x < -400) |
                 (bigdf.pix_y > 400) | (bigdf.pix_y < -400) ),
               ['pix_x', 'pix_y'] ] = np.nan;
    
    bigdf.to_csv('bigdf.csv', index=False);
    exit(0);
    ##REV: todo "plot" and show CC?

    DOPLOT=False;
    DOCORR=False;
    
    if(DOPLOT):
        nrow=len( idxdf.vid.unique() );
        rowhei=4;
        rowwid=8;
        
        ncol=len( idxdf.species.unique() );
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(rowwid*ncol, rowhei*nrow), sharey=True, sharex=True);
        ax=0;
        pass;
    
    print("Unique species: {}".format( idxdf.species.unique()));
    
    
    #REV: make pairwise distance plots here too.
    #REV: I could do "groupby", but better to do for each timepoint (in each video), subtract distance from all other timepoints
    #REV: in pairwise manner...huge. Note within vid of course.
    
    #REV: need to ensure "number of timepoints" is similar? Or "distance" is kind of pointless on a per-video thing.
    
        
    corrlist=list();
    distlist=list();
    nulllist=list();
    ncorrlist=list();
    
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
        

        if(DOPLOT):
            for i, (spec, specdf) in enumerate(vgazedf.groupby('species')):
                print("Species: {}".format(spec));
                #REV: ah no error estimator -> Fast
                sns.lineplot(data=specdf, x='movie_ts', y='pix_x', hue='subj', style='trialidx', lw=0.2, ax=axs[ax][i]);
                pass;
            axs[ax][i].set_ylim([-400,400]);
            ax+=1;

            pass;
                
            
        
        for tidx1, tdf1 in vgazedf.groupby('trialidx'):
            subj1=idxdf[ idxdf.trialidx==tidx1 ].iloc[0].subj;
            spec1=idxdf[ idxdf.trialidx==tidx1 ].iloc[0].species;
                    
            for tidx2, tdf2 in vgazedf.groupby('trialidx'):
                if(tidx1 > tidx2):
                    subj2=idxdf[ idxdf.trialidx==tidx2 ].iloc[0].subj;
                    spec2=idxdf[ idxdf.trialidx==tidx2 ].iloc[0].species;
                    
                    
                    
                    #REV: should not interpolate across NAN times...
                    tdf = pd.merge(left=tdf1, left_on='movie_ts',
                                   right=tdf2, right_on='movie_ts',
                                   how='outer',
                                   suffixes=('_1', '_2')
                                   ).reset_index(drop=True)[['movie_ts', 'pix_x_1', 'pix_y_1', 'pix_x_2', 'pix_y_2']];
                    
                    tdf = tdf.sort_values(by='movie_ts').reset_index(drop=True);
                    tdf = tdf.interpolate(method='linear', limit=1); #REV: limit 1 nan filled.

                    mylen=len(tdf.index);
                    
                    tdf1 = tdf[['movie_ts', 'pix_x_1', 'pix_y_1']].reset_index(drop=True).rename(columns={'pix_x_1':'pix_x','pix_y_1':'pix_y'});
                    tdf2 = tdf[['movie_ts', 'pix_x_2', 'pix_y_2']].reset_index(drop=True).rename(columns={'pix_x_2':'pix_x','pix_y_2':'pix_y'});
                    toshuffle=['pix_x', 'pix_y'];
                    
                    ntdf1 = tdf1.copy();
                    ntdf1[toshuffle] = ntdf1[toshuffle].sample(frac=1).values;
                    
                    #df[cols_to_shuffle] = df[cols_to_shuffle].sample(frac=1).values
                    #tdf1=tdf1.iloc[:mylen].reset_index(drop=True);
                    #tdf2=tdf2.iloc[:mylen].reset_index(drop=True); #pd corr uses index?
                    
                    #tdf1['t'] = tdf2.movie_ts.values;
                    
                    tdiff = np.sum(abs(tdf1.movie_ts - tdf2.movie_ts));
                    if(tdiff != 0):
                        print( tdf1[ (tdf1.movie_ts - tdf1.t) != 0 ][['movie_ts', 't']] );
                        raise Exception("TDIFF not zero {}".format(tdiff));
                    
                    x1=tdf1.pix_x;
                    x2=tdf2.pix_x;
                    xcc = x1.corr( x2 ); #REV: these values are LOW. Anyways, plot them to see...
                    
                    y1=tdf1.pix_y;
                    y2=tdf2.pix_y;
                    ycc = y1.corr( y2 );

                    nxcc = x2.corr( ntdf1.pix_x );
                    nycc = y2.corr( ntdf1.pix_y );
                    nxycc = (nxcc+nycc)/2;
                    
                    #REV: should randomly sample N times and take mean dist? Mean of each timepoint? Will approach the mean dist.
                    #REV: right, problem is mean distance is different...how about mean and stddev of X/Y?
                    null_noverlap=(np.isfinite(x2) & np.isfinite(ntdf1.pix_x)).sum();
                    
                    ntps1 = np.isfinite(x1).sum();
                    ntps2 = np.isfinite(x2).sum();
                    #minpts=np.min([ntps1, ntps2]);
                    
                    noverlap = (np.isfinite(x1) & np.isfinite(x2)).sum();
                    
                    #xcc = np.corrcoef(x1, x2)[0][0];
                    #ycc = np.corrcoef(y1, y2)[0][0];
                    xycc = (xcc+ycc)/2;

                    corrlist.append( dict(npts=mylen, vid=v, t1=tidx1, t2=tidx2, xcc=xcc, ycc=ycc, xycc=xycc, subj1=subj1, subj2=subj2, spec1=spec1, spec2=spec2, ntps1=ntps1, ntps2=ntps2, overlap=noverlap) );

                    ncorrlist.append( dict(npts=mylen, vid=v, t1=tidx1, t2=tidx2, xcc=nxcc, ycc=nycc, xycc=nxycc, subj1=subj1, subj2=subj2, spec1=spec1, spec2=spec2, ntps1=ntps1, ntps2=ntps2, overlap=null_noverlap) );
                    
                    print("{}:{} ({}/{}={:3.1f}%)  -  {}:{} ({}/{}={:3.1f}%)  OVERLAP:{}/{}    X: {:3.2f}  Y: {:3.2f}   XY: {:3.2f}".format(spec1,subj1,ntps1,mylen,100*ntps1/mylen,spec2,subj2,ntps2,mylen, 100*ntps2/mylen, noverlap,mylen, xcc, ycc, xycc));


                    #REV: compute distance too.

                    #REV: figure out timepoints (and videos) for which we are "far apart" versus "close together" and figure out WHY
                    # (what kind of stimuli? What videos?).
                    ## Show videos of "most similar" or "most different" eye movements? (separate by words/faces...?)

                    #REV: we need to assume that pixel size is full (i.e. it's effectively normalized). Videos are NEARLY same size in pixels
                    # Although DVA is significantly different...
                    pxdist=np.sqrt( (tdf1.pix_x-tdf2.pix_x)**2 +
                                    (tdf1.pix_y-tdf2.pix_y)**2); #REV: chunk in 5 px distances? Or, save per ?

                    npxdist = np.sqrt( (ntdf1.pix_x-tdf2.pix_x)**2 +
                                    (ntdf1.pix_y-tdf2.pix_y)**2); #REV: chunk in 5 px distances? Or, save per ?
                    #REV: then need to "merge" add one for each, copying the others down...
                    mydict = dict(subj1=subj1,
                                  subj2=subj2,
                                  spec1=spec1,
                                  spec2=spec2,
                                  vid=v,
                                  t1=tidx1,
                                  t2=tidx2,
                                  );
                    #print(tdf1.columns);
                    nddf = pd.DataFrame( { 'dist_px':npxdist, 'movie_ts':tdf.movie_ts } );
                    nddf = nddf.assign( **mydict );
                    nulllist.append(nddf);
                    
                    ddf = pd.DataFrame( { 'dist_px':pxdist, 'movie_ts':tdf.movie_ts } );
                    ddf = ddf.assign( **mydict );
                    distlist.append(ddf);
                    pass;
                pass;
            pass;
        pass;

    if(DOPLOT):
        fig.savefig('manualplot.pdf');
        pass;
    
    corrdf = pd.DataFrame( corrlist );
    corrdf.to_csv('gazecorrs.csv', index=False);

    ncorrdf = pd.DataFrame( ncorrlist );
    ncorrdf.to_csv('nullcorrs.csv', index=False);
    
    distdf = pd.concat( distlist );
    distdf.to_csv('gazedists.csv', index=False);
    
    nulldf = pd.concat( nulllist );
    nulldf.to_csv('nulldists.csv', index=False);

    
    
    
    

    ## 4D cube, of distance of source/dest for each distance.
    # I need a plot of distribution distance...for each video?

    # is each video quite different?

    
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
