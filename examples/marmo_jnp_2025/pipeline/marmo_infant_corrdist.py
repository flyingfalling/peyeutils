import sys;
import os;
import numpy as np;
import pandas as pd;


def make_bigdf(widxdf,wgazedf,oidxdf,ogazedf,savebigdf=True):

    winfomap = widxdf.drop_duplicates('trialidx').set_index('trialidx')[['Eyetracking_mode','subj','vid']];
    wgazedf = wgazedf.merge(winfomap, on='trialidx', how='left');
    wgazedf['species'] = 'marmo';

    wgazedf=wgazedf[wgazedf['Eyetracking_mode']=='Binocular'].copy();

    wgazedf=wgazedf[['trialidx','subj','vid','species','timems','pix_x','pix_y','dva_x','dva_y']].copy();


    oinfomap = oidxdf.drop_duplicates('trialidx').set_index('trialidx')[['binocularmode','subj','vid','species']];
    ogazedf = ogazedf.merge(oinfomap, on='trialidx', how='left');
    
    #Few samples in ogazedf in binocular mode...
    #ogazedf=ogazedf[ogazedf['binocularmode']=='Binocular'].copy();

    ogazedf=ogazedf[['trialidx','subj','vid','species','timems','pix_x','pix_y','dva_x','dva_y']].copy();

    bigdf=pd.concat([wgazedf,ogazedf],axis=0).reset_index(drop=True);


    marmo_movie_settings=pd.DataFrame([{'species':'marmo','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':24.1,'vid_dva_y':17.6},
                             {'species':'macaq','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':17.0,'vid_dva_y':12.4},
                             {'species':'human','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':21.7,'vid_dva_y':15.8}]);

    bigdf=bigdf.merge(marmo_movie_settings, on='species', how='left');


    bigdf['xcnorm01']=bigdf['pix_x']/(bigdf['vid_pix_x']/2);
    bigdf['ycnorm01']=bigdf['pix_y']/(bigdf['vid_pix_y']/2);

    bigdf.loc[bigdf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
    bigdf.loc[bigdf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

    bigdf=bigdf.melt(id_vars=['trialidx','subj','vid','species','timems']
                       ,value_vars=['xcnorm01','ycnorm01'],var_name='xy',value_name='xyvalue');
    
    if savebigdf:
        bigdf.to_csv('marmo_biggazedf.csv');

    return bigdf;


def align_time(bigdf,sample_hz,relative= True):

    #unify the time and downsample if necessary
    
    if relative:
        bigdf["movie_ts"] = bigdf["movie_ts"] - bigdf.groupby('trialidx',as_index=False)["movie_ts"].transform('min');
    
    bigdf["time"] = np.round(bigdf["movie_ts"]*sample_hz);
    bigdf=bigdf.drop(columns='movie_ts');
    
    bigdf=bigdf.groupby(['trialidx','time','xy','subj','vid','species'], as_index=False)['xyvalue'].mean();

    return bigdf;


def _corr_one(w, key, min_periods=2):
    #REV you only keep timepoints shared by ALL trials (any trial ever has an NAN at timepoint=3, no pair ever uses it)
    #REV: this may reduce the actual number of timepoints used for correlation to a very small number.
    #REV: can you add here also the number of remaining non-nan rows ("percent-data")
    nonan =  w.dropna(axis=1, how='all');
    nonan_ratio =  nonan.shape[0]/w.shape[0];
    c = nonan.corr(min_periods=min_periods); 
    iu = np.triu_indices(c.shape[0], k=1);
    return pd.DataFrame({key+'1': c.index[iu[0]], key+'2': c.columns[iu[1]],
                         'corr': c.to_numpy()[iu], 'nonan_ratio': nonan_ratio});


def pairwise_corr(bigdf, group, key, values, min_periods=2):
    wide = bigdf.pivot_table(index=group+['time', 'xy'], columns=key, values=values);
    corrdf = (wide.groupby(level=group+['xy'])
                  .apply(_corr_one, key=key, min_periods=min_periods)
                  .reset_index(level=-1, drop=True)
                  .set_index([key+'1', key+'2'], append=True)['corr']
                  .unstack('xy'));
    corrdf.columns = 'corr_' + corrdf.columns;
    corrdf['corr_xy'] = corrdf.mean(axis=1);
    return corrdf;


def _dist_one(w,key, min_periods=2):

    #extract valid x and y
    x = w['xcnorm01'].dropna(axis=1, how='all');
    y = w['ycnorm01'].reindex(columns=x.columns);

    # 1 if x and y are not na
    M = (x.notna() & y.notna()).to_numpy(float);
    X = np.nan_to_num(x.to_numpy()) * M;
    Y = np.nan_to_num(y.to_numpy()) * M;
    Q = X**2 + Y**2;

    # S[i, j] = Σ{(xi−xj)² + (yi−yj)²} when the trials are valid
    S = Q.T @ M + M.T @ Q - 2 * (X.T @ X + Y.T @ Y);

    # distances should be positive 
    S = np.clip(S, 0, None);

    # the number of shared samples
    N = M.T @ M;

    iu = np.triu_indices(x.shape[1], k=1);
    out = pd.DataFrame({key+'1': x.columns[iu[0]], key+'2': x.columns[iu[1]],
                        'n': N[iu].astype(int), 'dist': np.sqrt(S[iu])});
    out.loc[out['n'] < min_periods, 'dist'] = np.nan;

    #normalized by the number of samples
    out['rms'] = out['dist'] / np.sqrt(out['n']);
    return out;


def pairwise_dist(bigdf,group,key,values):

    wide = bigdf.pivot_table(index=group+['time'], columns=['xy', key], values=values);
    distdf = (wide.groupby(level=group).apply(_dist_one, key=key)
                .reset_index(level=-1, drop=True)
                .set_index([key+'1', key+'2'], append=True));

    return distdf;


def subj_corrdist(bigdf):

    corrdf=pairwise_corr(bigdf, group=['vid'], key="trialidx",values="xyvalue");

    print('correlation was calculated!');

    distdf=pairwise_dist(bigdf,group=['vid'], key="trialidx",values="xyvalue");

    print('distance was calculated!');

    corrdf=corrdf.reset_index();

    corrdistdf = corrdf.merge(distdf, on=['vid','trialidx1','trialidx2'], how='outer');

    subjmap = bigdf.drop_duplicates('trialidx').set_index('trialidx')['subj'];
    corrdistdf['subj1'] = corrdistdf['trialidx1'].map(subjmap);
    corrdistdf['subj2'] = corrdistdf['trialidx2'].map(subjmap);

    specmap = bigdf.drop_duplicates('trialidx').set_index('trialidx')['species'];
    corrdistdf['species1'] = corrdistdf['trialidx1'].map(specmap);
    corrdistdf['species2'] = corrdistdf['trialidx2'].map(specmap);

    return corrdistdf;


def main(): 

    #widxdf=pd.read_csv(sys.argv[1]);
    #wgazedf=pd.read_csv(sys.argv[2]);
    #oidxdf=pd.read_csv(sys.argv[3]);
    #ogazedf=pd.read_csv(sys.argv[4]);

    #bigdf=make_bigdf(widxdf,wgazedf,oidxdf,ogazedf);

    #rint('bigdf was made!');

    marmo_bigdf=pd.read_csv(sys.argv[1]);
    infant_bigdf=pd.read_csv(sys.argv[2]);

    marmo_bigdf['timems']/=1000;

    infant_bigdf=infant_bigdf.rename(columns={'agegroup':'species'});
    marmo_bigdf=marmo_bigdf.rename(columns={'timems':'movie_ts'});

    #prevent confusing trial indices of marmo and infants 
    infant_bigdf['trialidx'] += marmo_bigdf['trialidx'].max() + 1;

    bigdf=pd.concat([marmo_bigdf,infant_bigdf],axis=0,ignore_index=True);

    #mismatch in sample hz
    bigdf=align_time(bigdf,sample_hz=30);

    print('time aligned!');

    #bigdf.to_csv('marmo_infant_bigdf.csv');

    subj_corrdistdf=subj_corrdist(bigdf);

    subj_corrdistdf.to_csv('marmo_infant_corrdist.csv');

    return 0;


if __name__ == "__main__":

    main();
    sys.exit(0);
