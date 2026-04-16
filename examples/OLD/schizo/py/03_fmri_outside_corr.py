## "Null model" should be random within-subject correlation (for different videos?).

## How about "distance" between gaze targets? I.e. inter-observer model?

## Hm, corrcoef may not be great if e.g. they only look at one location, it will not correlate?
## Whereas, interobserver may be >50%.
## But, problem is that with error in gaze position, we want to be agnostic to translations.


## Compute statistics : number of non-bad datapoints, per video/group, inside/outside FMRI.

## Extract CSV for each "trial" individually. Save as numpy for example.

import pandas as pd;
import numpy as np;
import peyeutils as pu;




import sys;
import os;

def main():
    rowcsv=sys.argv[1];
    csvdir=sys.argv[2];
    
    rowdf=pd.read_csv(rowcsv);
    
    trialsdf=list();
    
    for i,row in rowdf.iterrows():
        if( False==row['edferror'] and
            row['haseyetracking'] ):
            mytrials = pd.read_csv( os.path.join(csvdir, row['trials_csv']) );
            if(len(mytrials.index)<1):
                continue;
            for r in row.keys():
                if( r in mytrials.columns ):
                    if( mytrials.iloc[0][r] != row[r] ):
                        print("Replacing {}: {}->{}".format(r,mytrials.iloc[0][r],row[r]));
                        raise Exception("Failed");
                    pass;
                mytrials[r] = row[r];
                pass;
            
            #print(mytrials.columns);
            if('blkidx' not in mytrials.columns):
                print(mytrials);
                raise Exception("WTF no blkidx?");
            
            mytrials['ntrials'] = mytrials.groupby('blkidx')['blkidx'].transform('size');
            #for i, gdf in mytrials.groupby('blockidx').apply:
            #    if( len(gdf.index) ):
            
            trialsdf.append(mytrials); #[mytrials['ntrials']==30]);
            pass;
        
        pass;

    trialsdf = pd.concat(trialsdf);
    #print(trialsdf);
    #print(trialsdf.columns);

    ccresults=list();
    tcol='Tsec';
    
    #print(trialsdf[trialsdf.kind=='fmri']);
    #print(trialsdf[trialsdf.kind=='outside']);
    
    for name, ndf in trialsdf.groupby('name'):
        fmris=ndf[ndf.kind=='fmri'];
        outsides=ndf[ndf.kind=='outside'];
        if(len(fmris.index) < 1 ):
            print("NO FMRIs {}".format(name));
            continue;
        if(len(outsides.index) < 1 ):
            print("NO OUTSIDEs {}".format(name));
            continue;
        sampdict=dict();
        
        
        for xxx, row in ndf.iterrows():
            if( row['edffile'] not in sampdict ):
                #print("Adding {} to sampdict".format(row['edffile']));
                sampdict[ row['edffile'] ] = pd.read_csv( os.path.join(csvdir, row['samples_csv']) );
                pass;
            pass;
        
        
        namelist=list();
        #print(ndf[['name', 'video', 'ntrials', 'kind']]);
        #print(ndf.video.unique(), len(ndf.video.unique()));
        for vid, sdf in ndf.groupby('video'):
            #print("VID: {}".format(vid));
            print(sdf);
            if( len(sdf.index) > 1 and
                len(sdf['kind'].unique()) > 1 ):
                
                #print(name, vid, sdf[['kind','ntrials','edffile','edfpath']]);
                
                fmri=sdf[sdf.kind=='fmri'];
                outs=sdf[sdf.kind=='outside'];
                tmplst=list();
                xcol='cgx_dva';
                ycol='cgy_dva';
                for fi, frow in fmri.iterrows():
                    #fsamps = pd.read_csv(os.path.join(csvdir, frow['samples_csv']));
                    fsamps = sampdict[ frow['edffile'] ];
                    #print(fsamps);
                    #print(frow.keys());
                    fsamps = fsamps[ (fsamps[tcol] >= frow.start_s) & (fsamps[tcol] <= frow.end_s) &
                                     (fsamps['eye']=='B') ];
                    fsamps = pu.preproc.preproc_SHARED_D_exclude_bad(fsamps, xcol=xcol, ycol=ycol);
                    for fo, orow in outs.iterrows():
                        #osamps = pd.read_csv(os.path.join(csvdir, orow['samples_csv']));
                        osamps = sampdict[ orow['edffile'] ];
                        #print(osamps);
                        osamps= osamps[ (osamps[tcol] >= orow.start_s) & (osamps[tcol] <= orow.end_s) &
                                        (osamps['eye']=='B') ];
                        osamps = pu.preproc.preproc_SHARED_D_exclude_bad(osamps, xcol=xcol, ycol=ycol);
                        
                        
                        xcc, xpts, n = pu.utils.pearson_gaze_CC(osamps[xcol].to_numpy(), fsamps[xcol].to_numpy());
                        
                        ycc, ypts, n = pu.utils.pearson_gaze_CC(osamps[ycol].to_numpy(), fsamps[ycol].to_numpy());
                        xycc=(xcc+ycc)/2;
                        tmplst.append(dict(xycc=xycc,xcc=xcc,ycc=ycc,pts=xpts,n=n));
                        pass;
                    pass;
                tmpdf=pd.DataFrame(tmplst);
                
                #print(tmpdf);
                
                TPTS_CUTOFF=0.2; #0.5;
                tmpdf=tmpdf[ tmpdf.pts/tmpdf.n > TPTS_CUTOFF ];
                if(len(tmpdf.index)>0):
                    tmpdf=tmpdf.mean();
                    xycc=tmpdf['xycc'];
                    prop=tmpdf['pts']/tmpdf['n'];
                    ccdict = dict(xycc=xycc, name=name, vid=vid);
                    print(ccdict);
                    ccresults.append(ccdict);
                    namelist.append(ccdict);
                    pass;
                else:
                    print("insufficient points");
                    pass;
                pass;
            
            pass;
        
        #print(namelist);
        
        namedf = pd.DataFrame(namelist);
        #print(namedf);
        print(name, "MEAN", namedf.mean(numeric_only=True));
        print(name, "STD", namedf.std(numeric_only=True));
        

        pass;
    ccdf=pd.DataFrame(ccresults);
    ccdf.to_csv('cc.csv', index=False);
    print(ccdf);
    return 0;




if __name__=='__main__':
    exit(main());
    pass;
