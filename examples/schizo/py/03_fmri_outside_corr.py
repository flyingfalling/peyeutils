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
    print(trialsdf);
    print(trialsdf.columns);

    ccresults=list();
    
    for (name, vid), sdf in trialsdf.groupby(['name', 'video']):
        
        if( len(sdf.index) > 1 and
            len(sdf['kind'].unique()) > 1 ):
            
            print(name, vid, sdf[['kind','ntrials','edffile','edfpath']]);
            
            fmri=sdf[sdf.kind=='fmri'];
            outs=sdf[sdf.kind=='outside'];
            tmplst=list();
            for fi, frow in fmri.iterrows():
                fsamps = pd.read_csv(os.path.join(csvdir, frow['samples_csv']));
                for fo, orow in outs.iterrows():
                    osamps = pd.read_csv(os.path.join(csvdir, orow['samples_csv']));
                    xcol='cgx_dva';
                    xcc, xpts, n = pu.utils.pearson_gaze_CC(osamps[xcol].to_numpy(), fsamps[xcol].to_numpy());
                    ycol='cgy_dva';
                    ycc, ypts, n = pu.utils.pearson_gaze_CC(osamps[ycol].to_numpy(), fsamps[ycol].to_numpy());
                    xycc=(xcc+ycc)/2;
                    tmplst.append(dict(xycc=xycc,xcc=xcc,ycc=ycc,pts=xpts,n=n));
                    pass;
                pass;
            tmpdf=pd.dataframe(tmplst);
            TPTS_CUTOFF=0.66;
            tmpdf=tmpdf[ tmpdf.pts/tmpdf.n > TPTS_CUTOFF ];
            tmpdf=tmpdf.mean();
            xycc=tmpdf['xycc'];
            
            print(dict(xycc=yxcc, name=name, vid=vid))
            ccresults.append( dict(xycc=yxcc, name=name, vid=vid) );
            pass;
        pass;
    ccdf=pd.DataFrame(ccresults);
    ccdf.to_csv('cc.csv', index=False);
    print(ccdf);
    return 0;




if __name__=='__main__':
    exit(main());
    pass;
