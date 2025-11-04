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
            trialsdf.append(mytrials);
            pass;
                
        pass;

    trialsdf = pd.concat(trialsdf);
    print(trialsdf);
    return 0;




if __name__=='__main__':
    exit(main());
    pass;
