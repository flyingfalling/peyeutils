import pandas as pd;
import numpy as np

def safe_df_concat(dflist):
    valid_dfs = list();
    cols = list();
    
    for df in dflist:
        cols = cols + list(df.columns);
        if not df.empty:
            all_na_cols = df.columns[df.isna().all()]
            
            if not all_na_cols.empty:
                df = df.drop(columns=all_na_cols);
                pass;
            valid_dfs.append(df)
            pass;
        else:
            #print("Skipping a DF because it was empty");
            pass;
        pass;
    
    cols = list(set(cols));
    if valid_dfs:
        resdf = pd.concat(valid_dfs, ignore_index=True);
        pass;
    else:
        resdf = pd.DataFrame()
        pass;
    
    for c in cols:
        if c not in resdf.columns:
            print("Re-adding missing (due to all NAN) column: [{}]".format(c));
            resdf[c] = np.nan; #REV: set all rows of col to NAN
            pass;
        pass;
    
    return resdf;

