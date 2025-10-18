import numpy as np;
import pandas as pd;

def recursive_lookup(disjoint_set, rowidx):
    if disjoint_set[rowidx] != rowidx:
        disjoint_set[rowidx] = recursive_lookup(disjoint_set, disjoint_set[rowidx]);
    return disjoint_set[rowidx];

def unique_clusters(df : pd.DataFrame):
    
    disjoint_set = {}
    value_lookup = {}
    output = df.copy();
    #print("DF");
    #print(df);
    #for rowidx in range(len(df.index)):
    for rowidx, row in df.iterrows():
        disjoint_set[rowidx] = rowidx;  # Mark it as independent set.
        #row = df.iloc[ rowidx ];
        #for key, value in list_1[row].items():  # not sure how to get key value with pandas
        #print("ROW");
        #print(row);
        #print("GO");
        for key in df.columns:
            value = row[key];
            #print(value);
            if (key, value) not in value_lookup:
                value_lookup[(key, value)] = rowidx;
            else:
                other_row = value_lookup[(key, value)];
                actual_other = recursive_lookup(disjoint_set, other_row);
                actual_row = recursive_lookup(disjoint_set, rowidx);
                disjoint_set[actual_row] = actual_other;
                pass;
            pass;#End for loop
    idxarray = list( range( len( df.index ) ) );
    for key in disjoint_set:
        idxarray[key] = disjoint_set[key];
        #output.loc[ key ] =
    output['cluster'] = idxarray;
    return output;
