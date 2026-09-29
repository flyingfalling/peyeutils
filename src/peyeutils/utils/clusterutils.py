import numpy as np;
import pandas as pd;

def recursive_lookup(disjoint_set, rowidx):
    """

    Parameters
    ----------
    disjoint_set :
        
    rowidx :
        

    Returns
    -------

    """
    if disjoint_set[rowidx] != rowidx:
        disjoint_set[rowidx] = recursive_lookup(disjoint_set, disjoint_set[rowidx]);
    return disjoint_set[rowidx];

def unique_clusters(df : pd.DataFrame):
    """Group rows into clusters via union-find, connecting any two rows that
    share the same value in ANY column.

    Two rows end up in the same cluster if they're directly connected
    (share a value in at least one column) or transitively connected
    through other rows. Useful for e.g. deduplicating/merging records that
    refer to the same underlying entity via any of several shared keys.

    Parameters
    ----------
    df : pandas.DataFrame
        Any index is supported (not just a default 0..N-1 RangeIndex).

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with an added 'cluster' column. Rows with the same
        'cluster' value are connected; the cluster id itself is arbitrary
        (it's one representative row's index label).

    Examples
    --------
    >>> import pandas as pd
    >>> from peyeutils.utils.clusterutils import unique_clusters
    >>> df = pd.DataFrame({'a': [1, 2, 1, 3], 'b': ['x', 'y', 'z', 'w']})
    >>> out = unique_clusters(df)
    >>> out.loc[0, 'cluster'] == out.loc[2, 'cluster']  # both have a==1
    True
    >>> out.loc[1, 'cluster'] == out.loc[3, 'cluster']  # unrelated singletons
    False
    """

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
    #REV: keyed by actual index LABELS (not 0..N-1 positions), so this is safe for
    #REV: any df.index (non-default, non-contiguous, non-integer, etc.), unlike indexing into a plain list by label.
    cluster_of = { rowidx: recursive_lookup(disjoint_set, rowidx) for rowidx in disjoint_set };
    output['cluster'] = output.index.map(cluster_of);
    return output;
