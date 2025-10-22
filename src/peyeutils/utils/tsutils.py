import numpy as np;
import pandas as pd;
import peyeutils as pu;
import math;

from scipy import ndimage;

def select_legal_timepoints4(ts, vals, maxdt):
    """

    Parameters
    ----------
    ts :
        
    vals :
        
    maxdt :
        

    Returns
    -------

    """
    
    
    df = pd.DataFrame();
    df['t'] = ts;
    df['v'] = vals;
    df = df.sort_values(by='t').reset_index(drop=True);
    #nona = df[ (False == df.isnull()) ]; ### REV: THIS DOES NOT WORK
    nona = df.dropna();
    DEBUG_LEVEL=0;
    if(DEBUG_LEVEL>0):
        print("Had {} non-null out of {}".format(len(nona.index), len(df.index)));
        pass;
    
    if( len(nona.index) > 0 ):
        goodt = np.array(nona['t']);
        badt = np.array(df[ pd.isnull(df).any(axis=1) ]['t']); #REV: need to do "any" on axis=1 (i.e. along columns?). DF NOT MATRIX ORDER!!
        if(DEBUG_LEVEL>0):
            print("Had {} null out of {}".format(len(badt), len(df.index)));
            pass;
        ######## IDEA ########
        # For each point, get the NEAREST NON-NULL VALUE (in time), and compute distance. If small enough, include
        
        ### For all MERGED TIMEPOINTS (i.e. include those with NAN for this guy)
        ### No, I only need to compute for badt
        ### compute CLOSEST NON-NAN ELEMENT. I.e. goodt are NON-NAN.
        alist=list();
        for bi, bt in enumerate(badt):
            closestidx = np.nanargmin( np.abs(goodt - bt) );
            alist.append(closestidx); #this is index of closest guy in goodt
            pass;
        
        myclosests = goodt[ alist ]; #Now have TIMES of closests.
        
        diffs = abs(badt - myclosests);

        # indices of NAN I will fill in from interpoalated (or slightly diff
        ## timesteps due to e.g. mag/acc arriving diff time than gaze.
        keeps = diffs <= maxdt;
        passed = badt[np.where(keeps)];
        passed = np.unique(passed);
        print('Kept {} passed badt out of orig {}'.format(len(passed), len(badt)));
        return goodt, passed;
    
    else:
        #REV: just return "all bad"
        return list(), list();
    return;


def interpolate_df_to_samplerate(df, tcol, targ_srhzsec, tcolunit_s, truesrs=dict(),
                                 maxtdelta_s=None, maxtdeltas_s=dict(),
                                 startsec=None, endsec=None,
                                 method='polynomial', order=2,
                                 tsecname='Tsec', tsec0name='Tsec0',
                                 zeroTsec=None,
                                 ):
    """

    Parameters
    ----------
    df :
        
    tcol :
        
    targ_srhzsec :
        
    tcolunit_s :
        
    truesrs :
         (Default value = dict())
    maxtdelta_s :
         (Default value = None)
    maxtdeltas_s :
         (Default value = dict())
    startsec :
         (Default value = None)
    endsec :
         (Default value = None)
    method :
         (Default value = 'polynomial')
    order :
         (Default value = 2)
    tsecname :
         (Default value = 'Tsec')
    tsec0name :
         (Default value = 'Tsec0')
    zeroTsec :
         (Default value = None)
    Returns
    -------

    """
    import numpy as np;
    import pandas as pd;
    
    if( tcolunit_s != 1 ):
        #raise Exception("Unit of tcol (time column) must be seconds (tcolunit_s=1) (is {})".format(tcolunit_s));
        print("WARNING: REV: still uncertain/untested when tcolunit_s is not 1 (second)");

    #REV: note units of DT is in units timecolunit_s
    dt = (1/tcolunit_s) / targ_srhzsec; #REV: e.g. (1/0.001) / 500 = 1000/500 = 2
        
    for c in df.columns:
        if c not in truesrs:
            truesrs[c] = targ_srhzsec; #REV: artificially set to sample value will grab any in that range UNLESS it's manually set per-sensor
            pass;
        pass;
    
    
    for c in df.columns:
        if c not in maxtdeltas_s:
            if( maxtdelta_s is None ):
                maxtdeltas_s[c] = (1/tcolunit_s) / truesrs[c]; # 1 * orig sr in dt. E.g. if true SR is 10, this will be (1000/10 = 100 off is OK?)
                pass;
            else:
                maxtdeltas_s[c] = maxtdelta_s / tcolunit_s; #REV: e.g. max of 0.001 sec, but timecol is 0.001, so "1" unit OK.
                pass;
            pass;
        else:
            #REV: it's already set manually by user per-sensor
            pass;
        pass;
    
    for c in df.columns:
        if c not in maxtdeltas_s:
            maxtdeltas_s[c] = dt;
            pass;
        pass;
    
    ##REV: only apply interpolation to values within maxtdelta time of valid values. Leave NAN further away alone.
    ##REV: this unfortunately must be done my masking each column individually (although they often do come in chunks).
    ##REV: thus full interpolation is run, but then we only take values masked as "OK"
    ##REV: to do this, we create "TRUE" for every row if it is within maxtdelta of a finite value.
    ##REV: I must check that in the "real, mixed arrivals" one, there is a real value within time delta of me IN THIS COLUMN.
    
    ##REV: to do this, FIND THE TIMESTAMPS OF NON-NAN VALUES (in this column).
    
    ## FUCK I DID THIS BEFORE. Ran out of memory building the full matrix of distances, since will be NxN
    ## Ended up iterating.
    
    
    if( np.sum(np.isnan(df[tcol])) > 0 ):
        print(df[tcol]);
        raise Exception("NANs in time column (not allowed)");
    
    df = df.sort_values(by=tcol).reset_index(drop=True);
    st = df[tcol].min();
    en = df[tcol].max();
    if( startsec is not None ):
        st = startsec;
        pass;
    if( endsec is not None ):
        en = ensec;
        pass;
    
    print("MAX T DELTAS of column [{}] (now in timeunit={} sec units) -- WILL LIN STEP ({}-{} @ {})".format(tcol, tcolunit_s, st, en, dt))
    
    if( ((en-st)/dt) > 1e9 ):
        raise Exception("you are attempting to create more than 1 billion time points at once...probably you will run out of memory");
    samps = pu.utils.linsteps( st, en, dt );
        
    tdf = pd.DataFrame();
    tdf[tcol] = samps;
    
    
    mdf = pd.merge( tdf, df, left_on=tcol, right_on=tcol, how='outer' );
    
    
    mdf = mdf.sort_values(by=tcol).reset_index(drop=True);
    
    
    maskdf = pd.DataFrame();
    maskdf[tcol] = mdf[tcol];
    nontcol = [ c for c in mdf.columns if c!=tcol ];
    
    
    ## CLUSTER columns by nan-like shape.
    #colnans = pd.isna(mdf) == False;
    colnans = pd.isna(mdf[nontcol]) == False;
    from itertools import combinations
    
    colpairs = [(i, j) for i,j in combinations(colnans, 2) if colnans[i].equals(colnans[j])];
    
    pool = set(map(frozenset, colpairs))
    groups = []
    while pool:
        group = set()
        groups.append([])
        while True:
            for candidate in pool:
                if not group or group & candidate:
                    group |= candidate
                    groups[-1].append(tuple(candidate))
                    pool.remove(candidate)
                    break
                pass;
            else:
                break
            pass;
        pass;

    finalgrps=list();
    for g in groups:
        myset=set();
        for pair in g:
            myset.add(pair[0]);
            myset.add(pair[1]);
            pass;
        finalgrps.append(list(myset));
        pass;
    unaccounted=list();
    for c in nontcol:
        found=False;
        for g in finalgrps:
            if c in g:
                found=True;
                pass;
            pass;
        if( not found ):
            unaccounted.append(c);
            pass;
        pass;
    
    for c in unaccounted:
        finalgrps.append([c]);
        pass;
    
    print("Groups of similar columns: ", finalgrps);
        
    for g in finalgrps:
        dtgrps = [maxtdeltas_s[c] for c in g];
        uniquedts = np.unique(dtgrps);
        #print("Group: ", g);
        #print("Unique maxdts for this group: ", uniquedts);
        
        for maxdt in uniquedts:
            mydtgrp = [c for c in maxtdeltas_s if ((maxtdeltas_s[c]==maxdt) and c in g)];
            #print("Group for maxdt={}".format(maxdt));
            #print(mydtgrp);
            
            c = mydtgrp[0];
            if( c == tcol ):
                raise Exception("WTF, c is the tcol even though it should not be?");
            print("For [{}] maxdt is [{}]".format(c, maxdt));
            good, bad = select_legal_timepoints4(mdf[tcol], mdf[c], maxdt);
            goodbad = np.concatenate([good, bad]);
            maskdf[c] = False;
            for c in mydtgrp:
                #print("Column is: [{}]".format(c));
                maskdf[c] = False;
                maskdf.loc[ maskdf[tcol].isin(goodbad), c ] = True;
                pass;
            pass;
        pass;

    
    #https://stackoverflow.com/questions/45757477/numpy-interpolation-using-pandas
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
    
    if( len(mdf[tcol]) != len(mdf[tcol].unique())):
        print(len(mdf[tcol]));
        print(len(mdf[tcol].unique()));
        print(mdf[tcol].iloc[np.where(mdf[tcol].duplicated())]);
        raise Exception("Tcol is not unique (you need to have unique times, suggest groupby(tcol).mean() for example)!");
    
    mdf = mdf.set_index( mdf[tcol] ); 
    mdf = mdf.interpolate(method=method, order=order);
    
    mdf = mdf.reset_index(drop=True);
    
    for c in nontcol:
        toset= (maskdf[c] == False);
        if(len(toset) > 0 ):
            mdf.loc[ toset, c ] = np.nan;
            pass;
        pass;
    
    mdf = pd.merge(tdf, mdf, how='inner', left_on=tcol, right_on=tcol).reset_index(drop=True);
    
    tsec=mdf[tcol] * tcolunit_s;
    if( zeroTsec is None ):
        tsec0=tsec - tsec.min();
        pass;
    else:
        print("Setting Tsec0 zero time to Tsec={} (to e.g. align with other sensor source such as video)".format(zeroTsec));
        tsec0=tsec - zeroTsec;
        pass;

    #REV: Tsec/Tsec0
    if( tsecname not in mdf ):
        mdf[tsecname] = tsec; #mdf[tcol] * tcolunit_s; #e.g. if tcolunit_s is 0.001 (msec), then tcol of 1000 msec will become 1 sec.
        pass;
    else:
        if( False == np.all(np.isclose(tsec, mdf[tsecname])  ) ):
            print(tsec);
            print(mdf[tsecname]);
            print(mdf);
            raise Exception("ERROR -- tsec exists but does not contain expected data!");
        pass;
    
    if( tsec0name not in mdf ):
        mdf[tsec0name] = tsec0;
        pass;
    else:
        if( False == np.all(np.isclose(tsec0, mdf[tsec0name])  ) ):
            raise Exception("ERROR -- tsec0 exists but does not contain expected data! (maybe it already existed and you used zeroTsec this time?)");
        pass;
    
    return mdf;




def get_dilated_nan_mask(arr, iterations, max_ignore_size=None):
    """

    Parameters
    ----------
    arr :
        
    iterations :
        
    max_ignore_size :
         (Default value = None)

    Returns
    -------

    """
    from scipy import ndimage
    clusters, nclusters = ndimage.label(np.isnan(arr))
    # go through all clusters and remove any cluster that is less
    # the max_ignore_size

    if( max_ignore_size is not None ):
        for i in range(nclusters):
            # cluster index is base1
            i = i + 1;
            if (clusters == i).sum() <= max_ignore_size:
                clusters[clusters == i] = 0;
                pass;
            pass;
        pass;
        
    mask = ndimage.binary_dilation(clusters > 0, iterations=iterations);
    return mask;

def get_dilated_mask(arr, iterations, dilateval, max_ignore_size=None):
    """

    Parameters
    ----------
    arr :
        
    iterations :
        
    dilateval :
        
    max_ignore_size :
         (Default value = None)

    Returns
    -------

    """
    clusters, nclusters = ndimage.label( (arr == dilateval) );
    # go through all clusters and remove any cluster that is less
    # the max_ignore_size
    
    if( max_ignore_size is not None ):
        for i in range(nclusters):
            # cluster index is base1
            i = i + 1;
            if (clusters == i).sum() <= max_ignore_size:
                clusters[clusters == i] = 0;
                pass;
            pass;
        pass;
        
    mask = ndimage.binary_dilation(clusters > 0, iterations=iterations);
    return mask;






#REV: creates a RLE of True/False for condition (val==val). I.e. for input x, creates boolean array x==val, then
# runs RLE on that, then otuputs a df, with some useful columns, possibly related to time (t)
def cond_rle_df( x, val, t=None):
    """Compute lengths of chunks and put in df.
    Note that the final chunk will have time of one less than expected (since no defined DT)
    Note that INDEXES are NON INCLUSIVE of endpoint.

    Parameters
    ----------
    x :
        
    val :
        
    t :
         (Default value = None)

    Returns
    -------

    """
    
    #if( t is None ):
    #    t = range(0, len(x));
    #    pass;
    
    if( t is not None):
        t = np.array(t); #REV: remove weird indexing shit? Assume sorting is identical.
        if( len(t) != len(x) ):
            raise Exception("Time length is {}, neq X length {} (events_over_thresh)".format(len(t), len(x)));
        pass;
    else:
        t = [np.nan]*len(x);
        pass;
    
    #REV: don't force dts between rows to be uniform?
    '''
    tdiff=t.diff();
    if( len(np.unique(tdiff)) != 2 ):
        if( np.nanstd(tidff) / np.nanmean(tdiff) > 0.001 ):
            raise Exception("Bad STD/MEAN for diff in times? Implies skipped timepoints etc. you need to interpolate.");
        pass;
    '''
    
    if( len(x) < 1 ):
        raise Exception("Input x len is <1");
    
    
    
    #REV: I should be able to compare against a constant right?
    #vals, starts, lens = rle((np.array(x)==np.fill(len(x), val)));
    if( np.isnan(val) ):
        boolarr = np.isnan(x);
        pass;
    elif( np.isinf(val) ):
        boolarr = np.isinf(x);
        pass;
    else:
        boolarr = (x==val); #REV: will work because nan/inf never equal to value. Could use "isclose"
        pass;
    
    vals, starts, lens = rle(boolarr); #, include_end=include_end);
    #(np.array(x)==np.fill(len(x), val)));
    
    #if(len(t) > 0 ):
    #    print(t[0]);
    #    #exit(0);
    
    alist=list();
    #REV: -1 as end is the index AFTER it.
    
    #REV: sum of all durations should be start-to-end time? (assuming no skips).
    #REV: note intervening "NAN" will break things?
    
    #REV: the last len will be one short...just make it NAN? Make first one NAN also? Unknown lengths
    #if(len(lens) > 0 ):
    #    lens[-1] -= 1;
    #    pass;
    
    #REV: start and end TIME will be correct (not end inclusive, i.e. >= and <)
    #REV: however, length will be meaningless...? Since time "this" to "next" is not really defined...
    
    for v, s, l in zip(vals[:-1], starts[:-1], lens[:-1]):
        d = dict( v=v,
                  sidx=s,
                  lidx=l,
                  eidx=s+l,
                  st=t[s],
                  et=t[s+l], #REV: should end time "include" 
                  lent=(t[s+l]-t[s]) #REV: so this will cause them to be "short" but last one also, length 0.
                 );
        
        alist.append(d);
        pass;
    v=vals[-1];
    s=starts[-1];
    l=lens[-1];
    d = dict(v=v,
             sidx=s,
             lidx=l,
             eidx=s+l,
             st=t[s],
             et=t[s+l-1], #REV: should end time "include" 
             lent=(t[s+l-1]-t[s])
             );
    alist.append(d);
    
    evdf = pd.DataFrame(alist).sort_values(by='sidx').reset_index(drop=True);
    #evdf.iloc[-1]['lent'] = np.nan;
    #evdf.iloc[-1]['et'] = np.nan;
    
    return evdf;




def events_over_thresh( x, thresh, t=None, invert=False ):
    """

    Parameters
    ----------
    x :
        
    thresh :
        
    t :
         (Default value = None)
    invert :
         (Default value = False)

    Returns
    -------

    """
    if( t is None ):
        t = range(0, len(x));
        pass;
    else:
        if( len(t) != len(x) ):
            raise Exception("Time length is {}, neq X length {} (events_over_thresh)".format(len(t), len(x)));
        pass;
    b = x > thresh;
    
    #val = True;
    #if( invert ):
    #    val = False;
    #    pass;
    
    vals, starts, lens = rle(b);
    
    alist=list();
    #REV: -1 as end is the index AFTER it.
    for v, s, l in zip(vals, starts, lens):
        d = dict( v=v, sidx=s, lidx=l, eidx=s+l, st=t[s], et=t[s+l-1], lent=(t[s+l-1]-t[s]) );
        alist.append(d);
        pass;

    evdf = pd.DataFrame(alist);
    
    return evdf;



#REV: similar to RLE (before I knew about it).
def contiguous_identical_vals( xs ):
    """

    Parameters
    ----------
    xs :
        

    Returns
    -------

    """
    xs=np.array(xs);
    same2left = ((xs[1:]-xs[:-1])==0)
    same2right = ((xs[:-1]-xs[1:])==0)
    same2right = np.append(same2right, False);
    same2left = np.append(False, same2left);
    same = np.array(same2left | same2right);
    return same;


#REV: how to handle NAN chunks?
#REV: we can either include in our RLE (i.e. have sections that are "nan" together).
#REV: They are not normally recognized due to "notequal"
def rle(x, withnan=True):
    """Find runs of consecutive items in an array.

    Parameters
    ----------
    x :
        
    withnan :
         (Default value = True)

    Returns
    -------

    """
    """REV: modified for NAN/INF (now exception)"""
    
    # ensure array
    x = np.asanyarray(x)
    if(x.ndim != 1):
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    if( False==withnan and np.any(~np.isfinite(x) ) ):
        raise Exception("Some of the passed array to RLE is not finite (i.e. NAN or INF) but withnan not true");
    
    # handle empty array
    if(0 == n):
        return np.array([]), np.array([]), np.array([])
    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:]); #REV: nan not equal to next will always be true...
        run_starts = np.nonzero(loc_run_start)[0]
        
        # find run values
        run_values = x[loc_run_start]
        
        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
        
        #REV: final run length needs to be modified...
        #if(len(run_lengths) > 0 ):
        #    run_lengths[-1] -= 1;
        #    pass;

        import pandas as pd;
        df = pd.DataFrame( dict(starts=run_starts, values=run_values, lengths=run_lengths) );
        
        if( False and withnan ):
            isnans = np.isnan(x);
            v, s, l = rle(isnans, withnan=False);
            v[:] = np.nan; #REV: just make all nan (even if they were inf etc.)

            
            run_values = np.append(run_values, v);
            run_starts = np.append(run_starts, s);
            run_lengths = np.append(run_lengths, l);
            
            isinfs = np.isinf(x);
            v, s, l = rle(isinfs, withnan=False);
            v[:] = np.inf; #REV: just make all nan (even if they were inf etc.)
            
            run_values = np.append(run_values, v);
            run_starts = np.append(run_starts, s);
            run_lengths = np.append(run_lengths, l);
            pass;
        
        return run_values, run_starts, run_lengths
    raise Exception("ERROR: RLE (should never reach end...)");



#REV: build array X from rle values
def inverse_rle(run_values, run_starts, run_lengths):
    """

    Parameters
    ----------
    run_values :
        
    run_starts :
        
    run_lengths :
        

    Returns
    -------

    """
    totallen=np.sum(run_lengths); #REV: must be list?
    x=np.zeros(totallen, dtype=run_values.dtype); #REV: will make false if bool
    for val, st, size in zip(run_values, run_starts, run_lengths):
        x[st:st+size] = val; #REV: e.g. if 0 and 10, OK.
        pass;
    return x;




def remove_suspicious_repeats(df, params):
    """

    Parameters
    ----------
    df :
        
    params :
        

    Returns
    -------

    """
    import math;
    import numpy as np;
    
    sr=params['samplerate_hzsec'];
    minlensec=params['min_suspicious_repeats_sec'];
    minlensamp = int(minlensec * sr); #REV: ceil better? float round? ;(
    xn=params['xname'];
    yn=params['yname'];
    
    vals, sts, lens = rle(df[ xn ]);
    sts = np.array(sts, dtype=int); #.flatten();
    ens = np.array(sts + np.array(lens), dtype=int); #.flatten(); #REV e.g.  X Y Y -> [ X 0 1 ], [ Y 1 2 ]
    
    ilens = np.where( lens > minlensamp )[0];
    #REV: or can just use non zero idex of tuple, since I will just index sts and ens with it.
    
    #REV: must take zeroth index...because returns 2d array...
    
    #https://stackoverflow.com/questions/64481108/assign-a-value-between-index-values-ranges-in-pandas
    for s, e in zip(sts[ilens], ens[ilens]):
        #df[xn].iloc[s:e] = np.nan;
        #df[yn].iloc[s:e] = np.nan;
        df[xn][s:e] = np.nan;
        df[yn][s:e] = np.nan;
        #REV: are DF indices guaranteed to be indexed correctly?
        #https://stackoverflow.com/questions/49247739/pandas-set-value-range-of-cell-with-list
        #df.loc[np.r_[sts[i]:ens[i]], xn] = np.nan;
        #df.loc[np.r_[sts[i]:ens[i]], yn] = np.nan;
        pass;
    
    return df



def dilate_nans( df, cols, params ):
    """

    Parameters
    ----------
    df :
        
    cols :
        
    params :
        

    Returns
    -------

    """
    sr=params['samplerate_hzsec'];
    dilate_nan_win_samp = math.ceil(params['dilate_nan_win_sec'] * sr); # E.g. if 0.030 sec and SR=100, 0.030 * 100 = 3
    #REV: maybe better to interpolate barely missing values?
    #min_blink_samp = int(params['min_blink_sec'] * sr);
    mask = np.full( len(df.index), False );
    for col in cols:
        mask = mask | (get_dilated_nan_mask( df[col],
                                             dilate_nan_win_samp ) );
        pass;

    df = df.reset_index(drop=True);
    for col in cols:
        #REV: only works if df index is dense and from 0 to length?
        df.loc[mask, col] = np.nan;
        pass;
    
    return df;


def dilate_val( arr, val, winsamp ):
    """

    Parameters
    ----------
    arr :
        
    val :
        
    winsamp :
        

    Returns
    -------

    """
    #min_blink_samp = int(params['min_blink_sec'] * sr);
    mask = np.full( len(arr), False );
    mask = mask | (get_dilated_mask( arr,
                                     winsamp,
                                     val ) );
    arr[mask] = val;
        
    return arr;

def dilate_xy_nans( df, params ):
    """

    Parameters
    ----------
    df :
        
    params :
        

    Returns
    -------

    """
    sr=params['samplerate_hzsec'];
    dilate_nan_win_samp = math.ceil(params['dilate_nan_win_sec'] * sr);
    min_blink_samp = math.ceil(params['min_blink_sec'] * sr);
    xname=params['xname'];
    yname=params['yname'];
    mask = get_dilated_nan_mask( df[xname],
                                 dilate_nan_win_samp,
                                 min_blink_samp );
    df[xname][mask] = np.nan;
    df[yname][mask] = np.nan;
    return df;
