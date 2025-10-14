import numpy as np;

def mean_std_on_2d_axis( mat, flattento='x', norm=True ):
    if(len(mat.shape) != 2):
        raise Exception("Does not work on anything except for 2d matrix");
    
    axidx = 0 if flattento=='x' else 1; #REV; theres only x or not x lol
    flat = np.nansum( mat, axis=axidx );
    n=float(len(flat));
    nrmidxs = np.arange(n);
    if( len(flat) < 2 ):
        print("REV: error, zero or one size matrix in mean_std?");
        exit(1);
    
    totaldensity = np.nansum(flat);
    weights = np.ones( len(flat) ) / len(flat);
    if( totaldensity > 0 ):
        weights = flat / totaldensity;
    else:
        print("REV: Wtf matrix is zero or something? totaldensity is zero?");
    combined = nrmidxs * weights;
    meanx = np.nansum( combined ) / np.nansum(weights);
    offset = np.array((nrmidxs - meanx));
    stdx = np.sqrt( np.nansum(weights*offset*offset) * n/(n-1) ) / np.nansum(weights);
    if( norm ):
        meanx /= n;
        stdx /= n;
    return meanx, stdx;

## REV: calculate percentile of positive val within negative vals.
def calc_pctl(posval, negvals):
    #pctle = ((negvals < posval).sum()) / float(len(negvals)+1);
    if( (len(negvals) < 1) ):
        return np.nan;
    
    nlt = (negvals < posval).sum(); # / float(len(negvals)+1);                                                                         
    neq = (negvals == posval).sum();
    #### REV: HANDLE NAN HERE?!??!!
    pctle = (nlt + neq/2) / len(negvals); #REV: should I include myself? If zero neg vals we fucked anyways lol
    return pctle;

#REV: information gain
#REV: note, input is assumed to be unlogged by default
def calc_infogain(posval, negval, islog2ed=False):
    lposval = posval;
    lnegval = negval;
    if( not islog2ed ):
        lposval = np.log2(lposval);
        lnegval = np.log2(lnegval);
        pass;
    
    ig = lposval - lnegval;
    return ig;


#REV: returns change of signal indf[valcol] "per unit time" (whatever units are)
#REV: takes input with cols valcol and timecol.
#REV: returns "diff" df containing diffs of valcol and timecol, and new col valcol + "_tdiff", containing time-scaled differences,
#     which are maximum magnitude of change from previous or next time step.
def magnitude_change_over_time(indf, valcol, timecol='Tsec'):
    indf = indf.sort_values(by=timecol).reset_index(drop=True);
    diffdf = indf[[valcol, timecol]].diff(); #REV: default element previous row. t.diff(periods=1, axis=0) (rows=0, cols=1) This will be e.g. change of 1/0.001 = 1000/sec
    newvalcol=valcol+"_abs_tdiff";
    #REV: will return errors for NAN values? Whatever.
    if( diffdf[timecol].min() <= 0 ):
        raise Exception("ERROR: Mag change over time EYEUTILS -> **TIME** col ({}) has diff of <= 0: {}".format(timecol, diffdf[timecol].min()));
    
    diffdf[newvalcol] = diffdf[valcol].astype(float) / diffdf[timecol]; #This is DIFFERENCE WITH PREVIOUS VALUE (index 0 is empty i.e. NAN)
    diffdf[newvalcol] = abs( diffdf[newvalcol] ); #REV: absolute (magnitude) of difference.
    
    #REV: each value is the higher of THIS VALUE (diff with previous time point), and DIFF WITH NEXT TIME POINT (i.e. shift everything
    ## BACKWARDS by 1, will get next point in this point's location.
    #REV: not NANMAX because it is along axis=0 (i.e. max of each ROW?)
    diffdf[newvalcol] = np.max( [ diffdf.shift(0, fill_value=0)[newvalcol],
                                     diffdf.shift(-1, fill_value=0)[newvalcol] ]
                                   ,
                                   axis=0 );
    
    #[A B] is what shape? numpy assumes number of ROWS first, i.e.
    #[1 1] is assumed to be...? 
    
    outdf = indf.copy();
    outdf[newvalcol] = diffdf[newvalcol]; #REV: order better be same..
    
    
    #This is DPRIME.
    
    return outdf, newvalcol;

#REV: takes DELTAs and TIMEDELTAS as arguments.
#REV: each is diff from previous value...
##IN: dataframe with valcol column (time series probably)
##OUT: df with valcol + '_med_err' which is per-unit divergence from median of that signal, and
##     madval, which is single value representing MAD (median absolute deviation).
def MAD_timediff(indf, valcol):
    devdf = indf;
    med = devdf[valcol].median(); #REV: is NANMEDIAN.
    newcol = valcol + '_med_err';
    devdf[newcol] = devdf[valcol] - med; #REV: error (deviation) from median
    #madval = stats.median_abs_deviation(devdf[valcol], scale='normal', nan_policy='omit');
    madval = np.nanmedian(abs(devdf[newcol]));
    if(madval == 0 ):
        print("WARNING: A MAD (maximum absolute divergence?) of 0 is unrealistic...(should skip?)");
        pass;
    devdf[valcol] = indf[valcol];
    return devdf, madval;
