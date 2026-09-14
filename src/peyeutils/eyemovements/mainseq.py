import numpy as np;

#REV; takes vector of ampl/dur and returns true/false for each.
#REV: INCLUSIVE to lower/upper bound. Bounds are linear.
def mainseq_ampldur_linear(myampl_dva,
                           mydur_sec,
                           over_err_intercep_sec,
                           over_err_slope_secdeg,
                           under_err_intercep_sec,
                           under_err_slope_secdeg,
                           ):
    
    over = over_err_intercep_sec + (myampl_dva * over_err_slope_secdeg);
    under = under_err_intercep_sec + (myampl_dva * under_err_slope_secdeg);
    result = (mydur_sec <= over) & (mydur_sec >= under);

    return result;



def mainseq_ampldur_linear_wplot(myampl_dva,
                                 mydur_sec,
                                 over_err_intercep_sec,
                                 over_err_slope_secdeg,
                                 under_err_intercep_sec,
                                 under_err_slope_secdeg,
                                 ):
    
    result = mainseq_ampldur_linear( myampl_dva,
                                     mydur_sec,
                                     over_err_intercep_sec,
                                     over_err_slope_secdeg,
                                     under_err_intercep_sec,
                                     under_err_slope_secdeg );
    import pandas as pd;
    import matplotlib.pyplot as plt;
    import seaborn as sns;
    
    df = pd.DataFrame( {'ampldva':myampl_dva, 'dursec':mydur_sec, 'ismainseq':result} );
    
    g = sns.relplot(data=df, x='ampldva', y='dursec', hue='ismainseq', kind='scatter');
    g.ax.axhline(0);
    xmin, xmax = 0, 60;
    ymin, ymax = 0, 0.300;
    g.ax.set_ylim([ymin, ymax]);
    g.ax.set_xlim([xmin, xmax]);
    g.ax.set_xlabel('Saccade Amplitude (deg)');
    g.ax.set_ylabel('Saccade Duration (sec)');
    g.ax.plot([xmin, xmax], [under_err_intercep_sec, under_err_intercep_sec + (xmax*under_err_slope_secdeg) ] );
    g.ax.plot([xmin, xmax], [over_err_intercep_sec, over_err_intercep_sec + (xmax*over_err_slope_secdeg) ] );
    g.fig.tight_layout();
    return result, g;


def getparams_mainseq_ampldur_linear_95pctl_human_chen2021(error_gain=1):
    intercep_msec = 32.2;
    slope_msec = 3.72;
    
    dvavals = np.array([0, 16]);
    topvals_msec = np.array([45, 170]) * error_gain;
    botvals_msec = np.array([14, 60]) * (1/error_gain); #REV: was 18 msec? For low at 0.

    ts = topvals_msec * 1e-3;
    bs = botvals_msec * 1e-3;

    tslope = (ts[1]-ts[0]) / (dvavals[1]-dvavals[0]);
    tinter = ts[0];
    
    bslope = (bs[1]-bs[0]) / (dvavals[1]-dvavals[0]);
    binter = bs[0];
    
    return dict(
                over_err_intercep_sec = tinter,
                over_err_slope_secdeg = tslope,
                under_err_intercep_sec = binter,
                under_err_slope_secdeg = bslope,
                );
    

def mainseq_ampldur_linear_95pctl_human_chen2021( myampl_dva,
                                                  mydur_sec,
                                                  error_gain=1,
                                                 ):
    """Flag which (amplitude, duration) saccade pairs are plausibly real
    saccades, using linear 95th-percentile amplitude/duration bounds fit to
    human data (Chen et al. 2021).

    A candidate saccade is accepted if its duration falls within
    ``[under_bound(ampl), over_bound(ampl)]``, where both bounds are linear
    in amplitude (see
    :func:`getparams_mainseq_ampldur_linear_95pctl_human_chen2021`).

    Parameters
    ----------
    myampl_dva : array-like
        Saccade amplitudes, in degrees of visual angle.
    mydur_sec : array-like
        Saccade durations, in seconds. Same length as `myampl_dva`.
    error_gain : float
        Multiplier that widens (>1) or narrows (<1) the acceptable
        duration band around the reference main sequence; noisier eye
        trackers typically need a larger value. `error_gain <= 0` disables
        the check entirely and accepts every saccade.

    Returns
    -------
    numpy.ndarray of bool
        True where the saccade's duration is plausible for its amplitude.

    Examples
    --------
    >>> import numpy as np
    >>> from peyeutils.eyemovements.mainseq import mainseq_ampldur_linear_95pctl_human_chen2021
    >>> ampl = np.array([5.0, 5.0])
    >>> dur = np.array([0.045, 5.0])  # second one is an implausibly long "saccade"
    >>> mainseq_ampldur_linear_95pctl_human_chen2021(ampl, dur, error_gain=1).tolist()
    [True, False]
    """

    #REV: must check error_gain<=0 BEFORE computing params: getparams_...() divides by
    #REV: error_gain, so error_gain=0 (the documented "accept everything" sentinel) used
    #REV: to raise ZeroDivisionError before this shortcut ever ran.
    if(error_gain <= 0):
        return np.full(myampl_dva.shape, True);

    params = getparams_mainseq_ampldur_linear_95pctl_human_chen2021(error_gain=error_gain);

    return mainseq_ampldur_linear( myampl_dva = myampl_dva,
                                   mydur_sec = mydur_sec,
                                   **params
                                  );
    
    

    
def mainseq_ampldur_linear_95pctl_human_chen2021_wplot( myampl_dva,
                                                       mydur_sec,
                                                       error_gain=1,
                                                       ):
    
    #REV: same ordering fix as above -- must check error_gain<=0 before computing params.
    if(error_gain <= 0):
        print("Skipping plot, all saccades are accepted");
        return np.full(myampl_dva.shape, True), None; #REV: keep (result, graphics) arity consistent with the plotted path below.

    params = getparams_mainseq_ampldur_linear_95pctl_human_chen2021(error_gain=error_gain);

    return mainseq_ampldur_linear_wplot(myampl_dva = myampl_dva,
                                        mydur_sec = mydur_sec,
                                        **params
                                        );

    
    
