import numpy as np;

def linsteps(start, stop, step, pastend=True):
    """

    Parameters
    ----------
    start :
        
    stop :
        
    step :
        
    pastend :
         (Default value = True)

    Returns
    -------

    """
    if( pastend ):
        n = int(np.ceil((stop - start)/step));
    else:
        n = int((stop - start)/step); # floor by default.
        n2 = int(np.floor((stop - start)/step));
        if( n != n2 ):
            raise Exception("Python integer conversion and floor not equivalent...");
        pass;
    samps = np.arange(0, n+1); #step = 1
    samps = start + (samps * step);
    return samps;


def allnan(alist):
    """

    Parameters
    ----------
    alist :
        

    Returns
    -------

    """
    if( np.count_nonzero(~np.isnan(np.array(alist))) < 1 ):
        return True;
    return False;

def l2dist(x1, y1, x2, y2):
    """

    Parameters
    ----------
    x1 :
        
    y1 :
        
    x2 :
        
    y2 :
        

    Returns
    -------

    """
    return math.sqrt( (x1-x2)**2 + (y1-y2)**2 );

def l2distvec(x1,y1,x2,y2):
    """

    Parameters
    ----------
    x1 :
        
    y1 :
        
    x2 :
        
    y2 :
        

    Returns
    -------

    """
    return np.sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
