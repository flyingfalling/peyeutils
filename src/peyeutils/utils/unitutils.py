import math;
import numpy as np;


#REV: move to utils
def msec_to_sec( msec ):
    """

    Parameters
    ----------
    msec :
        

    Returns
    -------

    """
    return msec*1e-3;

#REV: move to utils
def sec_to_msec( msec ):
    """

    Parameters
    ----------
    msec :
        

    Returns
    -------

    """
    return msec*1e3;

#REV: calculated angle (in radians!) of something of width wid_meters at dist of eye_to_screen_meters
def meter_to_rad( eye_to_screen_meters, wid_meters ):
    """

    Parameters
    ----------
    eye_to_screen_meters :
        
    wid_meters :
        

    Returns
    -------

    """
    #REV: argctan
    #REV: arctan2( y, x ); Assume we are quadrant one (upper right, and we measure from normal X-axis anti-clockwise)
    angle_rad = np.arctan2( wid_meters, eye_to_screen_meters ); #REV: y/x
    return angle_rad;

def dist_at_angle_deg( mid_dist_m, angle_deg ):
    """

    Parameters
    ----------
    mid_dist_m :
        
    angle_deg :
        

    Returns
    -------

    """
    #REV: cos(a) = adj/hyp; cos(a)*hyp = adj; hyp = adj/cos(a)
    hyp = mid_dist_m / math.cos( deg_to_rad(angle_deg) );
    return hyp;

def deg_from_mid_to_meter( mid_dist_m, angle_deg ):
    """Converts degrees from center (straight ahead=0) to meters, given
    distance eye-to-screen in meters.

    Parameters
    ----------
    mid_dist_m :
        
    angle_deg :
        

    Returns
    -------

    """
    #REV: tana = opp/adj; tana*adj = opp
    opp = math.tan( deg_to_rad(angle_deg) ) * mid_dist_m;
    return opp;

deg_to_meter = deg_from_mid_to_meter;
wid_at_angle_deg = deg_from_mid_to_meter;

#REV: give me full value, not div 2
def flatscreen_dva( dm, wm ):
    """

    Parameters
    ----------
    dm :
        
    wm :
        

    Returns
    -------

    """
    return 2*math.degrees(math.atan2( wm/2, dm ));


#scaled linearly, not with tangent function. Only good to about 20deg.
def get_center_dva_per_meter( dm, ppm, reference_width_meters=0.01, reference_cutoff_dva=2.0 ):
    """

    Parameters
    ----------
    dm :
        
    ppm :
        
    reference_width_meters :
         (Default value = 0.01)
    reference_cutoff_dva :
         (Default value = 2.0)

    Returns
    -------

    """
    if( ppm <= 0 or dm <= 0 ):
        print("Error, dm,ppm<=0");
        exit(1);
        pass;
    
    rad = meter_to_rad( dm, reference_width_meters ); #REV: use only a very small area in the middle? 1cm...
    indeg = math.degrees(rad);
    
    if( indeg > reference_cutoff_dva ):
        print("Reference {} meters corresponds to more than {} degrees visual angle...you are very close to screen and errors will happen! (reference, 1 meter wide screen at 1 meter distance, right of screen is 50cm from middle and 45 degrees visual angle, so about 1.1ish dva/centimeter".format(reference_width_meters, reference_cutoff_dva ) );
        exit(1);
        pass;
    
    return (indeg/reference_width_meters);
