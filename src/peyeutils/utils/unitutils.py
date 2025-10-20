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



def vec3d_to_yawpitchroll_NWU(n,w,u):
    from eyeutils.nputils import compute_yaw_pitch_roll;
    
    if( len(n.shape) > 1 ):
        raise Exception("vec3d_to_yawpitchroll_NWU requires n,w,u vectors to be 1d...(np treats as horizontal vectors). Real shape is {}".format(n.shape));
    invec = np.vstack( (n, w, u) ).T;
    #Solve this way, as all vectors are by default "horizontal" vectors...
    
    #print(invec.shape)
    y, p, r = compute_yaw_pitch_roll( invec );
    return y, p, r;




def euler_angles_from_3d_vects2(vect1, vect2):
    """
    Calculates the rotation in Euler angles (XYZ convention) from vect1 to vect2.

    Args:
        vect1: A NumPy array representing the first 3D vector.
        vect2: A NumPy array representing the second 3D vector.

    Returns:
        A NumPy array representing the Euler angles (alpha, beta, gamma) in radians.
        Returns None if the input vectors are invalid or if a unique rotation cannot be determined.

    Raises:
       ValueError: If the input vectors are not valid 3D vectors or have zero magnitudes.
    """
    if( True ):
        raise Exception("Not sure if this should be used...REV");

    try:
        vect1 = np.array(vect1, dtype=float)
        vect2 = np.array(vect2, dtype=float)
    except ValueError:
        raise ValueError("Input vectors must be convertible to NumPy arrays.")


    if vect1.shape != (3,) or vect2.shape != (3,):
        raise ValueError("Input vectors must be 3-dimensional.")
    
    if np.linalg.norm(vect1) == 0 or np.linalg.norm(vect2) == 0:
        raise ValueError("Input vectors cannot have zero magnitude.")


    # Normalize the vectors
    vect1 = vect1 / np.linalg.norm(vect1)
    vect2 = vect2 / np.linalg.norm(vect2)


    # Calculate the rotation axis and angle
    v = np.cross(vect1, vect2)
    c = np.dot(vect1, vect2)
    s = np.linalg.norm(v)
    
    if s == 0:  # Vectors are collinear
        if c == 1:
            return np.array([0.0, 0.0, 0.0]) #No rotation
        else:
           return None #180 degree rotation around any axis perpendicular to vector 


    k = v / s

    # Calculate the rotation matrix (rodriguez's formula)
    k_x = np.array([[0, -k[2], k[1]],
                   [k[2], 0, -k[0]],
                   [-k[1], k[0], 0]])

    R = np.eye(3) + \
        k_x * s + \
        (np.dot(k_x, k_x) * (1-c))

    # Extract Euler angles from the rotation matrix (XYZ convention)
    alpha = np.arctan2(R[1, 0], R[0, 0])
    beta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    gamma = np.arctan2(R[2, 1], R[2, 2])

    return np.array([alpha, beta, gamma])




#REV: basically the same as above. Returns in order ZYX (yaw, pitch, roll).
#REV: necessary because we are not rotating from origin, but rather from weird direction maybe (forward/straight ahead).

# Compute for every element ?
def tait_bryan_angles(start_vector, end_vector):
    """
    Computes the NWU Tait-Bryan rotations (Z-Y'-X'') in intrinsic coordinates 
    from a starting vector to an ending vector using Rodrigues' formula.

    Args:
        start_vector: A NumPy array representing the starting vector (3x1).
        end_vector: A NumPy array representing the ending vector (3x1).

    Returns:
        A NumPy array containing the Tait-Bryan angles [yaw, pitch, roll] in radians.
        Returns None if the vectors are not valid or if a solution cannot be found
    """
    
    import numpy as np

    start_vector = np.array(start_vector, dtype=float)
    end_vector = np.array(end_vector, dtype=float)

    # Normalize vectors
    start_vector = start_vector / np.linalg.norm(start_vector)
    end_vector = end_vector / np.linalg.norm(end_vector)

    # Check if vectors are valid
    if np.isnan(start_vector).any() or np.isnan(end_vector).any():
        return np.array([np.nan, np.nan, np.nan])
        #print("Error: Invalid input vectors")
        #return None

    # Calculate the rotation axis and angle
    v = np.cross(start_vector, end_vector)
    c = np.dot(start_vector, end_vector)
    s = np.linalg.norm(v)

    if s == 0:  # Vectors are parallel or antiparallel
      if c == 1: # Vectors are parallel
          return np.array([0.0, 0.0, 0.0]) # No rotation needed
      else: # Vectors are antiparallel (180 degree rotation)
          # In this case, there are multiple solutions, 
          # we will define a rotation around the X-axis
          return np.array([0.0, np.pi, 0.0]) # rotation around pitch axis

    # Calculate rotation vector
    k = v / s
    theta = np.arctan2(s,c)

    # Rodrigues' rotation formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + \
        np.sin(theta)*K + \
        ((1 - np.cos(theta))*np.dot(K, K));
    
    # Extract yaw, pitch and roll.
    yaw = np.arctan2(R[1,0], R[0,0])
    pitch = np.arcsin(-R[2,0])
    roll = np.arctan2(R[2,1], R[2,2])

    return np.array([yaw, pitch, roll])



#REV: throw away roll if you don't care.
def compute_yaw_pitch_roll_single(target_pose):
    """
    Computes the yaw and pitch angles to reach a target pose from the current pose [1,0,0].
    
    Args:
        target_pose: A 3D vector representing the target pose.
    
    Returns:
        A tuple containing the yaw and pitch angles in degrees.
        Returns None if the input vector is invalid or if a unique rotation cannot be determined.
    """
    
    import numpy as np
    
    current_pose = np.array([1, 0, 0])
    try:
        #euler_angles = euler_angles_from_3d_vects2(current_pose, target_pose)
        euler_angles = tait_bryan_angles(current_pose, target_pose);
    except ValueError as e:
        print(f"Error computing Euler angles: {e}")
        return None
    
    if euler_angles is None:
        print("Could not compute unique Euler angles (vectors might be opposite)")
        return None
    
    yaw = np.degrees(euler_angles[0])  
    pitch = np.degrees(euler_angles[1]) 
    roll = np.degrees(euler_angles[2]) 

    return yaw, pitch, roll

#REV: expects columns to be x,y,z (NWU). Rows are timepoints.
def compute_yaw_pitch_roll_vec(target_vec):
    import numpy as np
    if( len(target_vec.shape) != 2 or
        target_vec.shape[1] != 3 ):
        raise Exception("Target vector must be of shape Nx3 (shape = {})".format(target_vec.shape));
    newsize=target_vec.shape[0];
    yaw=np.zeros(newsize);
    pitch=np.zeros(newsize);
    roll=np.zeros(newsize);
    for i in range(newsize):
        vec = target_vec[i,:]; #REV: row.
        y, p, r = compute_yaw_pitch_roll_single(vec);
        yaw[i] = y;
        pitch[i] = p;
        roll[i] = r;
        pass;
    return yaw, pitch, roll;

def compute_yaw_pitch_roll(target_vec):
    import numpy as np
    target_vec = np.array(target_vec);
    if( len(target_vec.shape) > 1 ):
        return compute_yaw_pitch_roll_vec(target_vec);
    else:
        return compute_yaw_pitch_roll_single(target_vec);
    pass;



def compute_lat_lon(vtarg):
    """Computes the latitude and longitude of a point on the Earth's surface.

    Args:
        vtarg: A 3D NumPy array representing the point's coordinates in the Earth-centered, Earth-fixed (ECEF) frame.

    Returns:
        A tuple containing the latitude (in degrees) and longitude (in degrees) of the point.
    """

    x = vtarg[0]
    y = vtarg[1]
    z = vtarg[2]

    # Calculate radius
    r = np.sqrt(x**2 + y**2 + z**2)

    # Calculate latitude
    latitude = np.degrees(np.arcsin(z / r))

    # Calculate longitude
    longitude = np.degrees(np.arctan2(y, x))

    return latitude, longitude
