## Creates both video with gaze overlaid (no gaze-centering?),
## as well as plots including gyro etc.

## Note we want to include head rotation/gaze/etc.

import argparse
import pandas as pd
import cv2
import numpy as np
import seaborn as sns;
from math import ceil, floor;
import matplotlib.pyplot as plt;

import peyeutils as pu;

import sys
import os


def gaussian_kernel_convolution(t, v, sigma):
  """Convolves a time series with a Gaussian kernel.
  
  Args:
    t: A NumPy array of irregularly sampled times.
    v: A NumPy array of corresponding sampled values.
    sigma: The standard deviation of the Gaussian kernel.

  """
  
  # Create a new time vector for the convolved signal
  #t_new = np.linspace(min(t), max(t), len(t) * 10)  # Increased resolution for smoother output
  
  #v_new = np.zeros_like(t_new, dtype=float)
  v_new = np.zeros_like(v, dtype=float)
  
  # Iterate over each new time point
  for i, t_i in enumerate(t):
    # Calculate the Gaussian kernel for the current time point
    weights = np.exp(-(t - t_i)**2 / (2 * sigma**2))
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Convolve the kernel with the signal values
    v_new[i] = np.sum(v * weights)
    pass;
  
  return v_new;




#https://stackoverflow.com/questions/18245076/optical-flow-and-focus-of-expansion-in-opencv
def focus_of_expansion(u, v, x, y):
    print(x.shape);
    print(v.shape);
    #A = np.array([ v.flatten(), flipy*u.flatten() ]); #REV: yes, flipped v,u on purpose.
    b = np.array(( x.flatten() * v.flatten() ) - ( y.flatten() * u.flatten() )) ; #REV: yes, flipped! On purpose x * Vy - y * Vx
    
    #print(x);
    #print(y);
    #print(v);
    #https://robotics.stackexchange.com/questions/15461/estimation-of-focus-of-expansion
    #https://www.youtube.com/watch?v=t9w_WuyVJ-A
    ut = -u.flatten();
    vt = v.flatten();
    foe_x = ( np.sum(vt * b) * np.sum(ut*ut) ) - ( np.sum(ut * b) * np.sum(ut*vt) );
    foe_y = ( np.sum(ut * b) * np.sum(vt*vt) ) - ( np.sum(vt * b) * np.sum(ut*vt) );
    
    #REV: ah, is this biased due to not being "center"?
    #REV: had to add the 1/ut term to it to scale correctly
    Q = 1 / (len(ut) * np.sum(vt*vt * ut*ut) - (np.sum(ut*vt)*np.sum(ut*vt)));
    #foe = b * A;, where A will be a n*2, foe is 2*1?, b is n*1
    #b is the coefficients, i.e. the weights of each vector. So, I will linreg to find the 
    return (foe_x*Q), (foe_y*Q);




#https://stackoverflow.com/questions/71990386/calculating-divergence-and-curl-from-optical-flow-and-plotting-it
#https://stackoverflow.com/questions/11435809/compute-divergence-of-vector-field-using-python
def divergence_nD_scalar(F):
    """ compute the divergence of n-D scalar field `F` """
    return reduce(np.add,np.gradient(F))


## f is list of ndarrays.
## u is X (horizontal) displacement, v is Y displacement (vertical).
def divergence2d(u, v):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    f = [u,v];
    num_dims = len(f); #REV: len(f) is same as f.shape[-1]?
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])



def extract_saccades_by_acceleration(df, acceleration_threshold=10000, deceleration_threshold=-10000):
    """
    Extracts saccades from a dataframe of time and x, y eye positions based on acceleration
    and deceleration criteria.
    
    Parameters:
    - df (pd.DataFrame): A dataframe with columns ['time', 'x', 'y'] representing 
                         time in seconds and x, y eye positions in degrees or pixels.
    - acceleration_threshold (float): The acceleration threshold for detecting saccade starts 
                                       (units depend on input, e.g., degrees/sec^2).
    - deceleration_threshold (float): The deceleration threshold for detecting saccade ends 
                                       (units depend on input, e.g., degrees/sec^2).
                                  
    Returns:
    - saccades (list): A list of dictionaries where each dictionary represents 
                       a saccade with start time, end time, peak acceleration, 
                       and duration.
    """
    if not {'time', 'x', 'y'}.issubset(df.columns):
        raise ValueError("Input dataframe must have columns: 'time', 'x', 'y'.")

    # Calculate time, velocity, and acceleration
    dt = np.diff(df['time'], prepend=df.iloc[0]['time']);
    dx = np.diff(df['x'], prepend=df.iloc[0]['x']);
    dy = np.diff(df['y'], prepend=df.iloc[0]['y']);
    
    
    velocity_x = dx / dt
    velocity_y = dy / dt
    velocity = np.sqrt(velocity_x**2 + velocity_y**2);
    
    #dvx = np.diff(velocity_x, prepend=velocity_x[0])
    #dvy = np.diff(velocity_y, prepend=velocity_y[0])
    
    acceleration = np.diff(velocity, prepend=velocity[0]) / dt; #np.sqrt(dvx**2 + dvy**2) / dt

    # Detect potential saccade start and end points
    is_high_acceleration = acceleration > acceleration_threshold
    is_high_deceleration = acceleration < deceleration_threshold

    # Identify saccades
    saccades = []
    in_saccade = False
    saccade_start = None
    
    for i, (acc_high, acc_low, time) in enumerate(zip(is_high_acceleration, is_high_deceleration, df['time'])):
        if acc_high and not in_saccade:
            # Start of a saccade
            in_saccade = True
            saccade_start = i
        elif acc_low and in_saccade:
            # End of a saccade
            in_saccade = False
            saccade_end = i
            peak_acceleration = acceleration[saccade_start:saccade_end].max()
            st = df.iloc[saccade_start-1].time;
            en = df.iloc[saccade_end].time;
            xvec=df.iloc[saccade_end].x - df.iloc[saccade_start-1].x;
            yvec=df.iloc[saccade_end].y - df.iloc[saccade_start-1].y;
            saccades.append({
                'start_time': st,
                'end_time': en,
                'peak_acceleration': peak_acceleration,
                'duration': (en-st),
                'xvec': xvec,
                'yvec': yvec,
            });
            pass;
        pass;
    saccades = pd.DataFrame(data=saccades);
    if( len(saccades.index) > 0 ):
        saccades = saccades[saccades.duration < 0.1].copy();
        pass;
    saccades['mag'] = np.sqrt( saccades.xvec**2 + saccades.yvec**2 );
    return saccades;


#REV: in weird cases we may be combining multiple "trials" of different people.
#REV: in which case, can't use "time" as a way of doing it...

#REV: this will be "mass", not mean.
#REV: need to know "scale",
#REV: sampscale is the scale (e.g. in time seconds) per sample. Note in some cases sample scale may be variable?
#REV: in that case need new function...
def plot_joint_marginal_histo_heatmap(df, tcol, xcol, ycol,
                                      xlab, ylab, cbarlab,
                                      xlim, ylim, xbinwid, ybinwid,
                                      aspect='equal',
                                      sampunit='sec',
                                      sampscale=1,
                                      sampdivscale=60,
                                      vmax=10,
                                      norm=True,
                                      figwid=8):
    nsamp = df[xcol].count(); #numeric_only=True);
    print(nsamp);
    # Build figure and axes
    fighei= figwid; #(ylim[1]-ylim[0])/(xlim[1]-xlim[0]) * figwid;
    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(figwid,fighei),
                            gridspec_kw=dict(height_ratios=[1, 3], width_ratios=[3, 1], hspace=0.1, wspace=0.1));

    myax = axs[1,0];
    xmarax = axs[0,0];
    ymarax = axs[1,1];
    emptyax=axs[0,1];
    
    emptyax.set_visible(False); #top-right gone.
    xmarax.set_box_aspect(1/3); #top marginal
    ymarax.set_box_aspect(3/1); #right marginal
    wcol="__weights";
    df[wcol] = sampscale; # default is number of "samples"; By setting this, we set it to number of 'sampleweight", e.g. seconds.
    ## WITHIN THE WHOLE DATA
    ## If we want to n
    
    if( norm ):
        df[wcol] /= float(nsamp);  #weight now represents total seconds per sample... I want PER SECOND. So need to multiply by 1/sampscale
        df[wcol] *= (1/sampscale); #now it is "seconds per second of total time".
        pass;
    
    df[wcol] *= sampdivscale;
    
    #REV: have to compute bin edges fuck this.
    xbins=range(xlim[0], xlim[1]+xbinwid, xbinwid);
    ybins=range(ylim[0], ylim[1]+ybinwid, ybinwid);
    
    h, xedge, yedge, img = myax.hist2d(data=df, x=xcol, y=ycol, weights=wcol, range=[xlim, ylim],
                                            bins=[xbins,ybins]
                                            ); #, vmin=0, vmax=vmax); #REV: some may be more than vmax??

    myax.set_xlabel(xlab);
    myax.set_ylabel(ylab);
    
    
    #from mpl_toolkits.axes_grid1 import make_axes_locatable;
    #divider = make_axes_locatable(axs[1,0]);
    #cax = divider.append_axes("left", size="5%", pad=0.05)
    cax=emptyax; #myax;
    #REV: pad will depend on label sizes...
    #https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.colorbar.html
    #fig.colorbar(img, ax=cax, fraction=0.05, pad=0.1, location='bottom', orientation='horizontal', label=cbarlab); #axs[1,0]);
    fig.colorbar(img, ax=cax, fraction=1, pad=0.1, aspect=10, anchor=(0.2,0.5), label=cbarlab); #axs[1,0]);
    
    #REV: fuck do I need to compute the marginal weights here? No...
    # Rotate the tick labels and set their alignment.
    ymarax.hist(data=df, x=ycol, weights=wcol, orientation='horizontal', bins=ybins, range=ylim, fill=False);
    ymarax.axhline(0, color='black', linestyle='--');
    xmarax.axvline(0, color='black', linestyle='--');
    myax.axhline(0, color='white', linestyle='--');
    myax.axvline(0, color='white', linestyle='--');
    
    xmarax.hist(data=df, x=xcol, weights=wcol, orientation='vertical', bins=xbins, range=xlim, fill=False);
    
    
    myax.set_aspect('equal');
    
        
    from matplotlib.transforms import Bbox
    (x0m, y0m), (x1m, y1m) = myax.get_position().get_points();
    (x0h, y0h), (x1h, y1h) = xmarax.get_position().get_points();
    (x0v, y0v), (x1v, y1v) = ymarax.get_position().get_points();
    
    xmarax.set_position(Bbox([[x0m, y0h], [x1m, y1h]]));
    ymarax.set_position(Bbox([[x0v, y0m], [x1v, y1m]]));

    #hh=y1h-y0h;
    #vh=x1v-x0v;
    #addit=0.07;
    #xmarax.set_position(Bbox([[x0m, y1m+addit], [x1m, y1m+addit+hh]]));
    #ymarax.set_position(Bbox([[x1m+addit, y0m], [x1m+addit+vh, y1m]]));
    
    xmarax.set_ylabel(cbarlab);
    ymarax.set_xlabel(cbarlab);
    
    #fig.tight_layout();
    return fig, axs;




def plot_gaze_histo( df, tcol, xcol, ycol ):
    #g = sns.histplot(data=df, x=xcol, y=ycol, binrange=((-40, 40), (-30, 30)), binwidth=2, );
    #g = sns.jointplot(data=df, x=xcol, y=ycol, kind='hist', binwidth=2, xlim=(-40,40), ylim=(-30,30), binrange=((-40, 40), (-30, 30)), marginal_kws=dict(binwidth=2 ));
    binwid=10;
    bxrange=(-40,40);
    byrange=(-30,30);
    g = sns.JointGrid(data=df, x=xcol, y=ycol );
    #g.ax_marg_x.set_aspect('equal')
    #g.ax_marg_y.set_aspect('equal')
    g.ax_joint.set_aspect('equal') 
    g = g.plot_joint(sns.histplot, binrange=(bxrange, byrange), binwidth=binwid );
    _ = g.ax_marg_x.hist(df[xcol], bins=range(bxrange[0], bxrange[1]+binwid, binwid)); #, range=bxrange)
    _ = g.ax_marg_y.hist(df[ycol], bins=range(byrange[0], byrange[1]+binwid, binwid), orientation='horizontal'); #range=byrange)
    from matplotlib.transforms import Bbox
    (x0m, y0m), (x1m, y1m) = g.ax_joint.get_position().get_points();
    (x0h, y0h), (x1h, y1h) = g.ax_marg_x.get_position().get_points()
    (x0v, y0v), (x1v, y1v) = g.ax_marg_y.get_position().get_points()
    #g.ax_marg_x.set_position(Bbox([[x0h, y0m], [x1h, ));
    g.ax_marg_y.set_position(Bbox([[x0v,y0m], [x1v,y1m]]));
    g.set_axis_labels(['X (DVA)', 'Y (DVA)']);
    ##https://stackoverflow.com/questions/71119762/resize-axes-of-top-and-right-joint-marginal-plots-to-match-central-plot-with-mat
    g.refline(x=0,y=None);
    g.refline(x=None,y=0);
    #https://stackoverflow.com/questions/29096632/getting-colorbar-in-a-hex-jointplot
    #g.set_aspect('equal');
    #g = sns.jointplot(data=df, x=xcol, y=ycol, kind='hist', binwidth=10, xlim=(-40,40), ylim=(-30,30), marginal_kws=dict(binwidth=10));
    
    #g.ax_marg_x.set_binrange(-40, 40)
    #g.ax_marg_y.set_binrange(-30, 30)
    #g.ax_marg_x.set_xlim(-40, 40)
    #g.ax_marg_y.set_ylim(-30, 30)

    #REV: make a function that allows me to show "mean" or "count" for cells and marginals.
    #REV: also set colorbar "scale" (i.e. normalize by sample rate to get "time" or total time, or e.g.
    #REV: time per second" etc.
    
    #https://stackoverflow.com/questions/48154910/seaborn-jointplot-fixing-bin-range-in-marginals
    #g.set_title('Gaze Distribution (DVA)');
    #plt.title('Gaze Distribution (DVA)');
    
    #plt.xlabel('X (DVA)');
    #plt.ylabel('Y (DVA)');
    #plt.xlim([-40,40]);
    #plt.ylim([-30,30]);
    #plt.axhline(0, color='red');
    #plt.axvline(0, color='red');
    
    #plt.show();
    return g.figure;

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

def paginate_timecourse(df, plot_func, output_filename, page_duration, time_col, **kwargs):
    """
    Slices a dataframe by time and saves plots to a multi-page PDF.
    
    Args:
        df (pd.DataFrame): The source data.
        plot_func (callable): Function taking (df, **kwargs) that creates a plot.
        output_filename (str): Name of the PDF file to save.
        page_duration (float/offset): The duration of time to show on one page.
        time_col (str, optional): The column to slice by. If None, uses index.
        **kwargs: Additional arguments passed to the plot_func.
    """
    # 1. Setup time boundaries
    time_series = df[time_col] if time_col else df.index
    start_time = time_series.min()
    end_time = time_series.max()
    
    with PdfPages(output_filename) as pdf:
        current_start = start_time
        
        while current_start < end_time:
            current_end = current_start + page_duration
            
            # 2. Slice the dataframe for the current "page"
            if time_col:
                mask = (df[time_col] >= current_start) & (df[time_col] < current_end)
                df_page = df.loc[mask]
            else:
                df_page = df.loc[current_start:current_end]
            
            # 3. Generate the plot if data exists for this window
            if not df_page.empty:
                # We pass the subsetted df and any extra styling kwargs
                fig = plot_func(df_page, page_duration=page_duration, **kwargs)
                
                # Ensure we are capturing the current figure if plot_func doesn't return one
                if fig is None:
                    fig = plt.gcf()
                
                # 4. Save page and close figure to save memory
                pdf.savefig(fig)
                plt.close(fig)
            
            current_start = current_end
            
    print(f"Successfully saved timecourse to {output_filename}")
    return;

def plot_head_eye_gaze(df2, page_duration):
    nrows=5;
    ncols=1;
    perrow=5;
    percol=24;
    pitch_color=(0.8, 0.5, 0.3);
    roll_color=(0.7,0.7,0.7);
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(perrow*nrows, percol*ncols));
    axs[0].plot(df2['Tsec0'], df2['gaze3d_yaw'], label='Yaw_EYE (Right=Positive)', color='red');
    axs[0].plot(df2['Tsec0'], df2['gaze3d_pitch'], label='Pitch_EYE', color=pitch_color);
    axs[0].plot(df2['Tsec0'], df2['gaze3d_roll'], label='Roll_EYE', color=roll_color);
    axs[0].legend();
    axs[0].set_ylim([-50, 50]);
    axs[0].set_xlim([df2['Tsec0'].min(), df2['Tsec0'].min()+page_duration]);
    axs[0].set_ylabel('Eye-in-Head (NWU deg)');
    axs[0].axhline(0, color='gray', linestyle='--');
    

    axs[1].plot(df2['Tsec0'], df2['eulerz'], label='Yaw_Z_HEAD (Right=Positive)', color='red'); #"real" tobii, X is LEFT, z is up? Y is forward?
    axs[1].plot(df2['Tsec0'], df2['eulerx'], label='Pitch_X_HEAD', color=pitch_color);
    axs[1].plot(df2['Tsec0'], df2['eulery'], label='Roll_Y_HEAD', color=roll_color);
    
    axs[1].legend();
    axs[1].set_ylim([-180, 180]);
    #axs[1].set_xlabel('Time (sec)');
    axs[1].set_xlim([df2['Tsec0'].min(), df2['Tsec0'].min()+page_duration]);
    axs[1].set_ylabel('Head-in-Compass (NWU deg)');
    axs[1].axhline(0, color='gray', linestyle='--');
    
    axs[2].plot(df2['Tsec0'], df2['eulerx']+df2['gaze3d_pitch'], label='Pitch_X_Gaze', color='red');
    axs[2].plot(df2['Tsec0'], df2['eulerz']+df2['gaze3d_yaw'], label='Yaw_Z_Gaze (Right=Positive)', color=pitch_color); #"real" tobii, X is LEFT, z is up? Y is forward?
    axs[2].plot(df2['Tsec0'], df2['eulery']+df2['gaze3d_roll'], label='Roll_Y_Gaze', color=roll_color);
    
    axs[2].legend();
    axs[2].set_ylim([-180, 180]);
    axs[2].set_xlim([df2['Tsec0'].min(), df2['Tsec0'].min()+page_duration]);
    axs[2].set_ylabel('Gaze-in-Compass (NWU deg)');
    axs[2].axhline(0, color='gray', linestyle='--');
    
    df2 = df2.sort_values(by='Tsec0').reset_index(drop=True);
    dt = df2['Tsec0'].diff();
    eyesigma=0.020;
    df2['seyeyaw'] = gaussian_kernel_convolution( df2['Tsec0'], df2['gaze3d_yaw'], eyesigma );
    
    df2['deyeyaw'] = df2['seyeyaw'].diff() / dt;
    
    CUTOFF=150;
    df2 = df2[ abs(df2['deyeyaw']) < CUTOFF ]; #dropping saccades... (should smooth first...)
    
    deyesigma=0.030;
    df2['deyeyaw'] = gaussian_kernel_convolution( df2['Tsec0'], df2['deyeyaw'], deyesigma );
    
    df2['dheadyaw'] = df2['eulerz'].diff() / dt;
    df2['gainyaw'] = -df2['deyeyaw'] / df2['dheadyaw'];
    

    axs[3].plot(df2['Tsec0'], df2['deyeyaw'], label='Yaw_EYE_VEL', color='orange');
    axs[3].plot(df2['Tsec0'], df2['dheadyaw'], label='Yaw_HEAD_VEL', color='purple');
    #axs[3].plot(df2['Tsec0'], df2['deyeyaw']+df2['dheadyaw'], label='Yaw_SUM_VEL', color='black');
    axs[3].legend();
    axs[3].set_xlim([df2['Tsec0'].min(), df2['Tsec0'].min()+page_duration]);
    axs[3].set_ylabel('Velocity (NWU deg)');
    axs[3].set_ylim([-100, 100]);
    axs[3].axhline(0, color='gray', linestyle='--');
    
    lpfsigma=0.500;
    df2.loc[abs(df2.gainyaw) > 2, 'gainyaw' ] = np.nan;
    sgain = gaussian_kernel_convolution( df2['Tsec0'], df2['gainyaw'], lpfsigma );
    axs[4].plot(df2['Tsec0'], sgain, label='Yaw_EYE_HEAD_gain', color='red');
    axs[4].set_ylabel('Gain (Eye/Head, LPF sigma={}s)'.format(lpfsigma));
    axs[4].set_xlabel('Time (sec)');
    axs[4].set_xlim([df2['Tsec0'].min(), df2['Tsec0'].min()+page_duration]);
    axs[4].set_ylim([-1, 2]);
    axs[4].axhline(0, color='gray', linestyle='--');
    return fig;
  


def create_video_overlay(recobj, df):
  vid = recobj.fullscenepath;
  cap = cv2.VideoCapture(vid);
  if( False == cap.isOpened() ):
    raise Exception("Can't open cap?");
  
  nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT);
  wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
  hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
  fps = int(round(cap.get(cv2.CAP_PROP_FPS)));
  if( fps < 2 or fps > 100 ):
    raise Exception("Something weird, FPS weirdly high/low {}".format(fps));
  
  vidlensec=(nframes/fps);
  vidtsdf = recobj.vidtsdf;
  print(vidtsdf.columns);
  overlaid = vid + '.gaze.mp4';
  print("Writing to {}".format(overlaid));
  fourcc = cv2.VideoWriter_fourcc(*'mp4v');
  vw = cv2.VideoWriter(overlaid, fourcc, fps, (wid, hei));
  
  fidx: int =0;
  while(True):
    ret, fr = cap.read();
    if( not ret ):
      print("Finished Video");
      break;
    myrow= vidtsdf[ vidtsdf['idx'] == fidx ];
    if( len(myrow.index) != 1 ):
      raise Exception("No row in vidtsdf");
    myrow = myrow.iloc[0];
    stt= (myrow['Tsec'] - 0.050)
    ent = myrow['Tsec'];
    #print("{} - {}".format(stt,ent));
    
    mydf=df[ (df['Tsec0'] > stt) & (df['Tsec0'] <= ent) ];
    #print(mydf);
    for i, row in mydf.iterrows():
      #print(row);
      if( np.isfinite(row.gaze2d_0) ):
        x=int(row.gaze2d_0*wid)
        y=int(row.gaze2d_1*hei);
        fr = cv2.circle(fr, (x,y), 24, (30,50,255), thickness=4);
        pass;
      pass;
    
    vw.write(fr);
    #cv2.imshow("Vid", fr);
    #key = cv2.waitKey(1);
    #if(key==ord('q')):
    #  exit(0);
    fidx+=1;
    pass;
  cap.release();
  vw.release();
  
  from moviepy import VideoFileClip, AudioFileClip
  print("Merging audio and encoding to H264...")
  original_clip = VideoFileClip(vid)
  
  processed_video = VideoFileClip(overlaid)
  
  # Attach original audio to processed video
  final_clip = processed_video.with_audio(original_clip.audio)

  mergedname=overlaid + '.merged.mp4';
  # Write final file with H264 encoding
  final_clip.write_videofile(mergedname, codec="libx264", audio_codec="aac")
  
  # Cleanup
  processed_video.close();
  original_clip.close();
  return;

def main():
    mypath = sys.argv[1]
    recobj = pu.tobiig3.tobiig3_official_recording(mypath, overwrite=True); 

    sr=100;
    recobj.resample_interpolate_dfs(create_csvs=True, sr_hzsec=sr);

    recobj.convert_to_nwu(create_csvs=True);
    recobj.gaze_to_ypr_deg(create_csvs=True);
    #Gaze is now in Yaw, Pitch, Roll (deg).
    ## Note this is GAZE IN HEAD-CENTERED
    
    
    vid = recobj.fullscenepath;
    
    cap = cv2.VideoCapture(vid);
    if( False == cap.isOpened() ):
      raise Exception("Can't open cap?");
    
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT);
    fps = cap.get(cv2.CAP_PROP_FPS);
    if( fps < 2 or fps > 100 ):
      raise Exception("Something weird, FPS weirdly high/low {}".format(fps));
    cap = None;
    
    vidlensec=(nframes/fps);
    
    df = recobj.gazeimudf;
    
    ahrsdf = pu.imu.ahrs_pose_heading( df, pretburnin=15, kind='tobiig3', srhzsec=100 ); #REV: other settings...
    df = pd.merge(df, ahrsdf, left_on='timestamp', right_on='tsec', how='outer');
    
    df2 = df[ abs(df.gaze2d_lr_01) < 2 ];
    df2 = df2[ abs(df2.gaze2d_du_01) < 2 ];
    
        
    print(df2.columns);
    
    savepath=os.path.join(mypath,'head_eye_gaze.pdf')
    print("Saving to: {}".format(savepath));
    #plt.savefig(savepath);
    paginate_timecourse(df2, plot_head_eye_gaze, savepath, time_col='Tsec0', page_duration=30);
    #plt.show();
    

    #REV: fuck need to handle audio too?
    create_video_overlay(recobj, df2); #could add plots to bottom e.g. with head acceleration etc?
    
    
    return 0;




if __name__=='__main__':
  exit(main());
  pass;

