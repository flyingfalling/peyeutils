import cv2;
import pandas as pd
import av;
import numpy as np;
    
def frametimes_with_pyav(videofn: str, index: int = 0, timename='Tsec') -> pd.DataFrame: # -> List[float]:
    """
    Link: https://pypi.org/project/av/
    My comments:
        Works really well, but it is slower than ffprobe.
        The big advantage is that ffmpeg does not have to be installed on the computer, because pyav installs it automatically

    Parameters:
        videofn (str): Video path
        index (int): Stream index of the video.
    Returns:
        List of timestamps in SECONDS (floating point)
    """
    
    
    VIDIDX=0;
    container = av.open(videofn);
    #print([s for s in container.streams]);
    video = container.streams.get(index)[VIDIDX];
    #if( video.type != 'video' ):
    #    raise Exception("WTF not video?");
    
    if video.type != "video":
        raise ValueError(
            f'The index {index} is not a video stream. It is an {video.type} stream.'
        );
    
    av_timestamps = [
        {'idx':idx, timename:float(packet.pts * video.time_base), 'PTS':packet.pts}
        for idx, packet in enumerate(container.demux(video))
        if packet.pts is not None
    ];

    cap = cv2.VideoCapture( videofn );
    
    if( False == cap.isOpened() ):
        raise Exception("VID file {} does not exist".format(vidfn));
    
    wpx  = cap.get(cv2.CAP_PROP_FRAME_WIDTH); #flt
    hpx = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    fps = cap.get(cv2.CAP_PROP_FPS); #flt
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5);
    
    ghetto_idxs = np.array(range(nframes));
    #fps = int(fps);
    #if( fps != 25 ):
    #    raise Exception("Wow, big problem, fps={}".format(fps));
    ghetto_tss = ghetto_idxs / float(int(fps+0.5)); #REV: FPS Is their "true" FPS (24.9 or whatever), not ideal 25...
    ## REV: these are evenly spaced...

    ## OK, according to tobii pro lab, timestamps in PTS are what is important. And 25FPS is a lie. Shit. They line up. And diverge from
    ## 25 FPS by about 1.5 seconds every 10 minutes. See 
    
    container.close();
    timestampdf = pd.DataFrame( av_timestamps );
    timestampdf = timestampdf.sort_values(by=timename);
    d = timestampdf.idx.diff();
    if( d.min() != 1 and d.max() != 1 ):
        print(timestampdf);
        raise Exception("Wat, df diff min/max for PTS is fucked ({}, {})".format(d.min(), d.max()));
    
    timestampdf['drift'] = timestampdf[timename].diff()-(1/25.0); #0.040
    timestampdf['drift'] = timestampdf.drift.cumsum();
    print("Drift is [{}] seconds".format(timestampdf.iloc[ len(timestampdf.index)-1 ].drift));
    got = timestampdf[timename].max() - timestampdf[timename].min();
    expected = len(timestampdf.index)/25.0;
    print("Total time is [{}] s, time with 25Hz is: [{}]   (diff [{}] s)".format( got, expected, got - expected ));

    
    timestampdf['gts'] = ghetto_tss;
    timestampdf['ds'] = timestampdf[timename] - timestampdf.gts;
    print("Min: {}   Max: {}".format(timestampdf.ds.min(), timestampdf.ds.max()));
    if( timestampdf.ds.abs().max() > 0.1 ):
        print(" +++++++ We have a problem, max offset is: {}".format(timestampdf.ds.abs().max()));
        pass;
    #av_timestamps.sort();
    
    return timestampdf; #av_timestamps;



def read_video_timestamps(scenefullpath, timename): #row, mypath):
    cap = cv2.VideoCapture( scenefullpath );
    if( False == cap.isOpened() ):
        raise Exception("VID file {} does not exist".format(scenefullpath));
    
    wpx  = cap.get(cv2.CAP_PROP_FRAME_WIDTH); #flt
    hpx = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    fps = cap.get(cv2.CAP_PROP_FPS); #flt
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT);

    vidtimedf = frametimes_with_pyav( scenefullpath, timename=timename );
    vidlensec = vidtimedf[timename].max() - vidtimedf[timename].min();
    print("Video: [{}] {}x{} ({} fps, {} frames, i.e. [{}] duration), (PTS [{}]s - [{}]s, [{}]s long)".format(
        scenefullpath, wpx, hpx, fps, nframes, nframes/fps, vidtimedf[timename].min(), vidtimedf[timename].max(), vidlensec));
    
    metadict = dict(wpx=wpx,hpx=hpx,fps=fps,nframes=nframes,vidlensec=vidlensec);
    
    return cap, vidtimedf, metadict;
