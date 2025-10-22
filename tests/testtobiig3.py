import peyeutils as pu;

import pandas as pd;
import os;
import matplotlib.pyplot as plt;
import numpy as np;
import sys;
import cv2;
import math;


def main():

    mydir=sys.argv[1];
    outcsv='tobiig3csvs';

    recobj = pu.tobiig3.tobiig3_official_recording(mydir, outcsv, overwrite=True);
    sr=100;
    
    recobj.resample_interpolate_dfs(create_csvs=True, sr_hzsec=sr);
    
    recobj.convert_to_nwu(create_csvs=True);
    recobj.gaze_to_ypr_deg(create_csvs=True);


    vid = recobj.fullscenepath;
    cap = cv2.VideoCapture(vid);
    if( False == cap.isOpened() ):
        raise Exception("Can't open cap?");
    
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT);
    fps = cap.get(cv2.CAP_PROP_FPS);
    if( fps < 2 or fps > 100 ):
        raise Exception("Something weird, FPS weirdly high/low {}".format(fps));
    
    vidlensec=(nframes/fps);
    if( len(sys.argv) > 2 ):
        stsec=float(sys.argv[2]);
        ensec=float(sys.argv[3]);
        if( ensec < 0 ):
            # ensec -1 means "end of video"
            ensec = vidlensec;
            pass;
        print("MANUAL SET START/END");
        pass;
    else:
        if( vidlensec < 5 ):
            raise Exception("Video too short... should be at least 5 sec");
        stsec=0;
        ensec=vidlensec;
        print("DEFAULT START/END");
        pass;
    
    tag = '{}_{}'.format(math.floor(stsec), math.ceil(ensec));
    
    if( len(sys.argv) > 4 ):
        tag+='_{}'.format(sys.argv[4]);
        pass;
    
    
    print("Start/End seconds: [{}]-[{}] (Tagging with [{}])".format(stsec, ensec, tag));
    
        
    df = recobj.gazeimudf;

    print(df);

    fn='tobiig3.csv';
    print('saving to: {}'.format(fn));
    df.to_csv(fn, index=False);


    
    import peyeutils.eyemovements.remodnav as rv;
    
    params1 = rv.make_default_preproc_params(samplerate_hzsec=sr, timeunitsec=1, dva_per_px=1, xname='gaze2d_lr_dva',
                                             yname='gaze2d_du_dva',
                                             tname='Tsec0');
    params2 = rv.make_default_params(samplerate_hzsec=sr);
    params = params1 | params2;


    #REV: need to do peyeutils preprocessing as well?
    
    rdf = rv.remodnav_preprocess_eyetrace2d(eyesamps=df, params=params);
    ev = rv.remodnav_classify_events(rdf, params);
    
    print(ev);
    ev.to_csv('tg3events.csv', index=False);
        
    return 0;

if __name__=='__main__':
    exit(main());
    

