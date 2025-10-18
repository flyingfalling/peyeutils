import os;

import peyeutils;

def create_dir( path, exist_ok=True ):
    success=True;
    try:
        os.makedirs(path, exist_ok=exist_ok);
        pass;
    except Exception as e:
        success=False;
        print("MKDIR {} excepted: [{}]".format(path, e));
        pass;
    
    return success;


def is_filename_vid( fname, videoexts=peyeutils.videoexts ):
    ext = fname.split('.')[-1];
    if( ext.lower() in videoexts ):
        return True;
    else:
        return False;
    return None;

def is_filename_img( fname, imgexts=peyeutils.imgexts ):
    ext = fname.split('.')[-1];
    if( ext.lower() in imgexts ):
        return True;
    else:
        return False;
    return None;

def isfile(fname):
    return os.path.isfile( fname );
