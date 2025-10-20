import os;

import peyeutils;

def create_dir( path, exist_ok=True ):
    """

    Parameters
    ----------
    path :
        
    exist_ok :
         (Default value = True)

    Returns
    -------

    """
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
    """

    Parameters
    ----------
    fname :
        
    videoexts :
         (Default value = peyeutils.videoexts)

    Returns
    -------

    """
    ext = fname.split('.')[-1];
    if( ext.lower() in videoexts ):
        return True;
    else:
        return False;
    return None;

def is_filename_img( fname, imgexts=peyeutils.imgexts ):
    """

    Parameters
    ----------
    fname :
        
    imgexts :
         (Default value = peyeutils.imgexts)

    Returns
    -------

    """
    ext = fname.split('.')[-1];
    if( ext.lower() in imgexts ):
        return True;
    else:
        return False;
    return None;

def isfile(fname):
    """

    Parameters
    ----------
    fname :
        

    Returns
    -------

    """
    return os.path.isfile( fname );


## REV: todo make it generate so I can handle very very large files.
def ungzip_text_file_to_lines(fn, create_ungzip=False):
    import gzip
    dictlines=list();
    with gzip.open(fn, 'rb') as f_in:
        dictlines = f_in.readlines();
        pass;
    return dictlines;
