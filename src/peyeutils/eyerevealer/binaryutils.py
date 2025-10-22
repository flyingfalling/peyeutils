import struct
import json
import pandas as pd
import numpy as np
import sys

#REV:  use normal char? E.g. use ascii.. so 'd' for double, etc.
#REV:  tell endianness etc.?
def get_next_u8_time_from_filehandle( f ):
    toread=1;
    bytes = f.read(toread);
    if( len(bytes) < toread ):
        return None;
    return struct.unpack('<b', bytes)[0];

def get_next_u64_time_from_filehandle( f ):
    toread=8;
    bytes = f.read(toread);
    if( len(bytes) < toread ):
        return None;
    return struct.unpack('<Q', bytes)[0];

def get_next_u32_time_from_filehandle( f ):
    toread=4;
    bytes = f.read(toread);
    if( len(bytes) < toread ):
        return None;
    return struct.unpack('<I', bytes)[0];

def get_next_f32_time_from_filehandle( f ):
    toread=4;
    bytes = f.read(toread);
    if( len(bytes) < toread ):
        return None;
    return struct.unpack('<f', bytes)[0];

def get_next_f64_time_from_filehandle( f ):
    toread=8;
    bytes = f.read(toread);
    if( len(bytes) < toread ):
        return None;
    return struct.unpack('<d', bytes)[0];


tagtype = { 0 : 'uint64', 1 : 'float64', 2 : 'uint32', 3 : 'float32' };

def read_unpack_next_tagtype( myf, ts ):
    if( ts in tagtype ):
        tag = tagtype[ts];
    else:
        print("Unrecognized type...?");
        return None
    if( tag == 'uint64' ):
        return get_next_u64_time_from_filehandle(myf);
    elif( tag == 'uint32' ):
        return get_next_u32_time_from_filehandle(myf);
    elif( tag == 'float32' ):
        return get_next_f32_time_from_filehandle(myf);
    elif( tag == 'float64' ):
        return get_next_f64_time_from_filehandle(myf);
    else:
        return None;

#REV: this could get LARGE
def timestamps_from_file( fn, tb_hz_sec ):
    times = open( fn, "rb" );
    tstype = get_next_u8_time_from_filehandle( times );
    if tstype is not None:
        print( "Type {}".format( int(tstype)) );
        pass;
    else:
        exit(0);
        pass;

    res = [];
    idx=0;
    while( True ):
        val = read_unpack_next_tagtype( times, tstype );
        if( val == None ):
            break;
        else:
            #if( 0 == idx ):
            #    zerotime = val;
            #    pass;
            #assec = (val - zerotime) / tb_hz_sec;
            assec = val / tb_hz_sec;
            #print("Timestamp: {}  (Zeroed Sec: {})".format( val, assec ) );
            #if outf is not None:
            #    outf.write( "{} {} {}\n".format( idx, assec, val ) );
            res.append( [ idx, assec, val ] );
            idx+=1;
            pass;
        pass;
    
    return res;


#REV: I now have idx, sec, val for each. Note first is "zeroed"...
def df_from_type( colnames, myname, data ):
    if( False == isinstance( data, list ) ):
        data = [data];
        pass;
    
    for d in range(len(data)):
        colnames.append(myname+"_"+str(d));
        pass;
    
    df = pd.DataFrame( columns=colnames );
    return df;

def extract_timestamps(tsfname, tb_hz_sec, tname='timestamp'):
    tss = timestamps_from_file( tsfname, tb_hz_sec );
    tsdf = pd.DataFrame(columns=["idx",tname,"PTS"], data=tss); #the last one is the "raw" timestamp (e.g. PTS units, or sec units, etc.)
    return tsdf;
