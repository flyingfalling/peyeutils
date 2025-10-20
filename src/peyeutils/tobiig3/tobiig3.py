'''
# REV: 2024/12/04
# "Extract" a Tobii G3 recording (i.e. on SD card)
# Specifically, convert gz/etc to JSON, then to CSV

## MAYBE:
# Resample and interpolate? I.e. asynchronous magnetometer samples etc.
'''

import pandas as pd;
import numpy as np;
import sys, os;
import cv2;
import json;

from peyeutils.utils import isfile;


## Problem is that "image" will always be fucked, since Y will be "up" and x
## is right...

## Tobii axes are X, Y, Z in WUN order... NWU means X forward, Y left, Z up
#REV: just reorders column names, not columns themselves.

def tobiig3_axes_to_NWU(collist):
    axis_order=[2,0,1]
    retord=[ collist[x] for x in axis_order ];
    return retord;
    
def tobii3_fix_imudf(normdf, prepend='data.'):

    df = normdf.copy(0);
    withmag = False;
    toconvert = [prepend+'accelerometer', prepend+'gyroscope', prepend+'magnetometer'];
    print(df.columns);
    
    if( prepend+'accelerometer' in df.columns ):
        tmpacc = [ [x[0], x[1], x[2]] if type(x) is not float
                   else [x, x, x] for x in
                   df[prepend+'accelerometer'].tolist() ];
        #df[['accx_ms2', 'accy_ms2', 'accz_ms2']] = pd.DataFrame(tmpacc, index=df.index);
        df[ ['accelerometer_{}'.format(x) for x in range(3)] ] = pd.DataFrame(tmpacc, index=df.index);
        pass;
    else:
        #REV: In case data missing, still need columns to exist.
        df[ ['accelerometer_{}'.format(x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        pass;

    
    if( prepend+'gyroscope' in df.columns ):    
        tmpgyr = [ [x[0], x[1], x[2]] if type(x) is not float
                   else [x, x, x] for x in
                   df[prepend+'gyroscope'].tolist() ];
        #df[['gyrx_degsec', 'gyry_degsec', 'gyrz_degsec']] = pd.DataFrame(tmpgyr, index=df.index);
        df[ ['gyroscope_{}'.format(x) for x in range(3)] ] = pd.DataFrame(tmpgyr, index=df.index);
        pass;
    else:
        df[ ['gyroscope_{}'.format(x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        pass;
        
    
    if( prepend+'magnetometer' in df.columns ):
        
        tmpmag = [ [x[0], x[1], x[2]] if type(x) is not float
                   else [x, x, x] for x in
                   df[prepend+'magnetometer'].tolist() ];
        
        #df[['magx_uT', 'magy_uT', 'magz_uT']] = pd.DataFrame(tmpmag, index=df.index);
        df[ ['magnetometer_{}'.format(x) for x in range(3)] ] = pd.DataFrame(tmpmag, index=df.index);
        pass;
    else:
        df[ ['magnetometer_{}'.format(x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        pass;
    
    
    df = df[ [ c for c in list(df.columns) if c not in toconvert ] ];
    
        
    return df;



def tobii3_fix_gazedf(normdf, prepend='data.'):
    
    df = normdf.copy(0);
    
    #toconvert = ['data.gaze2d', 'data.gaze3d', 'data.eyeleft.gazeorigin', 'data.eyeleft.gazedirection', 'data.eyeright.gazeorigin', 'data.eyeright.gazedirection'];
    toconvert = list();
    for testit in [prepend+'gaze2d',
                   prepend+'gaze3d',
                   prepend+'eyeleft.gazeorigin',
                   prepend+'eyeleft.gazedirection',
                   prepend+'eyeright.gazeorigin',
                   prepend+'eyeright.gazedirection']:
        if( testit in df.columns ):
            toconvert.append(testit);
            pass;
        pass;

    #if( len(df['data.gaze2d'].tolist()) != len(df.index) ):
    #    raise Exception("tolist on nested list in column is broken");
    
    if( prepend+'gaze2d' in toconvert ):
        tmp2d = [ [x[0], x[1]] if type(x) is not float
                  else [x, x] for x in
                  df[prepend+'gaze2d'].tolist() ];
        #df[['x01_2d', 'y01_2d']] = pd.DataFrame(tmp2d, index=df.index);
        df[ ['gaze2d_{}'.format(x) for x in range(2)] ] = pd.DataFrame(tmp2d, index=df.index);
        pass;
    else:
        df[ ['gaze2d_{}'.format(x) for x in range(2)] ] = pd.DataFrame([[np.nan, np.nan]]*len(df.index), index=df.index);
        pass;


    
    if( prepend+'gaze3d' in toconvert ):
        tmp3d = [ [x[0], x[1], x[2]] if type(x) is not float
                  else [x, x, x] for x in
                 df[prepend+'gaze3d'].tolist() ];
        #df[['xmm_3d', 'ymm_3d', 'zmm_3d']] = pd.DataFrame(tmp3d, index=df.index);
        df[ ['gaze3d_{}'.format(x) for x in range(3)] ] = pd.DataFrame(tmp3d, index=df.index);
        pass;
    else:
        df[ ['gaze3d_{}'.format(x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        pass;


    
    eye='left';
    if( prepend+'eyeleft.gazeorigin' in toconvert and
        prepend+'eyeleft.gazedirection' in toconvert ):
        tmplorig = [ [x[0], x[1], x[2]] if type(x) is not float
                     else [x, x, x] for x in
                     df[prepend+'eyeleft.gazeorigin'].tolist() ];
        tmpldir = [ [x[0], x[1], x[2]] if type(x) is not float
                    else [x, x, x] for x in
                    df[prepend+'eyeleft.gazedirection'].tolist() ];
        
        
        df[ ['eye{}_gazedirection_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame(tmpldir, index=df.index);
        df[ ['eye{}_gazeorigin_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame(tmplorig, index=df.index);
        pass;
    else:
        df[ ['eye{}_gazedirection_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        df[ ['eye{}_gazeorigin_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        pass;
    
    
    eye='right';
    if( prepend+'eyeright.gazeorigin' in toconvert and
        prepend+'eyeright.gazedirection' in toconvert ):
        tmprorig = [ [x[0], x[1], x[2]] if type(x) is not float
                     else [x, x, x] for x in
                     df[prepend+'eyeright.gazeorigin'].tolist() ];
        tmprdir = [ [x[0], x[1], x[2]] if type(x) is not float
                    else [x, x, x] for x in
                    df[prepend+'eyeright.gazedirection'].tolist() ];
        
        df[ ['eye{}_gazedirection_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame(tmprdir, index=df.index);
        df[ ['eye{}_gazeorigin_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame(tmprorig, index=df.index);
        pass;
    else:
        df[ ['eye{}_gazedirection_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        df[ ['eye{}_gazeorigin_{}'.format(eye, x) for x in range(3)] ] = pd.DataFrame([[np.nan, np.nan, np.nan]]*len(df.index), index=df.index);
        pass;
        #df[['origL_x', 'origL_y', 'origL_z']] = pd.DataFrame(tmplorig, index=df.index);
    #df[['origR_x', 'origR_y', 'origR_z']] = pd.DataFrame(tmprorig, index=df.index);
    #df[['dirL_x', 'dirL_y', 'dirL_z']] = pd.DataFrame(tmpldir, index=df.index);
    #df[['dirR_x', 'dirR_y', 'dirR_z']] = pd.DataFrame(tmprdir, index=df.index);
    
    df = df[ [ c for c in list(df.columns) if c not in toconvert ] ];
    
    #df.rename( columns={prepend+'eyeleft.pupildiameter':'pdL_mm', prepend+'eyeright.pupildiameter':'pdR_mm'}, inplace=True);
    #pupil diameter is 1d always?
    if( prepend+'eyeleft.pupildiameter' in df.columns and
        prepend+'eyeright.pupildiameter' in df.columns ):
        df.rename( columns={prepend+'eyeleft.pupildiameter':'eyeleft_pupildiameter_0',
                            prepend+'eyeright.pupildiameter':'eyeright_pupildiameter_0'},
                   inplace=True);
        pass;
    else:
        df['eyeleft_pupildiameter_0'] = np.nan;
        df['eyeright_pupildiameter_0'] = np.nan;
        pass;
    
    #print(df.columns);
    
    return df;



#REV: shit, need to refer to INTRINSIC versus EXTRINSIC axes.
def tobiig3_WUN_gaze3d_axes_to_NWU(df, axis_order=[2,0,1], dropold=False):
    toswap=['eye{}_gazedirection'.format(x) for x in ['left', 'right'] ];
    toswap += [ 'eye{}_gazeorigin'.format(x) for x in ['left', 'right'] ];
    toswap += [ 'gaze3d' ];

    #REV: colorder is order of NWU columns in WUN. I.e. to get [0] of NWU, take [2] of WUN.
    for col in toswap:
        for x in range(3):
            oldvar = "{}_{}".format(col, axis_order[x]);
            newvar = "{}_NWU_{}".format(col, x);
            df[ newvar ] = df[ oldvar ];
            pass;
        pass;

    if( dropold ):
        #REV: drop old columns...
        df = df[ [ col for col in df.columns if col not in toswap ] ];
        pass;
    
    return df;



########### CLASS TG3 OFFICIAL RECORDING ###################
class tobiig3_official_recording():
    availmodes=['official', 'eyerevealer'];
    def __init__(self, mypath, overwrite=False, mode='official'):
        if( mode not in tobiig3_official_recording.availmodes ):
            raise Exception("REV: Mode {} not in available modes: (Available: {})".format(mode, tobiig3_official_recording.availmodes));
        
        self.mode = mode;
        self.mypath = mypath;
        self.tcol = 'timestamp';
        self.tcolunit_s = 1.0; #second units.
        self.overwrite=overwrite;
        self.sc_widdva=95; #REV: should extract from camera params actually...
        self.sc_heidva=63;
        #REV: need to "preconvert" if it is an eyerevealer that needs to be extracted?
        self.set_params(); #REV: this "overwrites" mode
        self.set_fnames();
        self.open_tobiig3_recording();
        pass;
    
    def set_params(self):
        self.recordingfilepath = os.path.join(self.mypath, 'recording.g3');
        self.recparams = dict();
        if( os.path.isfile( self.recordingfilepath ) ):
            with open(self.recordingfilepath) as f:
                self.recparams = json.loads(f.read());
                print(self.recparams);
                pass;
            
            self.scenevidfn = self.recparams['scenecamera']['file'];
            self.gazedatafn = self.recparams['gaze']['file'];
            self.imudatafn =  self.recparams['imu']['file'];
            self.expectedgazesamps = self.recparams['gaze']['samples'];
            self.expectedgazevalidsamps = self.recparams['gaze']['valid-samples'];
            pass;
        else:
            print("SETTING MODE TO eyerevealer (no recfile?)");
            self.mode = 'eyerevealer';
            print("Recording file (.g3) does not exist, probably created via eyerevealer/rteye2 stream?");
            #self.recparams = dict(scenevidfn='tobii3_scene.mkv', gazedatafn='gazedata.gz', imudatafn='imudata.gz', );
            self.recparams=dict();
            vidbase = 'tobii3_scene';
            if( self.mode == 'eyerevealer' ):
                vidbase = 'tobii3_scene';
                pass;
            vidpath=os.path.join(self.mypath, vidbase);
            
            if( os.path.isfile(vidpath+'.mkv') ): #REV: depends on "version"
                self.scenevidfn = vidbase+'.mkv';
                pass;
            elif( os.path.isfile(vidpath+'.mp4') ):
                self.scenevidfn = vidbase+'.mp4';
                pass;
            else:
                print("WARNING no scene video found (expected [{}] + .mkv or .mp4)".format(vidpath));
                self.scenevidfn = None;
                pass;
            self.gazedatafn = 'gazedata.gz';
            self.imudatafn = 'imudata.gz';
            self.expectedgazesamps = -1;
            self.expectedgazevalidsamps = -1;
            pass;
                
        return;
    
    def open_tobiig3_recording(self):
        import json
        self.get_camera_info();
        self.get_timestamps();
        return;
        
    def get_camera_info(self):
        if( len(self.recparams) > 0 ):
            self.camerainfo = self.recparams['scenecamera']['camera-calibration'];
            self.campos = self.camerainfo['position'];
            self.camfoc = self.camerainfo['focal-length'];
            self.camrot = np.array(self.camerainfo['rotation']);
            self.camskew = self.camerainfo['skew'];
            self.campp = self.camerainfo['principal-point'];
            self.camraddist = self.camerainfo['radial-distortion'];
            self.camtangdist = self.camerainfo['tangential-distortion'];
            self.camresol = self.camerainfo['resolution'];
            pass;
        else:
            print("No rec params (maybe eyerevealer)");
            pass;

                
        return;
    
    def get_timestamps(self):
        
        if( self.overwrite == False and
            isfile(self.outvidtspath)
           ):
            print("Files exist so passing load vid ts step... (set overwrite==True to do anyways)");
            self.vidtsdf = pd.read_csv(self.outvidtspath);
            return;
        else:
            if( 'official' == self.mode ):
                from eyeutils.vidutils import read_video_timestamps;
                cap, self.vidtsdf = read_video_timestamps(self.fullscenepath);
                pass;
            elif( 'eyerevealer' == self.mode ):
                from eyeutils.dataconversion.tobiig3.resample_tobii3_data import resample_tobii3_vid_ts;
                self.vidtsdf = resample_tobii3_vid_ts(self.vidtspath, overwrite=self.overwrite, savecsvs=True);
                pass;
            else:
                raise Exception("Unrec mode");
            pass;
        if( 'Tsec' not in self.vidtsdf ):
            self.vidtsdf['Tsec'] = self.vidtsdf[self.tcol] * self.tcolunit_s;
            pass;
        if( 'Tsec0' not in self.vidtsdf ):
            self.vidtsdf['Tsec0'] = self.vidtsdf['Tsec'] - self.vidtsdf['Tsec'].min();
            pass;
        return;

    def set_fnames(self):
        
        self.fullgazepath = os.path.join( self.mypath, self.gazedatafn );
        self.fullimupath = os.path.join( self.mypath, self.imudatafn );
        self.fullscenepath = os.path.join( self.mypath, self.scenevidfn);
        
        self.vidtspath = self.fullscenepath + '.ts';
        self.outvidtspath = self.fullscenepath + '.ts.csv';
        self.outgazepath = self.fullgazepath + '.csv'; #REV: OK, so this outputs to "gaze.gz.csv". But how does it READ?
        self.outimupath = self.fullimupath + '.csv';
        self.outgazeimupath = os.path.join(self.mypath, 'gazeimudata.csv');
        self.outgazepath_re = self.fullgazepath + '.resampled.csv';
        self.outimupath_re = self.fullimupath + '.resampled.csv';
        self.outgazeimupath_re = os.path.join(self.mypath, 'gazeimudata.resampled.csv');
            
        self.outgazeimupath_nwu = os.path.join(self.mypath, 'gazeimudata.resampled.nwu.csv');
        self.outgazeimupath_nwuypr = os.path.join(self.mypath, 'gazeimudata.resampled.nwu.ypr.csv');
        return;
    
    def dfs_from_tobiig3_jsongzs(self, create_csvs=False):
        if( self.overwrite == False and
            (isfile(self.outgazepath) and
             isfile(self.outimupath) and
             isfile(self.outgazeimupath) and
             isfile(self.outvidtspath) )
           ):
            print("Files exist so passing original DF creation step... (set overwrite==True to do anyways)");
            self.gazedf = pd.read_csv(self.outgazepath);
            self.imudf = pd.read_csv(self.outimupath);
            self.gazeimudf = pd.read_csv(self.outgazeimupath);
            self.vidtsdf = pd.read_csv(self.outvidtspath); #REV: this should be set by "get_timestamps"
            return;
        
        
        gazeaslines = ungzip_text_file_to_lines(self.fullgazepath);

        imuaslines = ungzip_text_file_to_lines(self.fullimupath);
        
        gazejsonlines = [ json.loads(line) for line in gazeaslines ];
        imujsonlines = [ json.loads(line) for line in imuaslines ];
        
        #gazedf = pd.DataFrame( gazejsonlines );
        gazedf = pd.json_normalize( gazejsonlines );
        self.gazedf = tobii3_fix_gazedf(gazedf);
        
        imudf = pd.json_normalize( imujsonlines );
        self.imudf = tobii3_fix_imudf(imudf);
        
        gazeimudf = pd.merge(self.gazedf, self.imudf, on=['timestamp'], how='outer');
        #REV: type is pointless, was just "gaze" or "imu"
        self.gazeimudf = gazeimudf[ [c  for c in list(gazeimudf.columns) if c not in ('type_x', 'type_y') ] ];
        
        print(self.gazeimudf.columns);
                
        if(       create_csvs ):
            print("Saving to:\nGAZE: [{}]\nIMU: [{}]\nVIDTS: [{}]\nGAZEIMU: [{}]".format(self.outgazepath,
                                                                                         self.outimupath,
                                                                                         self.outvidtspath,
                                                                                         self.outgazeimupath));
            self.gazedf.to_csv(self.outgazepath, index=False);
            self.imudf.to_csv(self.outimupath, index=False);
            self.vidtsdf.to_csv(self.outvidtspath, index=False);
            self.gazeimudf.to_csv(self.outgazeimupath, index=False);
            pass;
        
        return;
    
    def convert_to_nwu(self, create_csvs=True):
        #self.gazeimudf
        if( self.overwrite == False and
            isfile(self.outgazeimupath_nwu)
           ):
            print("Files exist so passing gaze_to_nwu step... (set overwrite==True to do anyways)");
            self.gazeimudf = pd.read_csv(self.outgazeimupath_nwu);
            return;
        
        toconv = [
            'eyeleft_gazedirection',
            'eyeleft_gazeorigin',
            'eyeright_gazedirection',
            'eyeright_gazeorigin',
            'accelerometer',
            'gyroscope',
            'magnetometer',
            'gaze3d',
        ];

                
        
        oldnames=['{}'.format(x) for x in range(3)];
        newnames=['{}'.format(x) for x in ['n', 'w', 'u'] ];

        for prefix in toconv:
            collist = [ '{}_{}'.format(prefix,x) for x in oldnames ];
            newcollist = [ '{}_{}'.format(prefix,x) for x in newnames ];
            neword = tobiig3_axes_to_NWU(collist);
            swapped = self.gazeimudf[ neword ];
            
            #renamedict = { '{}_{}'.format(prefix,o):'{}_{}'.format(prefix,n) for o,n in zip(oldnames, newnames) };
            swappedcollist = list(swapped.columns); #REV: order should now be correct
            renamedict = { o:n for o,n in zip(swappedcollist, newcollist) };
            print(renamedict);
            swapped = swapped.rename( columns=renamedict );
            print(swapped.columns);
            #self.gazeimudf[ newcollist ] = swapped; #REV: will this work?
            #REV: simply setting will not respect names (will use ORDER) e.g. df[['c', 'd']] = df2 NOT same as
            # df[['d', 'c']] = df2
            for c in swapped.columns:
                if( c in self.gazeimudf.columns ):
                    raise Exception("BIG ERROR, column [{}] already exists in gazeimudf".format(c));
                self.gazeimudf[ c ] = swapped[ c ];
                pass;
            pass;
        
                
        self.gazeimudf[ 'gaze2d_lr_01' ] = self.gazeimudf.gaze2d_0; #left-right
        self.gazeimudf[ 'gaze2d_ud_01' ] = self.gazeimudf.gaze2d_1; #up-down (Y-positive is DOWN)
        self.gazeimudf[ 'gaze2d_du_01' ] = 1 - self.gazeimudf.gaze2d_ud_01; #down-up (Y-positive is up)
        
        self.gazeimudf[ 'gaze2d_lr_dva' ] = (self.gazeimudf.gaze2d_lr_01 - 0.5) * self.sc_widdva; #left-right
        self.gazeimudf[ 'gaze2d_du_dva' ] = (self.gazeimudf.gaze2d_du_01 - 0.5) * self.sc_heidva; #down-up
        
        if(             create_csvs ):
            print("Outputting Gaze/IMU df in NWU to [{}]".format(self.outgazeimupath_nwu));
            self.gazeimudf.to_csv(self.outgazeimupath_nwu, index=False);
            pass;
        return;
    
    
    def gaze_to_ypr_deg(self, create_csvs=True):
        ## Convert gaze3d, gazeorigin/gazedirection, and gaze2d
        if( self.overwrite == False and
            isfile(self.outgazeimupath_nwuypr)
           ):
            print("Files exist so passing gaze_to_ypr step... (set overwrite==True to do anyways)");
            self.gazeimudf = pd.read_csv(self.outgazeimupath_nwuypr);
            return;
        
        ## GAZE 2D:
        ## LINEAR Simply convert to angle.
        ### DOES NOT USE LENS DISTORTION!!!!
        ## Just use rough to compute.
        widdva=self.sc_widdva;
        heidva=self.sc_heidva;
        # Do not invert Y (UD) because NWU convention, positive pitch should be down.
        INVERTY = 1;
        
        # Invert X (LR) because positive yaw in NWU convention should rotate to LEFT
        INVERTX = -1;
        
        y = INVERTX * (self.gazeimudf.gaze2d_lr_01 - 0.5) * widdva;
        p = INVERTY * (self.gazeimudf.gaze2d_ud_01 - 0.5) * heidva;
        r = y * 0;
        
        self.gazeimudf['gaze2d_yaw'] = y;
        self.gazeimudf['gaze2d_pitch'] = p;
        self.gazeimudf['gaze2d_roll'] = r; 
        
        
        ## GAZE 3D:
        ## Simply use this
        c = ['gaze3d_n', 'gaze3d_w', 'gaze3d_u'];
        nwu=['n','w','u'];
        ypr=['yaw','pitch','roll']
        y,p,r = vec3d_to_yawpitchroll_NWU(self.gazeimudf[c[0]],
                                          self.gazeimudf[c[1]],
                                          self.gazeimudf[c[2]]
                                          );
        self.gazeimudf['gaze3d_yaw'] = y;
        self.gazeimudf['gaze3d_pitch'] = p;
        self.gazeimudf['gaze3d_roll'] = r; #REV: this should actually be 0.
        
        ## EYELEFT
        #REV: could compute gaze direction from left/right eye...i.e. vergence angles. This will depend on subtracting origin.
        prefix='eyeleft_gazedirection';
        prefix2='eyeleft_gazeorigin';
        c = ['{}_{}'.format(prefix, x) for x in nwu ];
        co = ['{}_{}'.format(prefix2, x) for x in nwu ];
        n = self.gazeimudf[c[0]] - self.gazeimudf[co[0]]; #REV: get true vector from origin by subtracting origin.
        w = self.gazeimudf[c[1]] - self.gazeimudf[co[1]]; #REV: get true vector from origin by subtracting origin.
        u = self.gazeimudf[c[2]] - self.gazeimudf[co[2]]; #REV: get true vector from origin by subtracting origin.
        
        y,p,r = vec3d_to_yawpitchroll_NWU(n,
                                          w,
                                          u
                                          );
        
        self.gazeimudf['{}_yaw'.format(prefix)] = y;
        self.gazeimudf['{}_pitch'.format(prefix)] = p;
        self.gazeimudf['{}_roll'.format(prefix)] = r;

        
        ## EYERIGHT
        prefix='eyeright_gazedirection';
        prefix2='eyeright_gazeorigin';
        c = ['{}_{}'.format(prefix, x) for x in nwu ];
        co = ['{}_{}'.format(prefix2, x) for x in nwu ];
        n = self.gazeimudf[c[0]] - self.gazeimudf[co[0]]; #REV: get true vector from origin by subtracting origin.
        w = self.gazeimudf[c[1]] - self.gazeimudf[co[1]]; #REV: get true vector from origin by subtracting origin.
        u = self.gazeimudf[c[2]] - self.gazeimudf[co[2]]; #REV: get true vector from origin by subtracting origin.
        
        y,p,r = vec3d_to_yawpitchroll_NWU(n,
                                          w,
                                          u
                                          );
        
        self.gazeimudf['{}_yaw'.format(prefix)] = y;
        self.gazeimudf['{}_pitch'.format(prefix)] = p;
        self.gazeimudf['{}_roll'.format(prefix)] = r;
        
        
        if(      create_csvs ):
            print("Outputting Gaze/IMU df in YPR to [{}]".format(self.outgazeimupath_nwuypr));
            self.gazeimudf.to_csv(self.outgazeimupath_nwuypr, index=False);
            pass;
        return;

    
    
    
    def resample_interpolate_dfs(self, sr_hzsec=100.0, create_csvs=True):
        #REV: create new equally(ish) spaced values?
        from eyeutils.timeseriesutils import interpolate_df_to_samplerate;
        
        if( self.overwrite == False and
            (isfile(self.outgazepath_re) and
            isfile(self.outimupath_re) and
            isfile(self.outgazeimupath_re))
           ):
            self.gazedf = pd.read_csv(self.outgazepath_re);
            self.imudf = pd.read_csv(self.outimupath_re);
            self.gazeimudf = pd.read_csv(self.outgazeimupath_re);
            print("Files exist so skipping resample step... (set overwrite==True to do anyways)");
            return;
        
            
        
        if( False == hasattr(self, 'gazedf') ): #self.gazeimudf ):
            print("Reading/Creating initial load");
            if( 'official' == self.mode ):
                self.dfs_from_tobiig3_jsongzs(create_csvs);
                pass;
            elif( 'eyerevealer' == self.mode ):
                from eyeutils.dataconversion.tobiig3.resample_tobii3_data import resample_tobii3_json_to_csv;
                jsonfname = os.path.join(self.mypath, 'tobii3_data.json');
                self.gazeimudf, dicts = resample_tobii3_json_to_csv( jsonfname, sr_hzsec,
                                                                     savecsvs=True, overwrite=self.overwrite
                                                                    );
                self.gazedf = dicts['gaze'];
                self.imudf = dicts['imu'];
                pass;
            else:
                raise Exception("Unrecognized mode: {}".format(self.mode));
            pass;
        
        if( sr_hzsec is None ):
            if( create_csvs ):
                print(
                    "Saving (raw) CSVs to:\nGAZE: [{}]\nIMU: [{}]".format(
                        self.outgazepath_re,
                        self.outimupath_re,
                    )
                );
                
                self.gazedf.to_csv(self.outgazepath_re, index=False);
                self.imudf.to_csv(self.outimupath_re, index=False);
                pass;
            pass;
        else:
            magsr = 10;
            othersr = 100;
            truesrs = { c:(magsr if ('magneto' in c) else othersr) for c in self.gazeimudf.columns  };
            print(truesrs);
            
            interptype='polynomial';
            interporder=1;
            #REV: must remove duplicate times (usually mutually exclusive, i.e. accidentally get same time point arriving of accel and gaze)

            print(self.gazeimudf.dtypes);
            self.gazeimudf = self.gazeimudf.groupby('timestamp', as_index=False).mean(numeric_only=True);
            self.gazeimudf = interpolate_df_to_samplerate(self.gazeimudf, self.tcol, sr_hzsec, startsec=None, endsec=None,
                                                          method=interptype, order=interporder, truesrs=truesrs, tcolunit_s=1 );
            
            truesrs = { c:(magsr if ('magneto' in c) else othersr) for c in self.imudf.columns  };
            
            print("IMU DTYPES", self.imudf.dtypes);
            self.imudf = self.imudf.groupby('timestamp', as_index=False).mean(numeric_only=True);
            self.imudf = interpolate_df_to_samplerate(self.imudf, self.tcol, sr_hzsec, startsec=None, endsec=None,
                                                      method=interptype, order=interporder, truesrs=truesrs, tcolunit_s=1 );
        
            truesrs = { c:(magsr if ('magneto' in c) else othersr) for c in self.gazedf.columns  };
            #REV: interpolates gaze even when it is not available (should fill in "max distance")
            print("GAZE DTYPES", self.gazedf.dtypes);
            self.gazedf = self.gazedf.groupby('timestamp', as_index=False).mean(numeric_only=True);
            self.gazedf = interpolate_df_to_samplerate(self.gazedf, self.tcol, sr_hzsec, startsec=None, endsec=None,
                                                       method=interptype, order=interporder, truesrs=truesrs, tcolunit_s=1 );
            if( create_csvs ):
                print(
                    "Saving resampled to:\nGAZE: [{}]\nIMU: [{}]\nGAZEIMU: [{}]".format(
                        self.outgazepath_re,
                        self.outimupath_re,
                        self.outgazeimupath_re)
                );
                
                self.gazedf.to_csv(self.outgazepath_re, index=False);
                self.imudf.to_csv(self.outimupath_re, index=False);
                self.gazeimudf.to_csv(self.outgazeimupath_re, index=False);
                pass;
            pass;
        
        return;

    def get_minmax_lr_du_gaze_dva(self):
        xmin = -self.sc_widdva/2;
        xmax = self.sc_widdva/2;
        ymin = -self.sc_heidva/2;
        ymax = self.sc_heidva/2;
        return xmin, xmax, ymin, ymax;
    
    
    pass;

############  END OF CLASS #################


'''
## Test?
def mymain():
    mydir = sys.argv[1];

    recobj = tobiig3_official_recording(mydir);
    recobj.resample_interpolate_dfs(create_csvs=True, sr_hzsec=100);
    return 0;

if __name__=='__main__':
    exit(mymain());
    pass;
'''
