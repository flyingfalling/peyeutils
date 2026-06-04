
## REV: simple python script to build a marmo (CSV) index file based on processing of directories with named FNAMES and
## CSV files describing saliency (or gap) experiments using Denis's software.


import sys
import os
import pandas as pd
import re #regexp

def main():
    basedir=sys.argv[1];
    dirpattern='([0-9][0-9][0-9][0-9])\-([0-9][0-9])\-([0-9][0-9])_([a-zA-Z]+)'; #2024-04-09_Julian etc.

    matchlist=list();
    dictlist=list();
    for d in os.listdir(basedir):
        #https://docs.python.org/3/library/re.html

        if( False == os.path.isdir( os.path.join(basedir, d) ) ):
            print("[{}] is not a directory (is a file), skipping".format(d));
            continue;
        print("Testing [{}]".format(d));
        m = re.fullmatch(dirpattern, d);
        #m = re.match(d, dirpattern);
        if m:
            matchlist.append(d);
            print(m.group(1), m.group(2), m.group(3), m.group(4));
            mydict = dict(year=m.group(1), month=m.group(2), day=m.group(3), subj=m.group(4), dirname=d );
            dictlist.append(  mydict );
            print(" -- Matched [{}]".format(d));
            pass;
        else:
            print(" -- Skipping [{}], did not match format".format(d));
            pass;
        pass;
    
    fulldf = pd.DataFrame.from_records(dictlist);
    print(fulldf);

    '''
    gap_taskWajd_flexibleLoc2_noRepetition_optogenetics
    csvFileName, gap_taskWajd_flexibleLoc2_noRepetition_optogenetics_Julian_202311_30115047.csv
    edfFileName, 30115047.edf
    Name of the marmoset, Julian
    Date of the experiment, 2023-11-30
    Diameter of the fixation stimulus, 2.0
    Reward criterion of successive frames for fixation task, 12
    Duration of gap task, 12
    Opacity of the gap task stimulus, 0.00
    Number of saccade locations, 8
    locations, 8
    targetLocAngle, 45
    Type of the fixation stimulus, 2
    Magnification factor for eye window in gap task, 1.20
    optostimDelay, 0
    optoStimTiming, answerTime
    Stimulus size:  2.0 degrees
    Eccentricity of the saccade target:  8.0 degrees
    optostimDelay :  0.0 msThe total number of trials:  24
    The number of saccade targets:  8
    trial success rate: 75.00%
    off-target saccades percentage: 0.00%
    did not look: 25.00%
    '''
    
    '''
    salient_movieclip_wajd5
    csvFileName, salient_movieclip_wajd5_Julian_202403_21125133.csv
    edfFileName, 21125133.edf
    Name of the marmoset, Julian
    Date of the experiment, 2024-03-21
    Eyetracking mode: Binocular
    Number of trials: 1
    optostimDelay, 0
    optoStimTiming, noOpto
    Movie horizontal flip: False
    Movie vertical flip: False
    
    optoStim interval, 0.200
    Movie width:  640.0 pixels
    Movie height:  480.0 pixels
    '''
    
    dictlist=list();
    for row in fulldf.itertuples():
        adir = row.dirname;
        fulldir = os.path.join( basedir, adir );
        # Will list base CSV files in here not ending in _log.csv
        for f in os.listdir(fulldir):
            if( False == os.path.isfile( os.path.join( fulldir, f ) ) ):
                #print("Skipping directory [{}] in [{}]".format(f, fulldir));
                continue;
            if( len(f) > 4 and
                f[-4:] == '.csv' ):
                if( len(f) > 8 and
                    f[-8:-4] == '_log' ):
                    #print("Skipping LOG file [{}]".format(f));
                    continue;
                if( len(f) > 12 and
                    f[-12:-4] == '_summary' ):
                    #print("Skipping SUMMARY file [{}]".format(f));
                    continue;
                
                ## Now, appropriate
                filepath=os.path.join( fulldir, f );
                print("Handling trial CSV: [{}]".format(filepath));
                if( os.path.isfile(filepath+'.BU') ):
                    print("Reading .BU file");
                    with open(filepath+'.BU', 'r') as fh:
                        data=fh.read();
                        #lines = fh.readlines();
                        pass;
                    pass;
                else:
                    with open(filepath, 'r') as fh:
                        data=fh.read();
                        #lines = fh.readlines();
                        pass;
                    pass;
                
                
                
                #REV: made start with \n because other lines have
                ## blah blah blah locations, XXX
                if( '\nlocations, ' in data ):
                    print("Wajd's weirdly formatted CSV, fixing...")
                    loc=data.index('\nlocations, ');
                    print(loc);
                    myline = data[loc:].split('\n')[1]; #0 to 1 for \nlocations,
                    linelen=len(myline) + 1;
                    oopsquotes = myline.replace('"', '');
                    oopsquotes = oopsquotes.replace(':', '@'); #REV: ghetto replace...
                    oopsquotes = oopsquotes[:len('locations, ')] \
                        + oopsquotes[len('locations, '):] + '\n';
                    
                    newdata = data[:loc] + '\n' + oopsquotes + \
                        data[(loc+1+linelen):];
                    print("Replacing (LOC,) with:\n--------\n{}\n-----------".format(newdata));
                    
                    with open(filepath+'.BU', 'w') as fh:
                        fh.write(data);
                        pass;

                    with open(filepath, 'w') as fh:
                        fh.write(newdata);
                        pass;
                    
                    pass;
                    
                if( 'msThe' in data ):
                    print("Found errorfully formatted output missing newline...replacing and writing to file");
                    loc=data.index('msThe');
                    newdata = data[:(loc+2)] + '\n' + data[(loc+2):];
                    print("Replacing with:\n--------\n{}\n-----------".format(newdata));
                    with open(filepath+'.BU', 'w') as fh:
                        fh.write(data);
                        pass;
                    
                    with open(filepath, 'w') as fh:
                        fh.write(newdata);
                        pass;
                    
                    pass;
                
                lines = data.split('\n');
                
                
                progname='';
                if(len(lines) > 0 ):
                    progname=lines[0].strip();
                    pass;
                
                colnames=['varname', 'varval'];
                print("Reading CSV: [{}]".format(filepath));
                csvdf = pd.read_csv( filepath, sep=', |: ' , header=0, names=colnames, skiprows=1, engine='python' ).reset_index(drop=True);
                for col in csvdf:
                    csvdf[col] = csvdf[col].str.strip();
                    pass;
                
                mydict = csvdf.set_index('varname').to_dict()['varval']; #transpose(); #to_dict(); #orient='records'); #, index=False);
                print(mydict);
                mydict['progname'] = progname;
                mydict['trialcsv'] = f;
                print(row);
                print(type(row));
                rowdict = row._asdict();
                mydict.update( rowdict );

                trodesdir='Trodes_recordings';
                edfdir = 'edf_files';
                saldir='Saliency_data';
                gapdir = 'Gap_saccade_training_data';
                memdir = 'Memory_saccade_training_data';
                posnerdir = 'Posner_data';
                
                fbase = f[:-4];
                mydict['trodesfile'] = os.path.join(trodesdir, fbase+'.rec');
                mydict['edffile'] = os.path.join(edfdir, fbase+'.edf');
                mydict['gapcsvfile'] = os.path.join(gapdir, fbase+'.csv');
                mydict['salcsvfile'] = os.path.join(saldir, fbase+'.csv');
                mydict['memcsvfile'] = os.path.join(memdir, fbase+'.csv');
                mydict['posnercsvfile'] = os.path.join(posnerdir, fbase+'.csv');
                mydict['tasktype'] = '';
                mydict['vid'] = '';
                if( 'Name of the movie' in mydict ):
                    vidname = mydict['Name of the movie'];
                    #vidpath = vidname.encode("string_escape").split('\\');
                    vidpath = vidname.split('\\');
                    
                    if( len(vidpath) > 1 ):
                        vidname = vidpath[-1];
                        
                        pass;
                    else:
                        raise Exception("Error, expected Wajd to have weird Windows-like C: format for path to video, but had [{}]".format(vidpath));
                    mydict['vid'] = vidname;
                    mydict['tasktype'] = 'freeviewing';
                    pass;
                elif( 'Duration of gap task' in mydict):
                    mydict['tasktype'] = 'gapsacc';
                    pass;
                elif( 'Duration of memory guided trial before target reappears' in mydict ):
                    mydict['tasktype'] = 'memorysacc';
                    pass;
                elif( 'Opacity of the gap task stimulus' in mydict ): #REV: note is also in gaps but duration is not for random
                    mydict['tasktype'] = 'randomgap';
                    pass;
                elif( 'Duration of the cue and stim flashing' in mydict ):
                    mydict['tasktype'] = 'posner';
                    pass;
                else:
                    print(mydict);
                    raise Exception("Exception, unknown type of trial [{}]?".format(filepath));
                
                
                
                
                
                #REV: looks like layout of dir is "Saliency_data" and "edf_files" and "Gap_saccade_training_data" and "Trodes_recordings"
                # Note in "edf_files" they have extra stuff, only last _XXXXXXX.edf matters.
                # For video shown, it has wajd's C:\home\desktop dir, so need to remove it and take only tail (basename?).
                #salient_movieclip_wajd5_Julian_202405_24131221.edf
                ## Trode_recording files are same name as .edf?,
                ## but with .rec (Trodes_recordings/salient_movieclip_wajd5_Julian_202405_24131221.rec)
                
                dictlist.append(mydict);
                pass;
            
            pass;
        pass;

    df = pd.DataFrame.from_records(dictlist);
    print(df);
    df.to_csv('wajd_marmo_index3.csv', index=False);
    
    return 0;

if __name__=='__main__':
    exit(main());
    pass;
