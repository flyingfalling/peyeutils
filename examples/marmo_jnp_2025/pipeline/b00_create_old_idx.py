
## This creates the "old" index --> Based on indexed files.
## I.e. we don't use "chen" or "EDF" data yet, just the per-frame CSVs
## Note, also extracts "metadata" of the old data, and re-saves as CSVs in my folder
## with a tag. Already contains "video" etc.


import re
import os
import sys
import numpy as np

def parse_trial_description_pythonic(file_path):
    """
    Parses a trial description file using regex to be robust against whitespace 
    variations, populating a dictionary with the results.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found [{file_path}]", file=sys.stderr)
        return None

    with open(file_path, 'r') as f:
        content = f.read()

    data = {}

    # 1. Extract values using Regex patterns based on the C++ fillers
    # format: ... csvFileName, [VALUE] ...
    # We use ([\S]+) to capture non-whitespace characters (like C++ >> str)
    
    patterns = {
        'csvFileName':   r"csvFileName,\s+([\S]+)",
        'edfFileName':   r"edfFileName,\s+([\S]+)",
        'subjectName':   r"marmoset,\s+([\S]+)",
        'binocularmode': r"mode:\s+([\S]+)",
        'hflip':         r"horizontal\s+flip:\s+([\S]+)", # Context specific
        'vflip':         r"vertical\s+flip:\s+([\S]+)",   # Context specific
        'vid':         r"movie,\s+([\S]+)",
        'width':         r"width:\s+([\d.]+)", # Capture digits/dots
        'height':        r"height:\s+([\d.]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val = match.group(1)
            # Convert numeric types
            if key in ['width', 'height']:
                data[key] = float(val)
            else:
                data[key] = val
        else:
            print(f"WARNING: Could not find value for {key} using pattern '{pattern}' (FILE={file_path})", file=sys.stderr)
            data[key] = np.nan;
            pass;
        pass;

    # 2. Logic Validations (mirroring C++ logic)
    if data.get('hflip') == "True":
        print("REV: error horiz flip not impl yet", file=sys.stderr)
        return None

    if data.get('vflip') == "True":
        print("REV: error vert flip not impl yet", file=sys.stderr)
        return None

    # 3. Filename Parsing Logic
    # Parse the csvFileName string: "salient_movieclip_macaca_Ojo_20191102_1430.csv"
    csv_name = data['csvFileName']
    
    # Remove extension
    if not csv_name.endswith('.csv'):
        print(f"REV: not .csv file! [{csv_name}]", file=sys.stderr)
        return None
        
    base_name = os.path.splitext(csv_name)[0]
    parts = base_name.split('_')

    # Basic structure check: salient_movieclip_...
    if len(parts) < 3 or parts[0] != "salient" or parts[1] != "movieclip":
        print(f"REV: error filename structure invalid [{csv_name}]", file=sys.stderr)
        return None

    # Parse monkey name, date, time
    # Handles "macaca_NAME" (6 parts) vs "NAME" (5 parts)
    try:
        if len(parts) == 6:
            # e.g. salient_movieclip_macaca_Ojo_20191102_1430
            monkey_name = f"{parts[2]}_{parts[3]}" # macaca_Ojo
            date_part = parts[4]
            time_part = parts[5]
        elif len(parts) == 5:
            # e.g. salient_movieclip_Sub1_20191102_1430
            monkey_name = parts[2]
            date_part = parts[3]
            time_part = parts[4]
        else:
            print(f"WARNING: Filename format unexpected [{csv_name}]", file=sys.stderr)
            return None
            
        # Subject Name consistency check
        # Extract just the name part (Ojo from macaca_Ojo) to compare with file header
        extracted_subj = monkey_name.split('_')[-1]
        if extracted_subj != data['subjectName']:
            print(f"Error: Subject name mismatch. Header: {data['subjectName']}, Filename: {extracted_subj}", file=sys.stderr)
            return None

        # 4. Populate derived fields
        data['year'] = date_part[:4]
        data['month'] = date_part[4:6]
        data['day'] = time_part[:2]
        data['time'] = time_part[2:]
        
        data['rawyearmonth'] = date_part
        data['rawdaytime'] = time_part
        
        # Trial Name construction
        data['trialName'] = f"{monkey_name}_{data['year']}_{data['month']}_{data['day']}_{data['time']}_{data['vid']}"
        
        # Saliency Paths
        data['trialTag'] = f"{monkey_name}_{date_part}_{time_part}"
        data['saliencyCsvFileName'] = f"saliency_marmoset_{data['trialTag']}.csv"
        data['saliencyCsvFileFullName'] = f"Saliency_data/{data['saliencyCsvFileName']}"

    except IndexError:
        print("Error parsing filename tokens", file=sys.stderr)
        return None

    return data


def print_trial_data(data):
    """
    Pretty prints the trial description dictionary.
    """
    if not data:
        print("No data to display.")
        return

    print("-" * 40)
    print("TRIAL DESCRIPTION DATA")
    print("-" * 40)
    
    # Sort keys for easier reading
    max_len = max(len(k) for k in data.keys())
    for key in sorted(data.keys()):
        value = data[key]
        print(f"{key:<{max_len + 2}}: {value}")
    print("-" * 40)
    return;


import sys;
import os;
import pandas as pd;

def main():
    datdir=sys.argv[1];
    idxs=sys.argv[2:];

    idxlist=list();
    for i in idxs:
        df = pd.read_csv(i, sep=' ');
        print(df);
        idxlist.append(df);
        pass;

    trialcsvdf = pd.concat( idxlist );
    print(trialcsvdf);

    dictlist=list();
    for idx, row in trialcsvdf.iterrows():
        fpath =  os.path.join( datdir, getattr(row, 'trialcsv'));
        prepath = os.path.dirname(fpath);
        print(fpath);
        if( False == os.path.isfile( fpath ) ):
            raise Exception("File does not exist? {}".format(fpath));
        trialcsvpath = os.path.join( datdir, getattr(row, 'trialcsv') );
        data = parse_trial_description_pythonic( trialcsvpath );
        trialsubpath = os.path.dirname(trialcsvpath);
        trialsubpath = os.path.relpath(trialsubpath, datdir);
        print("RELATIVE PATH: {}".format(trialsubpath));
        data['relpath'] = trialsubpath;
        
        print(data);
        species = getattr(row, 'species');
        data['species'] = species;
        if( species == 'macaq' ):
            toremove = "_macaca";
            print("REV: {} replacing incorrect filename for macaq species (removing {})".format(fpath, toremove));
            data['saliencyCsvFileFullName'] = data['saliencyCsvFileFullName'].replace(toremove, "");
            data['saliencyCsvFileName'] = data['saliencyCsvFileName'].replace(toremove, "");
            pass;
        #For some, need to replace "marmoset" with "movieclip"
        print_trial_data(data);
        salfile = os.path.join( prepath, data['saliencyCsvFileFullName'] );
        print("Expecting full file in {}".format(salfile));
        if( False == os.path.isfile( salfile ) ):
            print(" ++!!!!!!!!!!!    Trial has no CSV gaze data {} !!!!!!!!!!!!!!!!".format(salfile));
            continue;
        else:
            print("GOOD DATA");
            pass;
        
        dictlist.append(data);
        
        pass;
    
    fulldf=pd.DataFrame(dictlist);
    print(fulldf);

    fulldf = fulldf.rename( columns={'subjectName':'subj'} );

    print("Expected: {}".format(len(trialcsvdf.index)));
    #REV: some trials are missing vid wid/hei...why? Just didnt write to file before started I guess...
    
    #REV: just use pure pixel values haha.
    print(fulldf.columns);
    fulldf['vid'] = fulldf['vid'].str[:-4];
    result = fulldf.groupby(['species', 'subj', 'vid']).size();
    fulldf = fulldf.reset_index(drop=True);
    
    fulldf['trialidx'] = fulldf.index;

    biggazelist=list();
    for adf in fulldf.itertuples():
        csvfile = getattr( adf, 'saliencyCsvFileFullName' );
        trialdir = getattr( adf, 'relpath');
        csvpath = os.path.join( datdir,
                                trialdir,
                                csvfile );
        gazedf = pd.read_csv(csvpath, skipinitialspace=True);
        gazedf = gazedf.rename( columns={'# ms':'timems'} );
        gazedf['trialidx'] = getattr(adf, 'trialidx');
        biggazelist.append(gazedf);
        #print(gazedf);
        pass;
    
    biggazedf = pd.concat(biggazelist);
    biggazedf.to_csv('data/orig_biggazedf_jnp2025.csv', index=False);
    fulldf.to_csv('data/orig_freeviewing_trials_idx_jnp2025.csv', index=False);
    
    pd.set_option('display.max_rows', None)
    print(result);
    
    return 0;



def example():
    fname = sys.argv[1];
    data = parse_trial_description_pythonic(fname);
    print_trial_data(data);
    
    return 0;





if __name__=='__main__':
    exit(main());
    pass;
