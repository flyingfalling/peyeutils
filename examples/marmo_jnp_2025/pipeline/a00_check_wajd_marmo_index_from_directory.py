#REV: load/check index (for file existence etc.)


import pandas as pd;
import os;
import sys;

def main():
    fn = sys.argv[1];
    sourcedir = sys.argv[2];
    
    df = pd.read_csv(fn);
    df.columns = df.columns.str.replace(' ', '_');
    
    #['gapsacc' 'freeviewing' 'memorysacc' 'randomgap' 'posner']
    tasktypes=df.tasktype.unique();

    # 'edffile', 'trodesfile',
    specialdirs = ['gapcsvfile',
                   'salcsvfile',
                   'memcsvfile',
                   'posnercsvfile'];
    
    taskdf = pd.DataFrame(
        data=dict(
            tasktype=['gapsacc',
                      'freeviewing',
                      'memorysacc',
                      'randomgap',
                      'posner'],
            specialdir=['gapcsvfile',
                        'salcsvfile',
                        'memcsvfile',
                        'gapcsvfile',
                        'posnercsvfile']
        )
    );

    df = pd.merge(left=df, right=taskdf, on='tasktype');

    print(df.columns);
    for trow in taskdf.itertuples():
        task=trow.tasktype;
        myspecdir=trow.specialdir;
        toremove = [ a for a in specialdirs if a!=myspecdir ];
        print("Removing unnecessary for tasktype [{}]".format(task));
        '''
        for tr in toremove:
            print(" -- Removing [{}]".format(tr) );
            df.loc[ df.tasktype==task, toremove ] = [ '' for a in toremove ];
            pass;
        '''
        pass;

    print(df.columns);

    df['hasspecial'] = False;
    df['hastrodes'] = False;
    df['hasedf'] = False;
    df = df.reset_index(drop=True);
    if 'Index' in df.columns:
        df = df.drop(columns=['Index']);
        pass;
    
    
    for row in df.itertuples():
        task = row.tasktype;
        mydir = row.specialdir;
        trialdir=row.dirname;
        #trialdir=row['dirname']; #ERROR fuck pandas type.
        #print(mydir);
        #print(row);
        row = row._asdict();
        #print(row);
        tocheck = row[ mydir ];
        #print(tocheck);
        
        specpath = os.path.join(sourcedir, trialdir, tocheck);
        if( os.path.isfile(specpath) ):
            df.loc[ df.trialcsv==row['trialcsv'], 'hasspecial' ] = True;
            #print("SUCCESS: [{}]");
            pass;
        else:
            #print("FAILURE (SPECIAL): [{}]".format(specpath));
            findit=specpath.split('_')[-1];
            basedir=os.path.dirname(specpath);
            replacements=list();
            if( os.path.isdir( basedir ) ):
                for d in os.listdir(basedir):
                    if( d.endswith(findit) ):
                        #print("Found replacement! [{}]".format(d));
                        replacements.append(d);
                        pass;
                    pass;
                if( len(replacements) > 1 ):
                    print(row);
                    print(replacements);
                    raise Exception("WTF, more than one replacement?! [{}]");
                elif( len(replacements) == 1 ):
                    replacement = replacements[0];
                    dirpart = os.path.dirname(row[mydir]);
                    newfile = os.path.join(dirpart, replacement);
                    df.loc[ df.trialcsv==row['trialcsv'], 'hasspecial' ] = True;
                    df.loc[ df.trialcsv==row['trialcsv'], mydir ] = newfile;
                    pass;
                else:
                    print("FAILURE (SPECIAL): [{}]".format(specpath));
                    # Nothing (leave as false)
                    pass;
                pass;
            pass;
        
        # sanity check trodes and edf, which should exist for each.
        trodespath = os.path.join(sourcedir, trialdir, row['trodesfile']);
        if( os.path.isfile(trodespath) ):
            df.loc[ df.trialcsv==row['trialcsv'], 'hastrodes' ] = True;
            pass;
        else:
            #print("FAILURE (TRODES): [{}]".format(trodespath));
            findit=trodespath.split('_')[-1];
            basedir=os.path.dirname(trodespath);
            replacements=list();
            if( os.path.isdir( basedir ) ):
                for d in os.listdir(basedir):
                    if( d.endswith(findit) ):
                        #print("Found replacement! [{}]".format(d));
                        replacements.append(d);
                        pass;
                    pass;
                if( len(replacements) > 1 ):
                    print(row);
                    print(replacements);
                    raise Exception("WTF, more than one replacement?! [{}]");
                elif( len(replacements) == 1 ):
                    replacement = replacements[0];
                    dirpart = os.path.dirname(row['trodesfile']);
                    newfile = os.path.join(dirpart, replacement);
                    df.loc[ df.trialcsv==row['trialcsv'], 'hastrodes' ] = True;
                    df.loc[ df.trialcsv==row['trialcsv'], 'trodesfile' ] = newfile;
                    pass;
                else:
                    print("FAILURE (TRODES): [{}]".format(trodespath));
                    # Nothing (leave as false)
                    pass;
                pass;
            pass;
        
        edfpath = os.path.join(sourcedir, trialdir, row['edffile']);
        if( os.path.isfile(edfpath) ):
            df.loc[ df.trialcsv==row['trialcsv'], 'hasedf' ] = True;
            pass;
        else:
            
            findit=edfpath.split('_')[-1];
            if( findit != row['edfFileName'] ):
                raise Exception("EDF filename is not same as in CSV [{}] (in CSV) versus [{}] (fname)".format(row['edfFileName'], findit));
            basedir=os.path.dirname(edfpath);
            replacements=list();
            if( os.path.isdir(basedir) ):
                for d in os.listdir(basedir):
                    if( d.endswith(findit) ):
                        #print("Found replacement! [{}]".format(d));
                        replacements.append(d);
                        pass;
                    pass;
                if( len(replacements) > 1 ):
                    print(row);
                    print(replacements);
                    raise Exception("WTF, more than one replacement?! [{}]");
                elif( len(replacements) == 1 ):
                    replacement = replacements[0];
                    dirpart = os.path.dirname(row['edffile']);
                    newfile = os.path.join(dirpart, replacement);
                    df.loc[ df.trialcsv==row['trialcsv'], 'hasedf' ] = True;
                    df.loc[ df.trialcsv==row['trialcsv'], 'edffile' ] = newfile;
                    pass;
                else:
                    # Nothing (leave as false)
                    print("FAILURE (   EDF): [{}]".format(edfpath));
                    pass;
                pass;
            pass;

        ### END FOR row.
        pass;

    df.to_csv(fn + '.cleaned.csv', index=False);
    
    return 0;


if __name__=='__main__':
    exit(main());
    pass;
