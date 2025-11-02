

import pandas as pd;
import sys;
import os;


def main():
    szxlsx = sys.argv[1];
    hcxlsx = sys.argv[2];

    szdobetc_sheet='個人情報（プロフィール②）';
    szclinical_sheet='臨床情報（プロフィール①）';
    szPANSS_sheet='PANSS';

    hcdobetc_sheet='個人情報（プロフィール①）'; #szdobetc_sheet;
    
    szdobstartrow=3;
    hcdobstartrow=3;

    #szdict = pd.read_excel(szxlsx, sheet_name=None);
    szdobetcdf=pd.read_excel(szxlsx, sheet_name=szdobetc_sheet, header=szdobstartrow);
    szclinicaldf=pd.read_excel(szxlsx, sheet_name=szclinical_sheet, header=szdobstartrow);
    
    hcdobetcdf=pd.read_excel(hcxlsx, sheet_name=hcdobetc_sheet, header=hcdobstartrow);
    
    print("SZ");
    print(szdobetcdf);
    print(szdobetcdf.columns);
    
    print("HC");
    print(hcdobetcdf);
    print(hcdobetcdf.columns);

    szdobetcdf['group']='sz';
    hcdobetcdf['group']='hc';


    print(szdobetcdf['ID'].to_list());
    print(hcdobetcdf['ID'].to_list());
        
    df = pd.concat([szdobetcdf, hcdobetcdf]);
    print(df);
    print(df.columns);
        
    df['fmridate'] = df['画像検査日（MRI）'];
    df['sex'] = df['性別'];
    df['dob'] = df['生年月日'];
    df['age'] = df['年齢'];
    df['ID'] = df['ID'].str.upper();
    df['out1date'] = df['心理検査日'];
    df['out2date'] = df['心理検査日2回目（あれば）'];
    
    df['fmridate'] = pd.to_datetime(df['fmridate'], errors='coerce'); #REV: note "NaT" means "not a time". Use pd.isna or pd.isnone?
    df['out1date'] = pd.to_datetime(df['out1date'], errors='coerce'); #REV: note "NaT" means "not a time". Use pd.isna or pd.isnone?
    df['out2date'] = pd.to_datetime(df['out2date'], errors='coerce'); #REV: note "NaT" means "not a time". Use pd.isna or pd.isnone?
    
    
    
    fmriedfdir=sys.argv[3];
    outsideedfdir=sys.argv[4];
    
    inoutlist=list();
    
    edflist=list();
    
    
    edfdatepattern='.+_start_(\\d{4}-\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2})_end_.+'
    for i, row in df.iterrows():
        name=row['ID']
        print("[{}]: {}    (OUT1: {}   OUT2: {})".format(name, row['fmridate'], row['out1date'], row['out2date']));
        
        myfmriedfdir = os.path.join(fmriedfdir, name);
        print("ATTEMPTING FMRIDIR: [{}]".format(myfmriedfdir));
        ninside=0;
        fmriexpecteddate=row.fmridate;
        
        if( pd.notna(fmriexpecteddate) ):
            fmriexpecteddate=fmriexpecteddate.normalize();
            pass;
        
        outsideexpecteddate=row.out1date;
        if( pd.notna(outsideexpecteddate)):
            outsideexpecteddate=outsideexpecteddate.normalize();
            pass;
        
        if( os.path.exists(myfmriedfdir) ):
            print("[{}]  HAS AN FMRI EDF DIR!!".format(name));
            #fmriedfs = [x for x in os.listdir(myfmriedfdir) if x.lower()[-4:]=='.edf'];
            fmriedfs=list();
            subdirs=list();
            for root, dirs, files in os.walk(myfmriedfdir):
                if(len(dirs) != 0 ):
                    print("WARNING: One or more subdirs? Unexpected: {}".format(dirs));
                for filename in files:
                    #if( filename.lower()[-4:]=='.edf' ):
                    if( filename.lower().endswith('.edf') ):
                        if( filename in fmriedfs ):
                            raise Exception("Something wrong, doubled up filename? [{}]".format(filename));
                        
                        full_file_path = os.path.join(root, filename);
                        relative_file_path = os.path.relpath(full_file_path, myfmriedfdir);
                        subdir = os.path.dirname(relative_file_path);
                        subdirs.append(subdir);
                        fmriedfs.append(filename);
                        pass;
                    pass;
                pass;
            
            ninside=len(fmriedfs);

            for x,subx in zip(fmriedfs, subdirs):
                import re;
                mydatetime = re.match(edfdatepattern, x);
                if( not mydatetime ):
                    print("WARNING, edf {} is missing date/time (maybe fname is not normal)".format(x));
                    mydatetime = None;
                    pass;
                else:
                    mydatetime = mydatetime.group(1);
                    mydatetime = pd.to_datetime(mydatetime, format='%Y-%m-%d-%H-%M-%S', errors='coerce');
                    mydate = mydatetime.normalize();
                    pass;
                #if( mydate != fmriexpecteddate ):
                #    print(" ---------- FMRI WARNING : Dates do not line up (got {}, expected {})".format(mydate, fmriexpecteddate));
                #    pass;
                
                print("Processing {}/{} (date={}, time={})".format(subx,x,mydate, mydatetime));
                edflist.append( dict( name=name, edfdate=mydate,
                                      edfdatetime=mydatetime,
                                      expecteddate=fmriexpecteddate,
                                      edfbasedir=myfmriedfdir,
                                      edfsubdir=subx,
                                      edfpath=os.path.join(myfmriedfdir,subx),
                                      edffile=x, kind='fmri') );
                
                pass;
            pass;
        else:
            print("!! NO FMRI EYETRACKING ({})".format(name));
            pass;
        
        
        noutside=0;
        myoutsideedfdir = os.path.join(outsideedfdir, name);
        if( os.path.exists(myoutsideedfdir) ):
            #REV: walk the dir.
            
            #outsideedfs = [x for x  in os.listdir(myoutsideedfdir) if x.lower()[-4:]=='.edf'];
            outsideedfs=list();
            subdirs=list();
            for root, dirs, files in os.walk(myoutsideedfdir):
                if(len(dirs) != 0 ):
                    print("WARNING: More than one subdir?: {}".format(dirs));
                for filename in files:
                    
                    #if( filename.lower()[-4:]=='.edf' ):
                    if( filename.lower().endswith('.edf') ):
                        if( filename in outsideedfs ):
                            raise Exception("Something wrong, doubled up filename? [{}]".format(filename));
                                                
                        full_file_path = os.path.join(root, filename);
                        relative_file_path = os.path.relpath(full_file_path, myoutsideedfdir);
                        subdir = os.path.dirname(relative_file_path);
                        
                        subdirs.append(subdir);
                        outsideedfs.append(filename);
                        pass;
                    pass;
                pass;
            
            noutside=len(outsideedfs);
            
            for x,subx in zip(outsideedfs, subdirs):
                mydatetime = re.match(edfdatepattern, x);
                mydatetime = mydatetime.group(1);
                mydatetime = pd.to_datetime(mydatetime, format='%Y-%m-%d-%H-%M-%S', errors='coerce')
                mydate = mydatetime.normalize();
                #if( mydate != outsideexpecteddate ):
                #    print(" ---------- OUTSIDE WARNING : Dates do not line up (got {}, expected {})".format(mydate, outsideexpecteddate));
                #    pass;
                print("Processing {}/{} (date={}, time={})".format(subx,x,mydate, mydatetime));
                edflist.append( dict( name=name, edfdate=mydate,
                                      edfdatetime=mydatetime,
                                      expecteddate=outsideexpecteddate,
                                      edfbasedir=myoutsideedfdir,
                                      edfsubdir=subx,
                                      edfpath=os.path.join(myoutsideedfdir,subx),
                                      edffile=x, kind='outside') );
                pass;
            pass;
        else:
            print("!! NO OUTSIDE EYETRACKING ({})".format(name));
            pass;
        
        inoutlist.append( dict(name=name, ninside=ninside, noutside=noutside) );
        pass;
    
    summdf=pd.DataFrame(inoutlist);
    print(summdf);
    
    alledf_df=pd.DataFrame(edflist);
    print(alledf_df);
    
    df.to_csv('sz_index_simplified.csv', index=False);
    alledf_df.to_csv('sz_edf_index.csv', index=False);
    summdf.to_csv('sz_inout_summary.csv', index=False);
    
    #REV: do the opposite? Find EDF files for which I don't know who they are from?
    
    
    
    
    
    return 0;

if __name__=='__main__':
    exit(main());
    pass;
