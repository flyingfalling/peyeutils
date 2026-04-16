## I need to read in and get the data from each species (and age) in
## the same format, and sampled on average during a video frame (30
## fps).

## I also need to "split" infants into "ages" (based on DoB and date
## of experiment), and extract video and looking position (cleanly).

## Some data where one eye is tracked and the other is
## e.g. nostril...would like to filter that out somehow. No way to
## find it after the fact (except maybe lack of conjugate movements
## that are saccade-like, for some minimum period of time, e.g. 10
## seconds, while the other eye has them. But head movements may look
## similar...). For now just remove (> 2 dva separation, etc.)...

## I need an "index" built, specifying subject (and age?). If no
## subject, but has DOB in file, then can ESTIMATE rough age at time
## of recording. And can "cluster" them.

## Then, we want to know:

##   distance (in video space, pixels)
##   between subjects (of same species) watching same point in same video when
##   both are looking at something of "high" saliency versus "medium"
##   saliency versus "low" saliency. And take DIFFERENCE between HH-LL.

##   Could take mean distance from eyes of other subjects (OF OTHER SPECIES).

##   Unfortunately, I don't have multiple viewings of same stimulus
##   for same subject, so I can't estimate easily "intra-subject
##   correlation".

##   I can estimate raw "inter-subject" correlation however, and
##   correlation between species (pairs of people of same/different species).

##   I can also compute raw basic AUROC (curves and ROC) for every subject/age/species
##     in terms of saliency prediction (and for each sub-channel).

##   Note I could compute distance for each sub-channel as well (e.g. motion etc.). Do HH-LL.

## I may just want to do correlation over e.g. 1-second or 3-second "chunks" of videos. Instead of "whole videos".

## which is fine, but each subject viewed different numbers of videos different times, and not good tracking throughout.

#REV: need to add wajd marmo data? Problem is maybe its CSV format is slightly different (need to know video from meta CSV file).



## TODO:
## 1) plot gaze of schizo versus not-schizo

## 2) plot gaze of monkey/infant/marmo/adult for different ages for example videos.
## 3) Plot of amount/quality of data available for each infant at each each (which video and how much datapoints are good).
## 4) Same for monkeys and shit (just use CSV files for now).
## 5) Convolve with saliency and compute mean (NULL MODEL) same timepoint-vs-distance distribution (histo)
## 6) Compute saliency HH-MM-LL for within-species (do for every pair of subjects, with sufficient data)
## 7) ...



import peyeutils as pu;

import pandas as pd;
import os;
import matplotlib.pyplot as plt;
import numpy as np;
import sys;

from multiprocessing import Pool;



def plotit(edfrow, out_csv_path):
    msgdf = pd.read_csv( os.path.join(out_csv_path, edfrow['messages_csv']) );
    elparamdict = pu.peyefv.get_elparams(msgdf); # has samplerate etc.
    ELsr=elparamdict['samplerate'];
    sampdf = pd.read_csv( os.path.join(out_csv_path, edfrow['samples_csv']) );
    edftrials=pd.read_csv( os.path.join(out_csv_path, edfrow['trials_csv']) );
    edfblocks=pd.read_csv( os.path.join(out_csv_path, edfrow['blocks_csv']) );
    #sampdf = pd.read_csv( os.path.join(out_csv_path, row['samples_csv']) );
    #sampdf = sampdf[ sampdf['eye'] == 'B' ];
    print(edfblocks);
    
    for blocki, blockrow in edfblocks.iterrows():
        mytrials = edftrials[ (edftrials.start_s >= blockrow.blkstart_s) & (edftrials.end_s <= blockrow.blkend_s)
                             ].copy();
        
        blkstsec = blockrow.blkstart_s;
        blkensec = blockrow.blkend_s;
        
        #REV: if FMRI, find FMRI trigger key message.
        #REV: else, it will start instantly, so block start message :)
        if( True == blockrow.isfmri and
            np.isfinite(blockrow.fmrist_s) ):
            print("Block fixed from {} -> {} for FMRI".format(blkstsec, blockrow.fmrist_s));
            blkstsec = blockrow.fmrist_s; #or EDF? May be multiple FMRI in it though... #REV: OH no it is PER TRIAL
            ## I find it, so will be OK... (at least it will error when extracting blocks if >1 unique).
            pass;
        
        #print("Block {}/{} of EDF [{}] ({}-{}) has {} trials (IsPract?: {}) (FMRI start={})".format(blockrow.blkidx, len(edfblocks.index),
        #                                                                                            edfrow.edffile,
        #                                                                                            blockrow.blkstart_s, blockrow.blkend_s,
        #                                                                                            len(mytrials.index),
        #                                                                                            blockrow.ispract, blockrow.fmrist_s) );
        
        mytrials = mytrials.sort_values(by='start_s').reset_index(drop=True);
        
        if( len(mytrials.index) < 1 ):
            print("BLK {}, {} has no trials".format(blockrow.blkidx, edfrow.edffile));
            pass;
        
        if( 'trialidx' not in mytrials.columns ):
            raise Exception("No trial index");

        
        print("{} Samps time ({}-{}) sec, block {}-{} sec".format(len(sampdf.index), sampdf.Tsec.min(), sampdf.Tsec.max(), blkstsec, blkensec));
        
        blocksamps = sampdf[ (sampdf.Tsec >= blkstsec) &
                             (sampdf.Tsec < blkensec) ];
        
        
        nrow=1;
        ncol=1;
        fig, ax = plt.subplots(nrow, ncol, figsize=(15,7));
        
        
        ELunit=1e-3;
        
        ############# extract trials of this block ##################3
        blocktrials = list();
        for ti, trialrow in mytrials.iterrows():
            alltsamps = blocksamps[ (blocksamps.Tsec >= trialrow.start_s) &
                                    (blocksamps.Tsec < trialrow.end_s)];
            
            print("Doing for trial {} (had {} samps?)".format(ti, len(alltsamps.index)));
            print("{} samps -- Note trial {}-{} sec  (block data goes from {}-{})".format(len(blocksamps.index), trialrow.start_s, trialrow.end_s, blocksamps.Tsec.min(), blocksamps.Tsec.max()));
            
            ## FOR EACH EYE
            for eye, tsamps in alltsamps.groupby('eye'):
                nsamp = len(tsamps.index);
                
                ngood = len( tsamps[ False==tsamps.bad ].index );
                nsec = (ngood)/float(ELsr);
                print("EYE: [{}] Trial vid {} has {}/{} ({:3.2f}%) timepoints (@SR={} Hz = {} sec)".format(eye,
                                                                                                           trialrow.video,
                                                                                                           ngood,
                                                                                                           nsamp,
                                                                                                           (ngood/nsamp)*100,
                                                                                                           ELsr,
                                                                                                           nsec));
                #mytrials.loc[ ti, 'ngood' ] = ngood;
                #mytrials.loc[ ti, 'nsec' ] = nsec;
                #mytrials.loc[ ti, 'nsamp' ] = nsamp;
                
                mydict = { k:trialrow[k] for k in trialrow.keys() };
                mydict['eye'] = eye;
                mydict['ngood'] = ngood;
                mydict['nsec'] = nsec;
                mydict['nsamp'] = nsamp;
                #newtrials.append(mydict); #with eye separated, and various new info.
                blocktrials.append(mydict);
                pass; #end for EYES
            
            pass; #END for all TRIALS (in block)
        
        #print(blocktrials.columns);
        print("Num block trials", len(blocktrials));
        print(mytrials);
        '''
        if( len(blocktrials) == 0 ):
        raise Exception("WOW NO TRIALS?!?!??!");
        pass;
        else:
        print("{} trials in block".format(len(blocktrials)));
        pass;
        '''
        if( len(blocktrials) > 0 ):
            blocktrials = pd.DataFrame(blocktrials).sort_values(by='start_s').reset_index(drop=True);
            pass;
        else:
            if( len(mytrials.index) > 0 ):
                raise Exception("Should never happen (empty blocktrials, but mytrials>0? ={})".format(len(mytrials.index)));
            blocktrials = mytrials; #REV: missing some 
            pass;
        #exit(0);
        ## REV: plot here?
        #mytrials = mytrials.sort_values(by='start_el').reset_index(drop=True);
        
        #REV: plot LEFT, RIGHT, and chunks of "Bad time" (based on source of badness?)
        
        #REV: mark same point, bad for L/R/B.
        lineoff=0.0;
        ax2max=-1;
        
        #REV: only take binoc gaze...
        #blocksamps = blocksamps[blocksamps.eye=='B'];
        spaneyes=['B']; #['L', 'R']
        ploteyes=['B']; #,'L','R'];
        for eye, bsamps in blocksamps.groupby('eye'):
            lineoff+=0.6;
            tvar='Tsec';
            xvar='cgx_dva';
            yvar='cgy_dva';
            
            #badvals=['badpupilPRE', 'badpupil', 'badPRE', 'bad']
            badvals=['bad']
            #badvar='bad';
            off2=0;
            for badvar in badvals:
                baddf = pu.utils.cond_rle_df( bsamps[badvar], val=True, t=bsamps[tvar] );
                xlines=list();
                for i, arow in baddf.iterrows():
                    if( arow.v == True  and
                        eye in spaneyes):
                        #print("{} - {}".format(arow.st,arow.et));
                        ax.axvspan(arow.st, arow.et, alpha=0.35)
                        #ax.plot([arow.st, arow.et], [-8+lineoff, -8+lineoff], label="{} BAD".format(eye));
                        #xlines.append(arow.st);
                        #xlines.append(arow.et);
                        #xlines.append(np.nan);
                        pass;
                    pass;
                #ylines = list([-8+lineoff+off2]*len(xlines));
                #ax.plot(xlines, ylines, label="{} {}".format(eye, badvar ));
                off2+=0.12;
                pass;
            
            
            #'Tsec', 'Tsec0', 'Tmsec', 'Tmsec0', 'ELeye', 'useeye',
            #       'elblink', 'elhasblink', 'elsacc', 'elfix', 'elpurs', 'ELevlabel',
            #              'ELsimultevs', 'pa_abs_tdiff', 'pa_abs_tdiff_med_err',
            #                     'pa_abs_tdiff_mad', 
            
            if( eye in ploteyes ):
                ax.scatter(bsamps.Tsec, bsamps.cgx_dva, s=0.5, label='{} X'.format(eye));
                ax.scatter(bsamps.Tsec, bsamps.cgy_dva, s=0.5, label='{} Y'.format(eye));
                pass;

            if( 'pa_abs_tdiff_med_err' not in bsamps.columns ):
                print(bsamps.columns);
                print("NO pupil info for MAD here, probably there was no data?");
                continue;
            
            mederr=bsamps.iloc[0].pa_abs_tdiff_med_err;
            mad=bsamps.iloc[0].pa_abs_tdiff_mad;
            
            MEDCONST=5; #REV: they used MEDCONST as basically 1?
            NCONST=5;
            med=np.nanmedian(bsamps.pa_abs_tdiff);
            thresh = ( med * MEDCONST) + (mad  *  NCONST);
            print("MED {}   MAD {}   THRESH  {}".format(med, mad, thresh));
            
            '''
            ax2.axhline(med, label='med');
            ax2.axhline(thresh, label='med', color='red', linestyle='--');
            #if( np.nanmax(bsamps.pa_abs_tdiff) > ax2max ):
            if( thresh*10 > ax2max ):
            ax2max = thresh*10;
            pass;
            
            toplot=bsamps[['Tmsec', 'pa_abs_tdiff']].copy();
            toplot.loc[toplot.pa_abs_tdiff > ax2max, 'pa_abs_tdiff'] = ax2max;
            ax2.scatter(toplot.Tmsec, toplot.pa_abs_tdiff, label='{} dp/dt'.format(eye), s=0.4);
            ax2.axhline(mad, label='mad');
            ax2.axhline(mederr, label='mederr');
            '''
            
            #ax3.scatter( bsamps.Tmsec, bsamps['pa'], label='{} Pa'.format(eye), s=0.4);
            pass;
        
        
        #for vid, viddf in blocktrials.groupby('video'):
        
        ymax=13;
        text_y=ymax;
        for tridx, trdf in blocktrials.groupby('trialidx'):
            #print(trdf.columns);
            #print(trdf);
            vid=trdf.iloc[0].video;
            most = trdf.nsec.max();
            pct = (trdf.ngood.max()/trdf.nsamp.max())*100;
            if( len(trdf.index) != 3 ):
                print("WARNING not 3 for L/R/B as expected: {} (maybe B is missing)".format(len(trdf.index)));
                pass;
            ax.axvline( trdf.iloc[0].start_s, color='red', linestyle='--', alpha=0.5 );
            ax.axvline( trdf.iloc[0].end_s, color='black', linestyle='--', alpha=0.5 );
            
            ax.text(  trdf.iloc[0].start_s, text_y,
                      '{}\n{:4.1f}s   ({:4.1f}%)'.format(vid, most, pct),
                      rotation=-90.0, rotation_mode='anchor');
            
            pass;
        
        
        #REV: do some smoothing...
        eyedata = blocksamps[ blocksamps.eye == 'B' ];
        eyedata = eyedata.sort_values(by='Tsec').reset_index(drop=True);
        eyedata = pu.preproc.preproc_SHARED_D_exclude_bad( eyedata, xcol='cgx_dva', ycol='cgy_dva',
                                                           badcol='bad' );
        from datetime import timedelta
        timecol='Tsec';
        smoothtimename='__'+timecol;
        eyedata[smoothtimename] = pd.to_timedelta(eyedata[timecol], unit='s');
        eyedata = eyedata.set_index(smoothtimename);
        smoothcols=['cgx_dva', 'cgy_dva'];
        #smoothvalcol=valcol+'_lpf';
        #eyedata.loc[ (True==eyedata.bad),
        smoothtconst=0.150;
        for valcol in smoothcols:
            eyedata[valcol] = eyedata[valcol].rolling(center=True,
                                                      min_periods=3,
                                                      window=pd.to_timedelta(smoothtconst, unit='s'),
                                                      ).mean();
            pass;
        eyedata = eyedata.reset_index(drop=True);
        ax.scatter(eyedata.Tsec, eyedata.cgx_dva, s=2.0, label='LPF X');
        ax.scatter(eyedata.Tsec, eyedata.cgy_dva, s=2.0, label='LPF Y');
        
        ax.set_ylim([-ymax, ymax]);
        ax.legend();
        ax.set_xlabel('Time (s)');
        ax.set_ylabel('Gaze (DVA)');
        
        ax.axhline(0);

        #ax.set_title("{} {} {} {} {} {} {}\nBLK {} in {}".format(edfrow['name'], blockrow.edfdatetime, blockrow.kind,
        #                                                         blockrow.APPA, blockrow.FREEFIX, blockrow.grp,
        #                                                         "Practice" if blockrow.ispract=='yes' else "NotPractice",
        #                                                         blockrow.blkidx, blockrow.edffile));
        ax.set_title(f"{edfrow.edffile} {blockrow.blkidx}");
        
        fig.tight_layout();
        fname="{}_blk{}.png".format(edfrow.edffile, blockrow.blkidx);
        outfpath=os.path.join(out_csv_path, fname);
        print("Saving figure {}".format(outfpath));
        plt.savefig(outfpath, dpi=250);
        #plt.show();
        plt.close(fig);
        pass;

    
    return;







def preproc_file(fn, out_csv_path, doplot=False):
    print("Setting input EDF filename to [{}]".format(fn));
    
    row, s, m, bt, b = pu.preproc_peyefv_edf(fn, out_csv_path=out_csv_path);
    
    print(s);
    print(bt);
    print(b);
    print(row);

    row2 = { a:[row[a]] for a in row };
    df = pd.DataFrame(row2);
    
    if(False == row['edferror'] and doplot ):
        plotit(df.iloc[0], out_csv_path);
        pass;
    
    
    print(df);
    
    return df;

####### parallel func wrapper ########
def parallel_preproc( mytup ):
    fn = mytup[0];
    out_csv_path = mytup[1];

    return preproc_file( fn, out_csv_path );




####### end parallel func ###########



def main():
    idxfile=sys.argv[1];
    inpath=sys.argv[2];
    outcsv=sys.argv[3];  #'infant_outcsvs';
    
    idxdf = pd.read_csv(idxfile);
    print(idxdf);

    #wrapped = [ tuple((x[1], outcsv)) for x in idxdf.iterrows() ];
    
    
    wrapped=list();
    for i, idxrow in idxdf.iterrows():
        subj=idxrow['subj'];
        path=idxrow['path'];
        
        fullpath = os.path.join(inpath, path);
        files=os.listdir(fullpath);
        for fn in files:
            fn2 = os.path.join(fullpath, fn);
            if( pu.utils.isfile(fn2) and
                fn2.lower().endswith('.edf') ):
                
                print("Adding: Processing [{}] (SUBJECT: {})".format(fn2, subj));
                wrapped.append( (fn2, outcsv) );
                pass;
            pass;
        pass;
    
    print("Will proc: {}".format(wrapped));
    
    MULTIPROC=True;
    #MULTIPROC=False;
    NPROC=60;
    results=list();
    if(MULTIPROC):
        with Pool(processes=NPROC) as pool:
            results = pool.map(parallel_preproc, wrapped);
            print(results);
            pass;
        pass;
    else:
        for row in wrapped:
            results.append( parallel_preproc(row) );
            pass;
        pass;

    rows=list();
    for row in results:
        if( row.iloc[0]['edferror'] ):
            continue;
        else:
            row['datadir']=path;
            if('subj' in row ):
                raise Exception("WTF why subj here");
            
            row['subj'] = subj;
            
            import json
            print(json.dumps(row.iloc[0].to_dict(), indent=4, sort_keys=True));
            rows.append(row);
        pass;
    
    '''
    rows=list();
    for i, idxrow in idxdf.iterrows():
        subj=idxrow['subj'];
        path=idxrow['path'];
        
        fullpath = os.path.join(inpath, path);
        files=os.listdir(fullpath);
        print(files);
        for fn in files:
            fn2 = os.path.join(fullpath, fn);
            if( pu.utils.isfile(fn2) and
                fn2.lower().endswith('.edf') ):
                                
                print("Processing [{}] (SUBJECT: {})".format(fn2, subj));
                row = preproc_file(fn2, outcsv);
                
                if( row.iloc[0]['edferror'] ):
                    continue;
                else:
                    row['datadir']=path;
                    if('subj' in row ):
                        raise Exception("WTF why subj here");
                    
                    row['subj'] = subj;
                    
                    import json
                    print(json.dumps(row.iloc[0].to_dict(), indent=4, sort_keys=True));
                    
                    rows.append(row);
                    pass;
                pass;
            pass;
        
        ## Problem, I think some old infant stuff, the VB info and DVA sizes are not correct, so need to figure it out from
        ## pixels (Physical distances and measures should be correct?).
        
        pass;
    '''
    
    bigrows = pd.concat(rows);
    
    outidxcsv='final_infant_rows.csv';
    print("Writing to file: {}".format(outidxcsv));
    bigrows.to_csv(outidxcsv, index=False);
    
    return 0;

if __name__=='__main__':
    exit(main());
    
