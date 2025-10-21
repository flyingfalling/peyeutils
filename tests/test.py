import peyeutils as pu;

import pandas as pd;
import os;
import matplotlib.pyplot as plt;
import numpy as np;
import sys;


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
        
        ymax=9;
        text_y=ymax;
        for tridx, trdf in blocktrials.groupby('trialidx'):
            #print(trdf.columns);
            #print(trdf);
            vid=trdf.iloc[0].video;
            most = trdf.nsec.max();
            pct = (trdf.ngood.max()/trdf.nsamp.max())*100;
            if( len(trdf.index) != 3 ):
                raise Exception("not 3 for L/R/B as expected: {}".format(len(trdf.index)));
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
        print("Saving figure {}".format(fname));
        plt.savefig(fname, dpi=300);
        plt.show();
        plt.close(fig);
        pass;

    
    return;

def saccades_remodnav( edfrow, out_csv_path):
    msgdf = pd.read_csv( os.path.join(out_csv_path, edfrow['messages_csv']) );
    elparamdict = pu.peyefv.get_elparams(msgdf); # has samplerate etc.
    ELsr=elparamdict['samplerate'];
    sampdf = pd.read_csv( os.path.join(out_csv_path, edfrow['samples_csv']) );

    import peyeutils.eyemovements.remodnav as rv;
    
    params1 = rv.make_default_preproc_params(ELsr, 1, 'xcdva', 'ycdva', 'tsec0');
    params2 = rv.make_default_params();
    params = params1 | params2;

    rdf = remodnav_preprocess_eyetrace2d(eyesamps=sampdf, params=params);
    ev = remodnav_classify_events(rdf, params);

    print(ev);
    ev.to_csv('events.csv', index=False);
    
    return;





def test1(out_csv_path):
    #fn='/mnt/coishare/data/freeviewing/data/bigsmall/nakazawa20251003/PYFREE_nakazawa_SIZEDVA_10__endrec_start_2025-10-03-11-07-42_end_2025-10-03-11-12-52.edf';
    if( len(sys.argv) < 2 ):
        fn='/mnt/coishare/data/freeviewing/data/bigsmall/ozaki20250909/PYFREE_ozaki_SIZEDVA_10__endrec_start_2025-09-09-17-14-52_end_2025-09-09-17-20-28.edf';
        pass;
    else:
        fn=sys.argv[1];
        pass;
    print("Setting input EDF filename to [{}]".format(fn));
    
    row, s, m, bt, b = pu.preproc_peyefv_edf(fn, out_csv_path=out_csv_path);
    
    print(s);
    print(bt);
    print(b);
    print(row);
    
    #bt.to_csv('testbt.csv');
    
    return row;

def test2(row, out_csv_path):
    
    print(row);
    row = { a:[row[a]] for a in row };
    df = pd.DataFrame(row);
    print(df);
    plotit(df.iloc[0], out_csv_path);
    
    return;

def test3(row, out_csv_path):
    row = { a:[row[a]] for a in row };
    df = pd.DataFrame(row);

    saccades_remodnav( df.iloc[0], out_csv_path );
    
    return;

def main():
    outcsv='outcsvs';
    row = test1(out_csv_path=outcsv);
    print("MY ROW", row);
    if( False == row['edferror'] ):
        test2(row, out_csv_path=outcsv);
        test3(row, out_csv_path=outcsv);
        pass;
    else:
        print("Empty row, i.e. no file?");
        pass;
    return 0;

if __name__=='__main__':
    exit(main());
    

