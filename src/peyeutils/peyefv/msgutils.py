#from peyeutils.utils.unitutils import *;
#from peyeutils.preproc.preproc import *;

import peyeutils.utils as ut;
import peyeutils.preproc as pre;

import pandas as pd;
import numpy as np;


def separate_EDF_msg_tags(messages):
    if( 'tag' in messages.columns ):
        print("Already tagged EDF messages, returning identical");
        return messages;
    
    msgs =  messages['message'].to_list();
    sep=' ';
    taglist=[];
    bodylist=[];
    for msg in msgs:
        smsg=msg.split(sep);
        if( len(smsg) > 0 ):
            tag = smsg[0];
            #body = smsg[1:]; #REV: will properly return empty list...
            body = ' '.join(smsg[1:]); #REV: will properly return empty list...
            taglist.append(tag);
            bodylist.append(body);
            pass;
        pass;
    messages['body'] = bodylist;
    messages['tag'] = taglist;
    return messages;

def get_eye_params( mymessages ):
    tag='EYE_USED';
    msgs=mymessages[ mymessages.tag == tag ]['body'];#msg'];
    if( len(msgs) < 1 ):
        print("Couldnt find EYE_USED msg?!?!");
        exit(1);
        pass;
    m = msgs.iloc[0].split(' ');
    mydict={'EYE_USED_eyeidx':m[0], 'EYE_USED_mode':m[1]};
    return mydict
    

def get_tag_messages( mymessages, tag ):
    vbmsgs = mymessages[ mymessages['tag'] == tag ];
    msgs=[];
    for idx, row in vbmsgs.iterrows():
        msgs.append( ' '.join(row['body'].split(' ')) );
        pass;
    return msgs;

def get_tag_lists( mymessages, tag ):
    vbmsgs = mymessages[ mymessages['tag'] == tag ];
    mylists=[];
    for idx, row in vbmsgs.iterrows():
        mylists.append( row['body'].split(' ') );
        pass;
    return mylists;

def reccfg_params( cfg ):
    mydict={};
    mydict['mode'] = cfg[0]; #CR etc.
    mydict['samplerate'] = int(cfg[1]);
    mydict['filefilter'] = int(cfg[2]); # 0=off, 1=standard, 2=extra (to this file)
    mydict['linkanalogfilter'] = int(cfg[3]); #0=off, 1=standard, 2=extra (to link and to e.g. analogue card)
    mydict['eyes'] = cfg[4]; #L if left, R if right, LR if left and right
    return mydict;

def gazecoord_params( cfg ):
    mydict={};
    mydict['l']=float(cfg[0]); #REV: maybe should be int but meh
    mydict['t']=float(cfg[1]);
    mydict['r']=float(cfg[2]);
    mydict['b']=float(cfg[3]);
    return mydict;
    

#REV: don't let them change DURING A RECORDING the VB...
#REV: okey then?
#REV: or change it so it has like a "set at time" thing.
#REV: there are multiple rows of VB right now though (it wouldn't fit on one line messgge in eyelink)
def get_tag_params( mymessages, tag, filter1=None, sep='=', tcol='time' ):
    vbmsgs = mymessages[ mymessages['tag'] == tag ];
    mydict={};
    for idx, row in vbmsgs.iterrows():
        msg = row['body'].split(' ');
        if( filter1 is not None and len(msg) >= 1):
            tofilt = msg[0];
            if( filter1 != tofilt ):
                continue;
            msg = msg[1:];
            pass;
        for item in msg:
            #print(item);
            tup = item.split(sep);
            name=tag+"_"+tup[0];
            if( name in mydict ):
                print("WARNING!!! OVERWRITING VBOX PARAM {} {} IN SAME RECORIDNG FILE?!?!?! {}".format(name, tup[1], vbmsgs));
                #exit(1);
                pass;
            
            if( len(tup) == 1 ):
                #print(tag, row['body'])
                print("REV: TAG PARAMS NOT A=B? {}".format(item));
                mydict[name] = row[tcol]; #time; #REV: e.g. S=, E=, etc.
                #REV: will contain time info.
                pass;
            elif(len(tup) == 2):
                mydict[name] = tup[1];
                pass;
            else:
                print("Error, tuple length > 2? A=B=C?");
                exit(1);
                pass;
                        
            pass;
        pass;
    return mydict;

def get_available_tags( mymessages ):
    return mymessages['tag'].unique();

def get_elparams( mymessages ):
    elparams = get_tag_lists( mymessages, 'RECCFG' );
    eldict = reccfg_params( elparams[0] );
    #coorddict = gazecoord_params( get_tag_lists( mymessages, "GAZE_COORDS" )[0] );

    return eldict;

def get_gazecoords( mymessages ):
    coorddict = gazecoord_params( get_tag_lists( mymessages, "GAZE_COORDS" )[0] );
    return coorddict;
    

def get_recordingsession_info( mymessages ):
    if( 'tag' not in mymessages.columns ):
        mymessages = separate_EDF_msg_tags(mymessages);
        pass;
    
    mydict = {};
    
    if( 'tag' not in mymessages.columns and
        'body' not in mymessages.columns ):
        raise Exception("need to call separate EDF msg tags first");
    
    vbparams = get_tag_params(mymessages, 'VB');
    #print(vbparams);
    
    
    eldict = get_elparams(mymessages);
    
    coorddict = get_gazecoords(mymessages);
    
    #REV: oh shit this will be over written?
    sessdict = get_tag_params(mymessages, 'SESSION', filter1='S');
    #print(sessdict);
    
    screendict = get_tag_params(mymessages, 'SCREEN');
    
    eyedict = get_eye_params( mymessages ); #EYE_USED
    
    mydict.update(eyedict);
    mydict.update(screendict);
    mydict.update(vbparams);
    mydict.update(eldict);
    mydict.update(coorddict);
    mydict.update(sessdict);
    
    if( len(mydict) != len(vbparams)+len(eldict)+len(coorddict)+len(sessdict) +len(eyedict)+len(screendict)):
        print("Something got overwritten");
        exit(1);
        pass;
    
    
    return mydict;




def preproc_peyefreeviewing_dva_from_flatscreen(df, msgs):
    rdict = get_recordingsession_info(msgs);
    distm=float(rdict['VB_DM']);
    ppm=float(rdict['VB_PPM']);
    df = pre.preproc_SHARED_dva_from_flatscreen(df, ppm, distm);
    return df;







def import_fmri_trials( mymessages, includeAasE=False, fixvidlensec=None, fiximglensec=None, fixdvawid=None, noaborts=True ):
    #REV: can't use this for aborted trials.
    #REV: should do difference for each case but fuck it lol
    if( fixvidlensec is not None or fiximglensec is not None ):
        includeAasE = False;
        pass;
    
    vbparams = get_tag_params(mymessages, 'VB');
    vbdict = vbparams;
    dva_per_m = ut.get_center_dva_per_meter( float(vbdict['VB_DM']), float(vbdict['VB_PPM']) );
    dva_per_px = dva_per_m / float(vbdict['VB_PPM']); #REV: dva/m / px/m = dva/m * m/px = dva/px
    
    ################################################################
    ############# EXTRACT VID TRIALS FROM MESSAGES! ################
    ################################################################
    
    #REV: for each video (i.e. DOVID trial (S->E)! -- note may usually be more than 1 per FILE!
    #REV: Includes (S->A too!)
    #subfm = mymessages[ (mymessages.filename == fn) & (mymessages.tag == 'VID') ].copy();
    subfm = mymessages[ (mymessages.tag == 'VID') ].copy();
    subfm.sort_values(by='Tsec', inplace=True);
    
    blkm = mymessages[ (mymessages.tag == 'BLK') ].copy();
    if( len(blkm) > 0 ):
        print("BLOCK INFO");
        print(blkm);
        pass;
    
    fmridf = list();
    fmrim = mymessages[ (mymessages.tag == 'FMRI') ].copy();
    if( len(fmrim) > 0 ):
        print("FMRI INFO");
        print(fmrim);
        for i, row in fmrim.iterrows():
            fmrilst = row.body.split('=');
            if(fmrilst[0] != 'TIME' or len(fmrilst) != 2):
                print(fmrilst, fmrim.body);
                raise Exception("FMRI MSG FORMAT BAD");
            fmrit = float(fmrilst[1]);
            fmridf.append( pd.DataFrame( dict(fmrist_wall=[fmrit],
                                              fmrist_s=[row.Tsec],
                                              fmrist_el=[row.ELtime]) ) );
            pass;
        pass;
    if( len(fmridf) == 0 ):
        fmridf = pd.DataFrame( columns=['fmrist_wall', 'fmrist_s', 'fmrist_el'] );
        pass;
    else:
        fmridf = pd.concat( fmridf );
        pass;
    
    #if( len(fmrim) > 1 ):
    #    print("SHOULD NOT BE MORE THAN 1 (not implemented)");
    #    raise Exception("MULTI FMRI");
    
    
    
    #REV: this handles both E and A because it assumes any VID tags which are not S must be E (even though some will be A)
    if( includeAasE ):
        subfm['sten'] = [ 'S' if (row.body[0] == 'S') else 'E' for idx,row in subfm.iterrows() ];
        pass;
    else: #As remain As...
        #subfm['sten'] = [ 'S' if row.body[0] == 'S' else 'E' if row.body[0] == 'E' else 'A' for idx,row in subfm.iterrows() ];
        subfm['sten'] = [ row.body[0] for idx,row in subfm.iterrows() ];
        pass;

    subfm['orig'] = subfm.sten;
    origfm=subfm.copy();
    #REV: this will make uneven numbers! because we lose evenly matched S/E. But we should be able to fix them...based on video length?
    ## Eww too much work. Just most recent/next end/start.
    ### DO THIS AT THE END
    '''
    if( noaborts ):
        subfm = subfm[ (subfm['sten'] == 'E') | (subfm['sten'] == 'S') ]; #just drop all As. We will sort out and drop
        #those trials S's later (by only taking S right before an E in time).
        pass;
    else:
        subfm.loc[ (subfm.sten == 'A'), 'sten' ] = 'E'; #REV: replace all As with Es.
        pass;
    
    '''
    
        
    #REV: special case for recovery from my mistake of not writing VID S
    #REV: note, fixdvawid is literally its wid, not scaled target wid...
    if( not includeAasE and (fixvidlensec is not None or fiximglensec is not None) and fixdvawid is not None ):
        endf=subfm[subfm.sten=='E'].copy();
        
        #REV: insert the missing rows...
        for (enidx, enrow) in endf.iterrows():
            enmsg = enrow.body.split(' ');
            vid = enmsg[1];
            ent = float(enmsg[2].split('=')[1]);

            #REV get vid w/h/fps/frames.
            #w, h, fps, fr = cv_vid_file_exists( vidpathdict[vid] );
                        
            #REV: scale based on PPM and DM
            wpx = fixdvawid / dva_per_px;
            ratio=float(h)/w;
            hpx = ratio * wpx;
            
            # S $VID WPX=X HPX=Y XPX=X YPX=Y T=T
            
            if( 0==fps ):
                FIXLEN=fiximglensec;
            else:
                FIXLEN=fixvidlensec;
                pass;
            smsg = ['S', vid, "WPX={}".format(int(wpx)), "HPX={}".format(int(hpx)), "XPX=0", "YPX=0", "T={}".format(ent-FIXLEN) ];
            starttime = enrow.ELtime - sec_to_msec(FIXLEN);
            newrow = enrow.copy();
            newrow.ELtime = starttime;
            newrow.Tsec = starttime * EL_TIMEUNIT_SEC;
            newrow.Tmsec = starttime;
            
            newrow.body = smsg;
            newrow.sten='S';
            subfm.loc[ len(subfm.index) ] = newrow;
            pass;
        pass;
    
    #REV: sort them.
    subfm.sort_values(by='Tsec', inplace=True); #REV: I think...?
    
    #REV: only want if they are DIRECTLY PAST EACH OTHER!
    #subfm.drop_duplicates( subset=['sten'], keep='last', inplace=True );
    #print(subfm);
    
    #REV: horrible recursion...remove e.g. if S S S we keep only the last one... (because this should go in order?)
    #REV: keep going until I remove none in a pass.
    mylen=len(subfm.index) + 1;
    while( len(subfm.index) < mylen ):
        print("DROPPING S's!");
        mylen=len(subfm.index);
        # Get [ True False True True False True ] where each is is that location == S
        # Get same thing but shifted by 1 (forward?). Filling the original blank with false...?
        # Drop all rows where BOTH are true (i.e. I am 'S', and the one before me was 'S'. Never drop the first one.
        # Will this make it "longer" at the end?
        subfm.drop( subfm[ (subfm['sten'].eq('S').shift(1, fill_value=False))
                           &
                           (subfm['sten'].eq('S'))
                          ].index,
                    axis=0,
                    inplace=True );
        pass;
    
    mylen=len(subfm.index) + 1;
    while( len(subfm.index) < mylen ):
        print("DROPPING E's!");
        mylen=len(subfm.index);
        subfm.drop( subfm[ (subfm['sten'].eq('E').shift(1, fill_value=False)) & (subfm['sten'].eq('E')) ].index, axis=0, inplace=True );
        pass;
    
    #REV: I now need to filter it to make sure I throw away up until the last S if SSSSE for example (because was SASASASE)
    #REV: these will not be the same size!!!!
    stdf=subfm[subfm.sten=='S'].reset_index(drop=True); #REV: assume/guaranteed still sorted...
    endf=subfm[(subfm.sten=='E') | (subfm.sten=='A')].reset_index(drop=True); #REV: if we will keep, keep anyways?
    #endf=subfm[subfm.sten=='E'].reset_index(drop=True); #REV: if we will keep, keep anyways?
    
    #print(stdf);
    #print(endf);
    
    if( len(stdf.index) != len(endf.index) ):
        print(stdf);
        print(endf);
        print(origfm);
        raise Exception("REV: dropped duplicates incorrectly??! Start: [{}]   End: [{}]".format(len(stdf.index) , len(endf.index)) );
    
    #REV: just drop all with orig=='A', but must match starting "S" as well...
    if( noaborts ):
        #abortidxs = endf[ endf.orig=='A' ].index;
        #REV: cheating...they line up 1-to-1 so this just works...
        #REV: this will work because we zero-align them and make them dense first?
        endf = endf[ endf.orig != 'A' ];
        stdf = stdf.iloc[ endf.index ]; #REV: or...drop?
        pass;
    else:
        # Nothing to do...just keep all as they are.
        pass;

    if( len(stdf.index) != len(endf.index) ):
        print(stdf);
        print(endf);
        raise Exception("AFTER NOABORT -- REV: dropped duplicates incorrectly??! Start: [{}]   End: [{}]".format(len(stdf.index) , len(endf.index)) );
    
    #REV:could load other stuff from context, e.g. that file's other settings...in fact it should?
    #REV: wpx and hpx can change between trials (vid not shown same size scaled for DVA!).
    '''
    vidtrialdf = pd.DataFrame( columns=['start_el', 'end_el',
                                        'start_s', 'end_s',
                                        'start_wall', 'end_wall',
                                        'video', 'vidw_px', 'vidh_px',
                                        'vidxpos_px', 'vidypos_px', 'isabort'] );
    '''
    vidtrialdf=list();
    trialnum=0;
    
    for (stidx, strow), (enidx, enrow) in zip(stdf.iterrows(), endf.iterrows()):
        smsg=strow.body.split(' ');
        emsg=enrow.body.split(' ');
        trialnum  += 1; #REV: errr....this is within file?
        
        vid = smsg[1];
        if( len(vid) < 5 ):
            print("ERROR!");
            print(strow);
            print(enrow);
            raise Exception("WTF SOMETHING WRONG?!");
        
        w = int(smsg[2].split('=')[1]);
        h = int(smsg[3].split('=')[1]);
        x = int(smsg[4].split('=')[1]);
        y = int(smsg[5].split('=')[1]);
        t = float(smsg[6].split('=')[1]);
        
        envid = emsg[1];
        ent = float(emsg[2].split('=')[1]);
        
        if(vid != envid):
            print("Wtf vid not end vid (should match!)? {} {}".format(vid,envid));
            exit(1);
            pass;
        
        isabort = (enrow.orig=='A');

        if( np.isnan(strow.ELtime) ):
            raise Exception("WTF NA TIME");

                
        newvidrow = dict( start_el=strow.ELtime,
                          end_el=enrow.ELtime,
                          start_s=strow.Tsec,
                          end_s=enrow.Tsec,
                          start_wall=t,
                          end_wall=ent,
                          video=vid,
                          vidw_px=w,
                          vidh_px=h,
                          vidxpos_px=x,
                          vidypos_px=y,
                          isabort=isabort );
        newvidrow = { k:[newvidrow[k]] for k in newvidrow };
        '''
        newvidrow = [ strow.ELtime, enrow.ELtime,
                      strow.Tsec, enrow.Tsec,
                      t, ent,
                      vid, w, h, x, y, isabort ];
        '''
        #vidtrialdf.loc[len(vidtrialdf.index)] = newvidrow;
        vidtrialdf.append( pd.DataFrame(newvidrow) );
        pass;

    if(len(vidtrialdf)>0):
        vidtrialdf = pd.concat(vidtrialdf);
        pass;
    else:
        vidtrialdf = pd.DataFrame( columns=['start_el', 'end_el',
                                            'start_s', 'end_s',
                                            'start_wall', 'end_wall',
                                            'video', 'vidw_px', 'vidh_px',
                                            'vidxpos_px', 'vidypos_px', 'isabort'] );
        pass;
    
    
    if( len(fmridf.index) > 0 ):
        vidtrialdf['isfmri'] = True;
        vidtrialdf['fmrist_el'] = np.nan;
        vidtrialdf['fmrist_s'] = np.nan;
        vidtrialdf['fmrist_wall'] = np.nan;
        for i, r in vidtrialdf.iterrows():
            vidst = r.start_s;
            #print("myidx", r.index);
            #print("vididx", vidtrialdf.index);
            availfmri = fmridf[ fmridf.fmrist_s <= vidst ].copy();
            if( len(availfmri.index) > 1 ):
                print("ERROR/WARNING: FMRI started multiple times before me, I will use the most recent <= vidstart");
                pass;
            if( len(availfmri.index) == 0 ):
                print("THIS VIDEO TRIAL HAD NO FMRI YET IT SEEMS?!");
                vidtrialdf.loc[ vidtrialdf.index==i, 'isfmri' ] = False;
                vidtrialdf.loc[ vidtrialdf.index==i, 'fmrist_el' ] = np.nan;
                vidtrialdf.loc[ vidtrialdf.index==i, 'fmrist_s' ] = np.nan;
                vidtrialdf.loc[ vidtrialdf.index==i, 'fmrist_wall' ] = np.nan;
                pass;
            else:
                availfmri = availfmri[ availfmri.fmrist_s == availfmri.fmrist_s.max() ];
                if( len(availfmri.index) != 1 ):
                    raise Exception("I fucked up");
                fmridata = availfmri.iloc[0];
                vidtrialdf.loc[ vidtrialdf.index==i, 'isfmri' ] = True;
                vidtrialdf.loc[ vidtrialdf.index==i, 'fmrist_el' ] = fmridata.fmrist_el
                vidtrialdf.loc[ vidtrialdf.index==i, 'fmrist_s' ] = fmridata.fmrist_s;
                vidtrialdf.loc[ vidtrialdf.index==i, 'fmrist_wall' ] = fmridata.fmrist_wall;
                pass;
            pass;
        #vidtrialdf['fmrist_el'] = fmridf.iloc[0].fmrist_el;
        #vidtrialdf['fmrist_s'] = fmridf.iloc[0].fmrist_s;
        pass;
    else:
        vidtrialdf['isfmri'] = False;
        vidtrialdf['fmrist_el'] = np.nan;
        vidtrialdf['fmrist_s'] = np.nan;
        vidtrialdf['fmrist_wall'] = np.nan;
        pass;

    vidtrialdf['fmri_offset_s'] = vidtrialdf.start_s - vidtrialdf.fmrist_s;
    vidtrialdf['fmri_offset_el'] = vidtrialdf.start_el - vidtrialdf.fmrist_el;
    vidtrialdf['fmri_offset_wall'] = vidtrialdf.start_wall - vidtrialdf.fmrist_wall;
    
    return vidtrialdf;








def import_fmri_blocks(msgdf, sampdf, trialsdf):
    blockdf=list();
    blockedtrialsdf=list();
    
    if( 'Tsec' not in msgdf.columns ):
        raise Exception("File (MSG DF) has no Tsec?");
    blkm = msgdf[ (msgdf.tag == 'BLK') ].copy();
    blkm = blkm.sort_values(by='Tsec');
    blkm['sten'] = [ row.body[0] for idx,row in blkm.iterrows() ];
    
    fmrim = msgdf[ (msgdf.tag == 'FMRI') ].copy();
    fmrim = fmrim.sort_values(by='Tsec');

    abortedblks = blkm[blkm.sten=='A'];
    if(len(abortedblks.index)>0):
        print("Contained ABORTED blocks. Treating as END");
        pass;
    blkm.loc[ (blkm.sten=='A'), 'sten' ] = 'E';
    
    stdf=blkm[blkm.sten=='S'].reset_index(drop=True); #REV: assume/guaranteed still sorted...
    endf=blkm[blkm.sten=='E'].reset_index(drop=True);
    if( len(stdf.index) != len(endf.index ) ):
        print(stdf);
        print(endf);
        #raise Exception("Uneven number of S/E for BLK ({})".format(row.edffile));
        
        
        #REV: add extra at very end just to be sure
        tmprow = stdf.iloc[ len(stdf.index)-1 ]; #stdf.tail(1); #iloc[ len(stdf.index)-1 ];
        tmprow['sten'] = 'E';
        #sampdf = pd.read_csv( samppath );
        lastsamp = sampdf['Tsec'].max();
        tmprow['Tsec'] = lastsamp;
        print("Uneven number of S/E for BLK ({}). Adding missing. (Artificial match is last sample [{}])".format(edfrow.edffile, lastsamp));
        
        endf.loc[ len(endf.index) ] = tmprow;
        
        #REV: make dummy "E" just before each "S" except the first.
        newEs = stdf.iloc[1:].copy();
        newEs['sten'] = 'E'
        newEs['Tsec'] -= 0.020; #20 msec, just so dividsible by 1k, 2k, 250hz, 500hz, 200hz?
        endf = pd.concat([endf, newEs]);
        #endf = endf.sort_values(by='time').reset_index(drop=True);
        
        #REV: recombine S/E and sort to interlace.
        bothdf=pd.concat([stdf,endf]);
        bothdf = bothdf.sort_values(by='Tsec').reset_index(drop=True);
        
        mylen=len(bothdf.index) + 1;
        while( len(bothdf.index) < mylen ):
            print("DROPPING E's!");
            mylen=len(bothdf.index); ## == len + 1, same as after drop+1
            bothdf.drop( bothdf[ (bothdf['sten'].eq('E').shift(1, fill_value=False)) & (bothdf['sten'].eq('E')) ].index, axis=0, inplace=True );
            
            #bothdf = bothdf.sort_values(by='time').reset_index(drop=True);
            pass;
        
        print("NEW COMBINED:");
        stdf = bothdf[ bothdf.sten == 'S' ];
        endf = bothdf[ bothdf.sten == 'E' ];
        print(stdf);
        print(endf);
        pass;
    
    #REV: must be "E" missing. Ignore trial data...could use it though.
    # The smallest timed "E" that is bigger than my S, but not bigger than any Ss after me.
    edfblocknum=0;
    for (stidx, strow), (enidx, enrow) in zip(stdf.iterrows(), endf.iterrows()):
        smsg=strow.body.split(' ');
        emsg=enrow.body.split(' ');
        SFREEFIX=smsg[1];
        #REV: can't use seconds because might be missing END MESSAGE
        #stsec = float(smsg[2].split('=')[1]);
        stel = strow.ELtime;
        enel = enrow.ELtime;
        stsec = strow.Tsec;
        ensec = enrow.Tsec;
        
        EFREEFIX=smsg[1];
        #etsec = float(emsg[2].split('=')[1]);
        
        if( SFREEFIX != EFREEFIX ):
            raise Exception("start/end type is not same (FREE/FIX)");
        
        blocktrialsdf = trialsdf[ (trialsdf.start_s >= stsec) & (trialsdf.end_s < ensec) ].sort_values(by='start_s').reset_index(drop=True);
        if( len(blocktrialsdf.index) < 1 ):
            print("BLOCK WOULD CONTAIN NO VIDEOS [{}] -- SKIPPING".format(edfrow.edffile));
            continue;

        tocheck = list(blocktrialsdf.columns);
        viddfcols=['start_el', 'end_el',
                   'start_s', 'end_s',
                   'start_wall', 'end_wall',
                   'video', 'vidw_px', 'vidh_px',
                   'vidxpos_px', 'vidypos_px', 'isabort']
        
        tocheck = [ c for c in tocheck if c not in viddfcols ];
        newblk = blocktrialsdf[ tocheck ].drop_duplicates();
                
        if( 1 != len(newblk.index) ):
            print(newblk);
            raise Exception("Non-unique items within block, this should be impossible");
        else:
            newblk = newblk.iloc[0];
            pass;

        ## REV: grab VBOX params here too (in case they change?)
        ## REV: I think VBOX has multiple rows ? SO need unique "command" types, last of each before start of block.
        newblk['blkstart_el'] = stel;
        newblk['blkstart_s'] = stsec;
        newblk['blkend_el'] = enel;
        newblk['blkend_s'] = ensec;
        newblk['blkidx'] = edfblocknum;
        
        #REV: find best FMRI for me.
        #REV: oh, note it should be also that I am best for that fmri block!
        if(len(fmrim.index)>0):
            myfmrim = fmrim.copy();
            print(myfmrim);
            print("DOING {} - {}".format(myfmrim['Tsec'], newblk['blkstart_s']));
            myfmrim['dist'] = myfmrim['Tsec'] - newblk['blkstart_s']; #will be negative if fmri before (smaller) than block start. Way it works is start block -> waits for FMRI key ('6') -> starts trial for sync.
            #print("HI!", myfmrim);
            myfmrim = myfmrim[ myfmrim['dist'] > 0 ];
            
            
            if(len(myfmrim) > 0):
                myfmrim = myfmrim.sort_values(by='dist').reset_index(drop=True);
                best = myfmrim.iloc[0];
                besttime = best['Tsec'];
                tmpstdf=stdf.copy();
                tmpstdf['dist'] = besttime - tmpstdf['Tsec']; #REV: smallest one.
                tmpstdf = tmpstdf.sort_values(by='dist').reset_index(drop=True);
                bestblk = tmpstdf.iloc[0];
                if( bestblk.Tsec != newblk.blkstart_s ):
                    print("There is no valid FMRI message for me");
                    newblk['fmrist_s'] = np.nan;
                    newblk['fmrist_el'] = np.nan;
                    newblk['fmrist_wall'] = np.nan;
                    pass;
                else:
                    print("FMRI trial for this block starts at: {}".format(best['Tsec']));
                    walltime = float(best.body.split('=')[1]);
                    newblk['fmrist_s']  = best['Tsec'];
                    newblk['fmrist_el']  = best['Tmsec'];
                    newblk['fmrist_wall']  = walltime;
                    pass;
                pass;
            else:
                print("No valid FMRI trial for this block");
                newblk['fmrist_s'] = np.nan;
                newblk['fmrist_el'] = np.nan;
                newblk['fmrist_wall'] = np.nan;
                pass;
            pass;
        else:
            print("This block has no FMRI");
            newblk['fmrist_s'] = np.nan;
            newblk['fmrist_el'] = np.nan;
            newblk['fmrist_wall'] = np.nan;
            pass;
        
        ######### SET NEW VALUES FOR TRIALS WITHIN THAT BLOCK....###########
        blocktrialsdf['trialidx'] = list(range(len(blocktrialsdf)));
        blocktrialsdf['blkidx'] = edfblocknum;
        for it, trow in blocktrialsdf.iterrows():
            trow['fmrist_s'] = newblk.fmrist_s;
            trow['fmrist_el'] = newblk.fmrist_el;
            trow['fmrist_wall'] = newblk.fmrist_wall;
            trow['fmri_offset_s'] = trow.start_s - newblk.fmrist_s;
            trow['fmri_offset_el'] = trow.start_el - newblk.fmrist_el;
            trow['fmri_offset_wall'] = trow.start_wall - newblk.fmrist_wall;
            
            blockedtrialsdf.append(pd.DataFrame([trow]));
            pass;
        
        
        blockdf.append(pd.DataFrame([newblk]));
        edfblocknum  += 1;
        pass; ### END for stidx/enidx
    
    blockdf = pd.concat(blockdf);
    blockedtrialsdf = pd.concat(blockedtrialsdf);
    
    return blockdf, blockedtrialsdf;
