
import peyeutils as pu;

#REV: blink returns "ampl" and "angle" of blink (as if it were a saccade), in case we want to include it.
#     Note, in cases where we go "off" the image, what do we do? We don't want to "reconnect" at the edge of the image...
#     We want to merge that saccade into the blink?
def compute_blinks_from_sampcol( samps,
                                 dva_per_px,
                                 badcol,
                                 tcol,
                                 xcol,
                                 ycol,
                                 eyecol='eye',
                                ):
    if eyecol not in samps.columns:
        print("Compute blinks, adding eyecol {} to samps, setting to empty string".format(eyecol));
        samps['eye']='';
        pass;
    
    blinkevs=list();
    for eye, eyedf in samps.groupby(eyecol, as_index=False):
        blinkev = pu.preproc.blink_df_from_samples(samps,
                                                   dva_per_px=dva_per_px,
                                                   badcol=badcol,
                                                   tcol=tcol,
                                                   xcol=xcol,
                                                   ycol=ycol,
                                                   );
        blinkev[eyecol] = eye;
        blinkevs.append(blinkev);
        pass;

    import pandas as pd;
    blinkev = pd.concat(blinkevs).reset_index(drop=True);
    return blinkev;




def add_blinks_to_events_from_sampcol( ev, samps,
                                       dva_per_px,
                                       badcol,
                                       tcol,
                                       xcol,
                                       ycol,
                                       eyecol='eye',
                                      ):
    
    blinkev = compute_blinks_from_sampcol( samps,
                                           dva_per_px=dva_per_px,
                                           badcol=badcol,
                                           tcol=tcol,
                                           xcol=xcol,
                                           ycol=ycol,
                                           eyecol=eyecol,
                                          );
        
    #REV: should I remove blinks in which eye did not move much (< 0.5 deg ?). I.e. fixation with intermediate lbink?
    # Vision is not happening during that time and physiologically it is equivalent...and then ISI is?
    import pandas as pd;
    ev = pd.concat([ev, blinkev]);
    ev = ev.sort_values(by='stsec').reset_index(drop=True);
    return ev;


#REV; merges blinks/saccades with very small ISI, or which are back-to-back.
#REV: note kind of want saccade to have "completed" then don't merge? I.e. levelled of.
#REV: or just detect that the blink is a "no movement" blink, in which case it's fixation... (or error?)
#REV: note period of "no data" should not be a blink, may be eyetracker error...blink should be based on
#REV: stereotyped pupil size change at beginning/end? What about ET that don't have pupil size for me?

def merge_blinkedge_saccades( ev, max_inter_sec=0.060 ):
    #REV: basically, assuming user has removed "too-small" ISI, we will merge subsequent blinks and saccades.
    #REV: look for blinks and saccades which have less than some threshold between them (e.g. 60 msec?), and
    ## compress them. Either before/after.
    #ev = ev.sort_values(by=);
    #REV: label them BLNKSACC or something?
    
    return ev;
