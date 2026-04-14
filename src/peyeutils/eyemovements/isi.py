
import pandas as pd;
import numpy as np;

def compute_ISIs_from_events( ev,
                              zerotime,
                              eventstouse=['SACC', 'BLNK'],
                              label='ISI',
                              stname='stsec',
                              enname='ensec',
                              durname='dursec',
                              ):
    saccblnks = ev[ ev.label.isin(eventstouse) ];
    saccblnks = saccblnks.sort_values(by=stname).reset_index(drop=True);
    
    isis = saccblnks.copy();
    
    isis[stname] = saccblnks.shift(1)[enname].copy();    #start of ISI is the "end" of the PREVIOUS one (will be null for first)
    print(isis);
    if( len(isis.index) > 0 ):
        isis.loc[ isis.index[0], stname ] = zerotime;
        pass;

    isis[enname] = saccblnks[stname];
    isis[durname] = isis[enname] - isis[stname];
    isis['label'] = label;
    
    return isis;



def add_ISIs_to_events( ev,
                        zerotime,
                        eventstouse=['SACC', 'BLNK'],
                        label='ISI',
                        stname='stsec',
                        enname='ensec',
                        durname='dursec',
                       ):
    
    isis = compute_ISIs_from_events( ev,
                                     zerotime,
                                     eventstouse=eventstouse,
                                     label=label,
                                     stname=stname,
                                     enname=enname,
                                     durname=dursec,
                                    );

    ev = pd.concat( [ev, isis] ).reset_index(drop=True);
    
    return ev;
