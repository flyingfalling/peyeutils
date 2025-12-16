import peyeutils as pu;
import numpy as np;
import pandas as pd;
import imufusion;

def extract_cols( df, tcol ):
    tsec = df[tcol];
    acc = df[ ['accelerometer_' + str(x) for x in range(3) ] ];
    gyr = df[ ['gyroscope_' + str(x) for x in range(3) ] ];
    if( 'magnetometer_0' in df.columns ):
        mag = df[ ['magnetometer_' + str(x) for x in range(3) ] ];
        pass;
    else:
        mag = None;
        pass;
    
    return tsec, acc, gyr, mag;


def ahrs_pose_heading(df, kind, srhzsec, tcol='timestamp', fusionsettings=None, pretburnin=0):
    import numpy as np
    import pandas as pd
    if( kind == 'tobiig3' ):
        res =  pu.tobiig3.wrangle_tobiig3_imu_sensor_data(df, tcol=tcol);
        pass;
    elif( kind == 'nwu_g_degsec' ):
        res =  extract_cols(df, tcol=tcol);
        pass;
    elif( kind == 'nwu_mss_degsec' ):
        res =  extract_cols(df, tcol=tcol);
        pass;
    else:
        raise Exception("Not impl kind of IMU data: {}".format(kind));

    
    if( res ):
        tsec, acc, gyr, mag = res;
        if( kind == 'nwu_mss_degsec' ):
            acc /= 9.807;
            pass;
        
        tsec_mat = tsec.values.flatten();
        acc_mat = np.array(acc.values);
        gyr_mat = np.array(gyr.values);
        
        gyr_range = (np.nanmin(gyr_mat), np.nanmax(gyr_mat));
        acc_range = (np.nanmin(acc_mat), np.nanmax(acc_mat));
        print("ACC RANGE: {}".format(acc_range));
        print("GYR RANGE: {}".format(gyr_range));
        with_mag=True;
        if( mag is not None and with_mag ):
            mag_mat = np.array(mag); #.values);
            with_mag = True;
            mag_range = (np.nanmin(mag_mat), np.nanmax(mag_mat));
            print("MAG RANGE: {}".format(mag_range));
            pass;
        else:
            with_mag = False;
            pass;


        offset = imufusion.Offset(srhzsec);
        ahrs = imufusion.Ahrs();

        if( with_mag ):
            if( fusionsettings == None ):
                #https://github.com/xioTechnologies/Fusion
                ahrs.settings = imufusion.Settings(
                    imufusion.CONVENTION_NWU,  # convention #REV: this means that X is ahead, Y is to left, Z is up.
                    0.5,  # gain of gyroscope trust relative to other sensors.
                    2000,  # gyroscope range
                    10,  # acceleration rejection
                    50,  # magnetic rejection # gyroscope (speed) above which I reject magnetic info?
                    3 * srhzsec,  # recovery trigger period = 5 seconds
                );
            else:
                ahrs.settings = fusionsettings;
                pass;
            pass;


        if( pretburnin > 0 ):
            mintsec = df[tcol].min();
            pret = np.array(np.arange(mintsec-pretburnin, mintsec, 1/srhzsec));
            
            preacc = np.array([ 0, 0, 1 ] * len(pret)).reshape([len(pret),3]);
            pregyr = np.array([ 0, 0, 0 ] * len(pret)).reshape([len(pret),3]);

            if(with_mag):
                #premag = np.array([ mag_mat[50,0], mag_mat[50,1], mag_mat[50,2] ] * len(pret)).reshape([len(pret),3]);
                premag = np.array([ np.nanmean(mag_mat), np.nanmean(mag_mat), np.nanmean(mag_mat) ] * len(pret)).reshape([len(pret),3]);
                pass;
            
            tsec_mat = np.concatenate([pret, tsec_mat]);
            acc_mat = np.concatenate([preacc, acc_mat]);
            gyr_mat = np.concatenate([pregyr, gyr_mat]);
            
            if(with_mag):
                mag_mat = np.concatenate([premag, mag_mat]);
                pass;
            pass;

        dt = np.diff(tsec_mat, prepend=tsec_mat[0]);
        euler = np.empty((len(tsec_mat), 3));
        internal_states = np.empty((len(tsec_mat), 6))
        flags = np.empty((len(tsec_mat), 4))

        for index in range(len(tsec_mat)):
            #REV: fuck, this expects a numpy 3xT...
            

            ##### SKIP NAN INPUTS?!
            if( np.isnan(gyr_mat[index]).any() or
                np.isnan(acc_mat[index]).any() ):
                continue;
            
            gyr_mat[index] = offset.update(gyr_mat[index])
            
            if( with_mag ):
                if( np.isnan(mag_mat[index]).any() ):
                    continue;
                ahrs.update(gyr_mat[index], acc_mat[index], mag_mat[index], dt[index]);
                pass;
            else:
                ahrs.update_no_magnetometer(gyr_mat[index], acc_mat[index], dt[index]);
                pass;
            
            euler[index] = ahrs.quaternion.to_euler()
            
            ahrs_internal_states = ahrs.internal_states
            internal_states[index] = np.array(
                [
                    ahrs_internal_states.acceleration_error,
                    ahrs_internal_states.accelerometer_ignored,
                    ahrs_internal_states.acceleration_recovery_trigger,
                    ahrs_internal_states.magnetic_error,
                    ahrs_internal_states.magnetometer_ignored,
                    ahrs_internal_states.magnetic_recovery_trigger,
                ] );
            
            ahrs_flags = ahrs.flags
            flags[index] = np.array(
                [
                    ahrs_flags.initialising,
                    ahrs_flags.angular_rate_recovery,
                    ahrs_flags.acceleration_recovery,
                    ahrs_flags.magnetic_recovery,
                ] );

            ## END TIMESTEP
            pass;

        #REV: does not check for "mag"
        dfdict = dict( tsec = tsec_mat,
                       gyrx = gyr_mat[:,0], gyry = gyr_mat[:,1], gyrz = gyr_mat[:,2],
                       accx = acc_mat[:,0], accy = acc_mat[:,1], accz = acc_mat[:,2],
                       eulerx = euler[:,0], eulery = euler[:,1], eulerz = euler[:,2],
                       accerr = internal_states[:,0], accign = internal_states[:,1], accrecovertrig = internal_states[:,2],
                       magerr = internal_states[:,3], magign = internal_states[:,4], magrecovertrig = internal_states[:,5],
                       init = flags[:,0], angrecover = flags[:,1], accrecover = flags[:,2], magrecover = flags[:,3]
                      );

        if( with_mag ):
            magdict = dict( magx = mag_mat[:,0], magy = mag_mat[:,1], magz = mag_mat[:,2]);
            dfdict.update(magdict); #REV: note does not return new thing
            pass;
        
        ahrsdf = pd.DataFrame( dfdict );
        
        pass;
    print(ahrsdf);
    return ahrsdf;
