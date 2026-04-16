import pandas as pd;
import matplotlib.pyplot as plt;

import sys;

def main():
    df = pd.read_csv(sys.argv[1]);
    df2 = df[df.label=='SACC'];
    fig,axs=plt.subplots(nrows=1,ncols=2, figsize=(14,7));
    axs[0].scatter(df2.ampl, df2.medvel);
    axs[0].set_title('Ampl-MedVel');
    axs[0].set_xlabel('Ampl(DVA)')
    axs[0].set_ylabel('MedVel (Deg/Sec)')
    axs[1].scatter(df2.ampl, df2.ensec-df2.stsec);
    axs[1].set_title('Ampl-Duration');
    axs[1].set_xlabel('Ampl(DVA)')
    axs[1].set_ylabel('Duration (Deg/Sec)')
    plt.show();
    pass;

if __name__=='__main__':
    exit(main());
    pass;
