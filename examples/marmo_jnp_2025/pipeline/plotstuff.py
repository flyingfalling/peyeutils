import sys;
import os;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import seaborn as sns;
from PIL import Image,ImageDraw, ImageFont;
from io import BytesIO;
import cv2;
import re;


def jointplot_to_image(cfg,img_size=4,xlim=None,ylim=None):

    print(cfg);

    g=sns.jointplot(data=cfg['data'],x=cfg['x'],y=cfg['y'],kind='hist',height=img_size,xlim=xlim,ylim=ylim);

    g.figure.suptitle(cfg['title']);
    g.ax_joint.set_aspect('equal');

    buf = BytesIO();
    g.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100);
    buf.seek(0);
    plt.close(g.figure);
    return Image.open(buf).copy();


def make_row_image(configs,img_size=4,xlim=None,ylim=None):

    #combine jointplots in a row
    images = [jointplot_to_image(cfg,img_size=img_size,xlim=xlim,ylim=ylim) for cfg in configs];
    
    row_width  = sum(img.width for img in images);
    row_height = max(img.height for img in images);
    row_img = Image.new("RGB", (row_width, row_height), "white");
    
    x_offset = 0;
    for img in images:
        row_img.paste(img, (x_offset, 0));
        x_offset += img.width;
    return row_img;


def add_title_to_row(row_img, title=None, fontsize=40,title_height=60):

    if title is None:
        return row_img
    
    new_img = Image.new("RGB", (row_img.width, row_img.height + title_height), "white")
    new_img.paste(row_img, (0, title_height))  
    
    draw = ImageDraw.Draw(new_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    
    # center the title
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    x = (row_img.width - text_width) // 2
    y = (title_height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), title, fill="black", font=font)
    
    return new_img


def marmohistplot(widxdf,oidxdf,wgazedf,ogazedf,iidxdf=None,igazedf=None):

    print('wajd unique subjects:{}'.format(widxdf['subj'].unique()));
    print('orig unique subjects:{}'.format(oidxdf['subj'].unique()));

    wsubjdf=pd.DataFrame([],columns=['subj','species','type']);

    wsubjdf['subj']=widxdf['subj'].unique();
    wsubjdf['species']='marmo';
    wsubjdf['type']='w';


    osubjdf=pd.DataFrame([],columns=['subj','species','type']);

    for osubj in oidxdf['subj'].unique():
        spec=oidxdf[oidxdf['subj']==osubj]['species'].unique().item();
        osubjdf.loc[len(osubjdf)]={'subj':osubj,'species':spec};
        pass;

    osubjdf['type']='o';

    subjdf=pd.concat([wsubjdf,osubjdf],axis=0).reset_index(drop=True);

    if (iidxdf is not None)&(igazedf is not None):

        isubjdf=pd.DataFrame([],columns=['subj','species','type']);
        isubjdf['subj']=iidxdf['subj'].unique();
        isubjdf['species']='infant';
        isubjdf['type']='i';

        subjdf=pd.concat([wsubjdf,osubjdf,isubjdf],axis=0).reset_index(drop=True);

        pass;    

    print(subjdf);

    #match species and plot rows
    plotrowdict={spec:i for i,spec in enumerate(subjdf['species'].unique())};

    #match species and number of subjects in each species
    nsubjdict={};
    for spec in plotrowdict.keys():
        nsubj=len(subjdf.loc[subjdf['species']==spec]);
        nsubjdict[spec]=nsubj;
        pass;

    plotcoldict=dict(zip(plotrowdict.keys(),[0]*len(plotrowdict)));

    fig=plt.figure(figsize=(12,16),constrained_layout=True);
    subfigs=fig.subfigures(nrows=len(plotrowdict),ncols=1);

    marmo_movie_settings=pd.DataFrame([{'species':'marmo','pix_x':640.,'pix_y':480.,'dva_x':24.1,'dva_y':17.6},
                             {'species':'macaq','pix_x':640.,'pix_y':480.,'dva_x':17.0,'dva_y':12.4},
                             {'species':'human','pix_x':640.,'pix_y':480.,'dva_x':21.7,'dva_y':15.8}]);


    for i,subfig in enumerate(subfigs):

        subfig.suptitle(list(plotrowdict.keys())[i], fontweight="bold");
        axs = subfig.subplots(1, list(nsubjdict.values())[i]); 

        if len(axs == 1):
            axs = [axs];
        
        pass;


    for _, row in subjdf.iterrows():

        if row['type']=='w':
            tidx=widxdf.loc[(widxdf['subj']==row['subj'])&(widxdf['Eyetracking_mode']=='Binocular'),'trialidx'].tolist();
            gazedf=wgazedf[wgazedf['trialidx'].isin(tidx)];


        elif row['type']=='o':
            #Few samples in orig data in binocular mode  

            #tidx=oidxdf[(oidxdf['subj']==row['subj'])&(oidxdf['binocularmode']=='Binocular')]['trialidx'].tolist();
            tidx=oidxdf.loc[oidxdf['subj']==row['subj'],'trialidx'].tolist();
            gazedf=ogazedf[ogazedf['trialidx'].isin(tidx)];
        
        elif row['type']=='i':
            tidx=iidxdf.loc[iidxdf['subj']==row['subj'],'trialidx'].tolist();
            gazedf=igazedf[igazedf['trialidx'].isin(tidx)];

        else:
            print('error in type');
            raise ValueError;
            
        axs=subfigs[plotrowdict[row['species']]].axes;
        ax=axs[plotcoldict[row['species']]];

        movie_setting=marmo_movie_settings[marmo_movie_settings['species']==row['species']];
        pix_deg_ratio_x=movie_setting['pix_x'].values/movie_setting['dva_x'].values;
        gazedf['modified_pix_x']=gazedf['dva_x']*pix_deg_ratio_x;

        pix_deg_ratio_y=movie_setting['pix_y'].values/movie_setting['dva_y'].values;
        gazedf['modified_pix_y']=gazedf['dva_y']*pix_deg_ratio_y;

        sns.histplot(data=gazedf,x='modified_pix_x',y='modified_pix_y',ax=ax);

        ax.set_xlim([-400,400]);
        ax.set_ylim([-400,400]);
        ax.set_title(row['subj']);
        ax.set_aspect('equal');

        plotcoldict[row['species']]+=1;

        pass;

    plt.savefig('histplot_persubj.png');

    return 0;


def marmojointplot(widxdf,wgazedf,oidxdf,ogazedf):

    
    print('wajd unique subjects:{}'.format(widxdf['subj'].unique()));
    print('orig unique subjects:{}'.format(oidxdf['subj'].unique()));

    wsubjdf=pd.DataFrame([],columns=['subj','species','wotype']);

    wsubjdf['subj']=widxdf['subj'].unique();
    wsubjdf['species']='marmo';
    wsubjdf['wotype']='w';


    osubjdf=pd.DataFrame([],columns=['subj','species','wotype']);

    for osubj in oidxdf['subj'].unique():
        spec=oidxdf[oidxdf['subj']==osubj]['species'].unique().item();
        osubjdf.loc[len(osubjdf)]={'subj':osubj,'species':spec};
        pass;

    osubjdf['wotype']='o';

    subjdf=pd.concat([wsubjdf,osubjdf],axis=0).reset_index(drop=True);

    subjdf['species']=subjdf['species'].replace('human','adult');

    subjdf['subj_maskname']=subjdf.groupby('species').cumcount()+1;
    subjdf['subj_maskname']=subjdf['species']+subjdf['subj_maskname'].astype(str);


    marmo_movie_settings=pd.DataFrame([{'species':'marmo','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':24.1,'vid_dva_y':17.6},
                             {'species':'macaq','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':17.0,'vid_dva_y':12.4},
                             {'species':'adult','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':21.7,'vid_dva_y':15.8}]);


    #match species and plot rows
    plotrowdict={spec:i for i,spec in enumerate(subjdf['species'].unique())};

    configs=list([] for _ in range(len(plotrowdict)));

    
    for _, row in subjdf.iterrows():

        if row['wotype']=='w':
            tidx=widxdf[(widxdf['subj']==row['subj'])&(widxdf['Eyetracking_mode']=='Binocular')]['trialidx'].tolist();
            gazedf=wgazedf[wgazedf['trialidx'].isin(tidx)];

        elif row['wotype']=='o':

            #Few samples in orig data in binocular mode  

            tidx=oidxdf[(oidxdf['subj']==row['subj'])&(oidxdf['binocularmode']=='Monocular')]['trialidx'].tolist();
            tidx=oidxdf[oidxdf['subj']==row['subj']]['trialidx'].tolist();
            gazedf=ogazedf[ogazedf['trialidx'].isin(tidx)];

        else:
            print('error in wotype');
            raise ValueError;

        gazedf['species']=row['species'];

        gazedf=gazedf.merge(marmo_movie_settings, on='species', how='left');

        gazedf['xcnorm01']=gazedf['dva_x']*2/gazedf['vid_dva_x'];
        gazedf['ycnorm01']=gazedf['dva_y']*2/gazedf['vid_dva_y'];

        gazedf.loc[gazedf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
        gazedf.loc[gazedf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

        subj_cfg={'data':gazedf,'x':'xcnorm01','y':'ycnorm01','title':row['subj_maskname']};
        configs[plotrowdict[row['species']]].append(subj_cfg);

        pass;

    """
    bigdf['subj']=bigdf.groupby('species').cumcount()+1;
    bigdf['subj']=bigdf['species']+bigdf['subj'].astype(str);

    for subj in bigdf['subj'].unique():
        
        sampdf=bigdf[bigdf['subj']==subj];

        subj_cfg={'data':sampdf,'x':'xcnorm01','y':'ycnorm01','title':subj};
        spec = re.sub(r'\d+$', '', subj);
        configs[plotrowdict[spec]].append(subj_cfg);
    """
     
    row_images = [];
    for i,row_cfg in enumerate(configs):
        row_img=make_row_image(row_cfg,xlim=(-1.25,1.25),ylim=(-1.25,1.25));
        row_img=add_title_to_row(row_img,list(plotrowdict.keys())[i]);
        row_images.append(row_img);


    max_width   = max(img.width for img in row_images);
    total_height = sum(img.height for img in row_images);

    final_image = Image.new("RGB", (max_width, total_height), "white");
    y_offset = 0;
    for row_img in row_images:
        final_image.paste(row_img, (0, y_offset));
        y_offset += row_img.height;

    final_image.save("jointplot_persubj_normed_fix.png", dpi=(100, 100));

    return 0;


def infantjointplot(trialdf,sampdf):

    trialdf = trialdf.rename(columns={'video': 'vid'}).copy()

    key=['subj', 'vid', 'agedays'];
    tidxmap = sampdf.drop_duplicates(key).set_index(key)['trialidx'];
    trialdf['trialidx'] = pd.MultiIndex.from_frame(trialdf[key]).map(tidxmap)
    trialdf = trialdf.dropna(subset=['trialidx'])


    binoc_tidx=trialdf.loc[trialdf['recinfo_eyes']=='LR','trialidx'].unique();

    sampdf=sampdf[sampdf['trialidx'].isin(binoc_tidx)].copy();

    #vidinfodf=trialdf[['trialidx','vidw_px', 'vidh_px', 'vidxpos_px', 'vidypos_px']].drop_duplicates('trialidx');
    #sampdf = sampdf.merge(vidinfodf, on='trialidx', how='left', validate='many_to_one');

    #grouping by subject ages
    sampdf['agemonths']=np.round(sampdf['agedays']/30);
    condlist=[sampdf['agemonths']<=12,sampdf['agemonths']<=30,sampdf['agemonths']>30];
    agegrouplist=['infant','toddler','preschooler']

    sampdf['agegroup']=np.select(condlist,agegrouplist,default=np.nan);
    sampdf.dropna(subset=['agegroup']);

    vidwpx=640;
    vidhpx=480;

    #normalize gaze positions into -1 to 1
    sampdf['xcnorm01']=sampdf['pix_x']/(vidwpx/2);
    sampdf['ycnorm01']=sampdf['pix_y']/(vidhpx/2);

    sampdf.loc[sampdf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
    sampdf.loc[sampdf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

    sampdf=sampdf.drop(['pix_x','pix_y'],axis=1);

    #sampdf=sampdf.melt(id_vars=['trialidx','movie_ts','vid', 'subj', 'agedays'],value_vars=['xcnorm01','ycnorm01'],var_name='xy',value_name='xyvalue');

    #configs=list([] for _ in range(1));

    configs=[];

    for i,agegroup in enumerate(sampdf['agegroup'].unique()):

        samps=sampdf[sampdf['agegroup']==agegroup];

        age_cfg={'data':samps,'x':'xcnorm01','y':'ycnorm01','title':agegroup};

        """
        if i<(len(sampdf['subj'].unique())//2+1):
            configs[0].append(subj_cfg);
        
        else:
            configs[1].append(subj_cfg);
        """

        configs.append(age_cfg);

        pass;

    row_images = [];
    for row_cfg in configs:
        print(row_cfg)
        row_img=make_row_image(row_cfg,xlim=(-1.25,1.25),ylim=(-1.25,1.25));
        row_img=add_title_to_row(row_img);
        row_images.append(row_img);


    max_width   = max(img.width for img in row_images);
    total_height = sum(img.height for img in row_images);

    final_image = Image.new("RGB", (max_width, total_height), "white");
    y_offset = 0;
    for row_img in row_images:
        final_image.paste(row_img, (0, y_offset));
        y_offset += row_img.height;

    final_image.save("infant_agegroup_gazeplot.png", dpi=(100, 100));

    return 0;


def load_samps(fnames):

    if not fnames:
        sampdf=None;
        pass;

    for i,fname in enumerate(fnames):

        if not os.path.isfile(fname):
            print(f'file directory was not valid:{fname}');
            continue;

        if i==0:
            sampdf=pd.read_csv(fname);
            sampdf=sampdf[sampdf['eye']==8];
            sampdf['ycnorm01']*=(-1);
            pass;

        else:
            trial_sampdf=pd.read_csv(fname);
            trial_sampdf=trial_sampdf[trial_sampdf['eye']==8];
            trial_sampdf['ycnorm01']*=(-1);
            sampdf=pd.concat([sampdf,trial_sampdf],axis=0).reset_index(drop=True);

        pass;

    return sampdf;


def bigsmall_jointplot(trialdf,sampsdir):

    if os.path.isdir(sampsdir)==False:
        print('Gaze sample directory was not valid');
        raise ValueError;

    #remove subjects who do not watch videos under big/small conditions
    trialdf = trialdf[trialdf.groupby("subj")["vidsizedva"].transform("nunique") == 2];

    subjlist=trialdf['subj'].unique();
    print(f'unique subjects:{subjlist}');

    conditions={'big':0,'small':1};

    configs=list([] for _ in range(len(conditions)));

    trialdf['vidsizedva']=trialdf['vidsizedva'].replace(25.,'big');
    trialdf['vidsizedva']=trialdf['vidsizedva'].replace(10.,'small');

    trialdf['fnames']=sampsdir+'/'+'trial'+trialdf['trialpk'].astype(str)+'_'+trialdf['vid']+'_normedsamples.csv';

    for vidsize in list(conditions.keys()):

        for subj in subjlist:

            tdf=trialdf[(trialdf['subj']==subj)&(trialdf['vidsizedva']==vidsize)];
            sampdf=load_samps(tdf['fnames'].tolist());

            if sampdf is None:
                continue;

            subj_cfg={'data':sampdf,'x':'xcnorm01','y':'ycnorm01','title':subj};
            configs[conditions[vidsize]].append(subj_cfg);

            pass;

        pass;


    row_images = [];
    for i,row_cfg in enumerate(configs):
        row_img=make_row_image(row_cfg);
        row_img=add_title_to_row(row_img,list(conditions.keys())[i]);
        row_images.append(row_img);

    max_width   = max(img.width for img in row_images);
    total_height = sum(img.height for img in row_images);

    final_image = Image.new("RGB", (max_width, total_height), "white");
    y_offset = 0;

    for row_img in row_images:
        final_image.paste(row_img, (0, y_offset));
        y_offset += row_img.height;

    final_image.save("bigsmall_jointplot_persubj.png", dpi=(100, 100));

    return 0;


def bigsmall_gazetimeplot(trialdf,sampsdir,vid):

    if os.path.isdir(sampsdir)==False:
        print('Gaze sample directory was not valid');
        raise ValueError;


    tdf=trialdf[(trialdf['vid']==vid)];

    tdf['vidsizedva']=tdf['vidsizedva'].replace(25.,'big');
    tdf['vidsizedva']=tdf['vidsizedva'].replace(10.,'small');

    conditions=list(tdf['vidsizedva'].unique());

    #remove subjects who do not watch videos under both conditions
    tdf = tdf[tdf.groupby("subj")["vidsizedva"].transform("nunique") == 2]

    subj_list=list(tdf['subj'].unique());
    print('unique subjects:{}'.format(subj_list));

    tdf['fnames']=sampsdir+'/'+'trial'+tdf['trialpk'].astype(str)+'_'+tdf['vid']+'_normedsamples.csv';

    samples=None;

    for vidsize in conditions:

        for i,subj in enumerate(subj_list):

            trials=tdf[(tdf['vidsizedva']==vidsize)&(tdf['subj']==subj)&(tdf['iscontrol']==False)];
            
            if len(trials['trialpk'].unique())!=1:
                print('several trials are included in the same condition');
                print('trialpk:{}'.format(trials['trialpk'].unique()));
                trials=trials[trials['trialpk']==list(trials['trialpk'].unique())[0]];
            
            sampdf=pd.read_csv(trials['fnames'].item());

            if sampdf is None:
                continue;
            
            sampdf=sampdf[sampdf['eye']==8];
            sampdf['ycnorm01']*=(-1);

            sampdf['subj']=subj;
            sampdf['vidsizedva']=vidsize;

            if samples is None:
                samples=sampdf;
            
            else:
                samples=pd.concat([samples,sampdf],axis=0).reset_index(drop=True);
        
            pass;

        pass;

    fig,axs=plt.subplots(nrows=2,ncols=2);

    sns.scatterplot(data=samples[samples['vidsizedva']=='big'],x='tsec',y='xcnorm01',hue='subj',ax=axs[0,0]);
    sns.scatterplot(data=samples[samples['vidsizedva']=='big'],x='tsec',y='ycnorm01',hue='subj',ax=axs[1,0]);
    sns.scatterplot(data=samples[samples['vidsizedva']=='small'],x='tsec',y='xcnorm01',hue='subj',ax=axs[0,1]);
    sns.scatterplot(data=samples[samples['vidsizedva']=='small'],x='tsec',y='ycnorm01',hue='subj',ax=axs[1,1]);

    axs[0,0].set_title('big');
    axs[0,1].set_title('small');

    plt.tight_layout();
    plt.savefig('gazetimeplot_{}.png'.format(vid),bbox_inches='tight');

    return 0;


def infant_gazetimeplot(trialdf,sampdf,vid):

    trialdf = trialdf.rename(columns={'video': 'vid'}).copy()

    trialdf=trialdf[trialdf['vid']==vid].copy();
    sampdf=sampdf[sampdf['vid']==vid].copy();

    key=['subj', 'vid', 'agedays'];
    tidxmap = sampdf.drop_duplicates(key).set_index(key)['trialidx'];
    trialdf['trialidx'] = pd.MultiIndex.from_frame(trialdf[key]).map(tidxmap)
    trialdf = trialdf.dropna(subset=['trialidx'])


    binoc_tidx=trialdf.loc[trialdf['recinfo_eyes']=='LR','trialidx'].unique();

    sampdf=sampdf[sampdf['trialidx'].isin(binoc_tidx)].copy();

    vidinfodf=trialdf[['trialidx','vidw_px', 'vidh_px', 'vidxpos_px', 'vidypos_px']].drop_duplicates('trialidx');
    sampdf = sampdf.merge(vidinfodf, on='trialidx', how='left', validate='many_to_one');

    #grouping by subject ages
    sampdf['agemonths']=np.round(sampdf['agedays']/30);
    condlist=[sampdf['agemonths']<=12,sampdf['agemonths']<=30,sampdf['agemonths']>30];
    agegrouplist=['infant','toddler','preschooler']

    sampdf['agegroup']=np.select(condlist,agegrouplist,default=np.nan);
    sampdf.dropna(subset=['agegroup']);

    #normalize gaze positions into -1 to 1
    sampdf['xcnorm01']=(sampdf['pix_x']-sampdf['vidxpos_px'])/(sampdf['vidw_px']/2);
    sampdf['ycnorm01']=(sampdf['pix_y']-sampdf['vidypos_px'])/(sampdf['vidh_px']/2);

    sampdf.loc[sampdf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
    sampdf.loc[sampdf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

    sampdf=sampdf.drop(['vidw_px', 'vidh_px', 'vidxpos_px', 'vidypos_px','pix_x','pix_y'],axis=1);

    subjmap={subj:'subj'+str(i+1) for i,subj in enumerate(sampdf['subj'].unique())};
    sampdf['subj_age']=sampdf['subj'].map(subjmap)+'_'+sampdf['agegroup'];

    print(sampdf['subj_age'].unique());

    fig,axs=plt.subplots(nrows=2,ncols=1);

    sns.scatterplot(data=sampdf,x='movie_ts',y='xcnorm01',hue='subj_age',s=5,ax=axs[0],legend=False);
    sns.scatterplot(data=sampdf,x='movie_ts',y='ycnorm01',hue='subj_age',s=5,ax=axs[1],legend='full');
    
    axs[0].set_title('gaze x position');
    axs[1].set_title('gaze y position');

    #axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0);
    #axs[1].legend(bbox_to_anchor=(1.05, 1), loc="lower right", borderaxespad=10);
    sns.move_legend(axs[1], loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout();
    plt.savefig('legend_infant_gazetimeplot_{}.png'.format(vid),bbox_inches='tight');

    return 0;


def make_marmo_bigdf(widxdf,wgazedf,oidxdf,ogazedf,savebigdf=True):

    winfomap = widxdf.drop_duplicates('trialidx').set_index('trialidx')[['Eyetracking_mode','subj','vid']];
    wgazedf = wgazedf.merge(winfomap, on='trialidx', how='left');
    wgazedf['species'] = 'marmo';

    #should also extract monocular data
    wgazedf=wgazedf[wgazedf['Eyetracking_mode']=='Binocular'].copy();

    wgazedf=wgazedf[['trialidx','subj','vid','species','timems','pix_x','pix_y','dva_x','dva_y']].copy();


    oinfomap = oidxdf.drop_duplicates('trialidx').set_index('trialidx')[['binocularmode','subj','vid','species']];
    ogazedf = ogazedf.merge(oinfomap, on='trialidx', how='left');
    
    ogazedf=ogazedf[ogazedf['binocularmode']=='Monocular'].copy();

    ogazedf=ogazedf[['trialidx','subj','vid','species','timems','pix_x','pix_y','dva_x','dva_y']].copy();

    bigdf=pd.concat([wgazedf,ogazedf],axis=0).reset_index(drop=True);

    bigdf['vid'] = bigdf['vid'].str.replace(r'\.(mp4|m4v)$', '', regex=True);

    marmo_movie_settings=pd.DataFrame([{'species':'marmo','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':24.1,'vid_dva_y':17.6},
                             {'species':'macaq','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':17.0,'vid_dva_y':12.4},
                             {'species':'human','vid_pix_x':640.,'vid_pix_y':480.,'vid_dva_x':21.7,'vid_dva_y':15.8}]);

    bigdf=bigdf.merge(marmo_movie_settings, on='species', how='left');


    bigdf['xcnorm01']=bigdf['dva_x']*2/bigdf['vid_dva_x'];
    bigdf['ycnorm01']=bigdf['dva_y']*2/bigdf['vid_dva_y'];

    bigdf.loc[bigdf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
    bigdf.loc[bigdf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

    bigdf=bigdf.melt(id_vars=['trialidx','subj','vid','species','timems']
                       ,value_vars=['xcnorm01','ycnorm01'],var_name='xy',value_name='xyvalue');
    
    if savebigdf:
        bigdf.to_csv('marmo_biggazedf.csv');

    return bigdf;


def marmo_gazetimeplot(bigdf,vid):

    bigdf=bigdf[bigdf['vid']==vid].copy();
    bigdf['subj']=bigdf.groupby('species').cumcount()+1;
    bigdf['subj']=bigdf['species']+bigdf['subj'].astype(str);

    bigdf['time']=bigdf['timems']/1000;

    xbigdf=bigdf[bigdf['xy']=='xcnorm01'].copy();
    ybigdf=bigdf[bigdf['xy']=='ycnorm01'].copy();

    fig,axs=plt.subplots(nrows=2,ncols=1);


    sns.scatterplot(data=xbigdf,x='time',y='xyvalue',hue='subj',s=5,ax=axs[0],legend=False);
    sns.scatterplot(data=ybigdf,x='time',y='xyvalue',hue='subj',s=5,ax=axs[1],legend='full');
    
    axs[0].set_title('gaze x position');
    axs[1].set_title('gaze y position');

    sns.move_legend(axs[1], loc='lower left', bbox_to_anchor=(1, 1))

    plt.tight_layout();
    plt.savefig('marmo_gazetimeplot_{}.png'.format(vid),bbox_inches='tight');

    return 0;


def process_samps(sampdf):

    sampdf=sampdf[sampdf['eye']==8];
    sampdf=sampdf[['pk','tsec','xcnorm01','ycnorm01']];
    sampdf['ycnorm01']*=(-1);
    
    sampdf[sampdf['xcnorm01'].abs()>1.25]=np.nan;
    sampdf[sampdf['ycnorm01'].abs()>1.25]=np.nan;

    sampdf=sampdf.melt(id_vars=['pk','tsec'],value_vars=['xcnorm01','ycnorm01'],var_name='xy',value_name='xyvalue');

    return sampdf;
    

def load_processed_samps(fnames):

    if not fnames:
        sampdf=None;
        pass;

    for i,fname in enumerate(fnames):

        if not os.path.isfile(fname):
            print(f'file directory was not valid:{fname}');
            continue;

        if i==0:
            sampdf=pd.read_csv(fname);
            sampdf=process_samps(sampdf);
            pass;

        else:
            trial_sampdf=pd.read_csv(fname);
            trial_sampdf=process_samps(trial_sampdf);

            sampdf=pd.concat([sampdf,trial_sampdf],axis=0).reset_index(drop=True);

        pass;

    return sampdf;


def bigsmall_catplot(trialdf,sampsdir):

    if os.path.isdir(sampsdir)==False:
        print('Gaze sample directory was not valid');
        raise ValueError;

    #remove subjects who do not watch videos under big/small conditions
    trialdf = trialdf[trialdf.groupby("subj")["vidsizedva"].transform("nunique") == 2]

    subjlist=list(trialdf['subj'].unique());
    print(f'unique subjects:{subjlist}');


    trialdf['vidsizedva']=trialdf['vidsizedva'].replace(25.,'big');
    trialdf['vidsizedva']=trialdf['vidsizedva'].replace(10.,'small');

    trialdf['fnames']=sampsdir+'/'+'trial'+trialdf['trialpk'].astype(str)+'_'+trialdf['vid']+'_normedsamples.csv';

    vidsizes=list(trialdf['vidsizedva'].unique());

    bigdf=None;

    for vidsize in vidsizes:

        for subj in subjlist:

            tdf=trialdf[(trialdf['subj']==subj)&(trialdf['vidsizedva']==vidsize)];
            sampdf=load_processed_samps(tdf['fnames'].tolist());

            if sampdf is None:
                continue;
            
            sampdf['subj']=subj;
            sampdf['vidsize']=vidsize;
            
            if bigdf is None:
                bigdf=sampdf;
            
            else:
                bigdf=pd.concat([bigdf,sampdf],axis=0).reset_index(drop=True);

            pass;

        pass;


    bigdf.to_csv('bigsmall_bigsampledf.csv');

    sns.catplot(data=bigdf,x='xyvalue',y='xy',hue='vidsize',kind='violin',errorbar=('ci',95));
    plt.savefig('distribution_catplot_bigsmall_pervidsize.png',bbox_inches="tight");
    plt.close();

    sns.catplot(data=bigdf,x='xyvalue',y='xy',hue='subj',kind='violin',errorbar=('ci',95));
    plt.savefig('distribution_catplot_bigsmall_persubj.png',bbox_inches="tight");
    plt.close();

    return 0;


def bigsmall_ecdfplot(trialdf,sampsdir):

    if os.path.isdir(sampsdir)==False:
        print('Gaze sample directory was not valid');
        raise ValueError;

    #remove subjects who do not watch videos under big/small conditions
    trialdf = trialdf[trialdf.groupby("subj")["vidsizedva"].transform("nunique") == 2]

    subjlist=list(trialdf['subj'].unique());
    print(f'unique subjects:{subjlist}');


    trialdf['vidsizedva']=trialdf['vidsizedva'].replace(25.,'big');
    trialdf['vidsizedva']=trialdf['vidsizedva'].replace(10.,'small');

    trialdf['fnames']=sampsdir+'/'+'trial'+trialdf['trialpk'].astype(str)+'_'+trialdf['vid']+'_normedsamples.csv';

    vidsizes=list(trialdf['vidsizedva'].unique());

    bigdf=None;

    for vidsize in vidsizes:

        for subj in subjlist:

            tdf=trialdf[(trialdf['subj']==subj)&(trialdf['vidsizedva']==vidsize)];
            sampdf=load_processed_samps(tdf['fnames'].tolist());

            if sampdf is None:
                continue;
            
            sampdf['subj']=subj;
            sampdf['vidsize']=vidsize;
            
            if bigdf is None:
                bigdf=sampdf;
            
            else:
                bigdf=pd.concat([bigdf,sampdf],axis=0).reset_index(drop=True);

            pass;

        pass;


    bigdf.to_csv('bigsmall_bigsampledf.csv');

    """
    fig,axs=plt.subplots(nrows=2,ncols=1)

    sns.kdeplot(data=bigdf[bigdf['xy']=='xcnorm01'],x='xyvalue',hue='vidsize',fill=False,ax=axs[0]);
    sns.kdeplot(data=bigdf[bigdf['xy']=='ycnorm01'],x='xyvalue',hue='vidsize',fill=False,ax=axs[1]);



    axs[0].set_title('x position');
    axs[1].set_title('y position');

    plt.savefig('kdedistributionplot_bigsmall_pervidsize.png',bbox_inches="tight");
    plt.close();


    fig,axs=plt.subplots(nrows=2,ncols=1)

    sns.kdeplot(data=bigdf[bigdf['xy']=='xcnorm01'],x='xyvalue',hue='subj',fill=False,ax=axs[0]);
    sns.kdeplot(data=bigdf[bigdf['xy']=='ycnorm01'],x='xyvalue',hue='subj',fill=False,ax=axs[1]);

    plt.savefig('kdedistributionplot_bigsmall_persubj.png',bbox_inches="tight");
    plt.close();
    """

    sns.displot(data=bigdf,x='xyvalue',y='xy',hue='vidsize',kind='ecdf');
    plt.savefig('distribution_ecdfplot_bigsmall_pervidsize.png',bbox_inches="tight");
    plt.close();

    sns.catplot(data=bigdf,x='xyvalue',y='xy',hue='subj',kind='ecdf');
    plt.savefig('distribution_ecdfplot_bigsmall_persubj.png',bbox_inches="tight");
    plt.close();

    return 0;


def make_infant_bigdf(trialdf,sampdf):

    trialdf = trialdf.rename(columns={'video': 'vid'}).copy()

    key=['subj', 'vid', 'agedays'];
    tidxmap = sampdf.drop_duplicates(key).set_index(key)['trialidx'];
    trialdf['trialidx'] = pd.MultiIndex.from_frame(trialdf[key]).map(tidxmap)
    trialdf = trialdf.dropna(subset=['trialidx'])


    binoc_tidx=trialdf.loc[trialdf['recinfo_eyes']=='LR','trialidx'].unique();

    sampdf=sampdf[sampdf['trialidx'].isin(binoc_tidx)].copy();

    #vidinfodf=trialdf[['trialidx','vidw_px', 'vidh_px', 'vidxpos_px', 'vidypos_px']].drop_duplicates('trialidx');
    #sampdf = sampdf.merge(vidinfodf, on='trialidx', how='left', validate='many_to_one');

    #grouping by subject ages
    sampdf['agemonths']=np.round(sampdf['agedays']/30);
    condlist=[sampdf['agemonths']<=12,sampdf['agemonths']<=30,sampdf['agemonths']>30];
    agegrouplist=['infant','toddler','preschooler']

    sampdf['agegroup']=np.select(condlist,agegrouplist,default=np.nan);
    sampdf=sampdf.dropna(subset=['agegroup']);

    sampdf['vid'] = sampdf['vid'].str.replace(r'\.(mp4|m4v)$', '', regex=True)

    vidwpx=640;
    vidhpx=480;

    #normalize gaze positions into -1 to 1
    sampdf['xcnorm01']=sampdf['pix_x']/(vidwpx/2);
    sampdf['ycnorm01']=sampdf['pix_y']/(vidhpx/2);

    sampdf.loc[sampdf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
    sampdf.loc[sampdf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

    sampdf=sampdf.drop(['pix_x','pix_y'],axis=1);

    sampdf=sampdf.melt(id_vars=['trialidx','movie_ts','vid', 'subj', 'agegroup'],
                       value_vars=['xcnorm01','ycnorm01'],var_name='xy',value_name='xyvalue');

    sampdf.to_csv('infant_biggazedf.csv');

    return 0;

def infant_catplot(trialdf,sampdf):

    trialdf = trialdf.rename(columns={'video': 'vid'}).copy()

    key=['subj', 'vid', 'agedays'];
    tidxmap = sampdf.drop_duplicates(key).set_index(key)['trialidx'];
    trialdf['trialidx'] = pd.MultiIndex.from_frame(trialdf[key]).map(tidxmap)
    trialdf = trialdf.dropna(subset=['trialidx'])


    binoc_tidx=trialdf.loc[trialdf['recinfo_eyes']=='LR','trialidx'].unique();

    sampdf=sampdf[sampdf['trialidx'].isin(binoc_tidx)].copy();

    #vidinfodf=trialdf[['trialidx','vidw_px', 'vidh_px', 'vidxpos_px', 'vidypos_px']].drop_duplicates('trialidx');
    #sampdf = sampdf.merge(vidinfodf, on='trialidx', how='left', validate='many_to_one');

    #grouping by subject ages
    sampdf['agemonths']=np.round(sampdf['agedays']/30);
    condlist=[sampdf['agemonths']<=12,sampdf['agemonths']<=30,sampdf['agemonths']>30];
    agegrouplist=['infant','toddler','preschooler']

    sampdf['agegroup']=np.select(condlist,agegrouplist,default=np.nan);
    sampdf.dropna(subset=['agegroup']);

    vidwpx=640;
    vidhpx=480;

    #normalize gaze positions into -1 to 1
    #sampdf['xcnorm01']=(sampdf['pix_x']-sampdf['vidxpos_px'])/(sampdf['vidw_px']/2);
    sampdf['xcnorm01']=sampdf['pix_x']/(vidwpx/2);
    #sampdf['ycnorm01']=(sampdf['pix_y']-sampdf['vidypos_px'])/(sampdf['vidh_px']/2);
    sampdf['ycnorm01']=sampdf['pix_y']/(vidhpx/2);

    sampdf.loc[sampdf['xcnorm01'].abs()>1.25,'xcnorm01']=np.nan;
    sampdf.loc[sampdf['ycnorm01'].abs()>1.25,'ycnorm01']=np.nan;

    sampdf=sampdf.drop(['pix_x','pix_y'],axis=1);

    sampdf=sampdf.melt(id_vars=['trialidx','movie_ts','vid', 'subj', 'agegroup'],
                       value_vars=['xcnorm01','ycnorm01'],var_name='xy',value_name='xyvalue');
    
    print(sampdf.groupby('xy')['xyvalue'].min());
    print(sampdf.groupby('xy')['xyvalue'].max());

    sns.catplot(data=sampdf,x='xyvalue',y='xy',hue='agegroup',hue_order=['infant','toddler','preschooler'],kind='violin',errorbar=('ci',95));
    plt.savefig('infant_catplot.png',bbox_inches="tight");
    plt.close();

    return 0;


def infant_corrdist_heatmap(corrdistdf):

    
    #extract identical subjects
    #corrdistdf=corrdistdf[corrdistdf['subj1']==corrdistdf['subj2']].copy();

    #have to sort index and columns
    
    #agegrouplist=['infant','toddler','preschooler']

    #corrdistdf['agegroup1'] = pd.Categorical(corrdistdf['agegroup1'], categories=agegrouplist, ordered=True);
    #corrdistdf['agegroup2'] = pd.Categorical(corrdistdf['agegroup2'], categories=agegrouplist, ordered=True);
    
    #corrdf=corrdistdf.pivot_table(index='agegroup1',columns='agegroup2',values='corr_xy');
    #distdf=corrdistdf.pivot_table(index='agegroup1',columns='agegroup2',values='rms');

    corrdistdf['species1']=corrdistdf['species1'].replace('human','adult');
    corrdistdf['species2']=corrdistdf['species2'].replace('human','adult');

    corrdistdf=corrdistdf.dropna(subset=['species1','species2']);

    specieslist=['marmo','macaq','adult','infant','toddler','preschooler'];

    corrdistdf['species1'] = pd.Categorical(corrdistdf['species1'], categories=specieslist, ordered=True);
    corrdistdf['species2'] = pd.Categorical(corrdistdf['species2'], categories=specieslist, ordered=True);

    print(corrdistdf.groupby(['species1','species2'], observed=True)[['corr_xy','rms']]
        .apply(lambda g: g.notna().sum()));

   
    corrdf=corrdistdf.pivot_table(index='species1',columns='species2',values='corr_xy');
    
    distdf=corrdistdf.pivot_table(index='species1',columns='species2',values='rms');
    

    #take the mean of each pair of agegroup1 and agegroup2 if they are not the same
    arr = corrdf.values.astype(float);
    mask = ~np.eye(len(arr), dtype=bool);

    sym_arr = arr.copy();
    sym_arr[mask] = np.nanmean([arr[mask], arr.T[mask]], axis=0);

    corrdf = pd.DataFrame(sym_arr, index=corrdf.index, columns=corrdf.columns);

    arr = distdf.values.astype(float);
    mask = ~np.eye(len(arr), dtype=bool);

    sym_arr = arr.copy();
    sym_arr[mask] = np.nanmean([arr[mask], arr.T[mask]], axis=0);

    distdf = pd.DataFrame(sym_arr, index=distdf.index, columns=distdf.columns);


    fig,axs=plt.subplots(nrows=1,ncols=2);

    sns.heatmap(corrdf,annot=True, annot_kws={"size": 8},fmt=".2f",cmap='coolwarm',square=True,ax=axs[0]);
    sns.heatmap(distdf,annot=True, annot_kws={"size": 8},fmt=".2f",cmap='coolwarm',square=True,ax=axs[1]);

    axs[0].set_title('Correlation');
    axs[1].set_title('Distance');

    plt.tight_layout();
    plt.savefig('marmo_infant_corrdist_heatmap.png',bbox_inches="tight");
    plt.close();

    return 0;


def marmo_infant_catplot(marmo_bigdf,infant_bigdf):

    marmo_bigdf['timems']/=1000;

    infant_bigdf=infant_bigdf.rename(columns={'agegroup':'species'});
    marmo_bigdf=marmo_bigdf.rename(columns={'timems':'movie_ts'});

    bigdf=pd.concat([marmo_bigdf,infant_bigdf],axis=0);

    bigdf['species']=bigdf['species'].replace('human','adult');

    sns.catplot(data=bigdf,x='xyvalue',y='xy',hue='species',
                hue_order=['marmo','macaq','adult','infant','toddler','preschooler'],
                kind='violin',errorbar=('ci',95));
    plt.savefig('marmo_infant_catplot_fix.png',bbox_inches="tight");
    plt.close();

    return 0;


def load_samps_gazedraw(fname):

    df = pd.read_csv(fname);
    df = df[df['eye']==8];
    df = df.dropna(subset=['tsec', 'xcnorm01', 'ycnorm01']);
    df = df.sort_values('tsec').reset_index(drop=True);
    df['ycnorm01'] *= -1;

    return df;


def draw_gaze(frame, df, t0, t1, color,width,height,radiuspx = 30,edgethicknesspx = 5,NEYESAMPS=None,MINWEIGHT=0):

    samps = df[(df['tsec'] >= t0) & (df['tsec'] < t1)];
    if len(samps) == 0:
        return frame;

    if NEYESAMPS is not None:
        samps = samps.iloc[-NEYESAMPS:];

    weights = ((samps['tsec'] - t0) / (t1 - t0)).clip(MINWEIGHT, 1.0).to_numpy();
    xs = (samps['xcnorm01'] * (width//2) + width//2).to_numpy();
    ys = (samps['ycnorm01'] * (height//2) + height//2).to_numpy();

    for x, y, w in zip(xs, ys, weights):
        overlay = frame.copy();
        cv2.circle(overlay, (int(round(x)), int(round(y))), radiuspx, color, edgethicknesspx);
        frame = cv2.addWeighted(frame, 1.0 - w, overlay, w, 0);
    
    return frame;


def gazedraw_videos(trialdf,sampsdir,viddir,tpk1,tpk2,maxframe=None):

    vid1 = trialdf.loc[trialdf['trialpk'] == tpk1, 'vid'].item();
    vid2 = trialdf.loc[trialdf['trialpk'] == tpk2, 'vid'].item();

    if vid1 != vid2:
        raise ValueError(f'Subjects did not watch the same video in these trials: {vid1}, {vid2}');
    vid = vid1;

    vidname = os.path.join(viddir, vid);
    cap = cv2.VideoCapture(vidname);

    if not cap.isOpened():
        raise IOError(f'Error in opening the input file [{vidname}]');

    fps = float(cap.get(cv2.CAP_PROP_FPS));
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
    print(f'video fps:{fps}, size:{width}x{height}');

    sampfname1 = os.path.join(sampsdir, f'trial{tpk1}_{vid}_normedsamples.csv');
    sampfname2 = os.path.join(sampsdir, f'trial{tpk2}_{vid}_normedsamples.csv');

    for f in (sampfname1, sampfname2):
        if not os.path.isfile(f):
            raise FileNotFoundError(f'Not valid path: {f}');


    sampdf1 = load_samps_gazedraw(sampfname1);
    sampdf2 = load_samps_gazedraw(sampfname2);

    #max samples in a frame
    NEYESAMPS = 10;
    starttime_sec = 0.0;

    # for drawing
    radiuspx = 30;
    color1 = (30, 30, 255);
    color2 = (255, 30, 30);
    edgethicknesspx = 5;
    MINWEIGHT = 0.1;


    output_fname = f'eyetrackingvideo_{vid}_{tpk1}_{tpk2}.mp4';
    fourcc = cv2.VideoWriter_fourcc(*'mp4v');
    writer = cv2.VideoWriter(output_fname, fourcc, fps, (width, height), True);
    if not writer.isOpened():
        raise IOError(f'Cannot open output file [{output_fname}]');

    frame_idx = 0;
    while maxframe is None or frame_idx < maxframe:
        ret, frame = cap.read();
        if not ret:
            break;

        t0 = starttime_sec + frame_idx / fps;
        t1 = starttime_sec + (frame_idx + 1) / fps;

        frame = draw_gaze(frame, sampdf1, t0, t1, color1,width=width,height=height,\
                          radiuspx=radiuspx,edgethicknesspx=edgethicknesspx,NEYESAMPS=NEYESAMPS,MINWEIGHT=MINWEIGHT);
        frame = draw_gaze(frame, sampdf2, t0, t1, color2,width=width,height=height,\
                          radiuspx=radiuspx,edgethicknesspx=edgethicknesspx,NEYESAMPS=NEYESAMPS,MINWEIGHT=MINWEIGHT);

        writer.write(frame);
        frame_idx += 1;
        if frame_idx % 100 == 0:
            print(f'processed frames:{frame_idx} ({t1:.2f}s)');

    cap.release();
    writer.release();
    print(f'done: {frame_idx} frames -> {output_fname}');

    return 0;


def main():

    #widxdf=pd.read_csv(sys.argv[1]);
    #wgazedf=pd.read_csv(sys.argv[2]);
    #oidxdf=pd.read_csv(sys.argv[3]); 
    #ogazedf=pd.read_csv(sys.argv[4]);
    
    #make_marmo_bigdf(widxdf,wgazedf,oidxdf,ogazedf);
    

    #trialdf=pd.read_csv(sys.argv[1]);
    #sampdf=pd.read_csv(sys.argv[2]);

    #make_infant_bigdf(trialdf,sampdf);

    #marmo_bigdf=pd.read_csv(sys.argv[1]);
    #infant_bigdf=pd.read_csv(sys.argv[2]);
    #marmo_gazetimeplot(marmo_bigdf,'beverly07');
    #marmojointplot(widxdf,wgazedf,oidxdf,ogazedf);
    #marmo_infant_catplot(marmo_bigdf,infant_bigdf);

    corrdistdf=pd.read_csv(sys.argv[1]);
    infant_corrdist_heatmap(corrdistdf);

    sys.exit(0);


if __name__=='__main__':
    main();
    pass;
