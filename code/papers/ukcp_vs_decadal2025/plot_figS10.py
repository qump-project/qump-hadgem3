import os
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import datetime

import iris

import qdcUtils as utils

from pathnames_v1 import *

############################################################
namefig= 'figS10'
ifig   = 1010

dpiarr=[150]

saveplot   = True
#saveplot   = False

detrend=True
detrend=False

percentiles = [10, 50, 90]

vararr=['AMV', 'NAO']

nresamp=4000

#stype='C'
stype='U'

nlump=1

#central = 'mean'
central = 'median'

cwil = '31-20-1' 

TTT = 'T1'


obscol='black'
obslw='2.0'

ukcpcol='blue'
ukcplw='0.75'
ukcplwmed='2.0'

dpscol='red'
dpslw='1.0'
dpslwmed='2.0'

fsleg=8.0
legframe=True
fmtleg = '%4.2f'   #'%5.3f' 


##################################################   
fig = plt.figure(ifig, figsize=(20/2.54, 9.5/2.54))     

matplotlib.rcParams.update({'font.size': 10.0})

plt.subplots_adjust(top=0.92, bottom=0.09, left=0.095, right=0.985, hspace=0.25,wspace=0.25)        

for ivar,var in enumerate(vararr):
    ax = plt.subplot(1, 2, ivar+1)
    if var == 'AMV':
        obslab    = 'Observations'
        ylab      = 'Anomaly ($\degree$C)'      
        tit       = 'T1 Annual AMV Anomaly (1971-2000 baseline)'
        tit       = 'T1 annual AMV'
        vsr       = 'tas_ann_amv'      
        ssn       = 'djfmamjjason'
        ylimset   = [-0.35, 0.65]
        yrbeg    =  1961.0417  
        yrend    =  2018.4167              
        name_dps  = 'T+1yr_None_amo_djfmamjjason_lastyr=2018.nc'
        name_obs  = 'hadcrut5_latlong_T+1yr_None_amo_djfmamjjason.nc'        
        name_ukcp = 'ukcp09_T+1yr_amo_air_temperature_djfmamjjason_ukwil=31-20-1.nc'
        name_score= 'fig6_scores_amo_djfmamjjason_None_T+1yr_nresamp='+str(nresamp)+'_lastyr=2018_ukwil='+cwil+'.txt'

    elif var == 'NAO':
        obslab    = 'Observations'
        ylab      = 'Anomaly (hPa)' 
        tit       = 'T1 DJF NAO Anomaly (1971-2000 baseline)'
        tit       = 'T1 winter NAO'        
        vsr       = 'psl_djf_nao' 
        ssn       = 'djf'
        ylimset   = [-13.0, 14.7]
        yrbeg    =  1961.0417        
        yrend    =  2018.0417         
        name_dps  = 'T+1yr_None_nao_stephenson_'+ssn+'_lastyr=2018.nc'                
        name_obs  = 'era5_T+1yr_None_nao_stephenson_'+ssn+'.nc'                
        name_ukcp = 'ukcp09_T+1yr_nao_stephenson_psl_'+ssn+'_ukwil='+cwil+'.nc'         
        name_score= 'fig6_scores_nao_stephenson_djf_None_T+1yr_nresamp='+str(nresamp)+'_lastyr=2018_ukwil='+cwil+'.txt'

    file_dps   = os.path.join(dpsdir,  name_dps)
    file_score = os.path.join(dpsdir,  name_score)
    file_ukcp  = os.path.join(ukcpdir, name_ukcp)
    file_obs   = os.path.join(obsdir,  name_obs)

    gotukcp=True

    print('>>> 1. Input from:', file_dps)
    dps=iris.load_cube(file_dps)

    print('>>> 2. Input from:', file_obs)
    obs=iris.load_cube(file_obs)

    try:
       print('>>> 3. Input from:', file_ukcp)
       ukcp=iris.load_cube(file_ukcp)
    except:
       gotukcp=False

    print('>>> 4. Input from:', file_score)
    sdata=numpy.genfromtxt(file_score, names=True, dtype=None, encoding=None)

    scorenames= list(sdata['SCORE'])
    scorevalue= sdata['VALUE']

    fbar = scorevalue[ scorenames.index('fbar') ]
    obar = scorevalue[ scorenames.index('obar') ]
    fmttit = '%5.2f'

    if stype == 'U':
        accname = 'ACC' 
        msssname= 'MSSS'
        if not detrend:
            acc      = scorevalue[ scorenames.index('ACCU_DPS') ]
            acc_p10  = scorevalue[ scorenames.index('ACCU_P10') ]
            acc_p90  = scorevalue[ scorenames.index('ACCU_P90') ]    
            msss     = scorevalue[ scorenames.index('MSSSU_DPS') ]
            msss_p10 = scorevalue[ scorenames.index('MSSSU_P10') ]
            msss_p90 = scorevalue[ scorenames.index('MSSSU_P90') ]
            acc_uk   = scorevalue[ scorenames.index('ACCU_UKCP') ]
            msss_uk  = scorevalue[ scorenames.index('MSSSU_UKCP') ]
        else:
            acc      = scorevalue[ scorenames.index('ACCU_DPS_DTR') ]
            acc_p10  = scorevalue[ scorenames.index('ACCU_P10_DTR') ]
            acc_p90  = scorevalue[ scorenames.index('ACCU_P90_DTR') ]    
            msss     = scorevalue[ scorenames.index('MSSSU_DPS_DTR') ]
            msss_p10 = scorevalue[ scorenames.index('MSSSU_P10_DTR') ]
            msss_p90 = scorevalue[ scorenames.index('MSSSU_P90_DTR') ]
            acc_uk   = scorevalue[ scorenames.index('ACCU_UKCP_DTR') ]
            msss_uk  = scorevalue[ scorenames.index('MSSSU_UKCP_DTR') ]
         

    elif stype == 'C':     
        accname = 'ACCC' 
        msssname= 'MSSSC'
        if not detrend:
            acc      = scorevalue[ scorenames.index('ACCC_DPS') ]
            acc_p10  = scorevalue[ scorenames.index('ACCC_P10') ]
            acc_p90  = scorevalue[ scorenames.index('ACCC_P90') ]    
            msss     = scorevalue[ scorenames.index('MSSSC_DPS') ]
            msss_p10 = scorevalue[ scorenames.index('MSSSC_P10') ]
            msss_p90 = scorevalue[ scorenames.index('MSSSC_P90') ]
            acc_uk   = scorevalue[ scorenames.index('ACCC_UKCP') ]
            msss_uk  = scorevalue[ scorenames.index('MSSSC_UKCP') ]
        else:
            acc      = scorevalue[ scorenames.index('ACCC_DPS_DTR') ]
            acc_p10  = scorevalue[ scorenames.index('ACCC_P10_DTR') ]
            acc_p90  = scorevalue[ scorenames.index('ACCC_P90_DTR') ]    
            msss     = scorevalue[ scorenames.index('MSSSC_DPS_DTR') ]
            msss_p10 = scorevalue[ scorenames.index('MSSSC_P10_DTR') ]
            msss_p90 = scorevalue[ scorenames.index('MSSSC_P90_DTR') ]
            acc_uk   = scorevalue[ scorenames.index('ACCC_UKCP_DTR') ]
            msss_uk  = scorevalue[ scorenames.index('MSSSC_UKCP_DTR') ]
        
    if central == 'median':
        dpssyr=dps.coord('season_year').points
        sy0=dpssyr[0]
        sy1=dpssyr[-1]    
        tConstraint = iris.Constraint(season_year=lambda y: y >= sy0 and y <= sy1 )
        obs2 = obs.extract(tConstraint)
        perccon = iris.Constraint(percentile_over_realization_index=50.0)
        dpsdata = dps.extract(perccon).data
        obsdata = obs2.data
        if detrend:        
            dpsdata = scipy.signal.detrend(dpsdata)
            obsdata = scipy.signal.detrend(obsdata)
        if stype == 'U':
            acc  = utils.ACC_MSSS(dpsdata, obsdata, score='acc',  sstype='uncentred')
            msss = utils.ACC_MSSS(dpsdata, obsdata, score='msss', sstype='uncentred')
        elif stype == 'C':          
            acc  = utils.ACC_MSSS(dpsdata, obsdata, score='acc',  sstype='centred')
            msss = utils.ACC_MSSS(dpsdata, obsdata, score='msss', sstype='centred')

    score_name =[]
    score_value=[]
    n_detrend  ='orig'
    if detrend: n_detrend='detrend' 
    
    # Plot UKCP  
    if gotukcp:               
        ukcplab='UKCP-pdf: '+accname+': '+str(fmtleg%acc_uk)+', '+msssname+': '+str(fmtleg%msss_uk)

        score_name.append( vsr+'_'+TTT+'_ukcp_'+central+'_acc'+stype.lower()+'_'+n_detrend )
        score_value.append(acc_uk)
        score_name.append( vsr+'_'+TTT+'_ukcp_'+central+'_msss'+stype.lower()+'_'+n_detrend )
        score_value.append(msss_uk)

        ukcpdata = utils.subset_ukcp(ukcp, dps, ssn, nlump, detrend=detrend)

        for perc in percentiles:
            cube=ukcp.extract(iris.Constraint(percentile=perc/100.0))
            time = utils.timefromsy(cube, ssn, nlump)
            lw=ukcplw
            #if perc == 50:   lw=ukcplwmed                    
            plt.plot(time, cube.data, color=ukcpcol, linewidth=lw)
            plt.ylabel(ylab)
        # Extract extremes :
        ukcpmax = ukcp.extract(iris.Constraint(percentile=90/100))
        ukcpmin = ukcp.extract(iris.Constraint(percentile=10/100))                
        ax=plt.gca()
        facecolor='grey'
        alpha=0.13
        ax.fill_between(time, ukcpmin.data, ukcpmax.data, facecolor=facecolor,alpha=alpha)
        fc_for_rectangle = matplotlib.colors.ColorConverter().to_rgba(facecolor, alpha=alpha)
        handle_ukcp      = plt.Rectangle( (0, 0), 0, 0, edgecolor=ukcpcol, fc=fc_for_rectangle, lw=ukcplw)
        print('VAR:',var,'uckp, time[-1]=',time[-1])
    
    # Plot OBS  
    cube = obs
    time = utils.timefromsy(cube, ssn, nlump)
    plt.plot(time, cube.data, color=obscol, linewidth=obslw)
    plt.ylabel(ylab)
    print('VAR:',var,'obs, time[-1]=',time[-1])

    # Plot DPS
    dpslab = 'MMDPE, 130 members\n'        
    dpslab = dpslab+accname +': '+str(fmtleg%acc) +' ('+str(fmtleg%acc_p10) +', '+str(fmtleg%acc_p90) +'), '
    dpslab = dpslab+msssname+': '+str(fmtleg%msss)+' ('+str(fmtleg%msss_p10)+', '+str(fmtleg%msss_p90)+')'

    slc=stype.lower()
    score_name.append( vsr+'_'+TTT+'_dps_'+central+'_acc'+slc+'_'+n_detrend )
    score_value.append(acc)
    score_name.append( vsr+'_'+TTT+'_dps_p10_acc'+slc+'_'+n_detrend )
    score_value.append(acc_p10)
    score_name.append( vsr+'_'+TTT+'_dps_p90_acc'+slc+'_'+n_detrend )
    score_value.append(acc_p90)

    score_name.append( vsr+'_'+TTT+'_dps_'+central+'_msss'+slc+'_'+n_detrend )
    score_value.append(msss)
    score_name.append( vsr+'_'+TTT+'_dps_p10_msss'+slc+'_'+n_detrend )
    score_value.append(msss_p10)
    score_name.append( vsr+'_'+TTT+'_dps_p90_msss'+slc+'_'+n_detrend )
    score_value.append(msss_p90)

    for perc in percentiles:
        cube = dps.extract(iris.Constraint(percentile_over_realization_index=perc))
        time = utils.timefromsy(cube, ssn, nlump)
        lw=dpslw
        if perc == 50:  lw=dpslwmed
        plt.plot(time, cube.data, color=dpscol, linewidth=lw)
    plt.ylabel(ylab)
    print('VAR:',var,'dps, time[-1]=',time[-1])

    plt.axhline(0.0,color='k',lw=0.75,ls=':')

    ax=plt.gca()
    ax.set_ylim(ylimset)
    
    xticks=[1965,1975,1985,1995,2005,2015]
    ax.set_xticks(xticks)
    xticklabels=[]
    for yyyy in xticks:
        xticklabels.append(str(yyyy))
    ax.set_xticklabels(xticklabels) 
    ax.set_xlim([1965., yrend]) 
    ax.set_xlim([yrbeg, yrend]) 
   
    labels= [obslab, dpslab]
    ll = []
    ll.append( matplotlib.lines.Line2D([], [], color=obscol,  lw=obslw) )
    ll.append( matplotlib.lines.Line2D([], [], color=dpscol,  lw=dpslw) )
    if gotukcp:    
        labels.append(ukcplab)
        ll.append( handle_ukcp )

    plt.title(tit, fontsize=10.75,pad=4)
    loc='upper left'
    legtit=''    
    leg=plt.legend(ll, labels, loc=loc, title=legtit, fontsize=fsleg, alignment='left',
                   handlelength=1.2, borderaxespad=0.4, handletextpad=0.6, labelspacing=0.6)    
    leg.draw_frame(legframe)


    # Save scores/data to file for table, future use etc
    score_name =numpy.array(score_name)
    score_value=numpy.array(score_value)
    data_filename = vsr+'_'+TTT+'_'+central+'_data_orig.npz'
    if stype == 'C':
        score_filename = vsr+'_'+TTT+'_centred_'+n_detrend+'.npz'
    else:
        score_filename = vsr+'_'+TTT+'_uncentred_'+n_detrend +'.npz'        
    outscore= os.path.join(scoredir, score_filename)
    outdata = os.path.join(scoredir, data_filename)    
    if not detrend:
        numpy.savez(outdata, time=time, dpsdata=dpsdata, obsdata=obsdata, ukcpdata=ukcpdata)
        print('Saved ',outdata)        
    if saveplot:
        numpy.savez(outscore, score_name=score_name, score_value=score_value)
        print('Saved ',outscore)
    else:
        print('NOT saved:',outscore)

if detrend:
    namefig = namefig+'_detrend'

for dpi in dpiarr:           
    cdpi  = str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
