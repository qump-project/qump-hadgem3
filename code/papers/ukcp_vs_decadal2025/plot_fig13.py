import os
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import datetime

import iris

import qdcUtils as utils

from pathnames_v1 import *

##############################
namefig= 'fig13'
ifig   = 13

dpiarr = [150]

saveplot = True
#saveplot = False

detrend=True
detrend=False

percentiles = [10, 50, 90]

vararr=['NEU DJF SAT', 'NEU JJA SAT', 'EAW DJF SAT', 'EAW JJA SAT']

nresamp=4000

stype='U'
#stype='C'

nlump=8

#central = 'mean'
central = 'median'

cwil = '31-20-8' 

plot_beg_time = 1964.5
plot_end_time = 2021.25
# last time for EAW JJA T2-9 is 2021.04167 
# last time for NEU JJA T2-9 is 2020.04167 

TTT = 'T2-9'


obscol='black'
obslw='2.0'

ukcpcol='blue'
ukcplw='0.75'
ukcplwmed='2.0'

dpscol='red'
dpslw='1.0'
dpslwmed='2.0'


fsleg=8.2          # Fig18 has 8.25
legframe=True
fmtleg = '%4.2f'   #'%5.3f' 

linespacing=1.0

#titfontsize=11     # Fig18 has 10.25
titfontsize=10.25
    
##################################################   
matplotlib.rcParams.update({'font.size': 9.5})

fig = plt.figure(ifig, figsize=(20/2.54, 18.5/2.54))     

plt.subplots_adjust(top=0.95, bottom=0.045, left=0.09, right=0.98, hspace=0.21,wspace=0.25)        

for ivar,var in enumerate(vararr):
    ax = plt.subplot(2, 2, ivar+1)
    if var == 'NEU DJF SAT':
        obslab    = 'HadCRUT5 Obs'
        obslab    = 'Observations'
        ylab      = 'Anomaly ($\degree$C)'
        #tit       = 'T2-9 northern Europe DJF Tair'
        #tit       = 'T2-9 northern Europe winter Tair anomaly'
        tit       = 'T2-9 NEU winter Tair'
        vsr       = 'tas_djf_neu'                    
        ssn       = 'djf'
        ylimset   = [-2.1, 3.7]
        name_dps  = 'T+2toT+9yr_NorthernEurope_air_temperature_'+ssn+'.nc'
        name_obs  = 'hadcrut5_latlong_T+2toT+9yr_NorthernEurope_air_temperature_'+ssn+'.nc'
        name_ukcp = 'ukcp09_T+2toT+9yr_NorthernEurope_air_temperature_'+ssn+'_ukwil='+cwil+'.nc'
        name_score= 'fig9_scores_air_temperature_djf_NorthernEurope_T+2to9yr_nresamp='+str(nresamp)+'_lastyr=2018_ukwil='+cwil+'.txt'

    elif var == 'NEU JJA SAT':
        obslab    = 'HadCRUT5 Obs'
        obslab    = 'Observations'
        ylab      = 'Anomaly ($\degree$C)'      
        #tit       = 'T2-9 northern Europe JJA Tair'
        #tit       = 'T2-9 northern Europe summer Tair anomaly'
        tit       = 'T2-9 NEU summer Tair'
        vsr       = 'tas_jja_neu' 
        ssn       = 'jja'
        ylimset   = [-1.0, 2.7]
        name_dps  = 'T+2toT+9yr_NorthernEurope_air_temperature_'+ssn+'.nc'
        name_obs  = 'hadcrut5_latlong_T+2toT+9yr_NorthernEurope_air_temperature_'+ssn+'.nc'
        name_ukcp = 'ukcp09_T+2toT+9yr_NorthernEurope_air_temperature_'+ssn+'_ukwil='+cwil+'.nc'
        name_score= 'fig9_scores_air_temperature_jja_NorthernEurope_T+2to9yr_nresamp='+str(nresamp)+'_lastyr=2018_ukwil='+cwil+'.txt'

    elif var == 'EAW DJF SAT':
        obslab    = 'NCIC Obs'
        obslab    = 'Observations'
        ylab      = 'Anomaly ($\degree$C)' 
        #tit       = 'T2-9 England-Wales DJF Tair'
        #tit       = 'T2-9 England-Wales winter Tair anomaly'        
        tit       = 'T2-9 EngWal winter Tair'        
        vsr       = 'tas_djf_eaw' 
        ssn       = 'djf'
        ylimset   = [-1.35, 2.3]
        name_dps  = 'T+2toT+9yr_EnglandandWales_air_temperature_'+ssn+'.nc'
        name_obs  = 'ncic_T+2toT+9yr_EnglandandWales_air_temperature_'+ssn+'.nc'
        name_ukcp = 'ukcp09_T+2toT+9yr_EnglandandWales_air_temperature_'+ssn+'_ukwil='+cwil+'.nc'
        name_score= 'fig9_scores_air_temperature_djf_EnglandandWales_T+2to9yr_nresamp='+str(nresamp)+'_lastyr=2018_ukwil='+cwil+'.txt'

    elif var == 'EAW JJA SAT':
        obslab    = 'NCIC Obs'
        obslab    = 'Observations'
        ylab      = 'Anomaly ($\degree$C)' 
        #tit       = 'T2-9 England-Wales JJA Tair'
        #tit       = 'T2-9 England-Wales summer Tair anomaly'
        tit       = 'T2-9 EngWal summer Tair'        
        vsr       = 'tas_jja_eaw' 
        ssn       = 'jja'
        ylimset   = [-1.0, 2.3]
        name_dps  = 'T+2toT+9yr_EnglandandWales_air_temperature_'+ssn+'.nc'
        name_obs  = 'ncic_T+2toT+9yr_EnglandandWales_air_temperature_'+ssn+'.nc'
        name_ukcp = 'ukcp09_T+2toT+9yr_EnglandandWales_air_temperature_'+ssn+'_ukwil='+cwil+'.nc'
        name_score= 'fig9_scores_air_temperature_jja_EnglandandWales_T+2to9yr_nresamp='+str(nresamp)+'_lastyr=2018_ukwil='+cwil+'.txt'
    
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
        # Can instead calculate the score from the ensemble median.        
        perccon = iris.Constraint(percentile_over_realization_index=50.0)
        dpsdata = dps.extract(perccon).data
        obsdata = obs.data
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
            plt.plot(time, cube.data, color=ukcpcol, linewidth=lw)
        # Extract extremes :
        ukcpmax = ukcp.extract(iris.Constraint(percentile=90/100))
        ukcpmin = ukcp.extract(iris.Constraint(percentile=10/100))                
        ax=plt.gca()
        facecolor='grey'
        alpha=0.12    #18 has 0.13
        ax.fill_between(time, ukcpmin.data, ukcpmax.data, facecolor=facecolor,alpha=alpha)
        fc_for_rectangle = matplotlib.colors.ColorConverter().to_rgba(facecolor, alpha=alpha)
        handle_ukcp      = plt.Rectangle( (0, 0), 0, 0, edgecolor=ukcpcol, fc=fc_for_rectangle, lw=ukcplw)
        sy0=cube.coord('season_year').points[0]
        print('UKCP:',var,'time[0]=',time[0],'time[-1]=',time[-1],', sy0=',sy0,', endtime=',plot_end_time)

    
    # Plot OBS  
    cube = obs
    time = utils.timefromsy(cube, ssn, nlump)
    plt.plot(time, cube.data, color=obscol, linewidth=obslw)
    plt.ylabel(ylab)
    sy0=cube.coord('season_year').points[0]
    print('OBS: ',var,'time[0]=',time[0],'time[-1]=',time[-1],', sy0=',sy0,', endtime=',plot_end_time)


    # Plot DPS
    dpslab = 'MMDPE, 150 members\n'        
    dpslab = dpslab+accname +': '+str(fmtleg%acc) +' ('+str(fmtleg%acc_p10) +', '+str(fmtleg%acc_p90) +')\n'
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
        lw   = dpslw
        if perc == 50:  lw=dpslwmed
        plt.plot(time, cube.data, color=dpscol, linewidth=lw)
    plt.ylabel(ylab)
    sy0=cube.coord('season_year').points[0]
    print('DPS: ',var,'time[0]=',time[0],'time[-1]=',time[-1],', sy0=',sy0,', endtime=',plot_end_time)

    ax.set_ylim(ylimset)

    xticks=[1965,1975,1985,1995,2005,2015]
    ax.set_xticks(xticks)
    xticklabels=[]
    for yyyy in xticks:
        xticklabels.append(str(yyyy))
    ax.set_xticklabels(xticklabels) 
    ax.set_xlim([plot_beg_time, plot_end_time])
   
    labels= [obslab, dpslab]
    ll = []
    ll.append( matplotlib.lines.Line2D([], [], color=obscol,  lw=obslw) )
    ll.append( matplotlib.lines.Line2D([], [], color=dpscol,  lw=dpslw) )
    if gotukcp:    
        labels.append(ukcplab)
        ll.append( handle_ukcp )

    plt.title(tit, fontsize=titfontsize, linespacing=linespacing, pad=4)

    loc='upper left'
    legtit=''    
    leg=plt.legend(ll, labels, loc=loc, title=legtit, fontsize=fsleg, alignment='left',
                   handlelength=1.1, borderaxespad=0.4, handletextpad=0.6, labelspacing=0.5)    
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
      
# End loop over variables/panels

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
