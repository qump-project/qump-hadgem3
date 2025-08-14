import os
import pickle
import numpy
import scipy
import scipy.stats.mstats

import matplotlib
import matplotlib.pyplot as plt

import iris

import qdcUtils as utils
from pathnames_v1 import *

#########################################
figno = input("Enter figure number, one of (S8, S17): ")
figno = figno.upper()
if figno not in ['S8', 'S17']:
    raise AssertionError('Incorrect figno, please enter one of (S8, S17)')        
namefig = 'fig'+figno

dpiarr  = [150]

saveplot   = True
#saveplot   = False

ensembles=['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3.1','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']


mdic={'BCC':'BCC',       'BSC':'BSC',                   'CAFE':'CAFE',     'CMCC':'CMCC',
      'CanESM5':'CCCma', 'Depresys4_gc3.1':'DePreSys4', 'IPSL':'IPSL',     'MIROC6':'MIROC', 
      'MPI':'MiKlip',    'NCAR40':'NCAR',               'NorCPM':'NCC-i1', 'NorCPMi2':'NCC-i2'}

stype='U'
#stype='C'

#central = 'mean'
central = 'median'

if namefig == 'figS8':    #T1 GLB ann tas
    ifig  = 1008
    reg   = 'glb' 
    name  = 'globalukcp09_annual_tas'
    ssn   = 'djfmamjjason'
    nlump = 8
    #yrlast= 2019
    yrlast= 2020
elif namefig == 'figS17':   #T1 NEU jja tas
    ifig  = 1017
    reg   = 'neu'
    name  = 'northerneurope_summer_tas'
    ssn   = 'jja'
    nlump = 1
    #yrlast= 2019
    # Now want same number of time points for all, so limit to 2018 
    # (last time in IPSL, CanESM5 is 2017.5416) 
    yrlast= 2018  

percentiles = [10, 50, 90]


ukcpcol   = 'blue'
ukcplw    = 0.5     # 0.75
ukcplwmed = 1.0      # 1.5
dpscol    = 'red'
obscol    = 'black'
obslw     = 1.5
xtick_interval   = 10
xtick_text_format= 'yyyy'
fmtleg           = '%4.2f'   #'%5.3f' 

fs=8.0
matplotlib.rcParams.update({'font.size': fs})
figsize=(17/2.54, 18.5/2.54)
plt.figure(ifig, figsize=figsize)
plt.subplots_adjust(top=0.935, bottom=0.035, left=0.07, right=0.985, hspace=0.28, wspace=0.23)        

for imod,model in enumerate(ensembles):
    pfile = os.path.join(dpsdir, 'fig1_'+model+'_'+name+'.pkl')
    print('Load file: ',pfile)
    with open(pfile, 'rb') as f:
       dic = pickle.load(f)       
    mname   =  dic['name']
    dpscube =  dic['dpscube']  
    obscube =  dic['obscube']  
    ukcpcube=  dic['ukcpcube'] 
    acc_dps =  dic['acc_dps']     #this is uncentred, see comment below                       
    fbaru   =  dic['fbar']  
    obaru   =  dic['obar']  
    msssu   =  dic['msssu'] 
    msssc   =  dic['msss']  
    msssw   =  dic['msssw'] 


    dpssyr = dpscube.coord('season_year').points
    dpsy0  = dpssyr[0]   
    dpsy1=dpssyr[-1]    
    obscube2 = obscube.extract( iris.Constraint(season_year=lambda y: y >= dpsy0 and y <= dpsy1) )
    
    obssyr = obscube2.coord('season_year').points
    obsy0  = obssyr[0]   
    obsy1=obssyr[-1]    
    dpscube2 = dpscube.extract( iris.Constraint(season_year=lambda y: y >= obsy0 and y <= obsy1) )
    
    if central == 'median':
        dps=dpscube2.collapsed('realization',iris.analysis.MEDIAN)
    else:
        dps=dpscube2.collapsed('realization',iris.analysis.MEAN)
    dpsdata = dps.data
    obsdata = obscube2.data
    
    if stype == 'U':
        acc_dps= utils.ACC_MSSS(dpsdata, obsdata, score='acc',  sstype='uncentred')
        msssu  = utils.ACC_MSSS(dpsdata, obsdata, score='msss', sstype='uncentred')
        # these give identical to eg dic['acc_dps'] above
    elif stype == 'C':          
        acc_dps= utils.ACC_MSSS(dpsdata, obsdata, score='acc',  sstype='centred')
        msssc  = utils.ACC_MSSS(dpsdata, obsdata, score='msss', sstype='centred')
   
    plt.subplot(4,3,imod+1)
    
    # Plot UKCP, fill bewteen limits
    ukcpmax = ukcpcube.extract(iris.Constraint(percentile=90/100))
    ukcpmin = ukcpcube.extract(iris.Constraint(percentile=10/100))                
    ax=plt.gca()
    facecolor='grey'
    alpha=0.13
    time = utils.timefromsy(ukcpcube, ssn, nlump)    
    ax.fill_between(time, ukcpmin.data, ukcpmax.data, facecolor=facecolor,alpha=alpha)

    # Plot UKCP lines 
    timeuk = utils.timefromsy(ukcpcube, ssn, nlump)    
    for perc in percentiles:
        cube= ukcpcube.extract(iris.Constraint(percentile=perc/100.0))
        lw  = ukcplw
        if perc == 50:   lw=ukcplwmed                    
        plt.plot(timeuk, cube.data, color=ukcpcol, linewidth=lw)
   
    # Plot DPS
    for ii,dpsslice in enumerate(dpscube.slices_over(['realization'])):
        timedps = utils.timefromsy(dpsslice, ssn, nlump)    
        iok=numpy.where(timedps < yrlast)[0]     
        plt.plot(timedps[iok], dpsslice.data[iok], lw=0, marker='o',
                 markersize=2.0, markerfacecolor='red', markeredgecolor='k', markeredgewidth=0.35, alpha=0.5)
                          
    # Plot OBS
    timeobs = utils.timefromsy(obscube, ssn, nlump)
    iok=numpy.where(timeobs < yrlast)[0]    
    plt.plot(timeobs[iok], obscube.data[iok], color=obscol, linewidth=obslw)

    print('> ',model,' timedps[0]=',timedps[0],' timeobs[0]=',timeobs[0])
    print('> ',model,' timedps[-1]=',timedps[-1],' timeobs[-1]=',timeobs[-1])
 
    xticks=[1965,1975,1985,1995,2005,2015]
    ax.set_xticks(xticks)
    xticklabels=[]
    xticklabels=['', '1975', '', '1995', '', '2015']
    ax.set_xticklabels(xticklabels) 
    
    if reg == 'glb': 
        ax.set_xlim([1964.91667, 2020.41667])    # Previous
        ax.set_xlim([1964.91667, 2020.9166])     # 1 year before/after first and last times plotted
    elif reg == 'neu': 
        #ax.set_xlim([1960.54167, 2019.54167])   # Previous, with diff number of time points
        ax.set_xlim([1960.54167, 2018.54167])    # 1 year before/after first and last times plotted
    
    ylim = ax.get_ylim()
    print(imod, model, 'ylim=',ylim)

    if reg == 'glb':    
        ax.set_ylim([-0.71, 1.21])
    elif reg == 'neu': 
        ax.set_ylim([-2.9, 4.2])
                    
    plt.title(mdic[model], pad=1)    
    panel_text = 'ACC: '+str("%.3f"%acc_dps)+ '\nMSSS: '+str("%.3f"%msssu)    
    plt.text(0.03, 0.82, panel_text, fontsize=7.5, weight=500, transform=ax.transAxes)  


if reg == 'glb':
    suptit = 'T2-9 annual GMST anomaly'
elif reg == 'neu':
    suptit='T1 NEU summer Tair anomaly'
plt.suptitle(suptit, fontsize=11.5, y=0.99)
    
for dpi in dpiarr:           
    cdpi  = str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
   
    
    
    
