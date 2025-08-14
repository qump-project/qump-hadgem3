import os
import copy
import numpy
import cf_units
import datetime

import matplotlib
import matplotlib.pyplot as plt

import iris

import naoUtils
import qdcUtils as utils
from pathnames_v1 import *

##############################################################
namefig= 'figS11'
ifig   = 1011

dpiarr = [150]

saveplot=True
#saveplot=False

#central='mean'
central='median'


normtype = 'lump_diff_em'         
norm_ssn = False     
renorm   = False


choices = [ ['djf','Smith'],  ['djf','Li'],  ['djf','Stephenson'] ]  

choicearr=numpy.array(choices)

modellist = ['BSC', 'CAFE', 'CanESM5', 'CMCC', 'Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NCAR', 'NorCPM', 'NorCPMi2']  
nmodels   = len(modellist) 


coldic={'Smith, djf':      'dodgerblue',   'Smith, djfm':      'blue', 
        'Li, djf':         'orange',       'Li, djfm':         'red',
        'Stephenson, djf': 'limegreen',    'Stephenson, djfm': 'darkgreen'}

tlimdic={'BCC': [1960, 2013],     'BSC': [1960, 2018],     'CAFE': [1960, 2019],     'CanESM5':[1960, 2016],
         'CMCC': [1960, 2019],    'CMCC-10': [1960, 2019], 'Depresys4_gc3.1': [1960, 2022],             
         'IPSL': [1960, 2016],    'MIROC6': [1960, 2018],  'MPI': [1960, 2017],      'NCAR': [1954, 2017],
         'NCAR-10': [1954, 2017], 'NorCPM': [1960, 2018],  'NorCPMi2': [1960, 2018], 'ALL': [1960, 2016]}   
           
clump    = 'T+2-9'
obsname  = 'era5'

if clump == 'T+2-9':
    t1 = 2 
    t2 = 9
    fp = '2to9'
elif clump == 'T+1-8':
    t1 = 1 
    t2 = 8
    fp = '1to8'
nlump = t2-t1+1

tinit0 = 1960
islice = slice(t1-1, t2) 

nchoice=len(choices)

colfracline=['limegreen', 'orange', 'dodgerblue']
colfracpt  =['purple',    'navy',   'red']

base1=1971
base2=2000

basearr = [ [1971,2000], [1981,2010] ]

lla     = []
labelsa = []
llb     = []
labelsb = []
for ichoice,choice in enumerate(choices):
    ssn=choice[0]
    region=choice[1]
    regssn= region+', '+ssn
    for ibase,basen in enumerate(basearr):
            
        base_1=basen[0]
        base_2=basen[1]

        cbase ='b'+str(base1)+'-'+str(base2)
        cbasen='b'+str(base_1)+'-'+str(base_2)

        # Load Obs    
        regname = region

        fnorth='nao_north_'+obsname+'_'+regname+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'
        fsouth='nao_south_'+obsname+'_'+regname+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'
   
        filenorth= os.path.join(naodir,fnorth)
        filesouth= os.path.join(naodir,fsouth)

        print('>>> Load Obs data from:',filenorth)
        obsnorth_in = iris.load_cube(filenorth)
        
        print('>>> Load Obs data from:',filesouth)
        obssouth_in = iris.load_cube(filesouth)

        timeobs1 = obsnorth_in.coord('season_year').points + 15./360.

        obsnorth, obssouth = naoUtils.rebase_obscube(obsnorth_in, obssouth_in, base_1, base_2)    

        ordersplit=normtype.split('_')
        o3=[]
        for o1 in ordersplit:
            if o1 != 'em':
                o3.append(o1)
        order_obs = '_'.join(o3)
        
        # Make index for Obs
        timeobs, naoobssm = naoUtils.obs_smoothdiff(obsnorth, obssouth, nlump, ssn=ssn, order='lump_diff', renorm=renorm)


        # Make index for MMDPE
        midyr, naomodel, nao_all, tlis, dlis = naoUtils.loadNAO(naodir, fp=fp, region=region, ssn=ssn, base1=base_1, base2=base_2,
                                                                ensemble=modellist, central=central, renorm=renorm)
                                
        iforobs, ifordec     = naoUtils.index_common(timeobs, midyr)
        
        mod_ff=naomodel[ifordec]
        obs_ff=naoobssm[iforobs]
        tim_ff=timeobs[iforobs]        
        nfrac=mod_ff.shape[0]
        kount=numpy.zeros(nfrac)
        same =numpy.zeros(nfrac)
        for kk, obs, model in zip(list(range(nfrac)),obs_ff,mod_ff):
            samesign = obs*model >= 0.0
            same[kk] = samesign
            if kk == 0:  
                if samesign: 
                    kount[kk]=1
                else:
                    kount[kk]=0
            else:   
                if samesign:
                    kount[kk]=kount[kk-1]+1
                else:
                    kount[kk]=kount[kk-1]
        frac = kount/numpy.arange(1,nfrac+1)

        # Now do fbar,obar
        
        nmax  = ifordec.shape[0]
        yfirst=1980
        found=False
        k0=0
        while not found:
            k0=k0+1
            if midyr[ifordec[k0]] > yfirst+1:
                found=True            
        klist = list(range(k0,nmax+1))     # k0=16 => 1980 
        nlast = len(klist)
        ylast = numpy.zeros(nlast)
        accu   = numpy.zeros(nlast)
        accc   = numpy.zeros(nlast)
        fbar   = numpy.zeros(nlast)
        obar   = numpy.zeros(nlast)               
        for ik,kk in enumerate(klist):
            idec = ifordec[:kk]
            iobs = iforobs[:kk]
            accu1 = naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='uncentred', score='acc') 
            accc1 = naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='centred', score='acc')
            fbar2 = numpy.mean(naomodel[idec])/numpy.sqrt(numpy.mean(naomodel[idec]**2)) 
            obar2 = numpy.mean(naoobssm[iobs])/numpy.sqrt(numpy.mean(naoobssm[iobs]**2))                         
            ylast[ik]=midyr[idec][-1]
            accu[ik]=accu1
            accc[ik]=accc1
            fbar[ik]=fbar2
            obar[ik]=obar2
                        
        if ichoice == 0 and ibase == 0:
            # set up Figure for first time
            plt.figure(ifig, figsize=(16.5/2.54, 7.7/2.54))
            matplotlib.rcParams.update({'font.size': 9.5})
            plt.subplots_adjust(hspace=0.30,wspace=0.30,top=0.91,bottom=0.11,left=0.103,right=0.967)
            alpha=0.75

        # Frac plot
        if ibase == 0:
            ax=plt.subplot(1,2,1)
            col1=coldic[regssn]
            plt.plot(tim_ff, frac, color=col1,lw=1.5,alpha=alpha)         
            labelsa.append(region+', '+ssn.upper())
            lla.append( matplotlib.lines.Line2D([], [], color=col1, lw=1.5, alpha=alpha) )

        # obar/fbar plot, only one region/ssn
        regssn_obar='Stephenson, djf'  ; regssntit='Stephenson, DJF' 
        if regssn == regssn_obar:
            ax=plt.subplot(1,2,2)  
            if cbasen == 'b1971-2000':
                col2=['limegreen', 'magenta']
            else:
                col2=['olive',     'darkcyan']   
            plt.plot(ylast, fbar, color=col2[0], alpha=alpha, lw=1.5, marker='', ms=3)
            plt.plot(ylast, obar, color=col2[1], alpha=alpha, lw=1.5, marker='', ms=3)               

            print('>>> ',cbasen) 
            print('fbar.mean()=',fbar.mean(),'obar.mean()=',obar.mean()) 
                       
            labelsb.append('Hindcast, '+cbasen[1:]+ ' baseline')                        
            llb.append( matplotlib.lines.Line2D([], [], color=col2[0], lw=1.5, alpha=alpha, marker='', ms=3) )

            labelsb.append('Observed, '+cbasen[1:]+ ' baseline')
            llb.append( matplotlib.lines.Line2D([], [], color=col2[1], lw=1.5, alpha=alpha, marker='', ms=3) )


ax=plt.subplot(1,2,1)    
plt.ylabel('Sign skill')
ax.set_xlim([1990,2020])
if central == 'mean':
    ax.set_ylim([0.56,0.82])
else:
    ax.set_ylim([0.50,0.80])
title= 'NAO, MMDPE, T2-9'
plt.title(title,fontsize=10.5, pad=5)
lega = plt.legend(lla, labelsa, loc='best', fontsize=8.25, frameon=True,
                  handlelength=1.0, borderaxespad=0.3, handletextpad=0.3, labelspacing=0.15)


ax=plt.subplot(1,2,2)  
plt.ylabel('Normalised average anomaly')
ax.set_xlim([1990,2020])

if central == 'mean':
    ax.set_ylim([-0.8, 0.0])
else:

    ax.set_ylim([-0.8, 0.2])

title='NAO, MMDPE, '+regssntit+', T2-9' #+'\n'+normtype+', renorm='+str(renorm)[0]
plt.title(title,fontsize=10.5, pad=5)
legb = plt.legend(llb, labelsb, loc='best', fontsize=8.25, frameon=True,
                  handlelength=1.0, borderaxespad=0.3, handletextpad=0.3,labelspacing=0.15)


########################################################################
# Save figure

for dpi in dpiarr:           
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
