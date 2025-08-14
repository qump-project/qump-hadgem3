import os
import numpy
import pickle
import copy
import scipy
import scipy.stats.mstats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import iris

import loadUtils as loadutils
import qdcUtils as utils

from pathnames_v1 import *

#########################################
namefig= 'fig8'
ifig   = 8

dpiarr = [150]

saveplot   = True
#saveplot   = False

detrend=True
detrend=False

if detrend:   ifig = 800

central = 'median'
#central = 'mean'

ensembles0 = ['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3.1','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']

namearr    = ['Globalukcp09_djfmamjjason_tas', 'amo_djfmamjjason_tas', 'nao_stephenson_djf_psl']

#xlabelarr  = ['Ann GMST',                      'Ann AMV',              'DJF NAO']
xlabelarr  = ['GMST',                          'AMV',                  'winter NAO']

fparr      = [1, 2, '2to9']

acc_mean_lis1  = []    ; acc_mean_lis2  = []    ; acc_mean_lis3  = []
msss_mean_lis1 = []    ; msss_mean_lis2 = []    ; msss_mean_lis3 = []
acc_arr_lis1   = []    ; acc_arr_lis2   = []    ; acc_arr_lis3   = []
msss_arr_lis1  = []    ; msss_arr_lis2  = []    ; msss_arr_lis3  = []
acc_ukcp_lis1  = []    ; acc_ukcp_lis2  = []    ; acc_ukcp_lis3  = []
msss_ukcp_lis1 = []    ; msss_ukcp_lis2 = []    ; msss_ukcp_lis3 = []



for iname,longname in zip(range(3),namearr):
    for fp in fparr:
        print('\n>>> Processing longname=',longname,'fp=',fp)
        ensembles=copy.deepcopy(ensembles0)

        # DPS
        yearmax=2019
        fac=1.0
        if '_psl' in longname:   fac=1./100
        
        dall, dlistok, modlist, syr, nlump = loadutils.loadDPS(longname, dpsdir, fp=fp, ensembles=ensembles, 
                                                               fac=fac, yearmax=yearmax)        
        print(longname,': number of DPS:',len(modlist))

        # OBS
        dobs= loadutils.loadOBS(longname, obsdir, syr, nlump)
         
        #WARNING - Need to truncate dall if dobs smaller than syr, fix: syr,dlistok,dall
        syr,dall,dlistok= loadutils.trimdata(dobs, syr,dall,dlistok)
        
        # If dobs longer, truncate that instead
        if dobs.shape[0] > dall.shape[0]:
            dobs = dobs[:dall.shape[0]]     
        #dall.shape = (57, 150),  (ntime,nrlz)
        if central == 'median':
            dallcent = numpy.median(dall,axis=1)
        else:
            dallcent = numpy.mean(dall,axis=1)

        # UKCP        
        if fp == '2to9': 
            ukwil = 'ukwil=31-20-8'
            if 'Globalukcp09' in longname: ukwil = 'ukwil=1-20-8'
        else:
            ukwil = 'ukwil=31-20-1'
            
        dukcp= loadutils.loadUKCP(longname, ukcpdir, syr, nlump, ukwil=ukwil)
                
        # Scores
        if detrend:
            acc_mean,msss_mean, acc_arr,msss_arr, acc_ukcp,msss_ukcp = loadutils.scores_detrend(dallcent, dobs, dukcp, dlistok, 
                                                                           sstype='uncentred', central=central)  
        else:
            acc_mean,msss_mean, acc_arr,msss_arr, acc_ukcp,msss_ukcp = loadutils.scores(dallcent, dobs, dukcp, dlistok, 
                                                                           sstype='uncentred', central=central)  
                
        print('>>> ',longname,fp)
        for ii,model,acc,msss in zip( range(len(modlist)), modlist,acc_arr,msss_arr):
             print(ii,model,acc,msss) 
        print('   ALLmean',acc_mean,msss_mean) 
        print('  UKCPmean',acc_ukcp,msss_ukcp) 

        if iname == 0:
            acc_mean_lis1.append(acc_mean)
            msss_mean_lis1.append(msss_mean)
            acc_arr_lis1.append(acc_arr) 
            msss_arr_lis1.append(msss_arr)
            acc_ukcp_lis1.append(acc_ukcp)
            msss_ukcp_lis1.append(msss_ukcp)
        elif iname == 1:
            acc_mean_lis2.append(acc_mean)
            msss_mean_lis2.append(msss_mean)
            acc_arr_lis2.append(acc_arr) 
            msss_arr_lis2.append(msss_arr)
            acc_ukcp_lis2.append(acc_ukcp)
            msss_ukcp_lis2.append(msss_ukcp)
        elif iname == 2:
            acc_mean_lis3.append(acc_mean)
            msss_mean_lis3.append(msss_mean)
            acc_arr_lis3.append(acc_arr) 
            msss_arr_lis3.append(msss_arr)
            acc_ukcp_lis3.append(acc_ukcp)
            msss_ukcp_lis3.append(msss_ukcp)

        if longname == 'nao_djf_psl' and fp == '2to9':
            isort=numpy.argsort(acc_arr)
            print('>>>',longname)
            for ii in range(len(ensembles)):           
                 print(ii, ensembles[isort[ii]], acc_arr[isort[ii]])
            #raise AssertionError('Stop for debugging...')


##############################
# Boxplot for 3: DPSmean, DPSmem, UKCP

lw=2.0
ms=5
fs=10
alpha=0.2
sscat=40
lwbox=0.5
lwmed=2.0


matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'legend.numpoints': 1})

fig = plt.figure(ifig, figsize=(18./2.54, 9./2.54) )

plt.subplots_adjust(hspace=0.25,wspace=0.23,top=0.915,bottom=0.11,left=0.075,right=0.98)

xrng0= 0.0
xrng1= 3.0
R    = xrng1-xrng0
xlimfix = [xrng0, xrng1]

fac = 0.12   #(d/D)
D   = R/(3.+4*fac)
d   = fac*D

dxfac  = 0.28
width  = 0.15
facline= 0.14

h1     = width/2.
xticks = [ d+D*1/2, 2*d+D*3/2, 3*d+D*5/2]

pcolarr=['darkorange', 'green',     'blue']
bcolarr=['orange',     'limegreen', 'dodgerblue']

alpha=0.15
useRect=True



### ACC plot
ax=plt.subplot(1,2,1)
if detrend:
    ylim=[-0.28, 0.90]
    ylim=[-0.65, 0.90]    
else:
    ylim=[-0.28, 1.05]

ax.set_xlim(xlimfix)
ax.set_ylim(ylim)

for iname,longname,xlabel in zip(range(3),namearr, xlabelarr):

    pos1 = xticks[iname]
    pos0 = pos1-dxfac*D
    pos2 = pos1+dxfac*D
    posa = [pos0,pos1,pos2]

    if iname == 0:
       score_mean_lis = acc_mean_lis1
       score_arr_lis  = acc_arr_lis1
       score_ukcp_lis = acc_ukcp_lis1
    elif iname == 1:
       score_mean_lis = acc_mean_lis2
       score_arr_lis  = acc_arr_lis2
       score_ukcp_lis = acc_ukcp_lis2
    elif iname == 2:
       score_mean_lis = acc_mean_lis3
       score_arr_lis  = acc_arr_lis3
       score_ukcp_lis = acc_ukcp_lis3

    nbox=len(score_mean_lis)
    
    for ibox in range(nbox):
        colpt  = pcolarr[ibox]
        colbox = bcolarr[ibox]  
      
        score_mean = score_mean_lis[ibox] 
        score_arr  = score_arr_lis[ibox] 
        score_ukcp = score_ukcp_lis[ibox]

        plo=score_arr.min()
        phi=score_arr.max()

        pos=posa[ibox]
        xxx=[pos-h1, pos+h1, pos+h1, pos-h1, pos-h1]
        yyy=[plo, plo, phi, phi, plo]

        # Plot box, either with Rectangle or with fill_between
        if useRect:
            ax.add_patch(Rectangle((xxx[0], yyy[0]), 2*h1, phi-plo,
                         edgecolor=colpt, facecolor=colbox, fill=True, lw=lwbox, alpha=alpha))
        else:
            plt.fill_between([pos-h1, pos+h1], [plo,plo],[phi,phi], color=colbox, alpha=alpha)

        # Plot box edge
        ax.plot(xxx, yyy, color=colbox, lw=lwbox, solid_capstyle='butt')

        # Plot median
        pmed = numpy.median(score_arr) 
        ax.plot( [pos-h1, pos+h1], [pmed, pmed], color=colbox, linewidth=lwmed, solid_capstyle='butt')
                 
        # Plot points
        np=score_arr.shape[0]
        xp=numpy.ones(np)*pos             
        ax.plot(xp, score_arr, lw=0,marker='o',ms=3.5, color=colbox,alpha=0.8)   #color=colpt,alpha=0.6)   
                 
        # Plot multimodel       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_mean, score_mean], color='red', linewidth=lwmed, solid_capstyle='butt')
      
        # Plot ukcp       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_ukcp, score_ukcp], color='purple', linewidth=lwmed, solid_capstyle='butt')

if ylim[0] < 0.0:
   plt.axhline(0.0, ls=':', lw=0.75, color='k')    
           
ax.set_xticks(xticks)
ax.set_xticklabels(xlabelarr)
if detrend:
    plt.title('ACC (detrended)')   
else:    
    plt.title('ACC')   

xlims=ax.get_xlim()
ylims=ax.get_ylim()
xleg=xlims[0]+0.02*(xlims[1]-xlims[0])
dxtxt=0.07*(xlims[1]-xlims[0])
dxbox=0.05*(xlims[1]-xlims[0])
dybox=0.032*(ylims[1]-ylims[0])
yfirst=0.22
dy=0.05
fstxt=9.0

yleg=ylims[0]+(yfirst-0*dy)*(ylims[1]-ylims[0])
col=bcolarr[0]
ax.add_patch(Rectangle((xleg, yleg), dxbox, dybox, edgecolor=colpt, facecolor=col, fill=True, lw=lwbox, alpha=alpha))
xxx=[xleg, xleg+dxbox, xleg+dxbox, xleg, xleg]
yyy=[yleg, yleg, yleg+dybox, yleg+dybox, yleg]
ax.plot(xxx, yyy, color=col, lw=lwbox, solid_capstyle='butt')
plt.text(xleg+dxtxt, yleg, 'T1',fontsize=fstxt) 
 
yleg=ylims[0]+(yfirst-1*dy)*(ylims[1]-ylims[0])
col=bcolarr[1]
ax.add_patch(Rectangle((xleg, yleg), dxbox, dybox, edgecolor=colpt, facecolor=col, fill=True, lw=lwbox, alpha=alpha))
xxx=[xleg, xleg+dxbox, xleg+dxbox, xleg, xleg]
yyy=[yleg, yleg, yleg+dybox, yleg+dybox, yleg]
ax.plot(xxx, yyy, color=col, lw=lwbox, solid_capstyle='butt')
plt.text(xleg+dxtxt, yleg, 'T2',fontsize=fstxt) 
 
yleg=ylims[0]+(yfirst-2*dy)*(ylims[1]-ylims[0])
col=bcolarr[2]
ax.add_patch(Rectangle((xleg, yleg), dxbox, dybox, edgecolor=colpt, facecolor=col, fill=True, lw=lwbox, alpha=alpha))
xxx=[xleg, xleg+dxbox, xleg+dxbox, xleg, xleg]
yyy=[yleg, yleg, yleg+dybox, yleg+dybox, yleg]
ax.plot(xxx, yyy, color=col, lw=lwbox, solid_capstyle='butt')
plt.text(xleg+dxtxt, yleg, 'T2-9',fontsize=fstxt) 
 
yleg=ylims[0]+(yfirst-3*dy)*(ylims[1]-ylims[0])
xxx=[xleg, xleg+dxbox]
yyy=[yleg+0.4*dy, yleg+0.4*dy]
ax.plot(xxx, yyy, color='red', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'MMDPE',fontsize=fstxt) 

yleg=ylims[0]+(yfirst-4*dy)*(ylims[1]-ylims[0])
xxx=[xleg, xleg+dxbox]
yyy=[yleg+0.4*dy, yleg+0.4*dy]
ax.plot(xxx, yyy, color='purple', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'UKCP-pdf',fontsize=fstxt) 
 
           


###############
# MSSS plot
ax=plt.subplot(1,2,2)
if detrend:
    ylim=[-8., 0.90]   #need -33 for T2-9 CAFE GMST!
else:
    ylim=[-0.4, 1.05]

ax.set_xlim(xlimfix)
ax.set_ylim(ylim)

for iname,longname,xlabel in zip(range(3),namearr, xlabelarr):

    pos1 = xticks[iname]
    pos0 = pos1-dxfac*D
    pos2 = pos1+dxfac*D
    posa = [pos0,pos1,pos2]

    if iname == 0:
       score_mean_lis = msss_mean_lis1
       score_arr_lis  = msss_arr_lis1
       score_ukcp_lis = msss_ukcp_lis1
    elif iname == 1:
       score_mean_lis = msss_mean_lis2
       score_arr_lis  = msss_arr_lis2
       score_ukcp_lis = msss_ukcp_lis2
    elif iname == 2:
       score_mean_lis = msss_mean_lis3
       score_arr_lis  = msss_arr_lis3
       score_ukcp_lis = msss_ukcp_lis3

    nbox=len(score_mean_lis)
    
    for ibox in range(nbox):
        colpt  = pcolarr[ibox]
        colbox = bcolarr[ibox]  
      
        score_mean = score_mean_lis[ibox] 
        score_arr  = score_arr_lis[ibox] 
        score_ukcp = score_ukcp_lis[ibox]

        #plo=scipy.stats.mstats.mquantiles(score_arr,prob=0.1,alphap=0.4,betap=0.4)[0]
        #phi=scipy.stats.mstats.mquantiles(score_arr,prob=0.9,alphap=0.4,betap=0.4)[0]

        plo=score_arr.min()
        phi=score_arr.max()

        pos=posa[ibox]
        xxx=[pos-h1, pos+h1, pos+h1, pos-h1, pos-h1]
        yyy=[plo, plo, phi, phi, plo]

        # Plot box, either with Rectangle or with fill_between
        if useRect:
            ax.add_patch(Rectangle((xxx[0], yyy[0]), 2*h1, phi-plo,
                         edgecolor=colpt, facecolor=colbox, fill=True, lw=lwbox, alpha=alpha))
        else:
            plt.fill_between([pos-h1, pos+h1], [plo,plo],[phi,phi], color=colbox, alpha=alpha)

        # Plot box edge
        ax.plot(xxx, yyy, color=colbox, lw=lwbox, solid_capstyle='butt')

        # Plot median
        pmed = numpy.median(score_arr) 
        ax.plot( [pos-h1, pos+h1], [pmed, pmed], color=colbox, linewidth=lwmed, solid_capstyle='butt')
                 
        # Plot points
        np=score_arr.shape[0]
        xp=numpy.ones(np)*pos             
        ax.plot(xp, score_arr, lw=0,marker='o',ms=3.5, color=colbox,alpha=0.8)   #color=colpt,alpha=0.6)   
                 
        # Plot multimodel       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_mean, score_mean], color='red', linewidth=lwmed, solid_capstyle='butt')
      
        # Plot ukcp       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_ukcp, score_ukcp], color='purple', linewidth=lwmed, solid_capstyle='butt')

if ylim[0] < 0.0:
   plt.axhline(0.0, ls=':', lw=0.75, color='k')    
            
ax.set_xticks(xticks)
ax.set_xticklabels(xlabelarr)    
if detrend:
    plt.title('MSSS (detrended)')   
else:
    plt.title('MSSS')   

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




