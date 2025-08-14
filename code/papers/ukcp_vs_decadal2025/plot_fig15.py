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
namefig= 'fig15'
ifig   = 15

dpiarr = [150]

saveplot   = True
#saveplot   = False

detrend=True
detrend=False

if detrend:   ifig = 1500


central = 'median'
#central = 'mean'

ensembles=['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3.1','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']


namearr   = ['Northern Europe_djf_tas',    'Northern Europe_djf_pr',    'Northern Europe_jja_tas',    'Northern Europe_jja_pr',
             'England and Wales_djf_tas',  'England and Wales_djf_pr',  'England and Wales_jja_tas',  'England and Wales_jja_pr']

xlabelarr = ['NEU\nDJF Tair',               'NEU\nDJF Precip',           'NEU\nJJA Tair',               'NEU\nJJA Precip',
             'EngWal\nDJF Tair',            'EngWal\nDJF Precip',        'EngWal\nJJA Tair',            'EngWal\nJJA Precip']

nv        = len(namearr)

fparr=[1, 2, '2to9']

acc_mean_lis1  = []  ; acc_mean_lis2  = []  ; acc_mean_lis3  = []  ; acc_mean_lis4  = []
msss_mean_lis1 = []  ; msss_mean_lis2 = []  ; msss_mean_lis3 = []  ; msss_mean_lis4 = []
acc_arr_lis1   = []  ; acc_arr_lis2   = []  ; acc_arr_lis3   = []  ; acc_arr_lis4   = []
msss_arr_lis1  = []  ; msss_arr_lis2  = []  ; msss_arr_lis3  = []  ; msss_arr_lis4  = []
acc_ukcp_lis1  = []  ; acc_ukcp_lis2  = []  ; acc_ukcp_lis3  = []  ; acc_ukcp_lis4  = []
msss_ukcp_lis1 = []  ; msss_ukcp_lis2 = []  ; msss_ukcp_lis3 = []  ; msss_ukcp_lis4 = []


acc_mean_lis5  = []  ; acc_mean_lis6  = []  ; acc_mean_lis7  = []  ; acc_mean_lis8  = []
msss_mean_lis5 = []  ; msss_mean_lis6 = []  ; msss_mean_lis7 = []  ; msss_mean_lis8 = []
acc_arr_lis5   = []  ; acc_arr_lis6   = []  ; acc_arr_lis7   = []  ; acc_arr_lis8   = []
msss_arr_lis5  = []  ; msss_arr_lis6  = []  ; msss_arr_lis7  = []  ; msss_arr_lis8  = []
acc_ukcp_lis5  = []  ; acc_ukcp_lis6  = []  ; acc_ukcp_lis7  = []  ; acc_ukcp_lis8  = []
msss_ukcp_lis5 = []  ; msss_ukcp_lis6 = []  ; msss_ukcp_lis7 = []  ; msss_ukcp_lis8 = []

for iname,longname in zip(range(nv),namearr):
    for fp in fparr:

        # DPS
        yearmax=2019
        fac=1.0
        dall, dlistok, modlist, syr, nlump = loadutils.loadDPS(longname, dpsdir, fp=fp, ensembles=ensembles,
                                                               fac=fac, yearmax=yearmax)        
        # OBS
        dobs= loadutils.loadOBS(longname, obsdir, syr, nlump) 

        # Need to truncate if dobs smaller than syr, fix: syr,dlistok,dall
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

        if iname == 0:
            acc_mean_lis1.append(acc_mean)   ; acc_arr_lis1.append(acc_arr)   ; acc_ukcp_lis1.append(acc_ukcp)           
            msss_mean_lis1.append(msss_mean) ; msss_arr_lis1.append(msss_arr) ; msss_ukcp_lis1.append(msss_ukcp)
        elif iname == 1:
            acc_mean_lis2.append(acc_mean)   ; acc_arr_lis2.append(acc_arr)   ; acc_ukcp_lis2.append(acc_ukcp)            
            msss_mean_lis2.append(msss_mean) ; msss_arr_lis2.append(msss_arr) ; msss_ukcp_lis2.append(msss_ukcp)
        elif iname == 2:        
            acc_mean_lis3.append(acc_mean)   ; acc_arr_lis3.append(acc_arr)   ; acc_ukcp_lis3.append(acc_ukcp)            
            msss_mean_lis3.append(msss_mean) ; msss_arr_lis3.append(msss_arr) ; msss_ukcp_lis3.append(msss_ukcp)
        elif iname == 3:
            acc_mean_lis4.append(acc_mean)   ; acc_arr_lis4.append(acc_arr)   ; acc_ukcp_lis4.append(acc_ukcp)            
            msss_mean_lis4.append(msss_mean) ; msss_arr_lis4.append(msss_arr) ; msss_ukcp_lis4.append(msss_ukcp)
        elif iname == 4:
            acc_mean_lis5.append(acc_mean)   ; acc_arr_lis5.append(acc_arr)   ; acc_ukcp_lis5.append(acc_ukcp)            
            msss_mean_lis5.append(msss_mean) ; msss_arr_lis5.append(msss_arr) ; msss_ukcp_lis5.append(msss_ukcp)
        elif iname == 5:
            acc_mean_lis6.append(acc_mean)   ; acc_arr_lis6.append(acc_arr)   ; acc_ukcp_lis6.append(acc_ukcp)            
            msss_mean_lis6.append(msss_mean) ; msss_arr_lis6.append(msss_arr) ; msss_ukcp_lis6.append(msss_ukcp)
        elif iname == 6:
            acc_mean_lis7.append(acc_mean)   ; acc_arr_lis7.append(acc_arr)   ; acc_ukcp_lis7.append(acc_ukcp)            
            msss_mean_lis7.append(msss_mean) ; msss_arr_lis7.append(msss_arr) ; msss_ukcp_lis7.append(msss_ukcp)
        elif iname == 7:
            acc_mean_lis8.append(acc_mean)   ; acc_arr_lis8.append(acc_arr)   ; acc_ukcp_lis8.append(acc_ukcp)            
            msss_mean_lis8.append(msss_mean) ; msss_arr_lis8.append(msss_arr) ; msss_ukcp_lis8.append(msss_ukcp)


##############################
# Boxplot 

lw=2.0
ms=5
fs=9.5
fstxt=9.0
alpha=0.2
sscat=40
lwbox=0.5
lwmed=2.0

matplotlib.rcParams.update({'font.size': fs})
matplotlib.rcParams.update({'legend.numpoints': 1})

fig = plt.figure(ifig, figsize=(16.5/2.54, 17./2.54) )
plt.subplots_adjust(hspace=0.27,wspace=0.24,top=0.96,bottom=0.07,left=0.09,right=0.98)

xrng0= 0.0
xrng1= 3.0
R    = xrng1-xrng0
xlimfix = [xrng0, xrng1]

fac = 0.12   #(d/D)
D   = R/(fac+nv*(fac+1))
d   = fac*D

dxfac  = 0.27
width  = 0.063
facline= 0.14

h1     = width/2.
#xticks = [ d+D*1/2, 2*d+D*3/2, 3*d+D*5/2]
xticks = [] 
for iv in range(1,nv+1):
   xticks.append( iv*d + D*(2*iv-1)/2 )


pcolarr=['darkorange', 'green',     'blue']
bcolarr=['orange',     'limegreen', 'dodgerblue']

alpha=0.15
useRect=True

### ACC plot
ax=plt.subplot(2,1,1)
if detrend:
    ylim=[-0.41, 1.01]
else:
    ylim=[-0.69, 1.09]

ax.set_xlim(xlimfix)
ax.set_ylim(ylim)

for iname,longname,xlabel in zip(range(nv),namearr, xlabelarr):

    pos1 = xticks[iname]
    pos0 = pos1-dxfac*D
    pos2 = pos1+dxfac*D
    posa = [pos0,pos1,pos2]

    if iname == 0:
       score_mean_lis=acc_mean_lis1 ; score_arr_lis =acc_arr_lis1 ; score_ukcp_lis=acc_ukcp_lis1
    elif iname == 1:
       score_mean_lis=acc_mean_lis2 ; score_arr_lis =acc_arr_lis2 ; score_ukcp_lis=acc_ukcp_lis2
    elif iname == 2:
       score_mean_lis=acc_mean_lis3 ; score_arr_lis =acc_arr_lis3 ; score_ukcp_lis=acc_ukcp_lis3
    elif iname == 3:
       score_mean_lis=acc_mean_lis4 ; score_arr_lis =acc_arr_lis4 ; score_ukcp_lis=acc_ukcp_lis4
    elif iname == 4:
       score_mean_lis=acc_mean_lis5 ; score_arr_lis =acc_arr_lis5 ; score_ukcp_lis=acc_ukcp_lis5
    elif iname == 5:
       score_mean_lis=acc_mean_lis6 ; score_arr_lis =acc_arr_lis6 ; score_ukcp_lis=acc_ukcp_lis6
    elif iname == 6:
       score_mean_lis=acc_mean_lis7 ; score_arr_lis =acc_arr_lis7 ; score_ukcp_lis=acc_ukcp_lis7
    elif iname == 7:
       score_mean_lis=acc_mean_lis8 ; score_arr_lis =acc_arr_lis8 ; score_ukcp_lis=acc_ukcp_lis8

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
        ax.plot(xp, score_arr, lw=0,marker='o',ms=3.0, color=colbox,alpha=0.8)   #color=colpt,alpha=0.6)   
                 
        # Plot multimodel       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_mean, score_mean], color='red', linewidth=lwmed, solid_capstyle='butt')
      
        # Plot ukcp       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_ukcp, score_ukcp], color='purple', linewidth=lwmed, solid_capstyle='butt')

if ylim[0] < 0.0:
   plt.axhline(0.0, ls=':', lw=0.75, color='k')    
           
ax.set_xticks(xticks)
ax.set_xticklabels(xlabelarr, fontsize=fs-1)    
if detrend:
    plt.title('ACC (detrended)',pad=2.0)   
else:
    plt.title('ACC',pad=2.0)   

xlims=ax.get_xlim()
ylims=ax.get_ylim()
xleg=xlims[0]+0.01*(xlims[1]-xlims[0])
dxtxt=0.045*(xlims[1]-xlims[0])
dxbox=0.035*(xlims[1]-xlims[0])
dybox=0.032*(ylims[1]-ylims[0])
if detrend:
    yfirst=0.94
else:
    yfirst=0.22
dy=0.05


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
yyy=[yleg+0.5*dy, yleg+0.5*dy]
ax.plot(xxx, yyy, color='red', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'MMDPE',fontsize=fstxt) 

yleg=ylims[0]+(yfirst-4*dy)*(ylims[1]-ylims[0])
xxx=[xleg, xleg+dxbox]
yyy=[yleg+0.5*dy, yleg+0.5*dy]
ax.plot(xxx, yyy, color='purple', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'UKCP-pdf',fontsize=fstxt) 
 
           


###############
# MSSS plot
ax=plt.subplot(2,1,2)
if detrend:
    ylim=[-1.82,  1.03]
else:
    ylim=[-0.69, 1.09]


ax.set_xlim(xlimfix)
ax.set_ylim(ylim)

for iname,longname,xlabel in zip(range(nv),namearr, xlabelarr):

    pos1 = xticks[iname]
    pos0 = pos1-dxfac*D
    pos2 = pos1+dxfac*D
    posa = [pos0,pos1,pos2]

    if iname == 0:
       score_mean_lis=msss_mean_lis1 ; score_arr_lis =msss_arr_lis1 ; score_ukcp_lis=msss_ukcp_lis1
    elif iname == 1:
       score_mean_lis=msss_mean_lis2 ; score_arr_lis =msss_arr_lis2 ; score_ukcp_lis=msss_ukcp_lis2
    elif iname == 2:
       score_mean_lis=msss_mean_lis3 ; score_arr_lis =msss_arr_lis3 ; score_ukcp_lis=msss_ukcp_lis3
    elif iname == 3:
       score_mean_lis=msss_mean_lis4 ; score_arr_lis =msss_arr_lis4 ; score_ukcp_lis=msss_ukcp_lis4
    elif iname == 4:
       score_mean_lis=msss_mean_lis5 ; score_arr_lis =msss_arr_lis5 ; score_ukcp_lis=msss_ukcp_lis5
    elif iname == 5:
       score_mean_lis=msss_mean_lis6 ; score_arr_lis =msss_arr_lis6 ; score_ukcp_lis=msss_ukcp_lis6
    elif iname == 6:
       score_mean_lis=msss_mean_lis7 ; score_arr_lis =msss_arr_lis7 ; score_ukcp_lis=msss_ukcp_lis7
    elif iname == 7:
       score_mean_lis=msss_mean_lis8 ; score_arr_lis =msss_arr_lis8 ; score_ukcp_lis=msss_ukcp_lis8

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
        ax.plot(xp, score_arr, lw=0,marker='o',ms=3.0, color=colbox,alpha=0.8)   #color=colpt,alpha=0.6)   
                 
        # Plot multimodel       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_mean, score_mean], color='red', linewidth=lwmed, solid_capstyle='butt')
      
        # Plot ukcp       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_ukcp, score_ukcp], color='purple', linewidth=lwmed, solid_capstyle='butt')

if ylim[0] < 0.0:
   plt.axhline(0.0, ls=':', lw=0.75, color='k')    
            
ax.set_xticks(xticks)
ax.set_xticklabels(xlabelarr, fontsize=fs-1)    
if detrend:
    plt.title('MSSS (detrended)',pad=2.0)   
else:   
    plt.title('MSSS',pad=2.0)   


xlims=ax.get_xlim()
ylims=ax.get_ylim()
xleg=xlims[0]+0.01*(xlims[1]-xlims[0])
dxtxt=0.045*(xlims[1]-xlims[0])
dxbox=0.035*(xlims[1]-xlims[0])
dybox=0.032*(ylims[1]-ylims[0])
yfirst=0.94
dy=0.05

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
yyy=[yleg+0.5*dy, yleg+0.5*dy]
ax.plot(xxx, yyy, color='red', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'MMDPE',fontsize=fstxt) 

yleg=ylims[0]+(yfirst-4*dy)*(ylims[1]-ylims[0])
xxx=[xleg, xleg+dxbox]
yyy=[yleg+0.5*dy, yleg+0.5*dy]
ax.plot(xxx, yyy, color='purple', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'UKCP-pdf',fontsize=fstxt) 
 
 
for dpi in dpiarr:           
    cdpi  = str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)

