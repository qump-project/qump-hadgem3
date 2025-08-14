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
namefig= 'fig19'
ifig   = 19

dpiarr = [150]

saveplot   = True
#saveplot   = False

ensembles=['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3.1','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']

namearr   = ['Northern Europe_djf_tas',    'Northern Europe_djf_pr',    'Northern Europe_jja_tas',    'Northern Europe_jja_pr',
             'England and Wales_djf_tas',  'England and Wales_djf_pr',  'England and Wales_jja_tas',  'England and Wales_jja_pr']

xlabelarr = ['NEU\nDJF Tair',               'NEU\nDJF Precip',           'NEU\nJJA Tair',               'NEU\nJJA Precip',
             'EngWal\nDJF Tair',            'EngWal\nDJF Precip',        'EngWal\nJJA Tair',            'EngWal\nJJA Precip']

nv        = len(namearr)

fparr=[1, 2, '2to9']

prob=[0.1, 0.9]    # look at fraction of observed events in this range


maketrendplot  = False
trendplotnames =  namearr

freq_mean_lis1 = []  ; freq_mean_lis2 = []  ; freq_mean_lis3 = []  ; freq_mean_lis4 = []
freq_arr_lis1  = []  ; freq_arr_lis2  = []  ; freq_arr_lis3  = []  ; freq_arr_lis4  = []
freq_ukcp_lis1 = []  ; freq_ukcp_lis2 = []  ; freq_ukcp_lis3 = []  ; freq_ukcp_lis4 = []

freq_mean_lis5 = []  ; freq_mean_lis6 = []  ; freq_mean_lis7 = []  ; freq_mean_lis8 = [] 
freq_arr_lis5  = []  ; freq_arr_lis6  = []  ; freq_arr_lis7  = []  ; freq_arr_lis8  = [] 
freq_ukcp_lis5 = []  ; freq_ukcp_lis6 = []  ; freq_ukcp_lis7 = []  ; freq_ukcp_lis8 = [] 

trendarr = []
for iname,longname in zip(range(nv),namearr):
    for fp in fparr:

        # DPS
        yearmax=2019
        fac=1.0
        dall, dlistok, modlist, syr, nlump = loadutils.loadDPS(longname, dpsdir, fp=fp, ensembles=ensembles, 
                                                               fac=fac, yearmax=yearmax)        

        # OBS
        dobs = loadutils.loadOBS(longname, obsdir, syr, nlump) 
        # Need to truncate if dobs smaller than syr, fix: syr,dlistok,dall
        syr,dall,dlistok= loadutils.trimdata(dobs, syr,dall,dlistok)
        # If dobs longer, truncate that instead
        if dobs.shape[0] > dall.shape[0]:
            dobs = dobs[:dall.shape[0]]
                 
        # UKCP
        if fp == '2to9':
            ukwil = 'ukwil=31-20-8'
        else:
            ukwil = 'ukwil=31-20-1'        
        qukcplo  = loadutils.loadUKCP(longname, ukcpdir, syr, nlump, prob=prob[0], ukwil=ukwil)
        qukcphi  = loadutils.loadUKCP(longname, ukcpdir, syr, nlump, prob=prob[1], ukwil=ukwil)        
        qukcp50  = loadutils.loadUKCP(longname, ukcpdir, syr, nlump, prob=0.5, ukwil=ukwil)

        freq_ukcp= utils.freqratio(dobs, qukcplo, qukcphi)        
        freq_mean= utils.rangefreq(dall, dobs, prob=prob, ab=0.4)

        freq_arr=[]
        for data in dlistok:
           freq1 = utils.rangefreq(data, dobs, prob=[0.1,0.9], ab=0.4)
           freq_arr.append(freq1)
        freq_arr=numpy.array(freq_arr)  
           

        # Linear trends, only for 2to9 data.
        if fp == '2to9':
            if '_pr' in longname:
                unit='%'
            else:
                unit='degC'                        

            syr2    = numpy.array([syr.min(), syr.max()])
            dallmed = numpy.median(dall,axis=1)    
                                
            nsamp    = 5000
            blocksize= 5   
            seed     = 101
            pcrit    = 0.025  # 0.025 will return [0.025, 0.975] quantiles (ie 95% range) for obs trend.
                        
            # block resample residuals (brr) to estimate trend uncertainty, for obs only though.

            # Hindcast Data 
            d_slope, d_intercept, d_cilo, d_cihi = utils.trend_uncert_brr(syr, dallmed, prob=pcrit, resamp=False, nsamp=nsamp, blocksize=blocksize, seed=seed)
            d_med2 = d_slope*syr2 + d_intercept
            b_dps  = 10*d_slope
  
            # UKCP Data 
            u_slope, u_intercept, u_cilo, u_cihi = utils.trend_uncert_brr(syr, qukcp50, prob=pcrit, resamp=False, nsamp=nsamp, blocksize=blocksize, seed=seed)
            u_med2 = u_slope*syr2 + u_intercept
            b_ukp  = 10*u_slope

            # Obs Data 
            o_slope, o_intercept, o_cilo, o_cihi = utils.trend_uncert_brr(syr, dobs, prob=pcrit, resamp=True, nsamp=nsamp, blocksize=blocksize, seed=seed)
            o_med2 = o_slope*syr2+ o_intercept
            b_obs  = 10*o_slope
            o_rng  = '(%5.3f'%(10*o_cilo)+', %5.3f'%(10*o_cihi)+')'
 
            out = numpy.array([b_dps, b_ukp, b_obs, 10*o_cilo, 10*o_cihi])
            trendarr.append(out)
            
            if  maketrendplot and longname in trendplotnames:
                plt.figure(2000+iname, figsize=(16/2.54, 12/2.54))
                plt.plot(syr, dobs, lw=0.75, ls=':', marker='o', ms=3.0, color='black',alpha=0.6)
                plt.plot(syr2, o_med2, lw=1.75, ls='-', color='black',label='Obs, slope=%5.3f'%b_obs+' '+unit+'/decade '+o_rng)

                plt.plot(syr, qukcp50, lw=0.75, ls=':', marker='o', ms=3.0, color='blue',alpha=0.6)
                plt.plot(syr2, u_med2, lw=1.75, ls='-', color='blue',label='UKCP, slope=%5.3f'%b_ukp+' '+unit+'/decade ')

                plt.plot(syr, dallmed, lw=0.75, ls=':', marker='o', ms=3.0, color='red',alpha=0.6)
                plt.plot(syr2, d_med2, lw=1.75, ls='-', color='red',label='MMDPE, slope=%5.3f'%b_dps+' '+unit+'/decade')

                cprob = str(int(100*(1.-2*pcrit) ))
                nnn=xlabelarr[iname].replace('\n', ' ')
                nnn=nnn.replace('EngWal', 'Eng-Wal ')
                tit = xlabelarr[iname] +', Obs Uncert: resample residuals\n('+cprob+'% range, blocksize='+str(blocksize)+', nsamp='+str(nsamp)+')'                                
                plt.title(tit,fontsize=10.5)
                leg=plt.legend(loc='best',fontsize=9.0,handlelength=1.5,borderaxespad=0.4,handletextpad=0.4,labelspacing=0.4)    
                leg.draw_frame(True)

           
        if iname == 0:
            freq_mean_lis1.append(freq_mean)   ; freq_arr_lis1.append(freq_arr)   ; freq_ukcp_lis1.append(freq_ukcp)
        elif iname == 1:
            freq_mean_lis2.append(freq_mean)   ; freq_arr_lis2.append(freq_arr)   ; freq_ukcp_lis2.append(freq_ukcp)    
        elif iname == 2:       
            freq_mean_lis3.append(freq_mean)   ; freq_arr_lis3.append(freq_arr)   ; freq_ukcp_lis3.append(freq_ukcp)
        elif iname == 3:
            freq_mean_lis4.append(freq_mean)   ; freq_arr_lis4.append(freq_arr)   ; freq_ukcp_lis4.append(freq_ukcp)
        elif iname == 4:
            freq_mean_lis5.append(freq_mean)   ; freq_arr_lis5.append(freq_arr)   ; freq_ukcp_lis5.append(freq_ukcp)
        elif iname == 5:
            freq_mean_lis6.append(freq_mean)   ; freq_arr_lis6.append(freq_arr)   ; freq_ukcp_lis6.append(freq_ukcp)
        elif iname == 6:
            freq_mean_lis7.append(freq_mean)   ; freq_arr_lis7.append(freq_arr)   ; freq_ukcp_lis7.append(freq_ukcp)
        elif iname == 7:
            freq_mean_lis8.append(freq_mean)   ; freq_arr_lis8.append(freq_arr)   ; freq_ukcp_lis8.append(freq_ukcp)

trendarr=numpy.array(trendarr)   # shape= (namearr.shape[0], 5) 


cprob = str(int(100*(1.-2*pcrit) ))
cccc  = 'prob='+cprob+'_blocksize='+str(blocksize)+'_nsamp='+str(nsamp)+'_seed='+str(seed)
trendfile = os.path.join(plotdir, 'trends_2to9_neu_eaw_'+cccc+'.txt')

if saveplot:
    f1 = open(trendfile, 'w')
    head = '                                     Dedadal Trends (per decade)'
    f1.write(head +'\n')
    head = 'VARIABLE                          MMDPE     UKCP      OBS   Obs_95%_Conf.Int'
    f1.write(head +'\n')
    for ii in range(trendarr.shape[0]):     
        linetuple = (namearr[ii], trendarr[ii,0], trendarr[ii,1], trendarr[ii,2], trendarr[ii,3], trendarr[ii,4])
        format = '%-30s' + ''.join( ['%9.4f']*(len(linetuple)-1) )     
        line1 = format%linetuple
        f1.write(line1 +'\n')
    f1.close()
    print('Written trendfile: ',trendfile)


##############################
# FREQ Boxplot 

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

fig = plt.figure(ifig, figsize=(16.5/2.54, 9/2.54) )
plt.subplots_adjust(hspace=0.25,wspace=0.25,top=0.93,bottom=0.12,left=0.07,right=0.985)

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
xticks = [] 
for iv in range(1,nv+1):
   xticks.append( iv*d + D*(2*iv-1)/2 )


pcolarr=['darkorange', 'green',     'blue']
bcolarr=['orange',     'limegreen', 'dodgerblue']

alpha=0.15
useRect=True

ax=plt.subplot(1,1,1)
ylim=[0.37, 1.03]

ax.set_xlim(xlimfix)
ax.set_ylim(ylim)

for iname,longname,xlabel in zip(range(nv),namearr, xlabelarr):

    pos1 = xticks[iname]
    pos0 = pos1-dxfac*D
    pos2 = pos1+dxfac*D
    posa = [pos0,pos1,pos2]

    if iname == 0:
       score_mean_lis=freq_mean_lis1 ; score_arr_lis =freq_arr_lis1 ; score_ukcp_lis=freq_ukcp_lis1
    elif iname == 1:
       score_mean_lis=freq_mean_lis2 ; score_arr_lis =freq_arr_lis2 ; score_ukcp_lis=freq_ukcp_lis2
    elif iname == 2:
       score_mean_lis=freq_mean_lis3 ; score_arr_lis =freq_arr_lis3 ; score_ukcp_lis=freq_ukcp_lis3
    elif iname == 3:
       score_mean_lis=freq_mean_lis4 ; score_arr_lis =freq_arr_lis4 ; score_ukcp_lis=freq_ukcp_lis4
    elif iname == 4:
       score_mean_lis=freq_mean_lis5 ; score_arr_lis =freq_arr_lis5 ; score_ukcp_lis=freq_ukcp_lis5
    elif iname == 5:
       score_mean_lis=freq_mean_lis6 ; score_arr_lis =freq_arr_lis6 ; score_ukcp_lis=freq_ukcp_lis6
    elif iname == 6:
       score_mean_lis=freq_mean_lis7 ; score_arr_lis =freq_arr_lis7 ; score_ukcp_lis=freq_ukcp_lis7
    elif iname == 7:
       score_mean_lis=freq_mean_lis8 ; score_arr_lis =freq_arr_lis8 ; score_ukcp_lis=freq_ukcp_lis8

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
        ax.plot(xp, score_arr, lw=0,marker='o',ms=3.0, color=colbox,alpha=0.6)   #color=colpt,alpha=0.6)   
                 
        # Plot multimodel       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_mean, score_mean], color='red', linewidth=lwmed, solid_capstyle='butt')
      
        # Plot ukcp       
        dh = facline*2*h1
        ax.plot( [pos-h1-dh, pos+h1+dh], [score_ukcp, score_ukcp], color='purple', linewidth=lwmed, solid_capstyle='butt')


plt.axhline(prob[1]-prob[0], ls=':', lw=0.75, color='k')    
           
ax.set_xticks(xticks)
ax.set_xticklabels(xlabelarr, fontsize=fs-1)    
plt.title('Fraction of Observed Events in 10-90% Hindcast Range',pad=2.0)   

xlims=ax.get_xlim()
ylims=ax.get_ylim()
xleg=xlims[0]+0.01*(xlims[1]-xlims[0])
dxtxt=0.036*(xlims[1]-xlims[0])
dxbox=0.028*(xlims[1]-xlims[0])
dybox=0.030*(ylims[1]-ylims[0])
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
yyy=[yleg+0.25*dy, yleg+0.25*dy]
ax.plot(xxx, yyy, color='red', lw=lwmed) 
plt.text(xleg+dxtxt, yleg, 'MMDPE',fontsize=fstxt) 

yleg=ylims[0]+(yfirst-4*dy)*(ylims[1]-ylims[0])
xxx=[xleg, xleg+dxbox]
yyy=[yleg+0.25*dy, yleg+0.25*dy]
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


