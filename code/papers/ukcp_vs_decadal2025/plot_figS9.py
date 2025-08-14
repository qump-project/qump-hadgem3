import os
import numpy
import scipy
import scipy.stats.mstats
import pickle
import copy
import datetime
import matplotlib
import matplotlib.pyplot as plt
import datetime

import iris

import loadUtils as loadutils
import qdcUtils as utils

from pathnames_v1 import *

##################################################   
namefig= 'figS9'
ifig   = 1009

dpiarr = [150]

saveplot   = True
#saveplot   = False

version = 2

models=['BSC', 'CAFE', 'CMCC', 'CanESM5','Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NCAR40', 'NorCPM', 'NorCPMi2']

coldic= utils.setcolours()
mdic  = utils.setmodelnames()

# First n2plot plotted, 1500 => plot all
# If less than full amount, will plot a random sample of size n2plot (see below)
#n2plot=500   ; alphagrey=0.3
n2plot=1500  ; alphagrey=0.15


namearr = ['Globalukcp09_djfmamjjason_tas', 'amo_djfmamjjason_tas', 'nao_stephenson_djf_psl'] 
fparr   = [1, 1, 1]
ylabels = ['Anomaly ($\degree$C)', 'Anomaly ($\degree$C)', 'Anomaly (hPa)']
nmaxa   = [58, 58, 58]

nvar=len(namearr)

seeddic={1:1,    2:2,    '2to9':9,      10:10}
titdic ={1:'T1', 1:'T2', '2to9':'T2-9', 10:'T10'}

nfac=100
cfac=str(nfac)

fs=8.5
alphar=0.30
fstit=10.0
fsleg1=8.5
fsleg2=8.0


matplotlib.rcParams.update({'font.size': fs})
matplotlib.rcParams.update({'legend.numpoints': 1})

if nvar == 1:
    fig = plt.figure(ifig, figsize=(22/2.54, 18/2.54) )
else:
    fig = plt.figure(ifig, figsize=(12.5/2.54, 18/2.54) )

plt.subplots_adjust(hspace=0.25,wspace=0.25,top=0.965,bottom=0.04,left=0.15,right=0.96)

for iname,longname,fp,ylabel,nmx in zip(range(5),namearr, fparr,ylabels,nmaxa):
    datamed=[]
    ndata=[]
    modellist=[]
    longname1=longname.replace('_stephenson','-stephenson')
    name,ssn,var = longname1.split('_')
    first=True
    for imod,model in enumerate(models):               
        # DPS
        yearmax=2019
        fac=1.0
        if 'nao' in name:   fac=1./100.         
        data1, dlistok, modlist, syr, nlump = loadutils.loadDPS(longname, dpsdir, fp=fp, ensembles=[model],
                                                                fac=fac, yearmax=yearmax, verbose=False)        
        
        # OBS
        if not data1 is None:
            dobs = loadutils.loadOBS(longname, obsdir, syr, nlump)             
            # Need to truncate if dobs smaller than syr, fix: syr,dlistok,dall
            syr,data1,dlistok= loadutils.trimdata(dobs, syr, data1, dlistok)       
            # If dobs longer, truncate that instead
            if dobs.shape[0] > data1.shape[0]:
                dobs = dobs[:data1.shape[0]]

        if data1 is None:
            pass
        else:
            print('>>> ',model,'data1.shape=',data1.shape)
            
            # Now have data1.shape=(nsyr, nrlz), and syr,rlz
            nsyr=data1.shape[0] 
            rlz=data1.shape[1] 

            loc  = numpy.median(data1,axis=1)
            scale= numpy.std(data1,axis=1,ddof=1)
            mscale=scale.mean()           

            ntime= data1.shape[0]
            nrlz = data1.shape[1]
            nsamp = nfac*nrlz
            iseed=1000*imod+100*iname+seeddic[fp]            
            numpy.random.seed(iseed)
            datar = numpy.random.normal(loc=loc, scale=scale, size=(nsamp, ntime) ).T

            # datar has shape (ntime,nsamp) similar to original (ntime,nrlz)
            datarmed=numpy.median(datar,axis=1)
            data1med=numpy.median(data1,axis=1)

            #nrlz0 = nrlz  #had this before
            nrlz0 = 10
            nsub  = nsamp//nrlz0 

            # nsub equals nfac if nrlz=10 (nfac=100 typically).
            # nsub equals 2*nfac with nrlz=20  (200 with nfac=100).
            # nsub equals 4*nfac with nrlz=40  (400 with nfac=100).
             
            # Finally decide to fix at 10 members for ALL models, incl CMCC and NCAR.
            # This means for NCAR (40 mem) we have 400 medians from 10 member samples.
            # For CMCC (20 mem) we have 200 medians from 10 member samples.
            # For all others (with 10 members) we have 100 (nfac) medians of 10 member samples.
            # Previously we had equiv of nrlz0=nrlz, so nsub would ALWAYS equal 100, but
            # for NCAR for example we would have had medians of 40 member resamples.

            # Recall datar is Gaussian resamp with shape (ntim, 100*nrlz)
            #  - Subset these into sets of 10 (nrlz0) => 100 subsets (200,400 for cmcc,ncar).
            #  - Find the median dsubmed of each subset (func of time only).
            #  - Estimate dif - the difference between the subset median and datarmed (the median 
            #    over all nsamp=1000, 2000, 4000). Note datarmed very similar to actual data1med.                                             
 
            dif=[]
            med=[]
            for j in range(nsub):
                dsubmed=numpy.median(datar[:,nrlz0*j:nrlz0*(j+1)],axis=1)
                dif.append(dsubmed-datarmed)
                med.append(dsubmed)
                    
            dif=numpy.array(dif).T    #shape=(ntim,nsub), eg (55,100), (55,200), (55,400)
            med=numpy.array(med).T    #shape=(ntim,nsub)

            datamed.append(data1med[:nmx])
            ndata.append(data1.shape[1])
            modellist.append(modlist[0])          
            if first:
                first=False
                dataall = data1[:nmx,:].copy()                
                datarall= datar[:nmx,:].copy()
                difall  = dif[:nmx,:].copy()
            else:
                dataall = numpy.ma.concatenate((dataall[:nmx,:], data1[:nmx,:]),   axis=1)
                datarall= numpy.ma.concatenate((datarall[:nmx,:],datar[:nmx,:]),   axis=1)
                difall  = numpy.ma.concatenate((difall[:nmx,:],  dif[:nmx,:]),     axis=1)
  
            #print(model, dif[:nmx,:].shape, dataall.shape, datarall.shape, difall.shape)

    datamed=numpy.array(datamed).T
    ndata=numpy.array(ndata)

    prob=[0.1, 0.5, 0.9]
    rrr = scipy.stats.mstats.mquantiles(datarall,prob=prob,axis=1)

    mmm = numpy.expand_dims(rrr[:nmx,1],-1) + difall
    mmq = scipy.stats.mstats.mquantiles(mmm,prob=prob,axis=1) 
    mmd = scipy.stats.mstats.mquantiles(difall,prob=prob,axis=1)   

    noutsum=0.0
    ntotsum=0.0
    for imod in range(datamed.shape[1]):
        nout, ntot= utils.noutside(datamed[:,imod], mmq[:nmx,0], mmq[:nmx,2])
        noutsum = noutsum + nout
        ntotsum = ntotsum + ntot
    freqout=noutsum/ntotsum  

    ### SUBPLOT
    
    ax=plt.subplot(nvar,1,iname+1)

    coldark='k'
    showobs=False
    lwp   =1.25
    alphap=0.85
    collite='darkgrey'

    tsyr = utils.timefromsy(syr[:nmx], ssn, nlump)

    if version == 1:                   
        if n2plot < mmm.shape[1]:
            # If fewer than the full number of realizations is to be plotted, plot a randomn sample.
            numpy.random.seed(123)
            i2plot = numpy.trunc( mmm.shape[1]*numpy.random.uniform(size=n2plot) ).astype(int)
            plt.plot(tsyr, mmm[:,i2plot], color=collite, alpha=alphagrey, lw=0.25)
        else:
            plt.plot(tsyr, mmm, color=collite, alpha=alphagrey, lw=0.25)
    elif version == 2: 
        facecolor='grey'
        ax.fill_between(tsyr, mmq[:,0], mmq[:,2], facecolor=facecolor,alpha=alphar)
   
    for icol in range(datamed.shape[1]):
        plt.plot(tsyr,datamed[:,icol],color=coldic[models[icol]],alpha=0.9,lw=1.0)

    if version == 1:
        plt.plot(tsyr,mmq[:nmx,:],color=coldark,alpha=alphap,lw=lwp)
        lab2='Resample: P10,P50,P90'
    else:
        plt.plot(tsyr,mmq[:nmx,[0,2]],color=coldark,alpha=alphap,lw=lwp)
        lab2='Resample: P10,P90'    

    print('mmq[:,2]-mmq[:,0]=',mmq[:,2]-mmq[:,0])
    print('Mean mmq[:,2]-mmq[:,0]=',numpy.mean(mmq[:,2]-mmq[:,0]))

    if showobs:
        plt.plot(tsyr,dobs[:nmx],color='k',alpha=1.0,lw=1.5)

    if iname == 0:
        tit='GMST, T1'
        ylim=[-0.55, 0.87]
        xlim=[1960,  2019]
        loc='upper left' 
    elif iname == 1:
        tit='AMV, T1'
        ylim=[-0.31, 0.50]
        xlim=[1960,  2019]
        loc='upper center'
    elif iname == 2:
        tit='NAO, T1'
        ylim=[-6.5, 8.0]
        xlim=[1960, 2019]
        loc='upper left'
        
    cfreqout=', Freq. outside P90-P10 range: %5.2f'%freqout
    plt.title(tit+cfreqout, fontsize=fstit,pad=3.0)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    
    ll = []
    m4col='MPI'
    labels = []
    if version == 1:
        ll.append( matplotlib.lines.Line2D([], [], color=collite,      lw=0.75, alpha=1.0) )
        ll.append( matplotlib.lines.Line2D([], [], color=coldark,      lw=lwp,  alpha=alphap) )
        ll.append( matplotlib.lines.Line2D([], [], color=coldic[m4col],lw=1.0,  alpha=1.0) )
        labels= ['Resample Gaussian subsets', lab2,
                 'Colours: individual medians']     #individual DPS medians
        if showobs:
            ll.append( matplotlib.lines.Line2D([], [], color='k',      lw=1.5,  alpha=1.0) )
            labels.append('Obs')
    else:
        facecolor='grey'
        edgecolor=coldark
        fc_for_rectangle = matplotlib.colors.ColorConverter().to_rgba(facecolor, alpha=alphar)
        handle_plume     = plt.Rectangle( (0, 0), 0, 0, edgecolor=edgecolor, fc=fc_for_rectangle, lw=1.0)
        ll.append(handle_plume)
        labels.append(lab2)
        ll.append( matplotlib.lines.Line2D([], [], color=coldic[m4col],lw=1.0,  alpha=1.0) )
        labels.append('Colours: individual medians')

    plt.ylabel(ylabel)
    leg1 = plt.legend(ll, labels, loc=loc, fontsize=fsleg1, handlelength=1.0,borderaxespad=0.05,handletextpad=0.25,labelspacing=0.20,frameon=False)
    if iname == 0:
        plt.gca().add_artist(leg1)
        ll2 = []
        labels2=[]
        for icol in range(datamed.shape[1]):
            col=coldic[models[icol]]
            mname=mdic[models[icol]]
            ll2.append( matplotlib.lines.Line2D([], [], color=col, lw=1.4, alpha=1.0) )
            labels2.append(mname)            
        leg2 = plt.legend(ll2, labels2, loc='lower right', fontsize=fsleg2, frameon=False, ncol=2, columnspacing=0.6,
                          handlelength=1.0, borderaxespad=0.05, handletextpad=0.25, labelspacing=0.15)


for dpi in dpiarr:           
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
