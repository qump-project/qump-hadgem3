import os
import numpy
import scipy
import scipy.stats.mstats
import pickle
import copy
import datetime
import matplotlib
import matplotlib.pyplot as plt

import iris

import loadUtils as loadutils
import qdcUtils as utils

from pathnames_v1 import *
 
##############################
namefig= 'fig20'
ifig   = 20

dpiarr = [150]

saveplot   = True
#saveplot   = False

percentiles = [10, 50, 90]

lastyr = 2018

models=['BSC', 'CAFE', 'CMCC', 'CanESM5','Depresys4_gc3.1', 'IPSL', 'MPI', 'MIROC6', 'NCAR40', 'NorCPM', 'NorCPMi2']

coldic= utils.setcolours()
mdic  = utils.setmodelnames()

namearr = ['Northern Europe_jja_tas', 'Northern Europe_djf_pr'] 
fparr   = [1, '2to9']
ylabels = ['Anomaly ($\degree$C)', 'Anomaly (%)']
nmaxa   = [57, 55]   

seeddic={1: 1,    '2to9': 2}
titdic ={1: 'T1', '2to9':'T2-9'}

showuk=True
showuk=False

showpoints=True

nfac=100
cfac=str(nfac)


obscol='black'
obslw='2.0'

ukcpcol='blue'
ukcplw='0.75'
ukcplwmed='2.0'

dpscol='red'
dpslw='1.0'
dpslwmed='2.0'

fsleg=8.25
legframe=True

##############################
fs=9.0
matplotlib.rcParams.update({'font.size': fs})
matplotlib.rcParams.update({'legend.numpoints': 1})

if showpoints:
    fig = plt.figure(ifig, figsize=(22/2.54, 10.5/2.54) )
    plt.subplots_adjust(hspace=0.25,wspace=0.19,top=0.92,bottom=0.16,left=0.065,right=0.985)
else:
    fig = plt.figure(ifig, figsize=(18/2.54, 8.5/2.54) )
    plt.subplots_adjust(hspace=0.25,wspace=0.25,top=0.92,bottom=0.085,left=0.095,right=0.98)

for iname,longname,fp,ylabel,nmx in zip(range(2),namearr, fparr,ylabels,nmaxa):
    datamed=[]
    ndata=[]
    modellist=[]
    nsubarr=[]
    name,ssn,var = longname.split('_')
    first=True
    for imod,model in enumerate(models):
               
        # DPS
        yearmax=2019
        fac=1.0
        data1, dlistok, modlist, syr, nlump = loadutils.loadDPS(longname, dpsdir, fp=fp, ensembles=[model], 
                                                                fac=fac, yearmax=yearmax, verbose=False)                     

        # OBS
        dobs = loadutils.loadOBS(longname, obsdir, syr, nlump)        
        # Need to truncate if dobs smaller than syr, fix: syr,dlistok,dall
        syr,data1,dlistok= loadutils.trimdata(dobs, syr, data1, dlistok)
        if dobs.shape[0] > data1.shape[0]:
            dobs = dobs[:data1.shape[0]]
     
        # UKCP
        if fp == '2to9':
            ukwil = 'ukwil=31-20-8'
        else:
            ukwil = 'ukwil=31-20-1'                    
        qukcplo  = loadutils.loadUKCP(longname, ukcpdir, syr, nlump, prob=0.1, ukwil=ukwil)            
        qukcpmd  = loadutils.loadUKCP(longname, ukcpdir, syr, nlump, prob=0.5, ukwil=ukwil)
        qukcphi  = loadutils.loadUKCP(longname, ukcpdir, syr, nlump, prob=0.9, ukwil=ukwil)

        if data1 is None:
            pass
        else:
            print(model,'data1.shape=',data1.shape,'syr[-1]=',syr[-1])
            
            # Now have data1.shape=(nsyr, nrlz), and syr,rlz
            nsyr=data1.shape[0] 
            rlz=data1.shape[1] 

            loc   = numpy.median(data1,axis=1)
            scale = numpy.std(data1,axis=1,ddof=1)           
            ntime = data1.shape[0]
            nrlz  = data1.shape[1]
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
            nsubarr.append(nsub)            
            dif=[]
            for j in range(nsub):
                dsubmed=numpy.median(datar[:,nrlz0*j:nrlz0*(j+1)],axis=1)
                dif.append(dsubmed-datarmed)
            dif=numpy.array(dif).T    #shape=(ntim,nsub), eg (55,100), (55,200), (55,400)   

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

            print(model, dif[:nmx,:].shape, dataall.shape, datarall.shape, difall.shape)

    datamed=numpy.array(datamed).T
    ndata=numpy.array(ndata)

    prob= [0.1, 0.5, 0.9]

    qqq = scipy.stats.mstats.mquantiles(dataall,prob=prob,axis=1)

    rrr = scipy.stats.mstats.mquantiles(datarall,prob=prob,axis=1)

    # Add deviations onto combined actual median (rrr is close to qqq but not quite identical)
    mmm = numpy.expand_dims(qqq[:nmx,1],-1) + difall

    # Estimate spread over these
    mmq = scipy.stats.mstats.mquantiles(mmm,prob=prob,axis=1)

    tsyr = utils.timefromsy(syr[:nmx], ssn, nlump)    

    ### SUBPLOT
    ax=plt.subplot(1,2,iname+1)
    if showuk:
        ukcplw=0.5
        ukcpcol='blue'
        plt.plot(tsyr, qukcplo[:nmx], color=ukcpcol, linewidth=ukcplw)
        plt.plot(tsyr, qukcpmd[:nmx], color=ukcpcol, linewidth=ukcplw)
        plt.plot(tsyr, qukcphi[:nmx], color=ukcpcol, linewidth=ukcplw)
        facecolor='grey'
        alpha=0.10
        ax.fill_between(tsyr, qukcplo[:nmx], qukcphi[:nmx], facecolor=facecolor,alpha=alpha)
        fc_for_rectangle = matplotlib.colors.ColorConverter().to_rgba(facecolor, alpha=alpha)
        handle_ukcp      = plt.Rectangle( (0, 0), 0, 0, edgecolor=ukcpcol, fc=fc_for_rectangle, lw=ukcplw)

    plt.plot(tsyr,qqq,lw=1.5,color='red')
    plt.plot(tsyr,rrr,lw=1.5,color='maroon')   #resamp
    plt.plot(tsyr,dobs[:nmx],lw=1.0,color='black') 

    m4col='CMCC' 
    ll = []
    ll.append( matplotlib.lines.Line2D([], [], color='red',   lw=1.5) )
    ll.append( matplotlib.lines.Line2D([], [], color='maroon',lw=1.5) )
    ll.append( matplotlib.lines.Line2D([], [], color='black', lw=1.0) )
    
    labels = ['MMDPE (N=150)', 'Gaussian resample ('+cfac+'*n for each)', 'Obs']
    labels = ['MMDPE',         'Gaussian resample ('+cfac+'*n for each)', 'Obs']
    labels = ['MMDPE',         'Gaussian resample of MMDPE',              'Obs']
    
    if showuk:
        ll.append( handle_ukcp )
        labels.append('UKCP-pdf') 

    xticks     = [ 1970,  1980,  1990,  2000,  2010]
    xticklabels= ['1970','1980','1990','2000','2010']
    nyearticks=len(xticks)
    if showpoints:
        n2p=250
        dely=2.5
        xtk=[]
        mtk=[]
        for imod in range(1,len(ndata)+1):        
            xx = numpy.ones(n2p)*(tsyr[-1]+dely*imod)
            ii = numpy.sum(ndata[:(imod-1)])*nfac + numpy.arange(n2p)
            imx= numpy.sum(ndata[:(imod)])*nfac
            yy=datarall[-1,ii]
            col=coldic[modellist[imod-1]]
            plt.plot(xx,yy,lw=0,marker='o',ms=1.25,alpha=0.3,color=col)
            x0=tsyr[-1]+dely*imod
            xticks.append(x0)
            xtk.append(x0)
            mname1= modellist[imod-1]
            mname = mdic[mname1] 
            mtk.append(mname)                 
            xticklabels.append(mname)
            x1 = numpy.ones(ndata[imod-1])*x0
            i1 = numpy.sum(ndata[:(imod-1)]) + numpy.arange(ndata[imod-1])
            y1 = dataall[-1, i1]    
            plt.plot(x1,y1,lw=0,marker='o',ms=3.5,alpha=0.65,color=col)
            print(imod, models[imod-1], ii[0], imx, i1[0], i1[-1])
            print(imod, models[imod-1],numpy.median(dataall[-1, i1]), numpy.mean(datarall[-1,ii[0]:imx]) )
        ll.append( matplotlib.lines.Line2D([], [], color=coldic[m4col], lw=0.0, marker='o',ms=3.5,alpha=0.65) )
        ll.append( matplotlib.lines.Line2D([], [], color=coldic[m4col], lw=0.0, marker='o',ms=1.25,alpha=0.30) )

        ctime='('+str(syr[nmx-1])+')'
        labels.append('Big points: hindcasts '+ctime) 
        labels.append('Small points: resampled '+ctime) 

    if iname == 0:
        ax.set_ylim([-2.0,4.5])
        ax.set_xlim([1960, x1[-1]+1.5])
        tit='NEU summer Tair, T1'
    else:
        ax.set_ylim([-11, 26])
        ax.set_xlim([1964, x1[-1]+1.5])
        tit='NEU winter precipitation, T2-9'

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,rotation=55, ha='right',fontsize=8.0)
    for imod,model in enumerate(models):
        kk = imod+nyearticks            
        plt.setp(ax.get_xticklabels()[kk], color=coldic[model])    
    ax.xaxis.set_tick_params(pad=0)
   
    plt.ylabel(ylabel)
    plt.title(tit)    
    leg = plt.legend(ll, labels, loc='best', fontsize=fs-0.5, handlelength=1.2, borderaxespad=0.25, handletextpad=0.25,labelspacing=0.25)
    leg.draw_frame(False)

    
for dpi in dpiarr:           
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
