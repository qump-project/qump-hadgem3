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
namefig= 'fig9'
ifig   = 9

dpiarr = [150]

saveplot   = True
#saveplot   = False

namearr=['globalukcp09_tas_djfmamjjason', 'globalukcp09_amo_djfmamjjason', 'globalukcp09_nao-stephenson_djf']

modarr= ['BSC', 'CAFE', 'CMCC', 'CanESM5','Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NCAR40', 'NorCPM', 'NorCPMi2']
mdic  = utils.setmodelnames()

fpstr= ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']

nmod = len(modarr)
prob = [0.1, 0.9]
ab   = 0.4

coldic = utils.setcolours()
colgc31= coldic['Depresys4_gc3.1']
colukcp= 'grey'
colobs = 'black'
colmm  = 'black'
fs     = 9.5

onelegend=True
#onelegend=False

version = 2 


##############################
matplotlib.rcParams.update({'font.size': fs})

if version == 1:
    figsize = (14.0/2.54, 22/2.54)   #first submission
    left=0.115
else:
    figsize = (12.0/2.54, 22/2.54)
    left=0.135
plt.figure(ifig, figsize=figsize)

plt.subplots_adjust(top=0.97, bottom=0.035, left=left, right=0.985, hspace=0.225, wspace=0.25)        

for iname,longname in enumerate(namearr):

    name,var,ssn = longname.split('_')
    var      = var.replace('-', '_')
    longname = longname .replace('-', '_')

    for imod,model in enumerate(modarr):
        pfile=os.path.join(dpsdir, model+'_'+longname+'.pickled')       
        print('>>> Load file:',pfile)
        a    = pickle.load(open(pfile, 'rb'))

        fpi=a[0].coord('forecast_period').points  #first is T+1
        rlz=a[0].coord('realization').points
        nsyr=len(a)
        nrlz=rlz.shape[0]
        nfpi=fpi.shape[0]
        syr =numpy.zeros(nsyr)
        data=numpy.ma.zeros( (nsyr,nrlz,nfpi))

        fac=1.0
        if 'nao' in var: fac=1./100
        
        for isyr in range(nsyr):
            a1        = a[isyr]
            syr[isyr] = a1.coord('season_year').points[0]
            cnames    = [c.name() for c in a1.coords()]
            if cnames[0] == 'realization':            
                data[isyr,:,:] = fac*a1.data
            else:
                data[isyr,:,:] = fac*a1.data.T
            
        # Now have data.shape=(nsyr, nrlz, nfpi), and syr,rlz,fpi   
        nsyr=data.shape[0] 
        nrlz=data.shape[1] 
        nfpi=data.shape[2] 
        
        if imod == 0:
            spread = numpy.ma.zeros( (nfpi,nmod) )
            dataall= data.copy()
        else:
            print('>',longname,imod,model,'dataall.shape=',dataall.shape,'data.shape=',data.shape)
            nmx=57
            dataall=numpy.ma.concatenate((dataall[:nmx,:,:],data[:nmx,:,:]), axis=1)
            
        for ifpi in range(nfpi):
            x = data[:,:,ifpi]
            spread[ifpi,imod]= utils.calcspread(x, prob=prob, ab=ab, method='mean')    
            #print(imod,model,ab,spread[:,imod])

    spreadall=numpy.ma.zeros( nfpi )
    for ifpi in range(nfpi):
        x = dataall[:,:,ifpi]
        spreadall[ifpi]= utils.calcspread(x, prob=prob, ab=ab, method='mean')    

    print('>',longname,nmx,'spreadall=',spreadall)
    if longname == 'globalukcp09_tas_djfmamjjason':
        tit1='Annual GMST spread'
        obslab='Obs.'        
        yunit='($\degree$C)'                
        ymin=0.0
        ymx=0.55
        
        txtfile=os.path.join(ukcpdir, 'ci80_tas_ann_GLB.txt') 
        print('Input data from: ',txtfile)
        usecols = list(range(4,14))
        d = numpy.genfromtxt(txtfile, skip_header=2, usecols=usecols)
        obs3 = [d[0], d[1], d[2]]
        ppe3 = [d[3], d[4], d[5]]
        gc31 = [d[6], d[7], d[8]]
        ukcp = numpy.ones(fpi.shape[0])*d[9]
        print(longname,', obs3=',obs3)
        print(longname,', ppe3=',ppe3)
        print(longname,', gc31=',gc31)
        print(longname,', ukcp=',ukcp[0])

    elif longname == 'globalukcp09_amo_djfmamjjason':
        tit1  ='Annual AMV spread'        
        obslab='Obs.'
        yunit ='($\degree$C)'
        if onelegend:
            ymin =  0.18    ; ymx  = 0.65       
        else:
            ymin =  -0.05   ; ymx  = 0.65
                   
        txtfile=os.path.join(ukcpdir, 'ci80_AMO_ANN.txt') 
        print('Input data from: ',txtfile)
        usecols = list(range(4,14))
        d = numpy.genfromtxt(txtfile, skip_header=2, usecols=usecols)
        obs3 = [d[0], d[1], d[2]]
        ppe3 = [d[3], d[4], d[5]]
        gc31 = [d[6], d[7], d[8]]
        ukcp = numpy.ones(fpi.shape[0])*d[9]
        print(longname,', obs3=',obs3)
        print(longname,', ppe3=',ppe3)
        print(longname,', gc31=',gc31)
        print(longname,', ukcp=',ukcp[0])
       
    elif longname == 'globalukcp09_nao_stephenson_djf':
        tit1  ='Winter NAO spread'
        obslab='Obs.'
        yunit ='(hPa)'
        if onelegend:
            ymin =  6.0   ; ymx  = 13.3      
        else:
            ymin =  4.5   ; ymx  = 13.3                                                

        txtfile=os.path.join(ukcpdir, 'ci80_nao_stephenson_djf.txt') 
        print('Input data from: ',txtfile)
        usecols = list(range(4,14))
        d = numpy.genfromtxt(txtfile, skip_header=2, usecols=usecols)
        obs3 = [d[0], d[1], d[2]]
        ppe3 = [d[3], d[4], d[5]]
        gc31 = [d[6], d[7], d[8]]
        ukcp = numpy.ones(fpi.shape[0])*d[9]
        print(longname,', obs3=',obs3)
        print(longname,', ppe3=',ppe3)
        print(longname,', gc31=',gc31)
        print(longname,', ukcp=',ukcp[0])


    ax=plt.subplot(3,1,iname+1)
    
    for imod,model in enumerate(modarr):
        plt.plot(fpi,spread[:,imod],marker='o',ms=3.5,lw=1.0,alpha=0.85,color=coldic[model],label=mdic[model])

    plt.plot(fpi,spreadall, color=colmm, ls='--', lw=1.0, marker='o', ms=3.5, label='MMDPE')

    dx0=10.1    
    if version == 1:    
        dx1=0.3
    else:
        dx1=0.35    
    plt.plot(fpi, ukcp, marker='o',ms=3.5,lw=1.0,alpha=0.85,color=colukcp,label='UKCP-pdf')
    x3=numpy.ones(3)*dx0+1*dx1 
    plt.plot(x3, ppe3, marker='s',ms=3.5,lw=1.0, alpha=0.95, color=colukcp, label='ESPPE')
    x3=numpy.ones(3)*dx0+2*dx1
    plt.plot(x3, gc31, marker='s',ms=3.5,lw=1.0, alpha=0.95, color=colgc31, label='GC3.1')
    x3=numpy.ones(3)*dx0+3*dx1
    plt.plot(x3, obs3, marker='s',ms=3.5,lw=1.0, alpha=0.95, color=colobs, label=obslab)

    ax.set_ylabel('P90-P10 Spread '+yunit)
    ax.set_ylim([ymin, ymx])
    if version == 1:  
        plt.xlim([fpi[0]-0.7, fpi[-1]+1.3])
    else: 
        plt.xlim([fpi[0]-1.5*dx1, dx0+4*dx1]) 
    plt.xticks(fpi, fpstr)        
    plt.title(tit1, fontsize=11.25, pad=4)
    loc='lower right'
    if 'nao' in var: loc='lower left'
    if (not onelegend) or (onelegend and iname == 0):
        leg = plt.legend(loc=loc, ncol=3, fontsize=8.0, handlelength=1.8, borderaxespad=0.30, handletextpad=0.30,labelspacing=0.30)
        #leg.draw_frame(False)

for dpi in dpiarr:           
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
