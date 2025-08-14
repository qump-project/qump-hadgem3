import os
import numpy
import pickle
import copy
import scipy
import scipy.stats.mstats
import matplotlib
import matplotlib.pyplot as plt

import iris

import qdcUtils as utils

from pathnames_v1 import *

#########################################
namefig= 'figS14'
ifig   = 1014

dpiarr=[150]

saveplot   = True
#saveplot   = False

detrend=True

namearr = ['tas_djf_neu',  'tas_djf_eaw', 
           'tas_jja_neu',  'tas_jja_eaw',
           'pr_djf_neu',   'pr_djf_eaw',  
           'pr_jja_neu',   'pr_jja_eaw']

npanel = len(namearr)

tttarr = ['T2-9' for ii in range(npanel)]

ydic = {'tas_djf_neu': [-1.85,  1.40, 'lower right',  'NEU winter Tair'],
        'tas_djf_eaw': [-0.85, 1.30,  'upper right',  'EngWal winter Tair'],
        'tas_jja_neu': [-0.47,  0.61, 'upper right',  'NEU summer Tair'], 
        'tas_jja_eaw': [-0.41, 0.65,  'upper right',  'EngWal summer Tair'],
        'pr_djf_neu':  [-10.0, 6.5,   'lower right',  'NEU winter precip.'],
        'pr_djf_eaw':  [-17.0, 23.5,  'upper left',   'EngWal winter precip.'],
        'pr_jja_neu':  [-8.5, 13.0,   'upper left',   'NEU summer precip.'],
        'pr_jja_eaw':  [-23.0, 34.0,  'upper left',   'EngWal summer precip.'] }
                      
cdtr='orig'

stype='U'   ;  sstype='uncentred'

ukcpcol= 'blue'
dpscol = 'red'
obscol = 'black'

lw      = 1.25
fsleg   = 7.00 #6.75
fstit   = 10.5
legframe= True
fmtleg  = '%4.2f' 
ylimfac = 1.40

##################################################   
matplotlib.rcParams.update({'font.size': 8.0})

fig = plt.figure(ifig, figsize=(16/2.54, 21/2.54))     
plt.subplots_adjust(top=0.97, bottom=0.0375, left=0.12, right=0.98, hspace=0.280,wspace=0.34)        
    
for ivar,name,ttt in zip(range(npanel), namearr, tttarr):
    ax = plt.subplot(npanel//2, 2, ivar+1)

    namet = name+'_'+ttt   
    var,ssn,reg = name.split('_')
    ylabel = ''
    if var == 'tas':   ylabel='Detrended anomaly ($\degree$C)'
    if var == 'psl':   ylabel='Detrended anomaly (hPa)'
    if var == 'pr':    ylabel='Detrended anomaly (%)'

    inname = name+'_'+ttt+'_median_data_'+cdtr+'.npz'
    #eg tas_djf_neu_T2-9_median_data_detrend.npz
    
    infile = os.path.join(scoredir, inname)
    print('>>> load file: ',infile)
    a= numpy.load(infile)
    time     = a['time']
    dpsdata  = a['dpsdata']
    obsdata  = a['obsdata']
    ukcpdata = a['ukcpdata']

    inscore = namet+'_uncentred_detrend.npz'
    a2      = numpy.load(os.path.join(scoredir, inscore))
    scorename  = list( a2['score_name'] )
    scorevalue = a2['score_value']
    keydps     = namet+'_dps_'
    key3 = [keydps+'median_accu_detrend',  keydps+'p10_accu_detrend',  keydps+'p90_accu_detrend']
    scores=[]
    for kkk in key3:
        ik = scorename.index(kkk)
        scores.append(scorevalue[ik])

    print('ivar=',ivar,'name=',name,'time[0]=',time[0],'time[-1]=',time[-1])
    #start_year= time[0]-0.5
    start_year= 1965.0 
    yrlast    = time[-1]+0.5 

    if detrend:    
        dpsdata = scipy.signal.detrend(dpsdata)
        obsdata = scipy.signal.detrend(obsdata)
        ukcpdata= scipy.signal.detrend(ukcpdata)

    acc_dps   = utils.ACC_MSSS(dpsdata, obsdata, score='acc',  sstype='uncentred')
    acc_ukcp  = utils.ACC_MSSS(ukcpdata, obsdata, score='acc',  sstype='uncentred')

    print('\n>',namet,', acc_dps=',acc_dps,', inscore=',scores[0])
    print('scores=',scores)

    plt.plot(time, ukcpdata, color=ukcpcol, linewidth=lw)
    plt.plot(time, obsdata, color=obscol, linewidth=lw)
    plt.plot(time, dpsdata, color=dpscol, linewidth=lw)
    plt.axhline(0.0,color='k',lw=0.75,ls=':')

    ax=plt.gca()
    #ax.set_ylim(ylimset)    
    xticks=[1965,1975,1985,1995,2005,2015]
    ax.set_xticks(xticks)
    xticklabels=[]
    for yyyy in xticks:
        xticklabels.append(str(yyyy))
    ax.set_xticklabels(xticklabels) 
    ax.set_xlim([start_year, yrlast])
    plt.ylabel(ylabel)

    #ylim = ax.get_ylim()
    #ylim = [ylim[0]*ylimfac, ylim[1]*ylimfac] 
    #ax.set_ylim(ylim)
    ax.set_ylim([ydic[name][0], ydic[name][1] ])

    obslab    = 'Observations'

    dpslab = 'MMDPE, ACC: '   +str(fmtleg%acc_dps)
    dpslab = 'MMDPE, ACC: '+str(fmtleg%scores[0]) +' ('+str(fmtleg%scores[1]) +', '+str(fmtleg%scores[2]) +')'

    ukcplab= 'UKCP-pdf, ACC: '+str(fmtleg%acc_ukcp)

    labels= [obslab, dpslab, ukcplab]
    ll = []
    ll.append( matplotlib.lines.Line2D([], [], color=obscol,   lw=lw) )
    ll.append( matplotlib.lines.Line2D([], [], color=dpscol,   lw=lw) )
    ll.append( matplotlib.lines.Line2D([], [], color=ukcpcol,  lw=lw) )

    tit=ttt+' '+ydic[name][3]
    plt.title(tit, fontsize=fstit,pad=2.5)
    
    legtit=''
    loc=ydic[name][2]    
    leg=plt.legend(ll, labels, loc=loc, title=legtit, fontsize=fsleg, alignment='left',
                   handlelength=1.0, borderaxespad=0.4, handletextpad=0.3, labelspacing=0.25)    
    leg.draw_frame(legframe)


for dpi in dpiarr:           
    cdpi  = str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)


