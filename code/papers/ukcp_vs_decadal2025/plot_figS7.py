import os
import numpy
import pickle
import copy
import iris

import scipy
import scipy.stats.mstats

import matplotlib
import matplotlib.pyplot as plt

import qdcUtils as utils
from pathnames_v1 import *


#########################################
namefig= 'figS7'
ifig   = 1007

dpiarr=[150]

saveplot   = True
#saveplot   = False

detrend=True


namearr = ['tas_ann_glb_T1', 'tas_ann_glb_T2-9', 'tas_ann_amv_T1', 'tas_ann_amv_T2-9', 'psl_djf_nao_T1', 'psl_djf_nao_T2-9']

ydic = {'tas_ann_glb_T1':     [-0.34,  0.34, 'upper left',   'T1 annual GMST'],
        'tas_ann_glb_T2-9':   [-0.16,  0.16, 'upper right',  'T2-9 annual GMST'],
        'tas_ann_amv_T1':     [-0.38,  0.24, 'lower left',   'T1 annual AMV'],
        'tas_ann_amv_T2-9':   [-0.115, 0.20, 'upper right',  'T2-9 annual AMV'],
        'psl_djf_nao_T1':     [-16.0,  10.5, 'lower left',   'T1 winter NAO'],    
        'psl_djf_nao_T2-9':   [-5.3,   4.8,  'lower left',   'T2-9 winter NAO'] }

cdtr='orig'

stype='U'   ;  sstype='uncentred'


ukcpcol= 'blue'
dpscol = 'red'
obscol = 'black'

lw      = 1.25
fsleg   = 7.5
fstit   = 10.5
legframe= True
fmtleg  = '%4.2f' 

npanel = len(namearr)
 
##################################################   

fig = plt.figure(ifig, figsize=(16/2.54, 21/2.54))     
matplotlib.rcParams.update({'font.size': 8.5})
plt.subplots_adjust(top=0.96, bottom=0.05, left=0.12, right=0.985, hspace=0.25,wspace=0.34)        
    

for ivar,namet in zip(range(npanel), namearr):

    ax = plt.subplot(npanel//2, 2, ivar+1)
    
    var,ssn,reg,ttt = namet.split('_')
    ylabel = ''
    if var == 'tas':   ylabel='Detrended anomaly ($\degree$C)'
    if var == 'psl':   ylabel='Detrended anomaly (hPa)'
    if var == 'pr':    ylabel='Detrended anomaly (%)'

    #eg tas_ann_glb_T2-9_median_data_detrend.npz
    inname = namet+'_median_data_'+cdtr+'.npz'
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


    print('ivar=',ivar,'namet=',namet,'time[0]=',time[0],'time[-1]=',time[-1])
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
    ax.set_ylim([ydic[namet][0], ydic[namet][1] ])    
    xticks=[1965,1975,1985,1995,2005,2015]
    ax.set_xticks(xticks)
    xticklabels=[]
    for yyyy in xticks:
        xticklabels.append(str(yyyy))
    ax.set_xticklabels(xticklabels) 
    ax.set_xlim([start_year, yrlast])
    plt.ylabel(ylabel)


    obslab    = 'Observations'

    #dpslab = 'MMDPE, ACC: '   +str(fmtleg%acc_dps)
    dpslab = 'MMDPE, ACC: '+str(fmtleg%scores[0]) +' ('+str(fmtleg%scores[1]) +', '+str(fmtleg%scores[2]) +')'

    ukcplab= 'UKCP-pdf, ACC: '+str(fmtleg%acc_ukcp)

    labels= [obslab, dpslab, ukcplab]
    ll = []
    ll.append( matplotlib.lines.Line2D([], [], color=obscol,   lw=lw) )
    ll.append( matplotlib.lines.Line2D([], [], color=dpscol,   lw=lw) )
    ll.append( matplotlib.lines.Line2D([], [], color=ukcpcol,  lw=lw) )

    tit = ydic[namet][3]
    plt.title(tit, fontsize=fstit,pad=2.5)
    
    legtit=''
    loc=ydic[namet][2]      
    leg=plt.legend(ll, labels, loc='best', title=legtit, fontsize=fsleg, alignment='left',
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


