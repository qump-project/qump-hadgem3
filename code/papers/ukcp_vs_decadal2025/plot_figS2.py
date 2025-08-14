import os
import numpy
import scipy
import scipy.stats.mstats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathnames_v1 import *

#########################################
'''
Resample standard normal for different samples sizes N, and use different interpolation
assumptions in scipy.stats.mstats.mquantiles to estimate spread (typically P90-P10).
This is compared with the true value for a standard normal and plotted.

Note - no input data read in. 

Some links to background
Hyndman & Fan (1996): https://doi.org/10.1080/00031305.1996.10473566
https://www.amherst.edu/media/view/129116/original/Sample+Quantiles.pdf

https://lorentzen.ch/index.php/2023/02/11/quantiles-and-their-estimation/
https://en.wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html#scipy.stats.mstats.mquantiles
'''

#########################################
def calcspread(data, prob=[0.1,0.9], alphap=0.4, betap=0.4):
    nsmp=data.shape[0]
    nrlz=data.shape[1]
    spr=numpy.ma.zeros(nsmp)       
    for iii in range(nsmp):   
        # loop over samples, estimating quants for the rlz (6-80 typically)
        x=data[iii,:]  
        qq=scipy.stats.mstats.mquantiles(x,prob=prob,alphap=alphap,betap=betap)
        spr[iii] = qq[1]-qq[0]
        #print(iii,spr)
    spread = numpy.ma.mean(spr)
    return spread, spr         

####################
namefig= 'figS2'
ifig   = 1002

dpiarr = [150]

saveplot   = True
#saveplot   = False

plotuncert= False

prob = [0.1,0.9]

nsmp = 10000
#nsmp = 1000   # faster running for testing

seed = 2      # seed used for submitted version

norm       = scipy.stats.norm(loc=0.0, scale=1.0)
spreadtrue = norm.ppf(0.9)-scipy.stats.norm.ppf(0.1)    #2.56310313109

namearr = ['R4',       'R5',      'R6',  'R7 (R default)',           'R8',       'R9',   'Cunnane']
abarr   = [[0,1], [0.5,0.5], [0.0,0.0],         [1.0,1.0], [0.333, 0.333], [3/8, 3/8],  [0.4, 0.4]]

try:
    iscipy=namearr.index('Cunnane')
except:
    iscipy=namearr.index('Scipy')   

colarr=[ 'red',        'purple', 'limegreen', 'darkorange', 'blue',       'olive',  
         'dodgerblue', 'green',  'magenta',   'gold',       'darksalmon', 'cyan']

coldic={'R7 (Iris)': 'red',   'R7 (R default)': 'red', 'R7': 'red',
        'Scipy': 'blue',      'Cunnane': 'blue',        
        'R4': 'magenta',     
        'R5': 'limegreen',       
        'R6': 'purple',  
        'R8': 'green',       
        'R9': 'darkorange' } 

rlzarr  = numpy.array([6,7,8,9, 10,12,14, 17,20,23, 26,30,34,38,42,46, 50,55,60,65,70,75,80])

sprarr = numpy.zeros( (rlzarr.shape[0], len(abarr)) )

qlo=numpy.zeros(rlzarr.shape[0])
qmd=numpy.zeros(rlzarr.shape[0])
qhi=numpy.zeros(rlzarr.shape[0])
for irlz,nrlz in enumerate(rlzarr):
    print('Working on nrlz=',nrlz)
    smp=norm.rvs(size=(nrlz,nsmp), random_state=seed)
    smp=smp.T
    for iab,ab1 in enumerate(abarr):
       alphap=ab1[0]
       betap =ab1[1]   
       spr,sprsamp = calcspread(smp, prob=prob, alphap=alphap, betap=betap)
       sprarr[irlz,iab]=spr
       print('nrlz=',nrlz,'iab=',iab,'ab1=',ab1,'spr/spreadtrue=',spr/spreadtrue)
       if namearr[iab] in ['Scipy', 'Cunnane']:
           qq=scipy.stats.mstats.mquantiles(sprsamp,prob=[0.05,0.5,0.95],alphap=0.4,betap=0.4)
           qlo[irlz]=qq[0]
           qmd[irlz]=qq[1]
           qhi[irlz]=qq[2]

# Normalise by spreadtrue to get units of rmse to be same as curves plotted. 
errarr = sprarr/spreadtrue -1.
rmse   = numpy.zeros( len(abarr) )
for iab,ab1 in enumerate(abarr):
    rmse1     = scipy.integrate.cumulative_trapezoid(errarr[:,iab]**2, x=rlzarr, initial=0.0)
    rmse[iab] = numpy.sqrt(rmse1[-1]/(rlzarr[-1]-rlzarr[0]) )


### Make plot

probtit = 'P'+str(int(100*prob[1])) +'-P'+str(int(100*prob[0])) 

fs=9.5
matplotlib.rcParams.update({'font.size': fs})
figsize = (12.5/2.54, 12.5/2.54)
plt.figure(ifig, figsize=figsize)
plt.subplots_adjust(top=0.91, bottom=0.10, left=0.135, right=0.97, hspace=0.25, wspace=0.25)        

ax=plt.subplot(1,1,1)
for iab in range(len(abarr)):
    labab = r'$\alpha$='+str(abarr[iab][0])+r', $\beta$='+str(abarr[iab][1])
    label = namearr[iab]+', '+labab+', RMSE='+'%5.3f'%rmse[iab]
    col=coldic[namearr[iab]]
    plt.plot(rlzarr,sprarr[:,iab]/spreadtrue,marker='o',ms=3.5,color=col,alpha=0.65,label=label)

if plotuncert:
    col=coldic[namearr[iscipy]]
    plt.fill_between(rlzarr, qlo/spreadtrue, qhi/spreadtrue, color='dodgerblue', alpha=0.10, label='90% range (Cunnane)')

plt.axhline(1.0, ls='--', lw=2.0, color='k', label='True Value for Std Normal')      #'True Value for N(0,1)'
ax.set_ylabel(probtit+' normalized by true value')
ax.set_xlabel('N')
ax.set_xlim([rlzarr[0], rlzarr[-1]])
if plotuncert:
    ax.set_ylim([0.50,1.70])
else:
    ax.set_ylim([0.73,1.28])

ylim=ax.get_ylim()
i10 = list(rlzarr).index(10)
plt.axvline(rlzarr[i10],color='k',ls=':',lw=0.75) 
plt.text(rlzarr[i10],ylim[0]+0.015*(ylim[1]-ylim[0]),' N=10',color='k',fontsize=fs)

tit1= probtit+' estimation for differing quantile interpolation.\n'
tit2= 'Mean of '+str(nsmp)+' random samples of size N from Std Norm.'
plt.title(tit1+tit2, fontsize=fs+1.0)
leg = plt.legend(loc='upper right', fontsize=8.5, handlelength=1.7, borderaxespad=0.30, handletextpad=0.30, labelspacing=0.15)
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



