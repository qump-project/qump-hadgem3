import os
import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt

from pathnames_v1 import *

#########################################
namefig= 'fig21'
ifig   = 21

dpiarr = [150]

saveplot   = True
#saveplot   = False

#spreadppe='all'
spreadppe='his'


obsfile = os.path.join(obsdir, 'ci80_2up_obs_tas_djf_NEU.npz')
print('Load file: ',obsfile)
o = numpy.load(obsfile)

ppefile = os.path.join(ukcpdir, 'ci80_2up_esppe_aldpf_tas_djf_NEU.npz')
print('Load file: ',ppefile)
p = numpy.load(ppefile)

time_obs        = o['time_obs']  
anom_obs        = o['anom_obs']  
anom_obs_low    = o['anom_obs_low']  
ac_obs          = o['ac_obs']
cinoise_obs_his = o['cinoise_obs_his']
ci_obs_rng      = o['ci_obs_rng']
unclo_obs_his   = o['unclo_obs_his']
unchi_obs_his   = o['unchi_obs_his']
                                                                          
time_ppe        = p['time_ppe']       
anom_ppe        = p['anom_ppe']       
anom_ppe_low    = p['anom_ppe_low']   
ac_ppe          = p['ac_ppe']         
cinoise_ppe_his = p['cinoise_ppe_his']
cinoise_ppe_all = p['cinoise_ppe_all']   
cinoise_ppe_fut = p['cinoise_ppe_fut']                                                                       
ci_ppe_his_rng  = p['ci_ppe_his_rng'] 
ci_ppe_fut_rng  = p['ci_ppe_fut_rng'] 
ci_ppe_all_rng  = p['ci_ppe_all_rng'] 
unclo_ppe_his   = p['unclo_ppe_his']  
unchi_ppe_his   = p['unchi_ppe_his']  


#########################################
# Make plot
fs=9.0
matplotlib.rcParams.update({'font.size': fs})
plt.figure(ifig, figsize=(17/2.54, 9/2.54))
plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.865, left=0.09, bottom=0.13, right=0.97)

# First panel
ax=plt.subplot(1, 2, 1)
ocol='black'
otext='P90-P10='+ '%5.2f'%cinoise_obs_his
ylow = anom_obs_low+ci_obs_rng[0]
yupp = anom_obs_low+ci_obs_rng[1]
 
plt.plot(time_obs, anom_obs,     color=ocol, linewidth=0.8, alpha=0.65)
plt.plot(time_obs, anom_obs_low, color=ocol, linewidth=2.0, alpha=1.0)
plt.plot(time_obs, ylow,         color=ocol, linewidth=1.5, alpha=1.0, ls='--')
plt.plot(time_obs, yupp,         color=ocol, linewidth=1.5, alpha=1.0, ls='--')
plt.xlabel('Year')
plt.ylabel('Anomaly ($\degree$C)')
tit='Observations'
plt.title(tit,fontsize=fs+1.5,pad=3.5)

ax.set_xlim([1850,2024])
ax.set_ylim([-4.2,3.8])

xlim=ax.get_xlim()
ylim=ax.get_ylim()
xout=xlim[-1]-0.02*(xlim[1]-xlim[0])
yout=ylim[0] +0.02*(ylim[1]-ylim[0])
plt.text(xout, yout, otext, ha='right', va='bottom', fontsize=fs+1.0, color='black')

labels=[]
ll = []
labels.append('Annual Obs (HadCRUT5)')
ll.append( matplotlib.lines.Line2D([], [], color=ocol, lw=0.8, alpha=0.65) )

labels.append('Low-pass filtered,\nbandpass cutoff=30yr')
ll.append( matplotlib.lines.Line2D([], [], color=ocol, lw=2.0, alpha=1.0) )

labels.append('P10, P90 ranges')
ll.append( matplotlib.lines.Line2D([], [], color=ocol, ls='--', lw=1.5, alpha=1.0) )

leg=plt.legend(ll, labels, loc='upper left', fontsize=fs-0.5, alignment='left',
                   handlelength=1.4, borderaxespad=0.2, handletextpad=0.5, labelspacing=0.5)    
leg.draw_frame(False)


# Second panel
ax=plt.subplot(1, 2, 2) 
pcol='blue'

if spreadppe == 'his':
    ylow = anom_ppe_low + ci_ppe_his_rng[0]
    yupp = anom_ppe_low + ci_ppe_his_rng[1]
    otext='P90-P10='+ '%5.2f'%cinoise_ppe_his
else:
    ylow = anom_ppe_low + ci_ppe_all_rng[0]
    yupp = anom_ppe_low + ci_ppe_all_rng[1]
    otext='P90-P10='+ '%5.2f'%cinoise_ppe_all
 
plt.plot(time_ppe, anom_ppe,     color=pcol, linewidth=0.8, alpha=0.65)
plt.plot(time_ppe, anom_ppe_low, color=pcol, linewidth=2.0, alpha=1.0)
plt.plot(time_ppe, ylow,         color=pcol, linewidth=1.5, alpha=1.0, ls='--')
plt.plot(time_ppe, yupp,         color=pcol, linewidth=1.5, alpha=1.0, ls='--')
plt.xlabel('Year')
plt.ylabel('Anomaly ($\degree$C)')
#tit='Single ESPPE member (aldpf)'
tit='Single ESPPE member'
plt.title(tit,fontsize=fs+1.5,pad=3.5)

ax.set_xlim([1860,2100])
ax.set_ylim([-6.0,12.0])

xlim=ax.get_xlim()
ylim=ax.get_ylim()
xout=xlim[-1]-0.02*(xlim[1]-xlim[0])
yout=ylim[0] +0.02*(ylim[1]-ylim[0])
plt.text(xout, yout, otext, ha='right', va='bottom', fontsize=fs+1.0, color='blue')

labels=[]
ll = []
labels.append('Annual data')
ll.append( matplotlib.lines.Line2D([], [], color=pcol, lw=0.8, alpha=0.65) )

labels.append('Low-pass filtered,\nbandpass cutoff=30yr')
ll.append( matplotlib.lines.Line2D([], [], color=pcol, lw=2.0, alpha=1.0) )

labels.append('P10, P90 ranges')
ll.append( matplotlib.lines.Line2D([], [], color=pcol, ls='--', lw=1.5, alpha=1.0) )

leg=plt.legend(ll, labels, loc='upper left', fontsize=fs-0.5, alignment='left',
                   handlelength=1.4, borderaxespad=0.2, handletextpad=0.5, labelspacing=0.5)    
leg.draw_frame(False)

plt.suptitle('NEU winter surface air temperature',fontsize=fs+2.0)

for dpi in dpiarr:            
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
