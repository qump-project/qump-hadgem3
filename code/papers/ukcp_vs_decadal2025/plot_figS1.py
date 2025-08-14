import os
import numpy
import scipy
import scipy.stats.mstats
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import iris

from pathnames_v1 import *

########################################
namefig= 'figS1'
ifig   = 1001

dpiarr = [150]

saveplot   = True
#saveplot   = False


ssn1='djf'  ; lmp1='l8' ; reg1 = 'NEU'  
ssn2='djf'  ; lmp2='l1' ; reg2 = 'NEU'

lmpdic={'l8': 'T2-9', 'l1': 'T1'}

ssndic={'djf': 'winter', 'jja': 'summer'}

# Load data

bname = 'tas_ued=T_esd=T_wgt=ALL_sens=STD_scen=rcp45_rs=1_rba=F_wqm=11_psd=20_nmonte=1000000_nwgt=50000_ns=3000' 

file1_i01 = os.path.join(ukcpdir, ssn1+'_'+bname+'_ni=1_cdf_w31_'+lmp1+'_regs.nc')
file1_i20 = os.path.join(ukcpdir, ssn1+'_'+bname+'_ni=20_cdf_w31_'+lmp1+'_regs.nc')

file2_i01 = os.path.join(ukcpdir, ssn2+'_'+bname+'_ni=1_cdf_w31_'+lmp2+'_regs.nc')
file2_i20 = os.path.join(ukcpdir, ssn2+'_'+bname+'_ni=20_cdf_w31_'+lmp2+'_regs.nc')


print('Load from:',file1_i01)
cube1_i01=iris.load_cube(file1_i01)

print('Load from:',file1_i20)
cube1_i20=iris.load_cube(file1_i20)

print('Load from:',file2_i01)
cube2_i01=iris.load_cube(file2_i01)

print('Load from:',file2_i20)
cube2_i20=iris.load_cube(file2_i20)


con1 = iris.Constraint(geo_region=reg1)
cube1_01 = cube1_i01.extract(con1)
cube1_20 = cube1_i20.extract(con1)

con2 = iris.Constraint(geo_region=reg2)
cube2_01 = cube2_i01.extract(con2)
cube2_20 = cube2_i20.extract(con2)

# Begin plot
fs=9.0
fsleg=9.0
fstit=11
alpha=0.75
lw=1.5

matplotlib.rcParams.update({'font.size': fs})

plt.figure(ifig, figsize=(18/2.54, 9.0/2.54))

plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.87, bottom=0.13, left=0.08, right=0.97)

# First panel
ax=plt.subplot(1,2,1)

y01=cube1_01.coord('year').points
d01=cube1_01.data[:,95]-cube1_01.data[:,15]
plt.plot(y01, cube1_01.data[:,15],color='darkorange',alpha=alpha,lw=lw)
plt.plot(y01, cube1_01.data[:,55],color='darkorange',alpha=alpha,lw=lw,label='No init. cond. realisations (n=1)')
plt.plot(y01, cube1_01.data[:,95],color='darkorange',alpha=alpha,lw=lw)
plt.plot(y01, d01, color='darkorange',ls='--',lw=lw,alpha=alpha,label='P90-P10')

y20=cube1_20.coord('year').points
d20=cube1_20.data[:,95]-cube1_20.data[:,15]
plt.plot(y20, cube1_20.data[:,15],color='dodgerblue',alpha=alpha,lw=lw)
plt.plot(y20, cube1_20.data[:,55],color='dodgerblue',alpha=alpha,lw=lw,label='With init. cond. realisations (n=20)')
plt.plot(y20, cube1_20.data[:,95],color='dodgerblue',alpha=alpha,lw=lw)
plt.plot(y20, d20, color='dodgerblue',ls='--',lw=lw,alpha=alpha,label='P90-P10')


ax.set_xlim([1950,2030])
ax.set_ylim([-1.9,5.0])
xticks      = [1960,1980,2000,2020]
xticklabels = ['1960','1980','2000','2020']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels,ha='center',fontsize=fs)
plt.xlabel('Year')
plt.ylabel('$\Delta$T ($\degree$C)' )

tit1 = reg1 +' '+ ssndic[ssn2]+' Tair, '+lmpdic[lmp1]+'\nUKCP plume (P10,P50,P90)'
plt.title(tit1, fontsize=fstit)
leg = plt.legend(loc='upper left',fontsize=fsleg, handlelength=1.5, borderaxespad=0.3, handletextpad=0.3, labelspacing=0.15)
leg.draw_frame(False)

# Second panel
ax=plt.subplot(1,2,2)

y01=cube2_01.coord('year').points
d01=cube2_01.data[:,95]-cube2_01.data[:,15]
plt.plot(y01, cube2_01.data[:,15],color='darkorange',alpha=alpha,lw=lw)
plt.plot(y01, cube2_01.data[:,55],color='darkorange',alpha=alpha,lw=lw,label='No init. cond. realisations (n=1)')
plt.plot(y01, cube2_01.data[:,95],color='darkorange',alpha=alpha,lw=lw)
plt.plot(y01, d01, color='darkorange',ls='--',lw=lw,alpha=alpha,label='P90-P10')

y20=cube2_20.coord('year').points
d20=cube2_20.data[:,95]-cube2_20.data[:,15]
plt.plot(y20, cube2_20.data[:,15],color='dodgerblue',alpha=alpha,lw=lw)
plt.plot(y20, cube2_20.data[:,55],color='dodgerblue',alpha=alpha,lw=lw,label='With init. cond. realisations (n=20)')
plt.plot(y20, cube2_20.data[:,95],color='dodgerblue',alpha=alpha,lw=lw)
plt.plot(y20, d20, color='dodgerblue',ls='--',lw=lw,alpha=alpha,label='P90-P10')


ax.set_xlim([1950,2030])
ax.set_ylim([-3.5,8.9])
xticks      = [1960,1980,2000,2020]
xticklabels = ['1960','1980','2000','2020']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels,ha='center',fontsize=fs)
plt.xlabel('Year')
plt.ylabel('$\Delta$T ($\degree$C)' )


tit2 = reg2+' '+ssndic[ssn2]+' Tair, '+lmpdic[lmp2]+'\nUKCP plume (P10,P50,P90)'
plt.title(tit2, fontsize=fstit)
leg = plt.legend(loc='upper left',fontsize=fsleg, handlelength=1.5, borderaxespad=0.3, handletextpad=0.3, labelspacing=0.15)
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











