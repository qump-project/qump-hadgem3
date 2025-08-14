import os
import copy
import numpy
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

import iris

import naoUtils
import qdcUtils as utils

from pathnames_v1 import *
 
##############################################################
def setcolours(choice=1):
    # Same as qdcUtils.py for 12 models, but with some extras.           
    coldic={'BCC':'saddlebrown', 'BSC':'olive',            'CAFE':'purple',       'CMCC':'magenta',        
            'CanESM5':'gold',    'Depresys4_gc3.1':'blue', 'IPSL':'darksalmon',   'MIROC6':'darkorange',                        
            'MPI':'red',         'NCAR40':'cyan',          'NorCPM':'darkgreen',  'NorCPMi2':'limegreen',            
            'NCAR':'cyan',       'ALL':'black',            
            'BEST':'grey',       'BESTDep':'grey',         'BEST4':'grey',         'BEST6':'grey'}
    return coldic


##############################################################
namefig = 'fig12'
ifig    = 12

dpiarr  = [150]

saveplot=True
#saveplot=False

#central='mean'
central='median'

sstype   = 'uncentred'  
norm_ssn = False   
order    = 'lump_diff_em'        
renorm   = True

nall     = 11   
nallrlz  = 150        
nameall  = 'ALL'

clump    = 'T+2-9' 
obsname  = 'era5'
score    = 'acc'


tlimdic={'BCC': [1960, 2013],     'BSC': [1960, 2018],     'CAFE': [1960, 2019],     'CanESM5':[1960, 2016],
         'CMCC': [1960, 2019],    'CMCC-10': [1960, 2019], 'Depresys4_gc3.1': [1960, 2022],             
         'IPSL': [1960, 2016],    'MIROC6': [1960, 2018],  'MPI': [1960, 2017],      'NCAR': [1954, 2017],
         'NCAR-10': [1954, 2017], 'NorCPM': [1960, 2018],  'NorCPMi2': [1960, 2018], 'ALL': [1960, 2016]}   

coldic= setcolours()
mdic  = utils.setmodelnames()

if clump == 'T+2-9':
    clumpt='T2-9'
    t1 = 2 
    t2 = 9
    fp = '2to9'
    time_xlabel = 'Centre year of lead-year range 2-9'
elif clump == 'T+1-8':
    clumpt='T1-8'
    t1 = 1 
    t2 = 8
    fp = '1to8'
    time_xlabel = 'Centre year of lead-year range 1-8'
    
nlump = t2-t1+1

tinit0 = 1960
islice = slice(t1-1, t2) 


#startyear = 1995
startyear = 2000

base1=1971
base2=2000

base1n=1971
base2n=2000

cbase='b'+str(base1)+'-'+str(base2)


############################################################
# PANEL 1
############################################################
print('\n>>> START PANEL 1')

ssn      = 'djf'     
region   = 'Stephenson'  

if region == 'Li':           creg='(Li & Wang)'
if region == 'Stephenson':   creg='(Stephenson)'
if region == 'Smith':        creg='(Smith)'


### Load obs

fnorth   = 'nao_north_'+obsname+'_'+region+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'
fsouth   = 'nao_south_'+obsname+'_'+region+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'

filenorth= os.path.join(naodir,fnorth)
filesouth= os.path.join(naodir,fsouth)

print('>>> Panel 1. Load obs from filenorth=',filenorth)
naonorth = iris.load_cube(filenorth)

print('>>> Panel 1. Load obs from filesouth=',filesouth)
naosouth = iris.load_cube(filesouth)

timeobs, naoobssm = naoUtils.obs_smoothdiff(naonorth, naosouth, nlump, ssn=ssn, order='lump_diff', renorm=renorm)

### Load Decadal ALL

aname   = 'ALL-nmod'+str(nall)+'-nrlz'+str(nallrlz)
inname0 = aname+'_' + 'naoindex_' +ssn+'_'+region+'_'+clump+'_'+cbase+'_normssn='+str(norm_ssn)[0]+'_renorm='+str(renorm)[0]+'_order='+order
inname  = inname0+'_ct='+central+'.npz'

infile  = os.path.join(naodir ,inname)

print('>>> Panel 1. Load MMDPE from:',infile)
a    = numpy.load(infile)
data = a['data']
midyr    = data[0,:]
naomodel = data[1,:]

#raise AssertionError('Stop for debugging...')   

iforobs, ifordec = naoUtils.index_common(timeobs, midyr, asint=True)

nmax  = ifordec.shape[0]           # eg 55
klist = list(range(17,nmax+1))     # 17 => 1980
nlast = len(klist)
ylast = numpy.zeros(nlast)
acc   = numpy.zeros(nlast)

headacc = 'ACC'
if sstype == 'centred': headacc='ACCc'
print('YEAR      '+headacc) 
for ik,kk in enumerate(klist):
    idec= ifordec[:kk]
    iobs= iforobs[:kk]
    acc1 = naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype=sstype, score=score)     
    print('%9.4f'%midyr[idec][-1], '%6.4f'%acc1)  
    ylast[ik]=midyr[idec][-1]
    acc[ik]=acc1

#tlast = 2018.5
tlast = 2019.5

msemod = (midyr[ifordec]-tlast)**2
i4mod=numpy.where(msemod == msemod.min())[0][0]
mseobs = (timeobs[iforobs]-tlast)**2
i4obs=numpy.where(mseobs == mseobs.min())[0][0]
acc_tit =naoUtils.ACC_MSSS(naomodel[ifordec][:i4mod+1], naoobssm[iforobs][:i4obs+1], sstype=sstype, score=score) 

#raise AssertionError('Stop for debugging...')   

########################################################################

fs=8.5
fsleg=8.0
fstit=10.0
titpad=3
matplotlib.rcParams.update({'font.size': fs})

plt.figure(ifig, figsize=(18/2.54, 17.5/2.5))   #try this for 4up
plt.subplots_adjust(hspace=0.27,wspace=0.230,top=0.965,bottom=0.11,left=0.08,right=0.98)        

########################################################################
# First subplot    

ax=plt.subplot(2,2,1)

label='MMDPE'
plt.plot(midyr[ifordec], naomodel[ifordec], color='darkorange', lw=1.0, marker='o', ms=3, label=label)  

plt.plot(timeobs[iforobs], naoobssm[iforobs], color='black', lw=1.0, marker='o', ms=3, label=obsname.upper())
plt.axhline(0.0, ls=':',lw=1.5,color='k')

plt.xlabel(time_xlabel)
plt.ylabel('Standardised NAO anomaly')

title='NAO '+creg+', '+ ssn.upper()+', '+ clumpt+ ', '+headacc+'=%5.3f'%acc_tit
plt.title(title, fontsize=fstit,pad=titpad)    

leg = plt.legend(loc='best', fontsize=fsleg, handlelength=1.4, borderaxespad=0.5, handletextpad=0.25,labelspacing=0.25)
ax.grid(axis='x',color='grey',ls='--',lw=0.5)   #'darkgoldenrod'
ax.grid(axis='y',color='grey',ls='--',lw=0.5)


############################################################
# PANEL 2
# Compare uncen vs cen and two baselines
############################################################
print('\n>>> START PANEL 2')


modellist = ['BSC', 'CAFE', 'CanESM5', 'CMCC', 'Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NorCPM', 'NorCPMi2', 'NCAR']  

nmodels   = len(modellist) 

order    = 'lump_diff_em'
renorm   = False
norm_ssn = False

regname='Stephenson'
ssn='djf'    

obsname  = 'era5'

clump = 'T+2-9'   ; t1=2    ; t2=9   ; clumpt = 'T2-9'   ; fp = '2to9'
nlump = t2-t1+1

islice = slice(t1-1, t2) 

fnorth='nao_north_'+obsname+'_'+regname+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'
fsouth='nao_south_'+obsname+'_'+regname+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'

filenorth= os.path.join(naodir,fnorth)
filesouth= os.path.join(naodir,fsouth)

print('>>> Panel 2. Load Obs data from:',filenorth)
obsnorth_in = iris.load_cube(filenorth)

print('>>> Panel 2. Load Obs data from:',filesouth)
obssouth_in = iris.load_cube(filesouth)

timeobs1 = obsnorth_in.coord('season_year').points + 15./360.

lastarr=numpy.arange(1990,2019+1)
nlast=lastarr.shape[0]

baselist = [ [1971,2000], [1981, 2010] ]
nbase = len(baselist)


# Load Decadal
narrlist=[]
sarrlist=[]
iyearlist=[]

for kmod,model in enumerate(modellist):
    #Default
    modelname = model
    nrlz      = 10
    tlim      = tlimdic[model]
         
    if model == 'BCC':   nrlz = 8
    if model == 'CMCC':  nrlz = 20
    if model == 'NCAR':  nrlz = 40

    name = 'NAO_'+modelname+'_nrlz='+str(nrlz)+'_region='+region+'_ssn='+ssn+'_aw=T_ib=T_north_south_'+str(tlim[0])+'_'+ str(tlim[1])+'.npz'
    file = os.path.join(naodir, name)
    
    print('>>> Panel 2. Load DPS data from:',file)
    a=numpy.load(file)

    iyear    = a['iyear']
    fcperiod = a['fcperiod']
    rlzarr   = a['rlzarr']
    northarr = a['northarr']     #shape=(nyear,nrlz,nfp)
    southarr = a['southarr'] 

    # Note - numpy.savez can lose the mask, so when we load we need to do numpy.ma.masked_invalid.
    # This works since we have set missing values equal to numpy.nan in make_decadal_nao.py.    
    northarr = numpy.ma.masked_invalid(northarr)
    southarr = numpy.ma.masked_invalid(southarr)

    anomnorth, anomsouth = naoUtils.calc_anom(northarr, southarr, base1n, base2n, tinit0=tlim[0], iyear=iyear, verbose=False)

    if kmod == 0:
        iyear_all    = iyear
        northarr_all = anomnorth   #anom now NOT northarr
        southarr_all = anomsouth
        print('kmod=',kmod,'northarr_all.shape=',northarr_all.shape,'iyear_all[-1]=',iyear_all[-1])      
    else:
        i1_all,i2_all  = naoUtils.index_common(iyear_all, iyear)
        iyear_all = iyear_all[i1_all] 
        northarr_all = numpy.ma.concatenate((northarr_all[i1_all,:,:], anomnorth[i2_all,:,:]), axis=1)
        southarr_all = numpy.ma.concatenate((southarr_all[i1_all,:,:], anomsouth[i2_all,:,:]), axis=1)
        print('kmod=',kmod,'northarr_all.shape=',northarr_all.shape,'iyear_all[-1]=',iyear_all[-1])  

northarr = numpy.ma.masked_invalid(northarr_all)
southarr = numpy.ma.masked_invalid(southarr_all)    
iyear    = iyear_all

nyr = northarr.shape[0]
nrlz= northarr.shape[1]
nfp = northarr.shape[2]
fcperiod = numpy.array(list(range(1,nfp+1)))

# Loop over base, and another loop over ylast

accu_arr=numpy.zeros( (nlast,nbase))
accd_arr=numpy.zeros( (nlast,nbase))
accc_arr=numpy.zeros( (nlast,nbase))
mbar_arr=numpy.zeros( (nlast,nbase))
obar_arr=numpy.zeros( (nlast,nbase))
mint_arr=numpy.zeros( (nlast,nbase))
oint_arr=numpy.zeros( (nlast,nbase))
for ibase,base in enumerate(baselist):
    base_1=base[0]
    base_2=base[1]
           
    # Rebase obs
    obsnorth, obssouth = naoUtils.rebase_obscube(obsnorth_in, obssouth_in, base_1, base_2)    
    timeobs, naoobssm = naoUtils.obs_smoothdiff(obsnorth, obssouth, nlump, ssn=ssn, order='lump_diff', renorm=renorm)
  
    indir = naodir
    midyr, naomodel, nao_all, tlis, dlis = naoUtils.loadNAO(indir, fp=fp, region=region, ssn=ssn, base1=base_1, base2=base_2, 
                                                            ensemble=modellist, central=central, renorm=renorm)

    print('OBS')
    print('nlump=',nlump,'ssn=',ssn,'norm_ssn=',norm_ssn,'renorm=',renorm)
    print('base1=',base_1,'base2=',base_2)
    print('DEC')
    print('indir=',indir,'fp=',fp,'region=',region,'ssn=',ssn,'base1=',base_1,'base2=',base_2)
    print('central=',central,'renorm=',renorm)
     
    iforobs, ifordec     = naoUtils.index_common(timeobs, midyr)    

    for iy,ylast in enumerate(lastarr):
        mse = (timeobs[iforobs]- (ylast+0.5))**2 
        iok = numpy.where(mse == mse.min())[0][0]

        idec=ifordec[:iok+1]
        iobs=iforobs[:iok+1]
        accu=naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='uncentred', score='acc') 
        accc=naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='centred', score='acc') 

        x = timeobs[iobs]
        yo = naoobssm[iobs]
        ym = naomodel[idec]
        anso = scipy.stats.linregress(x,yo)
        ansm = scipy.stats.linregress(x,ym)        
        yodt = yo - (x*anso.slope+anso.intercept)
        ymdt = ym - (x*ansm.slope+ansm.intercept)
        accd = naoUtils.ACC_MSSS(ymdt, yodt, sstype='uncentred', score='acc') 

        mbar = numpy.mean(naomodel[idec])
        obar = numpy.mean(naoobssm[iobs])
        mint = numpy.sqrt(numpy.mean(naomodel[idec]**2))
        oint = numpy.sqrt(numpy.mean(naoobssm[iobs]**2))
      
        accu_arr[iy,ibase]=accu
        accc_arr[iy,ibase]=accc
        accd_arr[iy,ibase]=accd        
        mbar_arr[iy,ibase]=mbar
        obar_arr[iy,ibase]=obar
        mint_arr[iy,ibase]=mint
        oint_arr[iy,ibase]=oint

        if ylast == lastarr[-1]:
            print('naomodel[idec]=',naomodel[idec])
            print('naoobssm[iobs]=',naoobssm[iobs])

########################################################################
# Second subplot    

ax=plt.subplot(2,2,2)
version=2

ms=3
alpha=0.75
lab='Uncentred, baseline='+ str(baselist[0][0]) +'-'+ str(baselist[0][1])
plt.plot(lastarr,accu_arr[:,0],color='limegreen',      lw=1.0, marker='o', ms=ms, alpha=alpha, label=lab)

lab='Uncentred, baseline='+ str(baselist[1][0]) +'-'+ str(baselist[1][1])
plt.plot(lastarr,accu_arr[:,1],color='olive',lw=1.0, marker='o', ms=ms, alpha=alpha, label=lab)

lab='Centred, baseline='+ str(baselist[0][0]) +'-'+ str(baselist[0][1])
plt.plot(lastarr,accc_arr[:,0],color='purple',       lw=1.0, marker='o', ms=ms, alpha=alpha, label=lab)

lab='Centred, baseline='+ str(baselist[1][0]) +'-'+ str(baselist[1][1])
plt.plot(lastarr,accc_arr[:,1],color='orchid',       lw=1.0, marker='o', ms=ms, alpha=alpha, label=lab)

if version == 1:
    lab='Detrended, Uncent., base='+ str(baselist[0][0]) +'-'+ str(baselist[0][1])
    plt.plot(lastarr,accd_arr[:,0],color='grey',lw=1.0, marker='o', ms=ms, alpha=alpha, label=lab)

if central == 'median':
    if version == 1:
        ax.set_ylim([0.33,0.76])
    else:
        ax.set_ylim([0.48,0.76])        
else:
    if version == 1:
        ax.set_ylim([0.27,0.77])  
    else:
        ax.set_ylim([0.42,0.77])        

leg = plt.legend(loc='best', fontsize=fsleg, handlelength=1.4, borderaxespad=0.3, handletextpad=0.3,labelspacing=0.3)

tit1='NAO ACC, MMDPE, '+regname+', '+ssn.upper()+', T2-9' 
plt.title(tit1,fontsize=fstit,pad=titpad)
plt.ylabel('Anomaly Correlation Coefficient')
plt.xlabel('Final T2-9 period for skill score calculation')


############################################################
# PANEL 3
############################################################
print('\n>>> START PANEL 3')

# Load Decadal All - different choices

#raise AssertionError('Stop for debugging...')  

cfig = 'regssn'

choices=[ ['djf', 'Smith'],      ['djfm', 'Smith'],
          ['djf', 'Li'],         ['djfm', 'Li'],
          ['djf', 'Stephenson'], ['djfm', 'Stephenson'] ]

coldic={'Smith, djf':      'dodgerblue',   'Smith, djfm':      'blue', 
        'Li, djf':         'orange',       'Li, djfm':         'red',
        'Stephenson, djf': 'limegreen',    'Stephenson, djfm': 'darkgreen'}


markerdic={'Smith, djf':      '*',    'Smith, djfm':      '*', 
           'Li, djf':         'p',    'Li, djfm':         'p',
           'Stephenson, djf': 'P',    'Stephenson, djfm': 'P'}

regdic = {'Li': 'Li & Wang', 'Stephenson': 'Stephenson', 'Smith': 'Smith'}  
       
nchoices=len(choices)

# For panel 2 and 3, now potentially use different setting for order, norm_ssn, sstype, renorm
# Panel 2,3
order    ='lump_diff_em'   
norm_ssn = False    
renorm   = False    
sstype   = 'uncentred'


model = 'ALL'
aname = 'ALL-nmod'+str(nall)+'-nrlz'+str(nallrlz)

acclist=[]
lablist1=[]
lablist2=[]
for choice in choices:
    ssn    = choice[0]
    region = choice[1]
    lablist1.append(region+', '+ssn)      
    lablist2.append(regdic[region]+', '+ssn.upper())

    fnorth='nao_north_'+obsname+'_'+region+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'
    fsouth='nao_south_'+obsname+'_'+region+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'

    filenorth= os.path.join(naodir,fnorth)
    filesouth= os.path.join(naodir,fsouth)
    
    print('PANEL 3. Load Obs from:',filenorth)
    naonorth = iris.load_cube(filenorth)
    
    print('PANEL 3. Load Obs from:',filesouth)   
    naosouth = iris.load_cube(filesouth)
    
    # Input data already baselined to 1971-2000, so no need to rebase    
        
    timeobs, naoobssm = naoUtils.obs_smoothdiff(naonorth, naosouth, nlump, ssn=ssn, order='lump_diff', renorm=renorm)

    # Load MMDPE ALL data           
    inname0 = aname+'_' + 'naoindex_' +ssn+'_'+region+'_'+clump+'_'+cbase+'_normssn='+str(norm_ssn)[0]+'_renorm='+str(renorm)[0]+'_order='+order
    inname  = inname0+'_ct='+central+'.npz'
    infile = os.path.join(naodir, inname)

    print('PANEL 3. Load MMDPE from:',infile)        
    a    = numpy.load(infile)
    data = a['data']
    midyr    = data[0,:]
    naomodel = data[1,:]

    indir = naodir
    midyr, naomodel, nao_all, tlis, dlis = naoUtils.loadNAO(indir, fp=fp, region=region, ssn=ssn, base1=base1n, base2=base2n, 
                                                            ensemble=modellist, central=central, renorm=renorm)

    #if region == 'Smith' and ssn == 'djf' and model == 'ALL':
    if False:
        print('PANEL 3 data for (Smith, djf, ALL):')
        print('Model data:',naomodel)
        print('Obs data:',naoobssm)
        print(ssn,norm_ssn,renorm)
        

    iforobs, ifordec = naoUtils.index_common(timeobs, midyr, asint=True)    

    nmax  = ifordec.shape[0]
    
    score2 = 'acc'   #'msss'
    yfirst=1990 
       
    found=False
    k0=0
    while not found:
        k0=k0+1
        if midyr[ifordec[k0]] > yfirst+1:
            found=True            
    klist = list(range(k0,nmax+1))     # k0=16 => 1980 
    
    nlast = len(klist)
    ylast = numpy.zeros(nlast)
    acc   = numpy.zeros(nlast)
    headacc = 'ACCu'
    if sstype == 'centred': headacc='ACCc'
    verbose=False
    if verbose:   print('YEAR      '+headacc) 
    for ik,kk in enumerate(klist):
        idec = ifordec[:kk]
        iobs = iforobs[:kk]
        acc1 = naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype=sstype, score=score2) 
        if verbose:   print('%9.4f'%midyr[idec][-1], '%6.4f'%acc1)  
        ylast[ik]=midyr[idec][-1]
        acc[ik]=acc1        
    acclist.append(acc)

########################################################################
# Third subplot    

ax=plt.subplot(2,2,3)
accdic ={'acc_uncentred': 'ACC (uncentred)', 'acc_centred':   'ACC (centred)'} 

ybeg=yfirst
acctype = 'acc_'+sstype

for k, acc, lab1,lab2 in zip(range(nchoices), acclist, lablist1, lablist2):
    plt.plot(ylast, acc, color=coldic[lab1], alpha=0.7, lw=0.75, marker='o', ms=3, label=lab2)

plt.xlabel('Final T2-9 period for skill score calculation')
plt.ylabel(accdic[acctype])
title = 'MMDPE NAO skill, '+clumpt
   
plt.title(title, fontsize=fstit,pad=titpad)
loc='best'
leg = plt.legend(loc=loc, fontsize=fsleg, handlelength=1.4, borderaxespad=0.4, handletextpad=0.3,labelspacing=0.3)
ax.set_xlim( [ybeg,2020] )

ystart = (((ybeg-1)//5) + 1)*5
xticks = list(range(ystart,2020+5,5))
xticklabels = [str(x) for x in xticks]

if central == 'mean':  
    ax.set_ylim([0.33,0.78])
else:    
    ax.set_ylim([0.29,0.70])
    dy=0.1    
    yticks = numpy.arange(0.3,0.7+dy,dy)
    yticklabels = ['%3.1f'%y for y in yticks]  
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=fs)
     
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels,fontsize=fs)


############################################################
# PANEL 4
# Do single sodel, single time processing, With equal-sized ensembles now.
############################################################
print('\n>>> START PANEL 4')

sstype   = 'uncentred'     
norm_ssn = True
norm_ssn = False

readlist=['BSC', 'CAFE', 'CMCC', 'CanESM5','Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NCAR', 'NorCPM', 'NorCPMi2', 
          'NCAR-10', 'CMCC-10',  'ALL']

eselist= ['BSC', 'CAFE', 'CMCC-10', 'CanESM5','Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NCAR-10', 'NorCPM', 'NorCPMi2']

ndrop=3  

if central == 'median':
    # Drop BSC, NorCPM, IPSL
    esekeep= ['CAFE', 'CMCC-10', 'CanESM5', 'Depresys4_gc3.1', 'MIROC6', 'MPI', 'NCAR-10',  'NorCPMi2']
else:
    # Drop BSC, NorCPM, MIROC6
    esekeep= ['CAFE', 'CMCC-10', 'CanESM5', 'Depresys4_gc3.1', 'IPSL', 'MPI', 'NCAR-10',  'NorCPMi2']


lablist = ['BSC', 'CAFE', 'CMCC', 'CanESM5','Depresys4_gc3.1', 'IPSL', 'MIROC6', 'MPI', 'NCAR', 'NorCPM', 'NorCPMi2',          
           'NCAR-10', 'CMCC-10', 'ALL', 'MMESE', 'MMESE_drop'] 

nmodels=len(lablist)

ylast=2019.54167

accarr = numpy.zeros( (nmodels,nchoices))
lablist1=[]
lablist2=[]
for kchoice,choice in enumerate(choices):
    ssn    = choice[0]
    region = choice[1]
    lablist1.append(region+', '+ssn)      
    lablist2.append(regdic[region]+', '+ssn.upper())

    startyear=2000
    crenorm = 'renorm='+str(renorm)[0]
    cnorm   = 'normssn='+str(norm_ssn)[0]
    corder  = 'order='+order

    base1n=1971
    base2n=2000

    ### First load obs
    fnorth='nao_north_'+obsname+'_'+region+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'
    fsouth='nao_south_'+obsname+'_'+region+'_'+ssn+'_mon=False_ssn=False_'+cbase+'.nc'

    filenorth= os.path.join(naodir,fnorth)
    filesouth= os.path.join(naodir,fsouth)

    print('>>> Panel 4. Load obs from:',filenorth)
    obsnorth=iris.load_cube(filenorth)

    print('>>> Panel 4. Load obs from:',filesouth)
    obssouth=iris.load_cube(filesouth)
              
    #obsnorth, obssouth = naoUtils.rebase_obscube(obsnorth, obssouth, base1n, base2n)
    # Input data already baselined to 1971-2000 so skiop rebase

    timeobs, naoobssm = naoUtils.obs_smoothdiff(obsnorth, obssouth, nlump, ssn=ssn, order='lump_diff', renorm=renorm)

    # Load Decadal
    narrlist=[]
    sarrlist=[]
    iyearlist=[]
    iyear_all = None
    iyear_drp = None
    for kread,model in enumerate(readlist):
        calcindex=True
        if model == 'ALL':   calcindex=False
        modeldir = model.split('-')[0]
            
        nrlz      = 10     
        nsub      =  0
        ctlab     = ''
        tlim      = tlimdic[model]
        
        if model == 'BCC':
            nrlz = 8
        elif model == 'CMCC':
            nrlz = 20
        elif model == 'CMCC-10':
            nsub = 10
            ctlab= '_ct='+central                        
        elif model == 'NCAR':
            nrlz = 40
        elif model == 'NCAR-10':
            nsub = 10
            ctlab= '_ct='+central                            

        if '-10' in model: 
            modelname = modeldir
        else:
            modelname = model

        if calcindex:
            name = 'NAO_'+modelname+'_nrlz='+str(nrlz)+'_region='+region+'_ssn='+ssn+'_aw=T_ib=T_north_south_'+str(tlim[0])+'_'+ str(tlim[1])+ctlab+'.npz'
            file = os.path.join(naodir, name)
            
            print('>>> Panel 4. Load DPS from:',file)
            a=numpy.load(file)

            iyear    = a['iyear']
            fcperiod = a['fcperiod']
            rlzarr   = a['rlzarr']
            northarr = a['northarr']     #shape=(nyear,nrlz,nfp)
            southarr = a['southarr'] 

            # WARNING - numpy.savez loses the mask, so when we load we need to do numpy.ma.masked_invalid.
            # This works since we have set missing values equal to numpy.nan in make_decadal_nao.py.
            
            northarr = numpy.ma.masked_invalid(northarr)
            southarr = numpy.ma.masked_invalid(southarr)

            # Anomalize, and make index for individual model that is read in.
            # Anomalies then concatenated for multimodel ensemble
            
            anomnorth, anomsouth = naoUtils.calc_anom(northarr, southarr, base1n, base2n, tinit0=tlim[0], iyear=iyear, verbose=False)

            indir = naodir
            midyr, naomodel, nao_all, tlis, dlis = naoUtils.loadNAO(indir, fp=fp, region=region, ssn=ssn, base1=base1n, base2=base2n,
                                                                    ensemble=model, central=central, renorm=renorm, nsub=nsub)

            iforobs, ifordec  = naoUtils.index_common(timeobs, midyr)    
            mse = (timeobs[iforobs]- (ylast+0.5))**2 
            iok = numpy.where(mse == mse.min())[0][0]
            idec=ifordec[:iok+1]
            iobs=iforobs[:iok+1]
            acc=naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='uncentred', score='acc') 

            if iyear_all is None:
                if model in eselist:  
                    iyear_all    = iyear
                    northarr_all = anomnorth 
                    southarr_all = anomsouth
            else:
                if model in eselist:              
                    i1_all, i2_all  = naoUtils.index_common(iyear_all, iyear)
                    iyear_all = iyear_all[i1_all] 
                    northarr_all = numpy.ma.concatenate((northarr_all[i1_all,:,:], anomnorth[i2_all,:,:]), axis=1)
                    southarr_all = numpy.ma.concatenate((southarr_all[i1_all,:,:], anomsouth[i2_all,:,:]), axis=1)

            if iyear_drp is None:
                if model in esekeep:
                    iyear_drp    = iyear                
                    northarr_drp = anomnorth 
                    southarr_drp = anomsouth                
            else:
                if model in esekeep:                   
                    i1_drp, i2_drp  = naoUtils.index_common(iyear_drp, iyear)
                    iyear_drp = iyear_drp[i1_drp] 
                    northarr_drp = numpy.ma.concatenate((northarr_drp[i1_drp,:,:], anomnorth[i2_drp,:,:]), axis=1)
                    southarr_drp = numpy.ma.concatenate((southarr_drp[i1_drp,:,:], anomsouth[i2_drp,:,:]), axis=1)

        else:
            if model == 'ALL':
                allname = 'ALL-nmod11-nrlz150'
                name = allname+'_naoindex_'+ssn+'_'+region+'_'+clump+'_'+cbase+'_'+cnorm+'_'+crenorm+'_'+corder+'_ct='+central+'.npz'                   
                file = os.path.join(naodir, name)
                print('>>> Panel 4. Load DPS from:',file)
                a = numpy.load(file)                
                data     = a['data']
                modyr    = data[0,:]            
                iforobs, ifordec = naoUtils.index_common(timeobs, modyr)    
                naomodel = data[1,:]
                iobs1=numpy.where(timeobs[iforobs] <= ylast)[0]
                imod1=numpy.where(modyr[ifordec] <= ylast)[0]
                acc=naoUtils.ACC_MSSS(naomodel[ifordec][imod1], naoobssm[iforobs][iobs1], sstype=sstype, score=score)                    

        # End conditional on calcindex. Now find idx for storing ACC

        if model in ['NCAR', 'NCAR-10']:
            idx=None
            if model == 'NCAR':   
                idx=lablist.index(model)
            else:
                idx=lablist.index('NCAR-10')           

        elif model in ['CMCC', 'CMCC-10']:
            idx=None
            if model == 'CMCC':   
                idx=lablist.index(model)
            else:
                idx=lablist.index('CMCC-10')
        else:
            idx=lablist.index(model)
        #print('model=',model,'idx=',idx)

        if not idx is None:
            accarr[idx,kchoice] = acc                     

    ### End loop over models

    # MMESE - Make NAO index and estimate skill 
    northarr = numpy.ma.masked_invalid(northarr_all)
    southarr = numpy.ma.masked_invalid(southarr_all)    
    iyear    = iyear_all    
    nyr = northarr.shape[0]
    nrlz= northarr.shape[1]
    nfp = northarr.shape[2]
    fcperiod = numpy.array(list(range(1,nfp+1)))
    
    anomnorth, anomsouth = northarr, southarr 
    indir = naodir
    midyr, naomodel, nao_ese, tlis, dlis = naoUtils.loadNAO(indir, fp=fp, region=region, ssn=ssn, base1=base1n, base2=base2n,
                                                            ensemble=eselist, central=central, renorm=renorm)
         
    iforobs, ifordec     = naoUtils.index_common(timeobs, midyr)    
    mse = (timeobs[iforobs]- (ylast+0.5))**2 
    iok = numpy.where(mse == mse.min())[0][0]
    idec= ifordec[:iok+1]
    iobs= iforobs[:iok+1]
    acc = naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='uncentred', score='acc') 
    idx=lablist.index('MMESE')
    accarr[idx,kchoice] = acc


    # MMESE_drop - Make NAO index and estimate skill 
    northarr = numpy.ma.masked_invalid(northarr_drp)
    southarr = numpy.ma.masked_invalid(southarr_drp)    
    iyear    = iyear_drp    
    nyr = northarr.shape[0]
    nrlz= northarr.shape[1]
    nfp = northarr.shape[2]
    fcperiod = numpy.array(list(range(1,nfp+1)))

    anomnorth, anomsouth = northarr, southarr
    indir = naodir
    midyr, naomodel, nao_drp, tlis, dlis = naoUtils.loadNAO(indir, fp=fp, region=region, ssn=ssn, base1=base1n, base2=base2n, 
                                                            ensemble=esekeep, central=central, renorm=renorm)
    
    iforobs, ifordec     = naoUtils.index_common(timeobs, midyr)    
    mse = (timeobs[iforobs] - (ylast+0.5))**2 
    iok = numpy.where(mse == mse.min())[0][0]
    idec= ifordec[:iok+1]
    iobs= iforobs[:iok+1]
    acc = naoUtils.ACC_MSSS(naomodel[idec], naoobssm[iobs], sstype='uncentred', score='acc') 
    idx=lablist.index('MMESE_drop')
    accarr[idx,kchoice] = acc

    #for lab,acc in zip(lablist,accarr[:,kchoice]):
    #    print(choice, lab, acc)
    #raise AssertionError('Stop for debugging...') 

ticklist=lablist.copy()
ticklist[ lablist.index('ALL') ]             = 'MMDPE'  
ticklist[ lablist.index('Depresys4_gc3.1') ] = 'Depresys4'   
ticklist[ lablist.index('NCAR') ]            = 'NCAR-40'
ticklist[ lablist.index('CMCC') ]            = 'CMCC-20'
if 'MMESE_drop' in lablist: 
    ticklist[ lablist.index('MMESE_drop') ]  = 'MMESE_drop'+str(ndrop)

accmean    = numpy.mean(accarr,axis=1)
ksort      = numpy.argsort(accmean)
xticknames = numpy.array(ticklist)[ksort]

# Want cases with MM in the name to be plotted last, ie MMESE, MMESE_drop3, MMDPE
ksort1=[]
ksort2=[]
for kk,model in zip(ksort, xticknames):
    if 'MM' in model:
        ksort2.append(kk)
    else:
        ksort1.append(kk)
ksort      = numpy.array(ksort1+ksort2)
xticknames = numpy.array(ticklist)[ksort]

for iname,name in enumerate(xticknames):
    if name in mdic.keys():
        print('Swap name ',name,' -> ',mdic[name]) 
        xticknames[iname] = mdic[name]
    else:
        print('Keep name ',name) 


########################################################################
# Fourth subplot (using above processing)

ax=plt.subplot(2,2,4)

accdic ={'acc_uncentred': 'ACC (uncentred)', 'acc_centred':   'ACC (centred)'} 

xx = 1+numpy.arange(accarr.shape[0])
for ilab,lab1,lab2 in zip(range(nchoices), lablist1, lablist2):
    marker=markerdic[lab1]  
    plt.plot(xx, accarr[ksort,ilab], color=coldic[lab1], alpha=0.6, lw=0.0, marker=marker, ms=6.0, label=lab2)

plt.axhline(0.0, ls=':',lw=1,color='k')
ax.set_xticks(xx)
ax.set_xticklabels(xticknames,rotation=50, ha='right',fontsize=8.00)
ax.xaxis.set_tick_params(pad=0)
cper  = str(int(timeobs[iforobs][0])) +'-'+ str(int(ylast)) 
title = 'NAO Skill, '+clumpt+', '+cper
plt.title(title,fontsize=fstit,pad=titpad)
plt.ylabel(accdic['acc_'+sstype])
leg = plt.legend(loc='best', fontsize=fsleg, handlelength=1.3, borderaxespad=0.3, handletextpad=0.4,labelspacing=0.3)


########################################################################
# Save figure

for dpi in dpiarr:           
    cdpi  = str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)

print('Successful completion of plot_fig12.py')




