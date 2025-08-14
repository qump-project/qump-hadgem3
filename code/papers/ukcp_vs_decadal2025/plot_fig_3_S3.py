##################################################
import os
import time
import sys
import scipy.io
import numpy
import matplotlib
import matplotlib.pyplot as plt

import iris

import qdcUtils as utils

from pathnames_v1 import *

##################################################
figno = input("Enter figure number, one of (3, S3): ")
figno = figno.upper()
if figno not in ['3', 'S3']:
    raise AssertionError('Incorrect figno, please enter one of (3, S3)')        
namefig = 'fig'+figno

dpiarr  = [150]

saveplot = True
#saveplot = False

plotvars=['glb_tas_ann', 'amo_tas_ann', 'naosteph_psl_djf']

if namefig == 'figS3':
    nlump = 1
    ifig  = 1003
elif namefig == 'fig3':
    nlump = 8
    ifig  = 3

regdic={'GLB':'GLB', 'AMO':'AMO', 'NAO':'NAO', 'NAOSTEPH':'NAO', 'NAOSMITH':'NAO', 'NEU':'Northern Europe', 'EAW':'England-Wales' }

basebeg=1971
baseend=basebeg+30

nboot   = 8000   #8000
quant   = [0.05, 0.95]
quant80 = [0.1, 0.9]

#ni  = 1
ni  = 20
cni = '_ni='+str(ni)

wcrit=30

npoolin   = 31
npooltglb = 1

if nlump == 1:
    tbeg=1960   ; tend=2022
elif nlump == 8:
    tbeg=1965   ; tend=2018

#titdic={'glb_tas_ann':      'Annual GMST, RCP45', 
#        'amo_tas_ann':      'Annual AMV, RCP45',
#        'naosteph_psl_djf': 'DJF NAO, RCP45'}

titdic={'glb_tas_ann':      'Annual GMST', 
        'amo_tas_ann':      'Annual AMV',
        'naosteph_psl_djf': 'Winter NAO'}

# Initialize figure
fs=9.0
matplotlib.rcParams.update({'font.size': fs})
plt.figure(ifig, figsize=(9/2.54, 18/2.54))
plt.subplots_adjust(hspace=0.29, wspace=0.27, top=0.95, bottom=0.05, left=0.21, right=0.94)

for iplot, plotvar in enumerate(plotvars):

    print('\n>>> Processing iplot=',iplot,'plotvar=',plotvar)
    pvsplit= plotvar.split('_')

    reg   = pvsplit[0].upper()
    fld   = pvsplit[1].lower()
    ssn   = pvsplit[2].lower()

    if fld == 'tas':
        unit='($\degree$C)'  
    elif fld == 'pr': 
        unit='(%)'    
    elif fld == 'psl': 
        unit='(hPa)'    

    npool = npoolin
    if   plotvar == 'glb_tas_ann':
        npool = npooltglb
        ymx = 1.8
    elif  plotvar == 'amo_tas_ann': 
        ymx = 1.0
    elif plotvar == 'naosteph_psl_djf': 
        ymx = 9.0
      
    ### OBS
    ssnhdr={'djf': 'win', 'mam':'spr', 'jja':'sum', 'son': 'aut'}
    yearhdr='year'
    if reg.upper() in ['GLB']:
        obsname = 'tglb_ann_mean_hadcrut5_anom2023_b19712000.nc'
        fileobs = os.path.join(obsdir, obsname)
        ha='left'
        if nlump == 8:
            ylim=[-1.0, 3.5]        
        else:
            ylim=[-1.1, 3.7]

    elif reg in ['AMO']:
        obsname = 'amo_hadcrut5_1851-2023_ann.nc'
        fileobs = os.path.join(obsdir, obsname)
        ha='left'
        if nlump == 8: 
            ylim=[-0.3, 1.15]
        else:
            ylim=[-0.36, 1.30]               

    else:   # ['NAO']        
        obsname ='era5_1851-2024_nao_stephenson_djf_notnorm.nc'
        fileobs = os.path.join(obsdir, obsname)        
        ha='right'
        if nlump == 8: 
            ylim=[-6, 10.5]
        else:
            ylim=[-13, 27.0]
        
    print('>>> Loading OBS file: ',fileobs)
    ocube = iris.load_cube(fileobs)
         
    odata = ocube.data
    otime = utils.time2real(ocube)

    testcond= (otime >= 1971) & (otime < 2001.0)
    ibase = numpy.where(testcond )[0]
    if ibase.shape[0] != 30:
        raise AssertionError('Stop for debugging...')
    baseval= numpy.mean(odata[ibase])
    odata  = odata - baseval
    
    otime = utils.lump(otime, nlump=nlump)
    odata = utils.lump(odata, nlump=nlump)

    # Obs variability
    obs_low   = utils.butterworth(odata, wcrit,axis=0)
    noise_obs = odata - obs_low
    ci_obs    = utils.ci_func(noise_obs, quant80) 

    # Need to use lumped times, estimate int var for ukcp over same period as obs.
    timeiv_obs = [max([otime[0], 1900.]), otime[-1] ]
    timeiv_ukcp= timeiv_obs

    tit1 = titdic[plotvar]

    if nlump == 1:
        tit=tit1+', T1'
    elif nlump == 8:
        tit=tit1+', T2-9'
    else:
        raise AssertionError('Stop for debugging...')

    ### UKCP residual spread
    suf      = 'ns=3000'+cni+'_l'+str(nlump)
    dataname = 'ukcp_res_quant_'+plotvar+'_'+suf+'.npz'
    sprdfile = os.path.join(ukcpdir, dataname)
    print('>>> Loading UKCP residual spread: ',sprdfile)
    a = numpy.load(sprdfile)
    pquant= a['pquant']
    quant = a['quant']
    ci_ukcp18 = utils.get_spread(0.1, 0.9, pquant, quant)
    print(plotvar,'ci_ukcp18=',ci_ukcp18)


    ### UKCP plume    
    basestr= 'b7100'
    regu=reg.upper()
    regl=reg.lower()    
    ssn1 = ssn
    if ssn == 'ann':   ssn1='annual' 
    regc=regu
    if reg == 'GLB':   regc=regl
    unitc='k'
    if fld == 'psl':   unitc='h'

    pname     = 'ukcp23_'+regu+'_'+fld+'_'+ssn1+'_'+unitc+'_' + basestr +'_w'+str(npool)+cni+'_l'+str(nlump)+ '.nc'    
    plumefile = os.path.join(ukcpdir, pname)    
    print('>>> Loading UKCP plume file: ',plumefile)
    cplume = iris.load_cube(plumefile)
    time_ukcp2  = utils.time2real(cplume)
    
    if fld == 'pr':
        iris.coord_categorisation.add_year(cplume, 'time')  

    # make ci80, recall c.coord('percentile').points = [0.025, 0.05 , 0.1  , 0.25 , 0.5  , 0.75 , 0.9  , 0.95 , 0.975]
    tConstraint = iris.Constraint(year=lambda y: y >= tbeg and y <= tend )
    pcon10 = iris.Constraint(percentile=0.1)
    pcon50 = iris.Constraint(percentile=0.5)
    pcon90 = iris.Constraint(percentile=0.9)

    cube10= cplume.extract( pcon10 )
    cube50= cplume.extract( pcon50 )
    cube90= cplume.extract( pcon90 )

    p10 = cube10.data
    p50 = cube50.data
    p90 = cube90.data
    # Used to have optional smoothing here
    ci_all = cube90.data-cube10.data   #all times

    cplumesub= cplume.extract( tConstraint )
    cube10sub= cplumesub.extract( pcon10 )
    cube90sub= cplumesub.extract( pcon90 )
    ci = cube90sub-cube10sub                 #cube
    # Used to have optional smoothing here
    ci_ukcp18_2 = float( ci.collapsed('time', iris.analysis.MEAN).data )
            
    print('>>> ci_ukcp18_2=',ci_ukcp18_2)
    print('>>> p10[-1],p50[-1], p90[-1] =', p10[-1],',',p50[-1],',',p90[-1] )


    imed=2
    if fld == 'tas':
        plumecol= 'red'
        plume_alpha=numpy.array([0.16, 0.29, 0.05 ])
    else:
        plumecol= 'royalblue'
        plume_alpha=numpy.array([0.21, 0.38, 0.05 ])

    fstit=10.5
    lwobs=1.25
    
    #########
    ax=plt.subplot(3,1, iplot+1)

    edgecolor='blue'
    alphaplume=0.20
    lwedge=0.75
    plt.fill_between(time_ukcp2, p10, p90, color='grey', alpha=alphaplume)   
    plt.plot(time_ukcp2, p10, color=edgecolor, linewidth=lwedge)
    plt.plot(time_ukcp2, p50, color=edgecolor, linewidth=lwedge)
    plt.plot(time_ukcp2, p90, color=edgecolor, linewidth=lwedge)

    plt.plot(otime, odata, color='k', linewidth=lwobs, alpha=0.8)

    ax.set_xlim([1900,2100])
    plt.axhline(0.0,ls=':',lw=0.75,color='k')
    ax.set_ylim(ylim)

    ylim=ax.get_ylim()
    
    spreadcol='darkorange'
    spreadcol='green'        
    
    fspt=8.0
    lwspread=1.0
    ms=3.0
    ivcol='purple'
    alphadark=0.7
    
    if fld == 'pr':
        ccco = '%3.f'%ci_obs  +'%'   
        ccc1 = '%3.f'%ci_ukcp18 +'%' 
        ccc2 = '%3.f'%ci_ukcp18_2 +'%'   
    elif fld == 'tas':
        ccco = '%4.2f'%ci_obs +'$\degree$C'
        ccc1 = '%4.2f'%ci_ukcp18 +'$\degree$C'
        ccc2 = '%4.2f'%ci_ukcp18_2 +'$\degree$C'
    elif fld == 'psl':
        ccco = '%4.2f'%ci_obs +'hPa'
        ccc1 = '%4.2f'%ci_ukcp18 +'hPa'
        ccc2 = '%4.2f'%ci_ukcp18_2 +'hPa'
          
    plt.plot(time_ukcp2, ci_all, lw=lwspread, ls='--', color=spreadcol)
    plt.plot(timeiv_ukcp, [ci_ukcp18,ci_ukcp18], lw=1.0, ls='--', color=ivcol,alpha=alphadark)
    plt.plot(timeiv_obs,  [ci_obs,ci_obs],       lw=1.0, ls='--', color='k',alpha=alphadark)            
    
    ll=[] ; labels=[]

    facecolor='grey'
    edgecolor='blue'
    fc_for_rectangle = matplotlib.colors.ColorConverter().to_rgba(facecolor, alpha=alphaplume)
    handle_ukcp      = plt.Rectangle( (0, 0), 0, 0, edgecolor=edgecolor, fc=fc_for_rectangle, lw=lwedge)
    ll.append( handle_ukcp )
    labels.append('UKCP-pdf (10,50,90%)')

    ll.append( matplotlib.lines.Line2D([], [], color=spreadcol, lw=lwspread, ls='--') )   
    labels.append('P90-P10 (UKCP-pdf): '+ccc2)
 
    ll.append( matplotlib.lines.Line2D([], [], color='k', lw=lwobs) )
    labels.append('Obs')

    ll.append( matplotlib.lines.Line2D([], [], color=ivcol, lw=1.0, ls='--', alpha=alphadark) )           
    labels.append('UKCP-pdf residual: '+ccc1)        
    ll.append( matplotlib.lines.Line2D([], [], color='k', lw=1.0, ls='--', alpha=alphadark) )   
    labels.append('Obs variability: '+ccco)
     
    plt.ylabel('Anomaly '+unit)       
    plt.title(tit,fontsize=fstit)

    loc='upper left'
    labelspacing=0.1
    if iplot in [0,1,2]:   #[0]
        if iplot == 2: labelspacing=0.24
        leg = plt.legend(ll, labels, loc=loc, fontsize=8.0, handlelength=1.20, borderaxespad=0.25, handletextpad=0.25, labelspacing=labelspacing)
        leg.draw_frame(False)

    print('>>> ',tit,'P50[-1]       =',p50[-1])
    print('>>> ',tit,'P50[-20:].mean=',p50[-20:].mean())

# End loop over var

for dpi in dpiarr:            
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)


