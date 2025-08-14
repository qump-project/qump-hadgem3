import os
import time
import sys
import scipy.io
import numpy
import matplotlib
import matplotlib.pyplot as plt

import iris
import iris.coord_categorisation

import qdcUtils as utils

from pathnames_v1 import *

##################################################
figno = input("Enter figure number, one of (4, 5, S5, S6): ")
figno = figno.upper()
if figno not in ['4', '5', 'S5', 'S6']:
    raise AssertionError('Incorrect figno, please enter one of (4, 5, S5, S6)')        
namefig = 'fig'+figno

dpiarr  = [150]

saveplot   = True
#saveplot   = False


if namefig == 'fig4':
    plotvars=['neu_tas_djf', 'neu_tas_jja', 'eaw_tas_djf', 'eaw_tas_jja' ]
    nlump = 8
    ifig  = 4

elif namefig == 'fig5':
    plotvars=['neu_pr_djf', 'neu_pr_jja', 'eaw_pr_djf', 'eaw_pr_jja' ]
    nlump = 8
    ifig  = 5

elif namefig == 'figS5':
    plotvars=['neu_tas_djf', 'neu_tas_jja', 'eaw_tas_djf', 'eaw_tas_jja']
    nlump = 1
    ifig  = 1005

elif namefig == 'figS6':
    plotvars=['neu_pr_djf', 'neu_pr_jja', 'eaw_pr_djf', 'eaw_pr_jja']
    nlump = 1
    ifig  = 1006



# Hardware y-axis range for paper
if namefig in ['figS5', 'figS6']:
    ylimdic={'neu_tas_djf': [-4.3, 10.2],  
             'neu_tas_jja': [-1.9, 7.5],   
             'eaw_tas_djf': [-4.8, 8.3],   
             'eaw_tas_jja': [-2.0, 7.1],
             'neu_pr_djf': [-35., 66.0],  
             'neu_pr_jja': [-45., 73.0],     
             'eaw_pr_djf': [-68., 118.0],   
             'eaw_pr_jja': [-71., 150.0] }

elif namefig in ['fig4', 'fig5']:
    ylimdic={'neu_tas_djf': [-2.8, 7.0],  
             'neu_tas_jja': [-1.1, 6.8],   
             'eaw_tas_djf': [-1.5, 4.1],   
             'eaw_tas_jja': [-1.1, 6.3],
             'neu_pr_djf': [-17., 42.0],  
             'neu_pr_jja': [-40., 50.0], 
             'eaw_pr_djf': [-24.5, 50.0],   
             'eaw_pr_jja': [-63., 60.0] }

titdic ={'neu_tas_djf': 'NEU winter Tair', 
         'neu_tas_jja': 'NEU summer Tair', 
         'eaw_tas_djf': 'EngWal winter Tair', 
         'eaw_tas_jja': 'EngWal summer Tair',
         'neu_pr_djf':  'NEU winter precipitation', 
         'neu_pr_jja':  'NEU summer precipitation', 
         'eaw_pr_djf':  'EngWal winter precipitation', 
         'eaw_pr_jja':  'EngWal summer precipitation' }

#ni  = 1
ni  = 20
cni = '_ni='+str(ni)

npool = 1 
npool = 31

   
wcrit=30
datestamp = '20240926'    
mdi       = 99999.

if nlump == 1:
    tbeg=1960   ; tend=2022
elif nlump == 8:
    tbeg=1965   ; tend=2018


regdic={'GLB':'GLB', 'NEU':'Northern Europe', 'EAW':'England-Wales'}

basebeg=1971
baseend=basebeg+30

nboot = 8000   #8000
quant = [0.05, 0.95]
quant80 = [0.1, 0.9]

fs=9.0
matplotlib.rcParams.update({'font.size': fs})
plt.figure(ifig,figsize=(16/2.54, 14.5/2.54))
plt.subplots_adjust(hspace=0.265, wspace=0.265, top=0.95, bottom=0.060, left=0.11, right=0.97)

# plot_fig4and5.py has
#plt.figure(ifig,figsize=(16/2.54, 15/2.54))
#plt.subplots_adjust(hspace=0.27, wspace=0.27, top=0.92, bottom=0.08, left=0.10, right=0.97)

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

    title=fld+'_'+ssn+'_'+reg
    
    if fld == 'tas':
        pv_in = ssn.upper() +' '+ regdic[reg.upper()] +' '+ fld
    else:
        pv_in = '%change '+ ssn.upper() +' '+ reg.upper() +' '+ fld

    ### OBS
    ssnhdr={'djf': 'win', 'mam':'spr', 'jja':'sum', 'son': 'aut'}
    yearhdr='year'
    
    if reg in ['EAW', 'SCO', 'SENG']:
        obsc='ncic'
        if fld == 'tas':
            obsname = obsc + '_tasmean_' + reg.lower() + '_'+datestamp+'.txt'
        else:
            obsname = obsc + '_pr_' + reg.lower() + '_'+datestamp+'.txt'

        fileobs = os.path.join(obsdir,obsname)        
        print('>>> Loading OBS file: ',fileobs)
        o = numpy.genfromtxt(fileobs, names=True, skip_header=5, missing_values='---', filling_values=mdi )
        iok = numpy.where(o[ssnhdr[ssn]] != mdi)[0]
        otime = o[yearhdr][iok]
        odata = o[ssnhdr[ssn]][iok]

    else:   #eg  ['NEU']
        obsc='obs'
        if fld == 'tas':
            obsname = 'tas_hadcrut5_1851-2023_reg2_ssn2.nc'                   
        else:
            obsname = 'pr_gpcc_1892-2023_reg2_ssn2.nc'

        fileobs = os.path.join(obsdir, obsname)
        print('>>> Loading OBS file: ',fileobs)
        clist=iris.load(fileobs)
        pv = [c.coord('Projection_variable').points[0] for c in clist]
        idx  = pv.index(pv_in)
        ocube = clist[idx]
        iris.coord_categorisation.add_season_year(ocube, 'time')
        
        ocubeb = ocube.extract(iris.Constraint(season_year=lambda cell: 1971 <= cell <= 2000))
        base=ocubeb.collapsed('season_year',iris.analysis.MEAN)
        ocube=ocube-base
        if fld == 'pr':  # rebase % anom
            ocube = ocube/(1.+base/100)            
        otime = utils.time2real(ocube)   
        odata = numpy.ma.filled(ocube.data)    #this removes mask

    
    testcond= (otime >= 1971) & (otime < 2001.0)
    ibase = numpy.where(testcond )[0]
    if ibase.shape[0] != 30:
        raise AssertionError('Stop for debugging...')
    baseval= numpy.mean(odata[ibase])
    
    otime = utils.lump(otime, nlump=nlump)
    odata = utils.lump(odata, nlump=nlump)

    # Need to re-estimate anoms after lumping
    # Uk variables are actual anoms, but NEU is % anom already
    if fld == 'tas':
        odata = odata - baseval
    else:
        if reg in ['EAW', 'SCO', 'SENG']:
            odata = 100.*(odata/baseval-1.)
        else:    #NEU already %
            odata = (odata - baseval) / (1. + baseval/100)
        
    # Obs variability
    obs_low   = utils.butterworth(odata, wcrit,axis=0)
    noise_obs = odata - obs_low
    ci_obs    = utils.ci_func(noise_obs, quant80) 

    # Need to use lumped times, and now estimate int var for ukcp over same period as obs.
    timeiv_obs = [max([otime[0], 1900.]), otime[-1] ]
    timeiv_ukcp= timeiv_obs

    tit1=titdic[plotvar]   
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
    print(plotvar,'ci_ukcp18=',ci_ukcp18,'ci_obs=',ci_obs)


    ### UKCP plume    
    basestr= 'b7100'
    regu=reg.upper()
    regl=reg.lower()    
    ssn1 = ssn
    if ssn == 'ann':   ssn1='annual' 
    regc=regu
    if fld == 'tas':   unitc='k'
    if fld == 'pr':    unitc='%'
  
    if fld == 'tas':
        pname = 'ukcp23_'+regl+'_'+fld+'_'+ssn1+'_'+unitc+'_' + basestr +'_w'+str(npool)+cni+'_l'+str(nlump)+ '.nc'
    elif fld == 'pr':
        pname = 'ukcp23_'+regl+'_'+fld+'_'+ssn1+'_'+unitc+'_' + basestr +'_w'+str(npool)+cni+'_l'+str(nlump)+ '.nc'

    plumefile = os.path.join(ukcpdir, pname)    
    print('>>> Loading UKCP file: ',plumefile)
    cplume = iris.load_cube(plumefile)
    time_ukcp2  = utils.time2real(cplume)
    if fld == 'pr':
        iris.coord_categorisation.add_year(cplume, 'time')  


    # make ci80, remember c.coord('percentile').points = [0.025, 0.05 , 0.1  , 0.25 , 0.5  , 0.75 , 0.9  , 0.95 , 0.975]
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
    ci_all = cube90.data-cube10.data   #all times

    cplumesub = cplume.extract( tConstraint )
    cube10sub= cplumesub.extract( pcon10 )
    cube90sub= cplumesub.extract( pcon90 )
    ci = cube90sub-cube10sub                 #cube    
    ci_ukcp18_2 = float( ci.collapsed('time', iris.analysis.MEAN).data )

    fstit=10.5
    lwobs=1.0
    
    #########
    ax=plt.subplot(2,2, iplot+1)

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

    ax.set_ylim(ylimdic[plotvar])
    ylim=ax.get_ylim()
    dy= 0.005*(ylim[1]-ylim[0])
    
    spreadcol='darkorange'
    spreadcol='green'            
    fspt     = 8.0
    lwspread = 1.0
    ms       = 3.0
    ivcol    = 'purple'
    alphadark= 0.7

    plt.plot(time_ukcp2, ci_all,                 lw=lwspread, ls='--', color=spreadcol)
    plt.plot(timeiv_ukcp, [ci_ukcp18,ci_ukcp18], lw=1.0, ls='--', color=ivcol,alpha=alphadark)
    plt.plot(timeiv_obs,  [ci_obs,ci_obs],       lw=1.0, ls='--', color='k',alpha=alphadark)

    if fld == 'pr':
        dx = 1.0
        ccco = '%3.1f'%ci_obs +'%'   
        ccc1 = '%3.1f'%ci_ukcp18 +'%'
        ccc2 = '%3.1f'%ci_ukcp18_2 +'%'  
    else:
        dx = 2.5
        ccco = '%4.2f'%ci_obs +'$\degree$C'
        ccc1 = '%4.2f'%ci_ukcp18 +'$\degree$C'
        ccc2 = '%4.2f'%ci_ukcp18_2 +'$\degree$C'

    ll=[] ; labels=[]
    facecolor='grey'
    edgecolor='blue'
    fc_for_rectangle = matplotlib.colors.ColorConverter().to_rgba(facecolor, alpha=alphaplume)
    handle_ukcp      = plt.Rectangle( (0, 0), 0, 0, edgecolor=edgecolor, fc=fc_for_rectangle, lw=lwedge)
    
    if namefig == 'figS6' and iplot in [2,3] and fld == 'pr':
        #drop a line from legend to save a bit of space for this fig
        pass
    else:
        ll.append( handle_ukcp )
        labels.append('UKCP-pdf (10,50,90%)')
    
    ll.append( matplotlib.lines.Line2D([], [], color=spreadcol, lw=lwspread, ls='--') )   
    labels.append('P90-P10 (UKCP-pdf): '+ccc2)
       
    ll.append( matplotlib.lines.Line2D([], [], color='k', lw=lwobs, alpha=0.8) )
    labels.append('Obs')
  
    ll.append( matplotlib.lines.Line2D([], [], color=ivcol, lw=1.0, ls='--', alpha=alphadark) )           
    labels.append('UKCP-pdf residual: '+ccc1)
        
    ll.append( matplotlib.lines.Line2D([], [], color='k', lw=1.0, ls='--', alpha=alphadark) )   
    labels.append('Obs variability: '+ccco)
    
    plt.ylabel('Anomaly '+unit)       
    plt.title(tit,fontsize=fstit,pad=4)

    loc='upper left'
    if namefig == 'fig5' and iplot == 3:
        loc='lower left'

    if fld == 'tas':  labelspacing=0.13         
    if fld == 'pr':   labelspacing=0.18  
    #borderaxespad=0.05
    borderaxespad=0.1
        
    if iplot in [0,1,2,3]: 
        leg = plt.legend(ll, labels, loc=loc, fontsize=8.0, handlelength=1.2, handletextpad=0.25, 
                         borderaxespad=borderaxespad, labelspacing=labelspacing)
        leg.draw_frame(False)
    
# End loop over ssn

for dpi in dpiarr:            
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)



















