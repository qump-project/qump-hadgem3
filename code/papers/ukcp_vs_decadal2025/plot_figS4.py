import os
import numpy
import glob
import scipy
import math
import copy
import cf_units
import datetime

import matplotlib
import matplotlib.pyplot as plt

import iris
import iris.coord_categorisation
import iris.analysis.cartography
import iris.plot as iplt
from iris.time import PartialDateTime

import loadUtils as loadutils
import qdcUtils as utils

from pathnames_v1 import *

###################################################
def extractTime(cube, begdate=None, enddate=None):
    if begdate == None and enddate == None:
        return cube 
    time_coord = cube.coord('time')  
    enddate_value = None    
    if enddate != None:
        if len(enddate) != 3:
            raise AssertionError('enddate not of len=3')
        else:
            enddate_value = PartialDateTime(year=enddate[0], month=enddate[1], day=enddate[2])
    begdate_value = None
    if begdate != None:
        if len(begdate) != 3:
            raise AssertionError('begdate not of len=3')
        else:            
            begdate_value = PartialDateTime(year=begdate[0], month=begdate[1], day=begdate[2])            
    if begdate_value:
        if enddate_value:
            time_constraint = iris.Constraint(time=lambda x:  begdate_value <= x <= enddate_value )
        else:
            time_constraint = iris.Constraint(time=lambda x:  begdate_value <= x )            
    else:
        if enddate_value:
            time_constraint = iris.Constraint(time=lambda x:  x <= enddate_value )                
    cube2     = copy.deepcopy(cube)
    hasbounds = cube.coord('time').has_bounds()
    if hasbounds: 
        cube2.coord('time').bounds=None
    newcube   = cube2.extract(time_constraint)
    if hasbounds and newcube.coord('time').shape[0] > 1: 
        newcube.coord('time').guess_bounds()    
    return newcube

######################################################################
namefig= 'figS4'
ifig   = 1004

dpiarr = [150]

saveplot   = True
#saveplot   = False

var = 'tas'

begdate = [1970,12,1]
enddate = [2000,12,1]

ocnmaskfile = os.path.join(ppedir, 'ocean.pp')
print('Load file: ',ocnmaskfile)
ocnmask_esppe = iris.load_cube(ocnmaskfile)

models=['aldpa','aldpb','aldpc','aldpe','aldpf','aldph','aldpi','aldpj','aldpk','aldpl',
        'aldpm','aldpn','aldpo','aldpp','aldpq','aldqa','aldqb','aldqc','aldqd','aldqe',
        'aldqg','aldqi','aldqj','aldql','aldqm','aldqn','aldqo','aldqp','aldqq','aldra',
        'aldrb','aldrc','aldre','aldrf','aldrg','aldrh','aldri','aldrj','aldrk','aldrl',
        'aldrm','aldrn','aldrq','aldsa','aldsb','aldsc','aldsd','aldse','aldsg','aldsh',
        'aldsi','aldsj','aldsl','aldsn','aldso','aldsp','aldsq']

nmodels=len(models)

naanom=[]
glbanom=[]
for imodel,model in enumerate(models):
    file = os.path.join(ppedir, model+'_tas_ann.nc')    
    print('Load file: ',file)
    gcm=iris.load_cube(file)

    gcm=iris.cube.CubeList([gcm])
    for i,ggg in enumerate(gcm):
        ggg = utils.addRealization(ggg, model)

    utils.tidy_cubes(gcm, set_var_name_None=True )
    utils.tidy_cubes(gcm, guess_bounds=['x','y'] )    
    cube = gcm[0]

    if (var == 'tas' and cube.data.max() > 290.):
        cube  = cube-273.15    
    base   = extractTime(cube, begdate=begdate, enddate=enddate )
    basemn = base.collapsed('time', iris.analysis.MEAN, mdtol=0.0)
    anom   = cube - basemn

    tCrit = iris.Constraint(year=lambda y: y >= 1985-1 )
    futanom=anom.extract(tCrit)

    anomocn = utils.applyMask(anom, ocnmask_esppe)

    #            W    S    E   N
    box_1  = [ -80,   0,   0, 60]   #N.Atl box
    box_2  = [-180, -60, 180, 60]   #Global 60N/60S box
    cube_1 = utils.boxmean(anomocn, box_1, areaweight=True, ignore_bounds=True)
    cube_2 = utils.boxmean(anomocn, box_2, areaweight=True, ignore_bounds=True)

    naanom.append(cube_1.data)
    glbanom.append(cube_2.data)

    grid_areas = iris.analysis.cartography.area_weights(futanom)
    tglb = futanom.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=grid_areas )
    tglbdata  = tglb.data
    tglbdata3 = numpy.expand_dims(tglbdata,axis=(1,2)) 
    futdata   = futanom.data

    denom = numpy.sum(tglbdata*tglbdata)
    numer = numpy.sum(futdata*tglbdata3, axis=0)
    norm  = numer/denom
    patcube = futanom[-1,:,:].copy(data=norm)
    if imodel == 0:
       patsum=patcube
    else:
       patsum=patsum+patcube

# End loop over models   

naanom_esppe  = numpy.array(naanom)
glbanom_esppe = numpy.array(glbanom)


patsum_esppe   = patsum/nmodels
patocn_esppe   = utils.applyMask(patsum_esppe, ocnmask_esppe)
constraint     = iris.Constraint(latitude=lambda cell: -60 <= cell <= 60)
patocn60_esppe = patocn_esppe.extract(constraint)

namean_esppe  = numpy.mean(naanom_esppe,axis=0)
glbmean_esppe = numpy.mean(glbanom_esppe,axis=0)
year_esppe    = anomocn.coord('year').points
 
inflate=0.3
nlev=21

#cmap = 'jet'        ; centre=True 
cmap = 'jet'        ; centre=False 

#cmap = 'RdBu_r'      ; centre=True    #maybe
#cmap = 'coolwarm'    ; centre=True    #maybe
#cmap = 'YlOrRd'      ; centre=False   #maybe
#cmap = 'Spectral_r'  ; centre=True    #maybe
#cmap = 'seismic'    ; centre=True     #maybe

cmap_use = cmap

data_esppe   = patocn60_esppe.data.data[~patocn60_esppe.data.mask]
levels_esppe = utils.setLevels(data_esppe, nlev=nlev, centre=centre, inflate=inflate)

##############################
# 12 CMIP5 ESMs used 
models=['BNU-ESM', 'CESM1-BGC', 'CanESM2', 'GFDL-ESM2G', 'HadGEM2-ES', 'IPSL-CM5A-LR', 
       'MIROC-ESM', 'MPI-ESM-LR', 'MRI-ESM1', 'bcc-csm1-1-m', 'bcc-csm1-1', 'inmcm4']


# Following is N96 (144x192, 1.25x1.875)
gridname    = 'sftlf_fx_HadGEM3-GC31-LL_piControl_r1i1p1f1_gn.nc'
scheme1     = iris.analysis.AreaWeighted()

landfracfile = os.path.join(ppedir, gridname)
print('Load file: ',landfracfile)
gridcube = iris.load_cube(landfracfile)

lndmask_cmip5 = utils.landfrac_to_mask(gridcube, threshold=0.5)
ocnmask_cmip5 = lndmask_cmip5.copy( data=utils.invertMask(lndmask_cmip5.data) )

nmodels=len(models)

naanom=[]
glbanom=[]
for imodel,model in enumerate(models):
    file=os.path.join(cmip5dir, model+'_tas_ann.nc')
    print('Load file: ',file)
    gcm=iris.load_cube(file)
    
    gcm=iris.cube.CubeList([gcm])
    lfscen='historical'
    if model in ['inmcm4']:  lfscen='rcp85'
    if model in ['bcc-csm1-1-m']:  lfscen='piControl'  
    
    for i,ggg in enumerate(gcm):
            ggg  = utils.addRealization(ggg, model)

    utils.tidy_cubes(gcm, set_var_name_None=True )
    utils.tidy_cubes(gcm, guess_bounds=['x','y'] )    
    cube  = gcm[0]

    if (var == 'tas' and cube.data.max() > 290.):
        cube  = cube-273.15    
    base  = extractTime(cube, begdate=begdate, enddate=enddate )
    basemn = base.collapsed('time', iris.analysis.MEAN, mdtol=0.0)
    anomin  = cube - basemn

    lffile= os.path.join(cmip5dir, 'sftlf_fx_'+model+'_'+lfscen+'_r0i0p0.nc')
    print('Load file: ',lffile)    
    lndfrac  = iris.load_cube(lffile)
    anom = utils.regrid_separately(anomin, lndfrac, gridcube, threshold=0.5, name=model)

    tCrit = iris.Constraint(year=lambda y: y >= 1985-1 )
    futanom=anom.extract(tCrit)

    anomocn = utils.applyMask(anom, ocnmask_cmip5)

    #            W    S    E   N
    box_1  = [ -80,   0,   0, 60]   #N.Atl box
    box_2  = [-180, -60, 180, 60]   #Global 60N/60S box
    cube_1 = utils.boxmean(anomocn, box_1, areaweight=True, ignore_bounds=True)
    cube_2 = utils.boxmean(anomocn, box_2, areaweight=True, ignore_bounds=True)

    naanom.append(cube_1.data)
    glbanom.append(cube_2.data)

    grid_areas= iris.analysis.cartography.area_weights(futanom)
    tglb      = futanom.collapsed(['latitude','longitude'], iris.analysis.MEAN, weights=grid_areas )
    tglbdata  = tglb.data
    tglbdata3 = numpy.expand_dims(tglbdata,axis=(1,2)) 
    futdata   = futanom.data

    denom = numpy.sum(tglbdata*tglbdata)
    numer = numpy.sum(futdata*tglbdata3, axis=0)
    norm  = numer/denom
    patcube = futanom[-1,:,:].copy(data=norm)
    if imodel == 0:
       patsum=patcube
    else:
       patsum=patsum+patcube

# End loop over models   

naanom_cmip5  = numpy.array(naanom)
glbanom_cmip5 = numpy.array(glbanom)


patsum_cmip5   = patsum/nmodels
patocn_cmip5   = utils.applyMask(patsum_cmip5, ocnmask_cmip5)
constraint     = iris.Constraint(latitude=lambda cell: -60 <= cell <= 60)
patocn60_cmip5 = patocn_cmip5.extract(constraint)

namean_cmip5  = numpy.mean(naanom_cmip5,axis=0)
glbmean_cmip5 = numpy.mean(glbanom_cmip5,axis=0)
year_cmip5    = anomocn.coord('year').points



cmap = 'jet'        ; centre=True 
cmap = 'jet'        ; centre=False 

#cmap = 'RdBu_r'      ; centre=True   
#cmap = 'coolwarm'    ; centre=True   
#cmap = 'YlOrRd'      ; centre=False  
#cmap = 'Spectral_r'  ; centre=True   
#cmap = 'seismic'     ; centre=True   

cmap_use = cmap

inflate=0.25   #0.05
nlev=21

data_cmip5   = patocn60_cmip5.data.data[~patocn60_cmip5.data.mask]
levels_cmip5 = utils.setLevels(data_cmip5, nlev=nlev, centre=centre, inflate=inflate)


######################################## 
# Plot 4up

xsize=25.
ysize=15.
matplotlib.rcParams.update({'font.size': 10.0})
xfig=xsize/2.54    #convert input size (cm) to inches
yfig=ysize/2.54
figsize = (xfig, yfig)  
fig=plt.figure(ifig, figsize=figsize, facecolor='white')  
plt.subplots_adjust(hspace=0.22, wspace=0.2, top=0.97, bottom=0.1, left=0.04, right=0.97)

subplot=(2,2,1)
shrink=0.95
orientation='vertical'
orientation='horizontal'
fstit=10.5
latlim=[-60,60]

fraction=0.08   #def 0.15
aspect=35       #def 20
padfrac=0.5     #def 1.0

dxax=0.05
dyax=0.10 
pfrac=0.33

###

subplot=(2,2,1)
barlabel='Warming per degree global warming'
title='Mean normalised future Tair response for 57 ESPPE'
lev= utils.plotCubeSubplot(patocn_esppe, levels=levels_esppe, latlim=latlim, cmap=cmap_use, orientation=orientation, 
              subplot=subplot, title=title, fstit=fstit, barlabel=barlabel,
              shrink=shrink, fraction=fraction, aspect=aspect, padfrac= padfrac )
xx=[box_1[0], box_1[2], box_1[2], box_1[0], box_1[0]]
yy=[box_1[1], box_1[1], box_1[3], box_1[3], box_1[1]]
plt.plot(xx, yy, color='k',lw=1.5, ls='--')
 
 
#location of the subplot is defined as [left, bottom, width, height] in figure-normalized units
ax = fig.add_axes([0.5+dxax,  0.5+dyax,  pfrac,  pfrac]) 
plt.plot(year_esppe, namean_esppe, color='red', label='North Atlantic')
plt.plot(year_esppe, glbmean_esppe, color='blue', label='Global ocean 60S-60N')
plt.axhline(0.0,color='k',ls=':',lw=1)
plt.ylabel('$\Delta$T ($\degree$C)')
plt.xlabel('Year')
plt.title('ESPPE ensemble mean RCP85 Tair response',fontsize=fstit)
leg = plt.legend(loc='best', fontsize=9, handlelength=1.3,borderaxespad=0.3,handletextpad=0.3,labelspacing=0.3) 
 
 
subplot=(2,2,3)
barlabel='Warming per degree global warming'
title='Mean normalised future Tair response for 12 CMIP5 ESM'
lev= utils.plotCubeSubplot(patocn_cmip5, levels=levels_esppe, latlim=latlim, cmap=cmap_use, orientation=orientation, 
              subplot=subplot, title=title, fstit=fstit, barlabel=barlabel,
              shrink=shrink, fraction=fraction, aspect=aspect, padfrac= padfrac )
xx=[box_1[0], box_1[2], box_1[2], box_1[0], box_1[0]]
yy=[box_1[1], box_1[1], box_1[3], box_1[3], box_1[1]]
plt.plot(xx, yy, color='k',lw=1.5, ls='--')
 
 
#location of the subplot is defined as [left, bottom, width, height] in figure-normalized units
ax = fig.add_axes([0.5+dxax,  0.0+dyax,  pfrac,  pfrac]) 
plt.plot(year_cmip5, namean_cmip5, color='red', label='North Atlantic')
plt.plot(year_cmip5, glbmean_cmip5, color='blue', label='Global ocean 60S-60N')
plt.axhline(0.0,color='k',ls=':',lw=1)
plt.ylabel('$\Delta$T ($\degree$C)')
plt.xlabel('Year')
plt.title('CMIP5-ESM ensemble mean RCP85 Tair response',fontsize=fstit)
leg = plt.legend(loc='best', fontsize=9, handlelength=1.3,borderaxespad=0.3,handletextpad=0.3,labelspacing=0.3) 


for dpi in dpiarr:           
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)

 
 
 
 
 
