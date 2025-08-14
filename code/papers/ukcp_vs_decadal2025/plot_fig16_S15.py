# Plot MAPS of AMO effects on precip AND the RANGE of these effects from all simulations put into a big pot
# This code produces 2 maps of precip: 1 from the decadal hindcasts, and the 2nd of obs,
# AMO here is annual surface air temperature for the N Atlantic minus global 60S to 60N air temp.
# Now also do plots conditional upon sign of NAO index. 

import os
import numpy

# Plotting functions:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('classic')
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import iris
import iris.plot as iplt
import iris.quickplot as qplt

import seasonclass
import regionclass
import weatherfieldclass

import amo_functions
import qdcUtils as utils

from pathnames_v1 import *

####################################################################
def extractbox(cube, box=[-40.0, 35.0, 40.0, 75.0], ignore_bounds=False):
    # Determine whether cube is a cube or a cubelist:
    if type(cube) is iris.cube.CubeList:
        # Have cubelist. Go through each cube:
        allcubes=[]
        for cube1 in cube:
            print('Processing: ',cube1.coord('season_year').points[0])
            cube1 = extractbox(cube1,box=box)
            allcubes.append(cube1)
        return iris.cube.CubeList(allcubes)
    else:
        ce_lon = iris.coords.CoordExtent('longitude', box[0], box[2])
        if ce_lon.minimum > ce_lon.maximum:
            ce_lon = iris.coords.CoordExtent('longitude', box[0], box[2]+360.0)

        ce_lat = iris.coords.CoordExtent('latitude', box[1], box[3])

        keep_all_longitudes=False
        if box[0] == 0 and box[2] == 0:       keep_all_longitudes=True
        if box[0] == 0 and box[2] == 360:     keep_all_longitudes=True
        if box[0] == -180 and box[2] == 180:  keep_all_longitudes=True

        if keep_all_longitudes:
            # Want to just extract over latitudes:
            ans = cube.extract(iris.Constraint(latitude=lambda cell: box[1] <= cell <= box[3]))
        else:
            ans = cube.intersection(ce_lon, ce_lat, ignore_bounds=ignore_bounds)
        return ans



########################################################################
# Start
########################################################################
figno = input("Enter figure number, one of (16, S15): ")
figno = figno.upper()
if figno not in ['16', 'S15']:
    raise AssertionError('Incorrect figno, please enter one of (16, S15)')        
namefig = 'fig'+figno


dpiarr = [150]

saveplot=False
#saveplot=True

maskocn = True   
#maskocn = False

forecast_period = 2

if namefig == 'fig16':
    amo_season  = 'djfmamjjason' 
    ppt_season  = 'jja'
    driver      = 'amo'
    ifig        = 16

if namefig == 'figS15':
    amo_season  = 'djf' 
    ppt_season  = 'djf'
    driver      = 'nao'
    ifig        = 1015
    


amo_season=seasonclass.InitSeason(amo_season)
ppt_season=seasonclass.InitSeason(ppt_season)

cdriver = driver.upper()
if driver == 'amo': cdriver='AMV'


fig    = None
basebeg= 1961
nbase  = 30

climate_periodstr = '_using'+str(basebeg)+'to'+str(basebeg+nbase-1)+'climate'

ppt_region='Globalukcp09' # global but with ukcp09 grid later used
ppt_region='England and Wales'
ppt_region='NW Europe'
print('ppt_region:',ppt_region)

ppt_region = regionclass.InitRegion(ppt_region)

obsdriver = driver

if driver == 'amo':
    #amoobssource='hadcrut5'
    #amoobssource='cowtanway_latlong'
    amoobssource='hadcrut5_latlong'  

if driver=='elnino':
    amoobssource='hadsst4'

if driver == 'nao':
    amoobssource= 'era5'    
    #obsdriver   = 'nao_stephenson'
    #obsdriver   = 'nao'


# Set the type of amo anomaly to use for the extraction of years of -ve and +ve amo:

## To use a baseline climate, eg 1971 to 2000:
#amo_anomaly_format = 'of_a_baseline_climate'

# To use a climate calculated over the entire period ofavailable data:
amo_anomaly_format = 'of_whole_period'

desired_precip_output_format = '+ve amo minus -ve amo as fraction of -ve'


if amo_anomaly_format == 'of_a_baseline_climate':
    anomalise_amo = True
    basebeg       = 1971
    nbase         = 30

if amo_anomaly_format == 'of_whole_period':
    anomalise_amo = True
    basebeg       = None
    nbase         = None

#weathersofinterest = ['air_pressure_at_sea_level']                   ; varlab='psl'
#weathersofinterest = ['precipitation']                               ; varlab='pr'
#weathersofinterest = ['temperature']                                 ; varlab='tas'
weathersofinterest = ['air_pressure_at_sea_level', 'precipitation']   ; varlab='psl_pr'

#regrid_target = 'coarsest' 
#regrid_target = 'ncar'  
regrid_target = 'depresys'  

start_forecast_reference_year = 1960

mdic={'BCC':'BCC',         'BSC':'BSC',      'CAFE':'CAFE',     'CMCC':'CMCC',    'CanESM5':'CCCma',
      'IPSL':'IPSL',       'MIROC6':'MIROC', 'MPI':'MiKlip',    'NCAR40':'NCAR',  'NorCPM':'NCC-i1', 'NorCPMi2':'NCC-i2', 
      'Depresys4_gc3':'DePreSys4', 'Depresys4_gc3.1':'DePreSys4'}

panelnumber=1


fs=9.5
matplotlib.rcParams.update({'font.size': fs})
fig = plt.figure(ifig, figsize=(18/2.54, 12.5/2.54))

plt.subplots_adjust(top=0.955, bottom=0.07, left=0.055, wspace=0.15, hspace=0.27, right=0.975) 

for weatherofinterest in weathersofinterest:

    if weatherofinterest == 'precipitation':             
         obssource = 'gpcc'
    
    if weatherofinterest == 'air_pressure_at_sea_level': 
        obssource='era5'
    
    if weatherofinterest == 'temperature':               
        obssource = 'hadcrut5_latlong'

    if desired_precip_output_format == '+ve amo minus -ve amo as fraction of -ve':
       # James wants +ve amo precip minus -ve amo precip so switch off precip anomalising 
       # (which returns precip as percentage of the climate):
       anomalise_ppt = False
    else:
       anomalise_ppt = True

    allobs = True

    nyrmax = None  #had this in dec 2023, use ALL years from each model
    #nyrmax = 57   #For T+2, equiv to 1962-2018 inclusive
   
    #start_year_of_obs = 1960   #RC
    start_year_of_obs = 1962   #First T+2 year

    if weatherofinterest == 'precipitation': 
        weather_field = weatherfieldclass.InitWeatherfield('precipitation_flux','','')
    if weatherofinterest == 'temperature':         
        weather_field = weatherfieldclass.InitWeatherfield('air_temperature',1.5,'m')
    if weatherofinterest == 'air_pressure_at_sea_level': 
        weather_field = weatherfieldclass.InitWeatherfield('air_pressure_at_sea_level','','')
        
        
    if allobs:
        if weatherofinterest == 'precipitation': 
            numberof_years_of_obs=63  # inc 2022
            numberof_years_of_obs=64  # ??? inc 2023  TRY for era5
            numberof_years_of_obs=62  # 62 = 2023-1962+1      
        if weatherofinterest == 'temperature': 
            numberof_years_of_obs=63  # inc 2022
            numberof_years_of_obs=64  # ??? inc 2023 TRY for era5
            numberof_years_of_obs=62  # 62 = 2023-1962+1   
        if weatherofinterest == 'air_pressure_at_sea_level': 
            numberof_years_of_obs=62  # inc 2021
            numberof_years_of_obs=64  # ??? inc 2023  TRY for era5?
            numberof_years_of_obs=62  # 62 = 2023-1962+1  
    else:
         numberof_years_of_obs=nyrmax   #not used now
        
    cubes_to_plot_when_amo_neg=[]
    cubes_to_plot_when_amo_pos=[]
    cubes_to_plot_when_amo_pos_minus_neg=[]
    
    # Start list for cubes of each model to be used for an ensemble mean:
    ensemble_cubes_when_amo_pos_minus_neg=[]

    panel_titles=[]

    #ensemblesall = ['BCC','BSC','CAFE','CanESM5','CMCC','Depresys4_gc3','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']   #12 
    ensemblesall = ['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']         #11 

    ensembles = ensemblesall
    if (driver=='nao' or driver=='amo') and forecast_period == 1:
        # Models with no DJF t+1 yr data
        ensembles.remove('BCC')
        ensembles.remove('CanESM5')
        ensembles.remove('IPSL')

    number_of_neg_amo_events = 0
    number_of_pos_amo_events = 0

    model_ppt_cubes_when_amo_pos = []
    model_ppt_cubes_when_amo_neg = []

    if ensembles is not None:
    
        for ensemble_index,ensemble in enumerate(ensembles):
           if ensemble=='Depresys4_gc3': ensemble='Depresys4_gc3.1' 
           print('ensemble:',ensemble)
  
           numberof_forecast_reference_years = amo_functions.number_of_years_of_initialised_forecasts(ensemble)
           print('number of ',ensemble,' forecast_reference_years:',numberof_forecast_reference_years)

           # Get all Index predictions from ensemble (AMO or NAO typically):
           dps_amo_all_years = amo_functions.get_model_idx(dpsdir, season=amo_season, driver=driver,
                                         forecast_period=forecast_period, anomalise_amo=anomalise_amo,
                                         basebeg=basebeg, nbase=nbase, ensemble=ensemble,
                                         start_forecast_reference_year=start_forecast_reference_year,
                                         numberof_forecast_reference_years=numberof_forecast_reference_years)
    
           # dps_amo_all_years is a cubelist of cubes of amo.
           # Each cube is from a single forecast_reference_time. Each cube has a
           # dim coordinate: realization and is of a single forecast_period (given as an integer 1,2 3 etc
            
           # Get all precip predictions from ensemble:
           dps_ppt_all_years = amo_functions.get_model_ppt(dpsdir, season=ppt_season, region=ppt_region,
                                         forecast_period=forecast_period, anomalise=anomalise_ppt,
                                         ensemble=ensemble,weather_field=weather_field,
                                         start_forecast_reference_year=start_forecast_reference_year,
                                         numberof_forecast_reference_years=numberof_forecast_reference_years)

           #raise AssertionError('Stop for debugging...') 
           if not nyrmax is None:
               dps_amo_all_years = dps_amo_all_years[:nyrmax]
               dps_ppt_all_years = dps_ppt_all_years[:nyrmax]


           # dps_ppt_all_years is a cubelist of cubes of precip.
           # Each cube is from a single forecast_reference_time. Each cube has dim coordinate:
           # realization, latitude, longitude, and is of a single forecast_period, given as an integer 1,2,3 etc

           # Each cube should correspond to the ith one of dps_amo_all_years 

           # Want to extract each individual member of each cube of dps_amo_all_years
           # and dps_ppt_all_years and, depending on the amo value, add
           # the corresponding prec field to one of either: a cubelist collating
           # the precip fields of -ve amo events or a cubelist collating the precip fields
           # of +ve amo events
           
           # zip through each cube of dps_amo_all_years and dps_ppt_all_years:
           
           for dps_amo_1year, dps_ppt_1year in zip(dps_amo_all_years, dps_ppt_all_years):
           
               # Check that the cube really is of the same season_year:
               if dps_amo_1year.coord('season_year').points[0] == dps_ppt_1year.coord('season_year').points[0]:
           
                   # Zip though slices over realization of dps_amo_1year and dps_ppt_1year:
                   #import iris.iterate

                   for dps_amo_1year_slice, dps_ppt_1year_slice in zip(dps_amo_1year.slices_over('realization'), 
                                                                       dps_ppt_1year.slices_over('realization')):
           
                       #print('dps_amo_1year_slice:',dps_amo_1year_slice)
                       #print('dps_ppt_1year_slice:',dps_ppt_1year_slice)
                       
                       # Check that the cube really is of the same realization:
                       if dps_amo_1year_slice.coord('realization').points[0] == dps_ppt_1year_slice.coord('realization').points[0]:

                           # Strip season_year, time and realization coords from the dps_ppt_1year_slice:
                           dps_ppt_1year_slice.remove_coord('season_year')
                           dps_ppt_1year_slice.remove_coord('time')
                           dps_ppt_1year_slice.remove_coord('realization')
                       
                           # Extract region of prec:
                           dps_ppt_1year_slice = amo_functions.extract_region(dps_ppt_1year_slice, region=ppt_region)
                                              
                           #print('dps_amo_1year:',dps_amo_1year.data)
                           if dps_amo_1year_slice.data < 0.0:
                               # +ve amo year.
                               # Add aux coord of the number_of_neg_amo_events so cubelist can be later merged:
                               
                               dps_ppt_1year_slice.add_aux_coord(iris.coords.AuxCoord(number_of_neg_amo_events,
                                   long_name='amo_neg_index', units='no_unit'))

                               model_ppt_cubes_when_amo_neg.append(dps_ppt_1year_slice)    
                               number_of_neg_amo_events += 1
                               
                           if dps_amo_1year_slice.data >= 0.0:
                               # +ve amo year.
                               # Add aux coord of the number_of_pos_amo_events so cubelist can be later merged:
                               
                               dps_ppt_1year_slice.add_aux_coord(iris.coords.AuxCoord(number_of_pos_amo_events,
                                                                 long_name='amo_pos_index', units='no_unit'))

                               model_ppt_cubes_when_amo_pos.append(dps_ppt_1year_slice)    
                               number_of_pos_amo_events += 1
                       
        #raise AssertionError('Stop for debugging...')
               

        model_ppt_cubes_when_amo_neg = amo_functions.regrid_to_target(model_ppt_cubes_when_amo_neg,
                                                 scheme='iris.analysis.Linear()', target=regrid_target,
                                                 coords_to_regrid=['latitude','longitude'])

        model_ppt_cubes_when_amo_pos = amo_functions.regrid_to_target(model_ppt_cubes_when_amo_pos,
                                                 scheme='iris.analysis.Linear()', target=regrid_target,
                                                 coords_to_regrid=['latitude','longitude'])


        # unify units:
        model_ppt_cubes_when_amo_neg = amo_functions.unify_cubelist(model_ppt_cubes_when_amo_neg)
        model_ppt_cubes_when_amo_pos = amo_functions.unify_cubelist(model_ppt_cubes_when_amo_pos)
        

        model_ppt_cubes_when_amo_neg = amo_functions.remove_cubelist_cell_methods(model_ppt_cubes_when_amo_neg)
        model_ppt_cubes_when_amo_pos = amo_functions.remove_cubelist_cell_methods(model_ppt_cubes_when_amo_pos)

        model_ppt_cubes_when_amo_neg = amo_functions.remove_cubelist_attributes(model_ppt_cubes_when_amo_neg)
        model_ppt_cubes_when_amo_pos = amo_functions.remove_cubelist_attributes(model_ppt_cubes_when_amo_pos)

        model_ppt_cubes_when_amo_neg = amo_functions.convert_cubelist_to_float32(model_ppt_cubes_when_amo_neg)
        model_ppt_cubes_when_amo_pos = amo_functions.convert_cubelist_to_float32(model_ppt_cubes_when_amo_pos)

        model_ppt_cubes_when_amo_neg = amo_functions.remove_cubelist_coords(model_ppt_cubes_when_amo_neg,['month','year','forecast_period','season'])              
        model_ppt_cubes_when_amo_pos = amo_functions.remove_cubelist_coords(model_ppt_cubes_when_amo_pos,['month','year','forecast_period','season'])

        if weather_field.standard_name == 'air_temperature':
             model_ppt_cubes_when_amo_neg = amo_functions.remove_cubelist_coords(model_ppt_cubes_when_amo_neg,['height'])
             model_ppt_cubes_when_amo_pos = amo_functions.remove_cubelist_coords(model_ppt_cubes_when_amo_pos,['height'])
      

        # Merge cubelists of prec during negative and positive amo events:        
        model_ppt_cube_when_amo_neg = iris.cube.CubeList(model_ppt_cubes_when_amo_neg).merge_cube()
        model_ppt_cube_when_amo_pos = iris.cube.CubeList(model_ppt_cubes_when_amo_pos).merge_cube()

        print('Total number of events from all ensembles combined:')
        print('number_of_neg_amo_events:',number_of_neg_amo_events)
        print('number_of_pos_amo_events:',number_of_pos_amo_events)

        # Calculate mean of both:
        model_ppt_cube_when_amo_neg = model_ppt_cube_when_amo_neg.collapsed('amo_neg_index',iris.analysis.MEAN)
        model_ppt_cube_when_amo_pos = model_ppt_cube_when_amo_pos.collapsed('amo_pos_index',iris.analysis.MEAN)

        # And calculate difference between them:
        model_ppt_cube_when_amo_pos_minus_neg = model_ppt_cube_when_amo_pos - model_ppt_cube_when_amo_neg

        if weather_field.standard_name == 'air_pressure_at_sea_level':
            model_ppt_cube_when_amo_pos_minus_neg.convert_units('hPa')

        print('model_ppt_cube_when_amo_pos_minus_neg units:',model_ppt_cube_when_amo_pos_minus_neg.units)
        
        print('weather_field.standard_name:',weather_field.standard_name)
        if weather_field.standard_name == 'precipitation_flux':
            model_ppt_cube_when_amo_pos_minus_neg = 100.0*(model_ppt_cube_when_amo_pos_minus_neg/model_ppt_cube_when_amo_neg)

        print('371 model_ppt_cube_when_amo_pos_minus_neg:',model_ppt_cube_when_amo_pos_minus_neg)

        if weather_field.standard_name=='precipitation_flux':           model_weather_field_str='precipitation (%)'
        if weather_field.standard_name=='air_pressure_at_sea_level':    model_weather_field_str='sea level pressure (hPa)'
        if weather_field.standard_name=='air_temperature':              model_weather_field_str='surf. air temperature ($\degree$C)'


        if len(ensembles) == 1:
            mmm = ', '+mdic[ensembles[0]]
        else:
            mmm = ', Nens='+str(len(ensembles))
            mmm = ''

        cnpos = '('+str(number_of_pos_amo_events)+'yrs)'
        cnneg = '('+str(number_of_neg_amo_events)+'yrs)'

        if ppt_season.mmm == 'jja':
            ssn_name='Summer'
        elif ppt_season.mmm == 'djf':
            ssn_name='Winter'
        else:
            raise AssertionError('Stop, ppt_season should be one of: (djf,jja)') 
        
        model_title = ssn_name+' hindcast '+ model_weather_field_str + mmm +'\n'+\
                      cdriver+' +ve '+cnpos +  ' minus ' + cdriver+' -ve '+cnneg

    percentage=False

    # Now process the obs prec during obs amo events:
    if desired_precip_output_format == '+ve amo minus -ve amo as fraction of -ve':

        if weather_field.standard_name == 'precipitation_flux':
            percentage = True
        
        obs_ppt_diff_amo_pos_minus_neg = amo_functions.mean_obs_ppt_conditioned_on_obs_amo(obsdir,
                                                       driver=obsdriver, amoobssource=amoobssource, amo_season=amo_season,
                                                       ppt_season=ppt_season, ppt_region=regionclass.InitRegion('None'),
                                                       amo_anomaly_format=amo_anomaly_format, percentage=percentage,
                                                       desired_precip_output_format=desired_precip_output_format,
                                                       weather_field=weather_field, obssource=obssource, anomalise_ppt=anomalise_ppt,
                                                       start_year=start_year_of_obs, number_of_years=numberof_years_of_obs)

        if obs_ppt_diff_amo_pos_minus_neg is not None:
            # obs_ppt_diff_amo_pos_minus_neg is a cube of dims latitude and longitude

            # Extract region:
            obs_ppt_diff_amo_pos_minus_neg = amo_functions.extract_region(obs_ppt_diff_amo_pos_minus_neg, region=ppt_region)

            print('455 obs_ppt_diff_amo_pos_minus_neg units:',obs_ppt_diff_amo_pos_minus_neg.units)

            if weather_field.standard_name=='air_pressure_at_sea_level':
                obs_ppt_diff_amo_pos_minus_neg.convert_units('hPa')

    else:
        obs_ppt_when_obs_amo_neg, obs_ppt_when_obs_amo_pos = amo_functions.mean_obs_ppt_conditioned_on_obs_amo(obsdir,
                                                                 driver=driver, amoobssource=amoobssource, amo_season=amo_season,
                                                                 ppt_season=ppt_season, ppt_region=regionclass.InitRegion('None'), 
                                                                 weather_field=weather_field, obssource=obssource, anomalise_ppt=anomalise_ppt,
                                                                 start_year=start_year_of_obs, number_of_years=numberof_years_of_obs)

        if obs_ppt_when_obs_amo_neg is not None:

            print('obs_ppt_when_obs_amo_neg:',obs_ppt_when_obs_amo_neg)
            print('obs_ppt_when_obs_amo_neg.data:',obs_ppt_when_obs_amo_neg.data)

            print('175 obs_ppt_when_obs_amo_neg:',obs_ppt_when_obs_amo_neg)
            print('obs_ppt_when_obs_amo_pos:',obs_ppt_when_obs_amo_pos)
    
            # obs_ppt_when_obs_amo_neg is a cube of lat,long
            # obs_ppt_when_obs_amo_pos is a cube of lat,long
    
            # Extract region:
            obs_ppt_when_obs_amo_neg = amo_functions.extract_region(obs_ppt_when_obs_amo_neg, region=ppt_region)

            obs_ppt_when_obs_amo_pos = amo_functions.extract_region(obs_ppt_when_obs_amo_pos, region=ppt_region)

            cubes_to_plot_when_amo_neg.append(obs_ppt_when_obs_amo_neg)
            cubes_to_plot_when_amo_pos.append(obs_ppt_when_obs_amo_pos)

    #raise AssertionError('Stop for debugging...') 


    #obs_title = ppt_season.mmm.upper()+' observed'
    #if obssource == 'gpcc':             
    #    obs_title += ' (GPCC)'
    #elif obssource == 'hadslp':           
    #    obs_title += ' (HadSLP2)'
    #elif obssource == 'era5':             
    #    obs_title += ' (ERA5)'
    #elif obssource == 'hadcrut5_latlong': 
    #    obs_title += ' (HadCRUT5)'

    if ppt_season.mmm == 'jja':
        obs_title = 'Summer observed'
    elif ppt_season.mmm == 'djf':
        obs_title = 'Winter observed'
    else:
        raise AssertionError('Stop, ppt_season should be one of: (djf,jja)') 

    if weather_field.standard_name == 'air_pressure_at_sea_level':
        obs_title += ' sea level pressure (hPa)'

    if weather_field.standard_name == 'precipitation_flux':
        obs_title += ' precipitation (%)'

    if weather_field.standard_name == 'air_temperature':
        obs_title += ' air temperature ($\degree$C)'
    
    npos = obs_ppt_diff_amo_pos_minus_neg.coord('number_of_pos_season_years').points[0]
    nneg = obs_ppt_diff_amo_pos_minus_neg.coord('number_of_neg_season_years').points[0]
    
    #obs_title += '\n('+npos+' +ve, '+nneg+' -ve '+driver.upper()+' years)'
    #obs_title += '\n'+driver.upper()+' +ve ('+str(npos)+'yrs) - ' + driver.upper()+' -ve ('+str(nneg)+'yrs)'

    cnpos = '('+str(npos)+'yrs)'
    cnneg = '('+str(nneg)+'yrs)'
    
    obs_title = obs_title+'\n'+cdriver+' +ve '+cnpos+' minus ' + cdriver +' -ve '+cnneg

    # Now decide the order of plotting, eg obs before model
    panel_titles = [obs_title, model_title]

    cubes_to_plot_when_amo_pos_minus_neg = [obs_ppt_diff_amo_pos_minus_neg, model_ppt_cube_when_amo_pos_minus_neg]

    # Plot the maps:

    if desired_precip_output_format == '+ve amo minus -ve amo as fraction of -ve':
        amo_modes_to_plot = ['pos_minus_neg']
    else:
        amo_modes_to_plot = ['negative','positive']
    
    for amo_mode_to_plot in amo_modes_to_plot:    
        if weather_field.standard_name=='precipitation_flux':        cmap=['red','blue']
        if weather_field.standard_name=='air_pressure_at_sea_level': cmap=['blue','red']
        if weather_field.standard_name=='air_temperature':           cmap=['blue','red']
        actual_cmap_to_use = amo_functions.load_cmap(cmap)

        title_prefixes=list('abcdefghijklmn')
        title_prefixes=['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
        
        #if amo_mode_to_plot=='negative':  cubes_to_plot=cubes_to_plot_when_amo_neg
        #if amo_mode_to_plot=='positive':  cubes_to_plot=cubes_to_plot_when_amo_pos
        if amo_mode_to_plot == 'pos_minus_neg':  cubes_to_plot = cubes_to_plot_when_amo_pos_minus_neg
               
        for cube_to_plot,panel_title in zip(cubes_to_plot,panel_titles):
            #Apply landmask        
            if 'hindcast precipitation' in panel_title and maskocn:
                # Hardwire in name of coarsest model. 
                if regrid_target == 'coarsest':
                    fracfile = os.path.join(dpsdir, 'Masks', 'CanESM5_land_area_fraction.nc')
                elif regrid_target == 'ncar':
                    fracfile = os.path.join(dpsdir, 'Masks', 'CESM1-1-CAM5_land_area_fraction.nc')                       
                elif regrid_target == 'depresys':
                    fracfile = os.path.join(dpsdir, 'Masks', 'HadGEM3-GC31-MM_land_area_fraction.nc')
                print('Load file: ',fracfile)
                landfrac = iris.load_cube(fracfile)
                landmask = utils.landfrac_to_mask(landfrac, threshold=0.5)
                if ppt_region.standard_name == 'NW Europe':
                    nweu_box = [-40.0, 35.0, 40.0, 75.0]
                else:
                    raise AssertionError('Abort, need to set correct box for:',ppt_region)
                landmask_box = extractbox(landmask, box=nweu_box, ignore_bounds=True)
                cube_to_plot = utils.applyMask(cube_to_plot, landmask_box)
    
            nrows=2
            ncols=2

            ax=plt.subplot(nrows, ncols, panelnumber)

            # for model ensemble means:
            levels=[-4,-3,-2,-1,0,1,2,3,4] # % diff from climate
            levels=[-8,-6,-4,-2,0,2,4,6,8]
        
            if panelnumber == 12:
                # for obs precip:
                levels=[-8,-6,-4,-2,0,2,4,6,8] # % diff from climate

            if weather_field.standard_name == 'precipitation_flux':
                levels=[-12,-9,-6,-3,0,3,6,9,12]
                if driver == 'nao':
                    levels=[-20,-16,-12,-8,-4,0,4,8,12,16,20]
                    #levels=[-24, -20,-16,-12,-8,-4,0,4,8,12,16,20,24]
                    #levels=[-25,-20,-15,-10,-5,0,5,10,15,20,25]

            if weather_field.standard_name == 'air_temperature':
                levels = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

            if weather_field.standard_name == 'air_pressure_at_sea_level':
                if panelnumber == 1 or panelnumber == 3:
                    # (model data)
                    levels=[-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8]
                else:
                    levels=[-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0]

                levels=[-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8]
                levels=[-1.2,-0.9,-0.6,-0.3,0.0,0.3,0.6,0.9,1.2]

                if driver == 'nao':
                    levels=[-8,-6,-4,-2,0,2,4,6,8]

            contour_precip=True
            if contour_precip:
                contour_result=iplt.contourf(cube_to_plot, cmap=actual_cmap_to_use, levels=levels, extend='both')
            else:
                contour_result=iplt.pcolormesh(cube_to_plot, cmap=actual_cmap_to_use, 
                                               norm=BoundaryNorm(levels, ncolors=actual_cmap_to_use.N, clip=True))    

            # Add latitude ticks:
            yticks=[40,50,60,70]
            plt.gca().set_yticks(yticks, crs=ccrs.PlateCarree())
            plt.gca().yaxis.set_major_formatter(LatitudeFormatter())

            # Add longitude ticks:
            xticks=[-40,-30,-20,-10,0,10,20,30,40]
            xticks=[-40,-20,0,20,40]               
            plt.gca().set_xticks(xticks, crs=ccrs.PlateCarree())
            lon_formatter=LongitudeFormatter(dateline_direction_label=True)
            plt.gca().xaxis.set_major_formatter(lon_formatter)

            # Reduce font size of the latitude and longitude ticks:
            plt.gca().tick_params(axis='x', labelsize=8)
            plt.gca().tick_params(axis='y', labelsize=8)

            # Add coastlines:
            plt.gca().coastlines()

            # Prevent the automatic 'Unknown' title being displayed:
            #plt.title(' ')
        
            #panel_title='('+title_prefixes[panelnumber-1]+') '+panel_title

            x = -0.02
            if panelnumber==1 or panelnumber==3: x=-0.06 # was -0.13
            #plt.title(panel_title,loc='left',x=x, fontsize=9.5, pad=3.5)

            plt.title(panel_title, fontsize=9.5, pad=3.5)

            # Add panel color bar
            #if weather_field.standard_name=='air_pressure_at_sea_level':
            #    fig.colorbar(contour_result, ax=plt.gca(),orientation='horizontal',pad=0.08)

            plot_row_bar=False
            print('618 panelnumber:',panelnumber)
            if panelnumber==2 or panelnumber==4:
                plot_row_bar=True
            
            if plot_row_bar: 
                # Plot color bar of whole row.
                print('Plotting a row bar')
    
                # Get position of most recent panel:
                pos1 = plt.gca().get_position()

                # Set width of colour bar:
                bar_width=0.015   #0.02
    
                # Set length
                bar_length=0.6 # 1.0 is whole left to right width of page
    
                # Set distance between plot and colour bar:
                vertical_pad_between_plot_and_colour_bar=0.069    #0.03
    
                print('pos1.y0:',pos1.y0)
    
                # Set axes for row bar:
                bar_axes_to_use=[0.5-(bar_length*0.5),  pos1.y0-vertical_pad_between_plot_and_colour_bar, bar_length, bar_width]

                format_for_color_bar_text="%.2f"
    
                # Plot color bar:
                #if weather_field.standard_name!='air_pressure_at_sea_level':
                
                print('bar_axes_to_use:',bar_axes_to_use)
                
                cbar = plt.colorbar(contour_result, cax=plt.gcf().add_axes(bar_axes_to_use), orientation='horizontal') # format=format_for_color_bar_text
                cbar.ax.tick_params(labelsize=9.0)

            panelnumber = panelnumber+1

# End loop over weathersofinterest

for dpi in dpiarr:           
    cdpi=str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)

print('plot_fig16_and_S15.py successfully completed')

