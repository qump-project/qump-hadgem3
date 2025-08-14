import os
import subprocess
import pickle
import numpy

import seasonclass
import regionclass
import weatherfieldclass

import iris
import iris.coords as icoords
from iris.coords import AuxCoord
import iris.coord_categorisation

########################################################################
def get_model_idx(dpsdir, season=None, driver='amo', amoregion=None, 
                  forecast_period=1, anomalise_amo=False,
                  basebeg=1971, nbase=30, ensemble=None,
                  start_forecast_reference_year=1960, numberof_forecast_reference_years=60):

    # Get a cubelist of model hindcasts of all members of a specific ensemble and forecast period. 
    # The differing cubes of the cubelist hold data from differing forecast refernce times.

    if anomalise_amo:
        anomalise_amo_str = 'anomalies_usingclimatefrom_'+str(basebeg)+'for'+str(nbase)+'years'
    else:
        anomalise_amo_str = ''

    dps_name = ensemble+'_'+season.mmm+'_'+\
               str(numberof_forecast_reference_years)+'years_'+driver+'_'+anomalise_amo_str+\
               '_all_forecast_periods.pickle'
                  
    dps_file = os.path.join(dpsdir, dps_name) 

    errorcode_of_dps_file = subprocess.call('find '+dps_file, shell=True)
    print('Summary file of '+driver+' data returned from depresys_driver_timeseries, (')
    print(dps_file,'),')
    print('error code: ',errorcode_of_dps_file)
                
    # If it exists, add to list:
    if errorcode_of_dps_file == 0:    
        # Summary file exists. Read from dps_idx from file:
        print('Load file: ',dps_file)
        with open(dps_file, 'rb') as fp:
            dps_idx = pickle.load(fp)
    else:
        print('Input file: ',dps_file)
        print('NOT FOUND => run plot_fig19_and_S18_full.py to create this')
        raise AssertionError('Stop for debugging...')
        
    # dps_idx is a cubelist of cubes of dims forecast_period, realization
    # Each cube is of a different forecast reference time.
    # Each cube has a time dim, correspnding to the available forecast periods of 1, 2,... 10 years 
    # Each cube also has a realization dim coordinate.
    # The cubes of dps_idx should NOT have latitude and longitude dim coordinates
    
    # Extract forecast_period instances from the cubes of dps_amo_all_years:
    #dps_idx = extract_from_dps_cubes(dps_idx, coords_to_extract=['forecast_period'],
    #                                 instances_to_extract=[forecast_period], return_cube=False)

    dps_idx = extract_from_dps_cubes(dps_idx, ['forecast_period'], [forecast_period], return_cube=False)
                
    # dps_idx is a cubelist of a single forecast_period
    # Each cube is multiple realizations for different forecast_reference_time

    return dps_idx

########################################################################
def get_model_ppt(dpsdir, season=None, region=None, 
                  forecast_period=1, anomalise=False,
                  weather_field=weatherfieldclass.InitWeatherfield('precipitation_flux','',''),
                  basebeg=1971, nbase=30, ensemble=None, nmembers=10,
                  start_forecast_reference_year=1960, numberof_forecast_reference_years=60):

    # Get cubelist of global lat long cubes of model prec.
    # Differing cubes will have differing forecast reference times
    # Each cube will have the dims: realization, latitude, longitude.
    # Each cube will have the hindcasts from a single forecast_period.

    # weather_field should be given as a weatherfieldclass
    # eg weatherfieldclass.InitWeatherfield('precipitation_flux','','')
    # or weatherfieldclass.InitWeatherfield('air_temperature',1.5,'m')
    # or weatherfieldclass.InitWeatherfield('air_pressure_at_sea_level','','')
    
        
    if anomalise:
        anomalise_str = 'anomalies_usingclimatefrom_'+str(basebeg)+'for'+str(nbase)+'years'
    else:
        anomalise_str=''

    dps_name = ensemble+'_'+season.mmm+'_'+\
                str(numberof_forecast_reference_years)+'years_'+\
                weather_field.standard_name+'_'+anomalise_str+\
                '_all_forecast_periods.pickle'
                
    dps_file = os.path.join(dpsdir, dps_name)
    dps_file1= dps_file.replace('_all_forecast_periods', 'T'+str(forecast_period)) 

    errorcode_of_dps_file1 = subprocess.call('find '+dps_file1, shell=True)

    if errorcode_of_dps_file1 == 0:
        print('Load file: ',dps_file1)        
        with open(dps_file1, 'rb') as fp:
            dps_ppt = pickle.load(fp)                
    else:
        print('Input file: ',dps_file1)
        print('NOT FOUND => run plot_fig19_and_S18_full.py to create this')
        raise AssertionError('Stop for debugging...')

    return dps_ppt


########################################################################
def extract_corresponding_ppt_slice(dps_amo_slice, dps_ppt_all_years):
    '''          
    Using information of dps_amo_slice, extract corresponding slice of cube and slice of dps_ppt
    '''
    # Step through all cubes of dps_ppt
    for dps_ppt in dps_ppt_all_years:
        # Look up forecast reference time of dps_ppt:    
        if dps_ppt.coord('season_year').points[0] == dps_amo_slice.coord('season_year').points[0]:            
            # Identified the corresponding sube of dps_ppt
            # Now Try to identify the corresponding realization:            
            for dps_ppt_slice in dps_ppt.slices_over('realization'):            
                if dps_ppt_slice.coord('realization').points[0] == dps_amo_slice.coord('realization').points[0]:                    
                    # Found corresponding realization of precip.
                    return dps_ppt_slice
    # If we have reached this line, we have nothing to return, so return None
    return None
            
	
########################################################################
def mean_model_ppt_conditioned_on_model_amo(driver='amo', amo_season=None, weather_field=None, 
                                            ppt_season=None, ppt_region=None, forecast_period=1,
                                            amo_anomaly_format=None, anomalise_ppt=False,
                                            desired_precip_output_format=None, percentage=False,
                                            start_forecast_reference_year=1960,
                                            numberof_forecast_reference_years=60,
                                            basebeg=1971, nbase=30, ensemble=None):
        
    ''' Calculate mean dps prec of years when amo is negative and positive
    '''

    if desired_precip_output_format=='+ve amo minus -ve amo as fraction of -ve':
        # Don't want to have precip expressed as an anomaly wrt climate:
        anomalise_ppt=False
    
    dps_ppt_when_amo_neg, dps_ppt_when_amo_pos = model_ppt_conditioned_on_model_amo(
        driver=driver, amo_season=amo_season, weather_field=weather_field, 
        ppt_season=ppt_season, ppt_region=ppt_region, forecast_period=forecast_period,
        amo_anomaly_format=amo_anomaly_format, anomalise_ppt=anomalise_ppt,
        start_forecast_reference_year=start_forecast_reference_year,
        numberof_forecast_reference_years=numberof_forecast_reference_years,
        basebeg=basebeg, nbase=nbase, ensemble=ensemble)

    # dps_ppt_when_amo_neg is a cube of precip during ALL events when amo was -ve 
    # dps_ppt_when_amo_pos is a cube of precip during ALL events when amo was +ve 

    print('dps_ppt_when_amo_neg:',dps_ppt_when_amo_neg)
    print('dps_ppt_when_amo_pos:',dps_ppt_when_amo_pos)
    
    if dps_ppt_when_amo_neg is not None:

        # Calculate means over the events:
        dps_ppt_when_amo_neg = dps_ppt_when_amo_neg.collapsed('amo_neg_index',iris.analysis.MEAN)

        dps_ppt_when_amo_pos = dps_ppt_when_amo_pos.collapsed('amo_pos_index',iris.analysis.MEAN)

        # dps_ppt_when_amo_neg and dps_ppt_when_amo_pos are cubes of dims latitude and longitude

        if desired_precip_output_format is not None:
            if desired_precip_output_format=='+ve amo minus -ve amo as fraction of -ve':
                # Calculate +ve amo minus -ve amo as a percentage of -ve amo prec
            
                dps_ppt_diff = dps_ppt_when_amo_pos - dps_ppt_when_amo_neg
        
                if percentage:
                    dps_ppt_diff = 100.0*(dps_ppt_diff/dps_ppt_when_amo_neg)
        
            # Return difference bteween -ve and +ve amo precip:
            return dps_ppt_diff
        else:        
            return dps_ppt_when_amo_neg, dps_ppt_when_amo_pos

                

########################################################################
def model_ppt_conditioned_on_model_amo(driver='amo', amo_season=None, weather_field=None, 
                                       ppt_season=None, ppt_region=None, forecast_period=1,
                                       amo_anomaly_format=None, anomalise_ppt=False,
                                       start_forecast_reference_year=1960,
                                       numberof_forecast_reference_years=60,
                                       basebeg=1971, nbase=30, ensemble=None):

    ''' Function to calculate model prec dependent on model amo 
    '''

    print('Welcome to model_ppt_conditioned_on_model_amo')
    if amo_anomaly_format is None:
        # Don't do any anomalising at all:
        anomalise_amo=False
    else:
        if amo_anomaly_format=='of_a_baseline_climate':
            # Anomalise wrt to a set period eg 1971 to 2000 (given by basebeg and nbase)
            anomalise_amo=True
        if amo_anomaly_format=='of_whole_period':
            anomalise_amo=True
            # Set the following 2 to be None. These will be used in function calculate_depresys_climates 
            # to calculate the climate over the whole period. 
            basebeg=None
            nbase=None


    # Get model amo:
    dps_amo_all_years = get_model_idx(season=amo_season,driver=driver,
            forecast_period=forecast_period, anomalise_amo=anomalise_amo,
            basebeg=basebeg, nbase=nbase, ensemble=ensemble,
            start_forecast_reference_year=start_forecast_reference_year,
            numberof_forecast_reference_years=numberof_forecast_reference_years)

    # Note that amo_anomaly_format was NOT given as an argument to get_model_idx.
    
    # dps_amo_all_years is a cubelist of cubes of amo.
    # Each cube is from a single forecast_reference_time. Each cube has
    # dim coordinate realization and is of a single forecast_period
    # (given as an integer 1,2 3 etc

    print('Number of cubes in dps_amo_all_years from get_model_idx:',len(dps_amo_all_years))
    print('(Each cube is from a different forecast reference time)')

    # Get model ppt:
    dps_ppt_all_years = get_model_ppt(season=ppt_season,
            forecast_period=forecast_period, region=ppt_region,
            ensemble=ensemble, weather_field=weather_field,           
            anomalise=anomalise_ppt, basebeg=basebeg, nbase=nbase,
            start_forecast_reference_year=start_forecast_reference_year,
            numberof_forecast_reference_years=numberof_forecast_reference_years)

    # forecast period check:
    print('forecast_period given to get_model_ppt:',forecast_period)
    for dps_ppt_singlecube in dps_ppt_all_years:
        print('dfp:',dps_ppt_singlecube.coord('forecast_period').points)
    

    # dps_ppt_all_years is a cubelist of cubes of precip.
    # Each cube is from a single forecast_reference_time. Each cube has
    # dim coordinate realization, latitude, longitude and is of a single forecast_period
    # (given as an integer 1,2 3 etc
    
    # Now need to extract forecasts of -ve and +ve AMO.
    # Do this by indentifying relevant years based on the AMO and then
    # exracting the precip for those particular years.
    
    # Go through each slice of each cube of dps_amo_all_years:

    amo_negative_years=[]
    amo_positive_years=[]
    
    dps_ppt_slices_when_amo_neg=[]
    amo_neg_index=0

    dps_ppt_slices_when_amo_pos=[]
    amo_pos_index=0

    # Could actually zip through the dps_amo_all_years AND dps_ppt_all_years at the same time, 
    # but there is a danger of year mismatches. So its best not to.
    
    for dps_amo in dps_amo_all_years:

        print('dps_amo.data:',dps_amo.data)
    
        # Need to slices_over the realization instances of dps_amo 
        for dps_amo_slice in dps_amo.slices_over('realization'):
    
            # Extract corresponding prec cube:
            dps_ppt_slice = extract_corresponding_ppt_slice(dps_amo_slice, dps_ppt_all_years)
    
            if dps_ppt_slice is not None:
    
                # dps_amo_slice is a scalar cube
                # dps_ppt_slice is a cube of latitude, longitude

                #print '202 dps_amo_slice:',dps_amo_slice
                #print 'dps_ppt_slice:',dps_ppt_slice

                # Extract season_year for later use:
                season_year_of_dps_amo_slice = str(int(dps_ppt_slice.coord('season_year').points[0]))

                # Get rid of redundant coords:
                dps_ppt_slice.remove_coord('season')
                dps_ppt_slice.remove_coord('season_year')
                dps_ppt_slice.remove_coord('realization')
                dps_ppt_slice.remove_coord('time')
                
                # Remove forecast_period because in one model it also has
                # a var_name which becomes a pain later on:
                dps_ppt_slice.remove_coord('forecast_period')
            
                if dps_amo_slice.data < 0.0:
                    # We have a negative amo slice.

                    # Increment a counter which will be used to index a
                    # resulting precip cube (to allow later cube merging):
                    amo_neg_index += 1
                
                    # Add coordinate of this counter to dps_ppt_slice:
                    dps_ppt_slice.add_aux_coord(iris.coords.AuxCoord(
                        amo_neg_index, long_name='amo_neg_index', units='no_unit'))

                    # Append year info to list for any later use:
                    amo_negative_years.append(season_year_of_dps_amo_slice)

                    # Append prec cube to list over all occurences of -ve amo:
                    dps_ppt_slices_when_amo_neg.append(dps_ppt_slice)
                    
                if dps_amo_slice.data >= 0.0:
                    # We have a positive amo slice.

                    # Increment a counter which will be used to index a
                    # resulting precip cube (to allow later cube merging):
                    amo_pos_index += 1
                
                    # Add coordinate of this counter to dps_ppt_slice:
                    dps_ppt_slice.add_aux_coord(iris.coords.AuxCoord(
                        amo_pos_index, long_name='amo_pos_index', units='no_unit'))

                    # Append year info to list for any later use:
                    amo_positive_years.append(season_year_of_dps_amo_slice)

                    # Append prec cube to list over all occurences of -ve amo:
                    dps_ppt_slices_when_amo_pos.append(dps_ppt_slice)


    # Convert list of prec cubes when amo is -ve, into a single cube and calculate means:
    
    print(len(dps_ppt_slices_when_amo_neg))
    
    # Get rid of attributes which might prevent merging:
    dps_ppt_slices_when_amo_neg = remove_cubelist_attributes(dps_ppt_slices_when_amo_neg)
    
    if len(dps_ppt_slices_when_amo_neg) > 1:
        dps_ppt_when_amo_neg = iris.cube.CubeList(dps_ppt_slices_when_amo_neg).merge_cube()
    elif len(dps_ppt_slices_when_amo_neg) == 1:
        dps_ppt_when_amo_neg = dps_ppt_slices_when_amo_neg[0]     
    elif len(dps_ppt_slices_when_amo_neg) == 0:
        dps_ppt_when_amo_neg = None

    # For positive amo events:

    # Get rid of attributes which might prevent merging:
    dps_ppt_slices_when_amo_pos = remove_cubelist_attributes(dps_ppt_slices_when_amo_pos)

    if len(dps_ppt_slices_when_amo_pos) > 1:
        dps_ppt_when_amo_pos = iris.cube.CubeList(dps_ppt_slices_when_amo_pos).merge_cube()
    elif len(dps_ppt_slices_when_amo_pos) == 1:
        dps_ppt_when_amo_pos = dps_ppt_slices_when_amo_pos[0]     
    elif len(dps_ppt_slices_when_amo_pos) == 0:
        dps_ppt_when_amo_pos = None

    print(ensemble+' '+driver+' occurences:')                
    print('When negative:',amo_neg_index)
    print('(',', '.join(amo_negative_years),')')                    

    print('When positive:',amo_pos_index)                    
    print('(',', '.join(amo_positive_years),')')                    

    # Note that, for the first year in the DJF and DJFMAMJJSSON seasons in some ensembles,
    # the prec during amo events will appear to be None because of the masking of
    # that season due to hindcast initializations beginnning after the season start
       
    # Return cube of precip during ALL events when amo was -ve and a cube when the amo was +ve:

    return dps_ppt_when_amo_neg, dps_ppt_when_amo_pos



########################################################################
def get_obs_idx(obsdir, amo_season=None, driver='amo', amoregion=None, amoobssource='hadcrut5',
                basebeg=1971, nbase=30, anomalise_amo=False, start_year=1960, number_of_years=60):

    print('anomalise_amo:',anomalise_amo)
    # Load obs air temperature, for amo:

    obs_amo_all_years = getobs(amo_season, start_year, number_of_years, 
                               obsdir=obsdir, anomalise=anomalise_amo, obssource=amoobssource,                                         
                               weatherfield=weatherfieldclass.InitWeatherfield(driver, 1.5, 'm'),            
                               region=regionclass.InitRegion('None'),                          
                               returnanomaliesaspercentages=False,
                               startyearofclimateperiod=basebeg,
                               nyearsofclimateperiod=nbase)    
        
    return obs_amo_all_years
	
########################################################################
def get_obs_ppt(obsdir, season=None, region=None, obssource=None, 
                weather_field=weatherfieldclass.InitWeatherfield('precipitation_flux','',''),
                basebeg=1971, nbase=30, anomalise=False, start_year=1960, number_of_years=60):
    
    if anomalise and weather_field.standard_name=='precipitation_flux':
        returnanomaliesaspercentages=True
    else:
        returnanomaliesaspercentages=False
    print(weather_field.standard_name) 
    print('returnanomaliesaspercentages:',returnanomaliesaspercentages)

    obs_ppt = getobs(season, start_year, number_of_years,
                     obsdir=obsdir, 
                     anomalise=anomalise, obssource=obssource, 
                     region=region, weatherfield=weather_field,                  
                     returnanomaliesaspercentages=returnanomaliesaspercentages,
                     startyearofclimateperiod=basebeg, 
                     nyearsofclimateperiod=nbase)    
    return obs_ppt


########################################################################
def mean_obs_ppt_conditioned_on_obs_amo(obsdir, amo_season=None, driver='amo', amoregion=None, amoobssource='hadcrut5',
                                        ppt_season=None, ppt_region=None, obssource=None, weather_field=None,
                                        amo_anomaly_format=None, desired_precip_output_format=None, anomalise_ppt=False,
                                        percentage=False, basebeg=1971, nbase=30, start_year=1960, number_of_years=60):

    '''Calculate mean obs prec of years when amo is negative and positive'''

    if desired_precip_output_format == '+ve amo minus -ve amo as fraction of -ve':
        # Don't want to have precip expressed as an anomaly wrt climate:
        anomalise_ppt=False

    #anomalise_ppt=True
    print('anomalise_ppt in call to obs_ppt_conditioned_on_obs_amo:',anomalise_ppt)
    obs_ppt_when_amo_neg, obs_ppt_when_amo_pos = obs_ppt_conditioned_on_obs_amo(obsdir,
            amo_season=amo_season, driver=driver, amoregion=amoregion,
            amoobssource=amoobssource, ppt_season=ppt_season, ppt_region=ppt_region,
            obssource=obssource, weather_field=weather_field,
            amo_anomaly_format=amo_anomaly_format, anomalise_ppt=anomalise_ppt,
            basebeg=basebeg, nbase=nbase,
            start_year=start_year, number_of_years=number_of_years)


    # obs_ppt_when_amo_neg is a cube of precip during ALL events when amo was -ve 
    # obs_ppt_when_amo_pos is a cube of precip during ALL events when amo was +ve 

    number_of_obs_ppt_when_amo_neg_years = obs_ppt_when_amo_neg.coord('season_year').points.size
    number_of_obs_ppt_when_amo_pos_years = obs_ppt_when_amo_pos.coord('season_year').points.size
        
    reg_rek = regionclass.InitRegion('Reykjavik')
    reg_azr = regionclass.InitRegion('Azores')

    verbose=False
    
    if verbose:
        print('obs_ppt_when_amo_neg:',obs_ppt_when_amo_neg)
        print('obs_ppt_when_amo_neg season_year:',obs_ppt_when_amo_neg.coord('season_year'))
    
        print('obs_ppt_when_amo_pos:',obs_ppt_when_amo_pos)
        print('obs_ppt_when_amo_pos season_year:',obs_ppt_when_amo_pos.coord('season_year'))

        # Look at Azores minus Iceland pressure in each season year
        
        # Years of -ve nao:
        for obs_ppt_when_amo_neg_slice in obs_ppt_when_amo_neg.slices_over('season_year'):
        
            iceland_obs_ppt_when_amo_neg_slice = extract_region(obs_ppt_when_amo_neg_slice, region=reg_rek)
            iceland_obs_ppt_when_amo_neg_slice = iceland_obs_ppt_when_amo_neg_slice.collapsed(['latitude','longitude'],iris.analysis.MEAN)

            azores_obs_ppt_when_amo_neg_slice = extract_region(obs_ppt_when_amo_neg_slice,region=reg_azr)
            azores_obs_ppt_when_amo_neg_slice = azores_obs_ppt_when_amo_neg_slice.collapsed(['latitude','longitude'],iris.analysis.MEAN)

            print(azores_obs_ppt_when_amo_neg_slice.coord('season_year').points[0],
                  ' azores minus iceland during -ve ',driver,' years:',
                  azores_obs_ppt_when_amo_neg_slice.data-iceland_obs_ppt_when_amo_neg_slice.data)

        # Years of +ve nao:
        for obs_ppt_when_amo_pos_slice in obs_ppt_when_amo_pos.slices_over('season_year'):
            
            iceland_obs_ppt_when_amo_pos_slice = extract_region(obs_ppt_when_amo_pos_slice,region=reg_rek)
            iceland_obs_ppt_when_amo_pos_slice = iceland_obs_ppt_when_amo_pos_slice.collapsed(['latitude','longitude'],iris.analysis.MEAN)
    
            azores_obs_ppt_when_amo_pos_slice = extract_region(obs_ppt_when_amo_pos_slice,region=reg_azr)    
            azores_obs_ppt_when_amo_pos_slice = azores_obs_ppt_when_amo_pos_slice.collapsed(['latitude','longitude'],iris.analysis.MEAN)
    
            print(azores_obs_ppt_when_amo_pos_slice.coord('season_year').points[0],
                  ' azores minus iceland during +ve ',driver,' years:',
                  azores_obs_ppt_when_amo_pos_slice.data-iceland_obs_ppt_when_amo_pos_slice.data)


    # Calculate mean prec whenever the amo is negative:
    obs_ppt_when_amo_neg = obs_ppt_when_amo_neg.collapsed('season_year',iris.analysis.MEAN)
    
    # Calculate mean prec whenever the amo is positive:
    obs_ppt_when_amo_pos = obs_ppt_when_amo_pos.collapsed('season_year',iris.analysis.MEAN)


    if verbose:
        # Quick look at +ve years of the driver:
        
        iceland_obs_ppt_when_amo_pos = extract_region(obs_ppt_when_amo_pos,region=reg_rek)
        print('iceland_obs_ppt_when_amo_pos lats:',iceland_obs_ppt_when_amo_pos.coord('latitude').points)
        print('iceland_obs_ppt_when_amo_pos lons:',iceland_obs_ppt_when_amo_pos.coord('longitude').points)
        iceland_obs_ppt_when_amo_pos = iceland_obs_ppt_when_amo_pos.collapsed(['latitude','longitude'],iris.analysis.MEAN)
        print('iceland_obs_ppt_when_amo_pos data:',iceland_obs_ppt_when_amo_pos.data)

        azores_obs_ppt_when_amo_pos = extract_region(obs_ppt_when_amo_pos,region=reg_azr)
        print('azores_obs_ppt_when_amo_pos lats:',azores_obs_ppt_when_amo_pos.coord('latitude').points)
        print('azores_obs_ppt_when_amo_pos lons:',azores_obs_ppt_when_amo_pos.coord('longitude').points)
        azores_obs_ppt_when_amo_pos = azores_obs_ppt_when_amo_pos.collapsed(['latitude','longitude'],iris.analysis.MEAN)
        print('azores_obs_ppt_when_amo_pos data:',azores_obs_ppt_when_amo_pos.data)

        print('azores_obs_ppt_when_amo_pos-iceland_obs_ppt_when_amo_pos:',
              azores_obs_ppt_when_amo_pos.data-iceland_obs_ppt_when_amo_pos.data)

        # Quick look at -ve years of the driver:

        iceland_obs_ppt_when_amo_neg = extract_region(obs_ppt_when_amo_neg,region=reg_rek)
        print('iceland_obs_ppt_when_amo_neg lats:',iceland_obs_ppt_when_amo_neg.coord('latitude').points)
        print('iceland_obs_ppt_when_amo_neg lons:',iceland_obs_ppt_when_amo_neg.coord('longitude').points)
        iceland_obs_ppt_when_amo_neg=iceland_obs_ppt_when_amo_neg.collapsed(['latitude','longitude'],iris.analysis.MEAN)
        print('iceland_obs_ppt_when_amo_neg data:',iceland_obs_ppt_when_amo_neg.data)

        azores_obs_ppt_when_amo_neg = extract_region(obs_ppt_when_amo_neg,region=reg_azr)
        print('azores_obs_ppt_when_amo_neg lats:',azores_obs_ppt_when_amo_neg.coord('latitude').points)
        print('azores_obs_ppt_when_amo_neg lons:',azores_obs_ppt_when_amo_neg.coord('longitude').points)
        azores_obs_ppt_when_amo_neg=azores_obs_ppt_when_amo_neg.collapsed(['latitude','longitude'],iris.analysis.MEAN)
        print('azores_obs_ppt_when_amo_neg data:',azores_obs_ppt_when_amo_neg.data)

        print('azores_obs_ppt_when_amo_neg-iceland_obs_ppt_when_amo_neg:',
              azores_obs_ppt_when_amo_neg.data-iceland_obs_ppt_when_amo_neg.data)

    
    if desired_precip_output_format is not None:
        if desired_precip_output_format == '+ve amo minus -ve amo as fraction of -ve':
            # Calculate +ve amo minus -ve amo as a percentage of -ve amo prec        
            obs_ppt_diff = obs_ppt_when_amo_pos - obs_ppt_when_amo_neg
        
            if percentage:
                obs_ppt_diff=100.0*(obs_ppt_diff/obs_ppt_when_amo_neg)
        
            # Add information about the number of +ve and -ve AMO years to the cube as a new coord:
            obs_ppt_diff.add_aux_coord(iris.coords.AuxCoord(
                str(number_of_obs_ppt_when_amo_neg_years),
                long_name='number_of_neg_season_years', units='no_unit'))

            obs_ppt_diff.add_aux_coord(iris.coords.AuxCoord(
                str(number_of_obs_ppt_when_amo_pos_years),
                long_name='number_of_pos_season_years', units='no_unit'))
        
            return obs_ppt_diff
    else:        
        # Return a cube of mean prec when amo is negative and a cube of mean prec when amo is positive:
        return obs_ppt_when_amo_neg, obs_ppt_when_amo_pos



########################################################################
def obs_ppt_conditioned_on_obs_amo(obsdir, amo_season=None, driver='amo', amoregion=None,
                                   amoobssource='hadcrut5', ppt_season=None, ppt_region=None,obssource=None, 
                                   weather_field=None, amo_anomaly_format=None,anomalise_ppt=False,
                                   basebeg=1971, nbase=30, start_year=1960, number_of_years=60):

    ''' Function to calculate obs prec dependent on obs amo 
    '''

    print('Welcome to obs_ppt_conditioned_on_obs_amo')    

    # Need to get the obs amo, but firstly need to set whether we get it as anomalies wrt a baseline period, 
    # or whether to simply get it as absolutes and then turn it into anomalies.    
    print('amoobssource:',amoobssource)
    print('amo_anomaly_format given to obs_ppt_conditioned_on_obs_amo:',amo_anomaly_format)

    if amo_anomaly_format is None:
        # Don't do any anomalising at all:
        anomalise_amo=False
    else:
        if amo_anomaly_format=='of_a_baseline_climate':
            # Anomalise wrt to a set period eg 1971 to 2000 (given by basebeg and nbase)
            anomalise_amo=True
        if amo_anomaly_format=='of_whole_period':
            anomalise_amo=True
            basebeg=None
            nbase=None

    print('anomalise_amo to be given to get_obs_idx:',anomalise_amo)

    # Get obs idx:
    obs_amo_all_years = get_obs_idx(obsdir, amo_season=amo_season, driver=driver, amoregion=None,
                                    amoobssource=amoobssource, basebeg=basebeg,
                                    nbase=nbase, anomalise_amo=anomalise_amo,
                                    start_year=start_year, number_of_years=number_of_years)
    
    #print('obs_amo_all_years:',obs_amo_all_years.data)
    #print('obs_amo_all_years season_year:',obs_amo_all_years.coord('season_year'))

    # Get obs ppt: 
    obs_ppt_all_years = get_obs_ppt(obsdir, season=ppt_season, obssource=obssource,
                                    region=ppt_region, weather_field=weather_field,
                                    basebeg=basebeg, nbase=nbase, anomalise=anomalise_ppt,
                                    start_year=start_year, number_of_years=number_of_years)
    
    # obs_ppt is a cube of dims time, latitude, longitude
    #print('obs_ppt_all_years season_years:',obs_ppt_all_years.coord('season_year').points)
    #print('obs_ppt_all_years season_years units:',obs_ppt_all_years.units)


    obs_ppt_slices_when_amo_neg=[]
    obs_ppt_slices_when_amo_pos=[]

    amo_neg_index=0
    amo_pos_index=0

    season_years_of_negative_amo=[]
    season_years_of_positive_amo=[]
   
    # Go through each obs_amo_all_years slice:
    for obs_amo in obs_amo_all_years.slices([]):
    
        # Prepare constraint for later precip extraction from obs_ppt_all_years:
        constraint = iris.Constraint(season_year=obs_amo.coord('season_year').points[0])

        # Extract slice of obs_ppt_all_years corresponding to obs_amo:
        obs_ppt = obs_ppt_all_years.extract(constraint)

        if obs_ppt is not None:

            # Check that obs_ppt really is of obs_amo.coord('season_year').points[0]:
            if obs_ppt.coord('season_year').points[0] != obs_amo.coord('season_year').points[0]:
                print('Error in obs_ppt extraction.')
                raise AssertionError('Stop for debugging...')

            if obs_amo.data < 0.0:
                # We have a negative amo slice. Increment counter which will be used to index a 
                # resulting precip cube (to allow later cube merging)
                amo_neg_index += 1
            
                # Append prec cube to list over all occurences of -ve amo
                obs_ppt_slices_when_amo_neg.append(obs_ppt)
            
            if obs_amo.data >= 0.0:
                # We have a positive amo slice. Increment counter which will be used to index a 
                # resulting precip cube (to allow later cube merging)
                amo_pos_index += 1
        
                # Append prec cube to list over all occurences of -ve amo
                obs_ppt_slices_when_amo_pos.append(obs_ppt)


    # cubelist and merge obs_ppt_when_obs_amo_negative and obs_ppt_when_obs_amo_positive:
    obs_ppt_when_amo_neg = iris.cube.CubeList(obs_ppt_slices_when_amo_neg).merge_cube()
    obs_ppt_when_amo_pos = iris.cube.CubeList(obs_ppt_slices_when_amo_pos).merge_cube()

    #print('842 obs_ppt_when_amo_neg:',obs_ppt_when_amo_neg)
    #print('843 obs_ppt_when_amo_pos:',obs_ppt_when_amo_pos)

    season_years_of_negative_amo = obs_ppt_when_amo_neg.coord('season_year').points
    season_years_of_positive_amo = obs_ppt_when_amo_pos.coord('season_year').points

    print('Number of obs negative '+driver+' years:',len(season_years_of_negative_amo),
        '(',season_years_of_negative_amo,')')
    print('Number of obs positive '+driver+' years:',len(season_years_of_positive_amo),
        '(',season_years_of_positive_amo,')')

    # Return cube of multiple years of obs prec during negative amo events, and
    # cube of multiple years of obs prec during positive amo events.

    return obs_ppt_when_amo_neg, obs_ppt_when_amo_pos


########################################################################
# NEW - imported from functions_for_this_dir.py
########################################################################

def getobs(season, startyear, nyears, obsdir=None,
           region=None, weatherfield=None, obssource='era5',
           percentiles=None, anomalise=False, returnanomaliesaspercentages=False, 
           area_avg=False, startyearofclimateperiod=None, nyearsofclimateperiod=None):

    ''' 
    Function to build filename and load in a cube of gridded observations or reanalyses. 

    Required args:
        season: season class (see seasonclass.py for info on setting up).
        startyear: integer - first year of obs.
        nyears: number of years to read in.
        obsdir: firectory for obs data.
    Keyword args:
        region: region class (see regionclass.py for info on setting up)
        obssource: Obs source ('erai')
        weatherfield: weather field class (see weatherfieldclass.py for info on setting up)
        obssource: Obs source ('era5')
        TBC...                           
    '''
     
    print('Welcome to getobs which will get ',season.mmm,weatherfield.standard_name,' obs, from ',obssource)
    print('percentiles:',percentiles)
    print('season:',season,' weatherfield:',weatherfield,' obssource:',obssource)

    # Build filename for obs, use RC name     
    obs_scratchfile = obssource+'_'+season.mmm+'_'+str(startyear)+'for'+str(nyears)+'yrs'    

    obs_scratchfile = obs_scratchfile + '_'+weatherfield.standard_name + '_norm='+str(weatherfield.norm)[0] 

    if area_avg:    obs_scratchfile += '_area_avg'
    if percentiles: obs_scratchfile += '_percentiles'
    if region.standard_name != 'None': 
        obs_scratchfile += '_'+region.standard_name.replace(' ','')
    if anomalise:
        anomstr = '_anomalies'
        if returnanomaliesaspercentages: 
            anomstr += '_aspercentages'
        anomstr += '_of'
        anomstr += str(startyearofclimateperiod)+'for'+str(nyears)+'yrs'
        obs_scratchfile = obs_scratchfile + anomstr

    obs_scratchfile = os.path.join(obsdir, obs_scratchfile+'.nc')

    # Check existance of scratchfile:    
    errorcode_of_obs_scratchfile = subprocess.call('find '+obs_scratchfile, shell=True)
              
    if errorcode_of_obs_scratchfile==0:
        # Summary file exists. Read it in and return the output to parent.
        print('obs_scratchfile\n',obs_scratchfile,'\n exists. Read it in:')
        obs = iris.load_cube(obs_scratchfile)
        print('obs from\n'+obs_scratchfile+'\n:',obs)

    else:    
        print('Input file: ',obs_scratchfile)
        print('NOT FOUND => run plot_fig19_and_S18_full.py to create this')
        raise AssertionError('Stop for debugging...')
    
    return obs

########################################################################
def regrid(cube=None, cubelist=None, cubetoregridto=None, scheme=iris.analysis.Linear(), remove_coord_system=False):
    ''' 
    Function to regrid a cube (cube) or list (cubelist) of cubes onto the grid of a 2nd cube (cubetoregridto)
    '''
    if cubelist != None:
        # Regrid cubelist recursively
        regriddedcubes=[]
        for cube1 in cubelist:
            regriddedcube = regrid(cube=cube1, cubetoregridto=cubetoregridto, 
                                   remove_coord_system=remove_coord_system)        
            regriddedcubes.append(regriddedcube)        
        # cubelist but dont merge:
        regriddedcubes = iris.cube.CubeList(regriddedcubes)
        return regriddedcubes
        
    if cube != None:
        # Regrid cube
        if cubetoregridto != None:
           # If coord systems differ, remove the coords. 
            if cube.coord('latitude').coord_system != cubetoregridto.coord('latitude').coord_system:
                cube.coord('latitude').coord_system=None
                cubetoregridto.coord('latitude').coord_system=None            
            if cube.coord('longitude').coord_system != cubetoregridto.coord('longitude').coord_system:
                cube.coord('longitude').coord_system=None
                cubetoregridto.coord('longitude').coord_system=None
            
        # Regrid single cube
        regriddedcube = cube.regrid(cubetoregridto, scheme)        
        # Copy grid attribute from cubetoregridto if it exists:
        try:
            regriddedcube.attributes['grid'] = cubetoregridto.attributes['grid']
        except:
            pass
        
        regriddedcube.name = cube.name
        return regriddedcube

########################################################################
def regrid_to_target(cubes, scheme=iris.analysis.Linear(), coords_to_regrid=None, target='coarsest'):
    '''
    Regrid cubes in cubelist to cube specified by target (one of 'depresys', 'ncar', 'coarsest')    
    Args:
        cubes: a list of cubes     
    KWArgs: 
        scheme: the regridding function (given as a string) to be used (default: iris.analysis.Linear())
        coords_to_regrid: coordinates to regrid (given as a list of strings)    
    '''

    # Follow RC and identify target in cubes from total number of grid-points
    if target == 'coarsest':
        number_of_latitudes = numpy.array( [len(cube.coord(coords_to_regrid[0]).points) for cube in cubes] )   
        index_min           = numpy.where( number_of_latitudes == number_of_latitudes.min() )[0][0]
        # Return cubes regridded to the coarsest:
        ans = regrid(cubelist=cubes, cubetoregridto=cubes[index_min], scheme=scheme, remove_coord_system=False)
    else:
        number_of_latitudes  = numpy.array( [len(cube.coord(coords_to_regrid[0]).points) for cube in cubes] )   
        number_of_longitudes = numpy.array( [len(cube.coord(coords_to_regrid[1]).points) for cube in cubes] )   
        nn        = number_of_latitudes*number_of_longitudes
        # Note, only setup for ncar, cmcc, depresys   
        if target == 'ncar':     
            nnrequired=43*65    # this is NCAR and CMCC resolution
        elif target == 'depresys':  
            nnrequired=72*96    # depresys resolution 
        else:    
            nnrequired=72*96    # depresys resolution   
        index_ok  = numpy.where( nn == nnrequired )[0][0]
        ans       = regrid(cubelist=cubes, cubetoregridto=cubes[index_ok], scheme=scheme, remove_coord_system=False)
    return ans    

########################################################################
def unify_cubelist(cubes):
    '''
    Convert long_names, var_names, standard_names, and units of all cubes to those of the first.  
    '''
    for cube in cubes[1:]:

        cube.long_name     = cubes[0].long_name
        cube.var_name      = cubes[0].var_name
        cube.standard_name = cubes[0].standard_name
 
        if cubes[0].units=='mm s-1':                
            if cube.units=='mm/day':
                cube.units='mm day-1'
                cube.convert_units('mm s-1')
        
        if cubes[0].units=='Pa':
            cube.convert_units('Pa')
        
        if cubes[0].units=='hPa':
            cube.convert_units('hPa')

        if cube.units != cubes[0].units:
            print('cubes[0].units is still different from that of cube')
            print('cubes[0].units:',cubes[0].units)
            print('cube.units:',cube.units)
            print('Stopping to allow modification of function unify_cubelist')
            raise AssertionError('Stop for debugging...')            

    return cubes

########################################################################
def remove_cubelist_attributes(cubelist):
    for cube in cubelist:
        cube.attributes=None        
    return cubelist

########################################################################
def remove_cubelist_cell_methods(cubelist):
    for cube in cubelist:
        cube.cell_methods=None        
    return cubelist

########################################################################
def remove_cubelist_attributes(cubelist):
    for cube in cubelist:
        cube.attributes=None        
    return cubelist

########################################################################
def convert_cubelist_to_float32(cubelist):
    for cube in cubelist:
        cube.data = cube.data.astype(numpy.float32)    
    return cubelist

########################################################################
def remove_cubelist_coords(cubelist, coords):    
    def hasCoord(cube, coordtofind):
        return coordtofind in set([c.name() for c in cube.coords()])
    for coord in coords:        
        for cube in cubelist:
            if hasCoord(cube, coord): 
                cube.remove_coord(coord)                
    return cubelist

########################################################################
def number_of_years_of_initialised_forecasts(ensemble):
    # Old  default of 58: [1960, 2017]
    numberof_forecast_reference_years=58
    
    if ensemble=='BCC':             numberof_forecast_reference_years=54    # 1960-2013     2013-1960+1 = 54  start year actually 1961
    if ensemble=='CanESM5':         numberof_forecast_reference_years=57    # 1960-2016     2016-1960+1 = 57  start year actually 1961
    if ensemble=='IPSL':            numberof_forecast_reference_years=57    # 1960-2016     2016-1960+1 = 57  start year actually 1961

    if ensemble=='BSC':             numberof_forecast_reference_years=59    # 1960-2018     2018-1960+1 = 59
    if ensemble=='CAFE':            numberof_forecast_reference_years=60    # 1960-2019     2019-1960+1 = 60
    if ensemble=='CMCC':            numberof_forecast_reference_years=60    # 1960-2019     2019-1960+1 = 60
    if ensemble=='MIROC6':          numberof_forecast_reference_years=59    # 1960-2018     2018-1960+1 = 59        
    if ensemble=='MPI':             numberof_forecast_reference_years=58    # 1960-2017     2017-1960+1 = 58
    if ensemble=='NorCPM':          numberof_forecast_reference_years=59    # 1960-2018     2018-1960+1 = 59   
    if ensemble=='NorCPMi2':        numberof_forecast_reference_years=59    # 1960-2018     2018-1960+1 = 59   
    if ensemble=='NCAR40':          numberof_forecast_reference_years=58    # 1960-2017     2017-1960+1 = 58
    if ensemble=='Depresys4_gc3.1': numberof_forecast_reference_years=60    # 1960-2019     2019-1960+1 = 60     
    # Depresys4_gc3.1 actually goes to 2023 now, but restrict to 2019 (max of others), since these extra years not needed
    
    return numberof_forecast_reference_years
    
########################################################################
def extract_region(cube, region=None):
    '''
    Function to extract a region from a cube or cubelist.    
    Args:
        cube: cube of data with ['latitude','longitude'] coordinates
        region: Region class. See regionclass.py for info.            
    '''
    if region.standard_name=='None':
        print('region.standard_name given to extract_region is None. Returning cube as given')
        return cube
    # Determine whether cube is a cube or a cubelist:
    if type(cube) is iris.cube.CubeList:
        # Have cubelist, go through each cube:
        truecubes=[]
        for truecube in cube:
            truecube = extract_region(truecube, region=region)
            truecubes.append(truecube)
        return iris.cube.CubeList(truecubes)        
    else:               
        ce = iris.coords.CoordExtent('longitude', region.coords[0],region.coords[2])
        if ce.minimum > ce.maximum:
            ce = iris.coords.CoordExtent('longitude', region.coords[0],region.coords[2]+360.0)
        ce2 = iris.coords.CoordExtent('latitude', region.coords[1], region.coords[3])            
        keep_all_longitudes = False
        if region.coords[0] == 0 and region.coords[2] == 0:      keep_all_longitudes=True
        if region.coords[0] == 0 and region.coords[2] == 360:    keep_all_longitudes=True
        if region.coords[0] == -180 and region.coords[2] == 180: keep_all_longitudes=True    
        #print 'region.coords[0],region.coords[2]:',region.coords[0],region.coords[2]    
        if keep_all_longitudes:
            # Extract over latitudes only:
            cube = cube.extract(iris.Constraint(latitude=lambda cell: region.coords[1]<=cell<=region.coords[3]))
        else:
            cube = cube.intersection(ce, ce2)
            cube.data        
    return cube

########################################################################
def extract_from_dps_cubes(dps_cubes, coords_to_extract, instances_to_extract, 
                           return_cube=True, remove_time_coord=False):
    '''
    From a list of cubes, extract cubes of requested instances of given coords.
    Note - this is only set up for 'forecast_period' and 'realization' so far.
    '''    
    # Extract instances for each cube in dps_cubes.
    dps_cubes_out = []    
    for cube in dps_cubes:
        for coord, instance in zip(coords_to_extract, instances_to_extract):
            constraint = None
            if coord == 'forecast_period':
                constraint=iris.Constraint(forecast_period=instance)
            if coord == 'realization':
                constraint=iris.Constraint(realization=instance)
            if constraint is None:
                print('Need to add functionality for ',coord,' coord in function extract_from_dps_cubes')
            # Extract:
            cube = cube.extract(constraint)
            if remove_time_coord:
                if coord_existance(cube,'time'):
                    cube.remove_coord('time')
        dps_cubes_out.append(cube)

    if return_cube:
        print('len(dps_cubes_out):',len(dps_cubes_out))        
        dps_cubes_out = unify_coord(dps_cubes_out, coord_to_extract_from, verbose=verbose)
        # Return a merged cube to parent        
        return iris.cube.CubeList(dps_cubes_out).merge_cube()
    else:
        # Return a cubelist to parent 
        return iris.cube.CubeList(dps_cubes_out)
    

########################################################################
def load_cmap(palette, middle='white', whiteinmiddle=False, middle_fraction=0.5):
    '''
    Sort out a colour map, ie a palette.   
    Args:
        palette: a string of either a single colour (eg 'green') or a list of 2 colours for a bicolor color map    
    Keyword Args:
        whiteinmiddle: a boolean. If True, will impose the colour given by middle KWArg in the middle of the bar. (default: False)
        middle: a string of colour (eg 'white') for the middle of the colour map 
    Example:
        load_cmap('blue','green',middle='white',whiteinmiddle=True) will give a colour map ranging from dark blue to white to dark green
    '''
    def monocolor_cmap(color):
        from matplotlib.colors import LinearSegmentedColormap    
        return LinearSegmentedColormap.from_list(None, [(0.0, 'white'),(1.0, color)])

    def bicolor_cmap(left,right,middle='white',whiteinmiddle=False,middle_fraction=0.50):
        from matplotlib.colors import LinearSegmentedColormap    
        if whiteinmiddle:
            return LinearSegmentedColormap.from_list(None, [(0.0, left),(0.45, middle),(0.55, middle),(1.0, right)])
        else:
            return LinearSegmentedColormap.from_list(None, [(0.0, left),(middle_fraction, middle),(1.0, right)])

    if type(palette) is list:
        if len(palette) == 1: 
            cmap = monocolor_cmap(palette[0])
        if len(palette) == 2: 
            cmap = bicolor_cmap(palette[0], palette[1], middle_fraction=middle_fraction)            
    else:
        cmap = palette
    return cmap

