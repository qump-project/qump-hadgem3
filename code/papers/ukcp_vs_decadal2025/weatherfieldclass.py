class InitWeatherfield:
    '''
    Class of the weather field to consider. 
    Class has properties standard_name (eg wind_speed), height (eg 10) and height units(eg m)
    '''     
    def __init__(self, standard_name, height, heightunits, norm=False):

        '''
        Function to set up weatherfieldclass. A class which holds a range of useful information about 
        an individual weather diagnostic (eg precip).
        Set it by: weatherfield=weatherfieldclass.InitWeatherfield(individualseason). An example is:
            weatherfield=weatherfieldclass.InitWeatherfield('air_pressure_at_sea_level',None,None)
        
        The resulting class has attributes:
        season.standard_name:    The cube.standard_name which iris uses, eg 'wind_speed, 'air_temperature', 'precipitation_flux'.
        season.long_name:        Similar to standard_name but with appropriate spaces added.
        season.nameforfilenames: Lowercase, all spaces removed version for filenames.
        season.cmip_name:        The appropriate name of the diagnostic used in the CMIP5 archive. eg 'tas' for 'air_temperature'.
        season.height:           Height for diagnostics of height, eg 10m wind.    
        season.heightunits:      units of height for diagnostics of height, eg 10m wind.      
        '''

        if standard_name!=None:
        
            self.standard_name = standard_name
            self.long_name     = standard_name
            self.short_name    = standard_name
            
            if self.standard_name=='wind_speed':                          self.long_name='wind speed'
            if self.standard_name=='wind_power':                          self.long_name='wind power'
            if self.standard_name=='power_density':                       self.long_name='power density'
            if self.standard_name=='power_density_using_fixed_density':   self.long_name='power density using fixed density'
            if self.standard_name=='power_density_using_dynamic_density': self.long_name='power density using dynamic density'
            if self.standard_name=='eastward_wind':                       self.long_name='westerly wind speed'
            if self.standard_name=='air_temperature':                     self.long_name='air temperature'
            if self.standard_name=='surface_temperature':                 self.long_name='surface temperature'
            if self.standard_name=='sea_surface_temperature':             self.long_name='sea surface temperature'
            if self.standard_name=='air_pressure_at_sea_level':           self.long_name='air pressure at sea level'
            if self.standard_name=='precipitation':                       self.long_name='precipitation'
            if self.standard_name=='precipitation_flux':                  self.long_name='precipitation'
            if self.standard_name=='geopotential_height':                 self.long_name='geopotential height'

            if self.standard_name=='wind_speed':                          self.short_name='wspeed'
            if self.standard_name=='wind_power':                          self.short_name='wpower'
            if self.standard_name=='power_density':                       self.long_name='power density'
            if self.standard_name=='power_density_using_fixed_density':   self.short_name='power density'
            if self.standard_name=='power_density_using_dynamic_density': self.short_name='power density'
            if self.standard_name=='eastward_wind':                       self.short_name='u'
            if self.standard_name=='northward_wind':                      self.short_name='v'
            if self.standard_name=='vertical_wind':                       self.short_name='ascent'
            if self.standard_name=='air_temperature':                     self.short_name='air temperature'
            if self.standard_name=='surface_temperature':                 self.short_name='surface temperature'
            if self.standard_name=='sea_surface_temperature':             self.short_name='SST'
            if self.standard_name=='air_pressure_at_sea_level':           self.short_name='PMSL'
            if self.standard_name=='precipitation':                       self.short_name='precipitation'
            if self.standard_name=='precipitation_flux':                  self.short_name='precipitation'
            if self.standard_name=='geopotential_height':                 self.short_name='Geop. height'
            if self.standard_name=='amo':                                 self.short_name='AMO'
            if self.standard_name=='enso':                                self.short_name='ENSO'
            if self.standard_name=='nao':                                 self.short_name='NAO'
            if self.standard_name=='nao_hurrell':                         self.short_name='NAO'

            self.nameforfilenames = self.standard_name
            self.heightunits      = heightunits                                       
            self.height           = height
            
            if self.height != None: self.long_name=str(height)+self.long_name
            if self.height != None: self.nameforfilenames=str(height)+self.nameforfilenames

            #print self.standard_name,',', self.long_name,',', self.height,',', self.heightunits

            self.standard_nameinrawumfiles = self.standard_name
            if self.standard_name == 'precipitation':self.standard_nameinrawumfiles='precipitation_flux'

            if self.standard_name=='wind_speed':                          self.cmip_name='wind speed'
            if self.standard_name=='wind_power':                          self.cmip_name='wind power'
            if self.standard_name=='power_density':                       self.cmip_name='wind power'
            if self.standard_name=='power_density_using_fixed_density':   self.cmip_name='power density using fixed density'
            if self.standard_name=='power_density_using_dynamic_density': self.cmip_name='power density using dynamic density'
            if self.standard_name=='eastward_wind':                       self.cmip_name='ua'
            if self.standard_name=='northerly_wind':                      self.cmip_name='va'
            if self.standard_name=='air_temperature':                     self.cmip_name='tas'
            if self.standard_name=='tmax':                                self.cmip_name='tasmax'
            if self.standard_name=='tmin':                                self.cmip_name='tasmin'
            if self.standard_name=='surface_temperature':                 self.cmip_name='surface temperature'
            if self.standard_name=='sea_surface_temperature':             self.cmip_name='sea surface temperature'
            if self.standard_name=='air_pressure_at_sea_level':           self.cmip_name='psl'
            if self.standard_name=='precipitation':                       self.cmip_name='pr'
            if self.standard_name=='precipitation_flux':                  self.cmip_name='pr'
            if self.standard_name=='nao':                                 self.cmip_name='nao'
            if self.standard_name=='nao_hurrell':                         self.cmip_name='nao_hurrell'
            if self.standard_name=='amo':                                 self.cmip_name='amo'
            if self.standard_name=='enso':                                self.cmip_name='enso'

            #New nao
            self.norm = False
            if self.standard_name[:4] == 'nao_':
                self.short_name = 'NAO'
                self.cmip_name  = self.standard_name
                self.norm       = norm
            
            
            








