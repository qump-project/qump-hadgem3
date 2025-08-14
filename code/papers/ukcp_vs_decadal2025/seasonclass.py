class InitSeason:
    def __init__(self, name, minimum_forecast_leadtime_in_days=19, maximum_forecast_leadtime_in_days=40):
        '''
        Function to set up seasonclass. A class which holds a range of useful information about an  
        individual season. Set by: season=seasonclass.InitSeason(individualseason). An example is:
            season = seasonclass.InitSeason('djf')
        
        The resulting class has attributes .months, .standard_name, standard_name2, 
        desired_forecast_reference_times, .mmm, .MMM and .notmmm.
        
        Examples:
        season.months could be [12,1,2]
        season.standard_name could be 'winter'
        season.standard_name2 could be 'Winter'
        season.mmm could be 'djf'
        season.MMM could be 'DJF'
        season.desired_forecast_reference_times could be ['000402','0004010'] meaning April 2nd and April 10th
        (used when retrieving operational season hindcasts)
        season.notmmm could be [3,4,5,6,7,8,9,10,11] for season djf. This is typically used by iris when extracting 
        slices of cubes for particular seasons.            
        '''        
        
        self.mmm = name
        self.MMM = name.upper()

        if name == 'dj':self.months=[12,1]
        if name == 'jf':self.months=[1,2]
        if name == 'fm':self.months=[2,3]
        if name == 'ma':self.months=[3,4]
        if name == 'am':self.months=[4,5]
        if name == 'mj':self.months=[5,6]
        if name == 'jj':self.months=[6,7]
        if name == 'ja':self.months=[7,8]
        if name == 'as':self.months=[8,9]
        if name == 'so':self.months=[9,10]
        if name == 'on':self.months=[10,11]
        if name == 'nd':self.months=[11,12]

        if name == 'djf':self.months=[12,1,2]
        if name == 'jfm':self.months=[1,2,3]
        if name == 'fma':self.months=[2,3,4]
        if name == 'mam':self.months=[3,4,5]
        if name == 'amj':self.months=[4,5,6]
        if name == 'mjj':self.months=[5,6,7]
        if name == 'jja':self.months=[6,7,8]
        if name == 'jas':self.months=[7,8,9]
        if name == 'aso':self.months=[8,9,10]
        if name == 'son':self.months=[9,10,11]
        if name == 'ond':self.months=[10,11,12]
        if name == 'ndj':self.months=[11,12,1]

        if name == 'djfm':self.months=[12,1,2,3]
        if name == 'jfma':self.months=[1,2,3,4]
        if name == 'fmam':self.months=[2,3,4,5]
        if name == 'mamj':self.months=[3,4,5,6]
        if name == 'amjj':self.months=[4,5,6,7]
        if name == 'mjja':self.months=[5,6,7,8]
        if name == 'jjas':self.months=[6,7,8,9]
        if name == 'jaso':self.months=[7,8,9,10]
        if name == 'ason':self.months=[8,9,10,11]
        if name == 'sond':self.months=[9,10,11,12]
        if name == 'ondj':self.months=[10,11,12,1]
        if name == 'ndjf':self.months=[11,12,1,2]

        if name == 'djfma':self.months=[12,1,2,3,4]
        if name == 'jfmam':self.months=[1,2,3,4,5]
        if name == 'fmamj':self.months=[2,3,4,5,6]
        if name == 'mamjj':self.months=[3,4,5,6,7]
        if name == 'amjja':self.months=[4,5,6,7,8]
        if name == 'mjjas':self.months=[5,6,7,8,9]
        if name == 'jjaso':self.months=[6,7,8,9,10]
        if name == 'jason':self.months=[7,8,9,10,11]
        if name == 'asond':self.months=[8,9,10,11,12]
        if name == 'sondj':self.months=[9,10,11,12,1]
        if name == 'ondjf':self.months=[10,11,12,1,2]
        if name == 'ndjfm':self.months=[11,12,1,2,3]

        if name == 'jfmam':self.months=[1,2,3,4,5]
        if name == 'jfmamj':self.months=[1,2,3,4,5,6]
        if name == 'jfmamjj':self.months=[1,2,3,4,5,6,7]
        if name == 'jfmamjja':self.months=[1,2,3,4,5,6,7,8]
        if name == 'jfmamjjas':self.months=[1,2,3,4,5,6,7,8,9]
        if name == 'jfmamjjaso':self.months=[1,2,3,4,5,6,7,8,9,10]
        if name == 'jfmamjjason':self.months=[1,2,3,4,5,6,7,8,9,10,11]
        if name == 'jfmamjjasond':self.months=[1,2,3,4,5,6,7,8,9,10,11,12]

        if name == 'djfmamjjason':self.months=[12,1,2,3,4,5,6,7,8,9,10,11]

        if name == 'mjjas':self.months=[5,6,7,8,9]
        if name == 'ondjfm':self.months=[10,11,12,1,2,3]

        if name == 'jan':self.months=[1]
        if name == 'feb':self.months=[2]
        if name == 'mar':self.months=[3]
        if name == 'apr':self.months=[4]
        if name == 'may':self.months=[5]
        if name == 'jun':self.months=[6]
        if name == 'jul':self.months=[7]
        if name == 'aug':self.months=[8]
        if name == 'sep':self.months=[9]
        if name == 'oct':self.months=[10]
        if name == 'nov':self.months=[11]
        if name == 'dec':self.months=[12]

        # Compile list of abbreviated names of months based on self.months:        
        self.months_mmm=[]
        for month in self.months:
            if month==1: self.months_mmm.append('jan')
            if month==2: self.months_mmm.append('feb')
            if month==3: self.months_mmm.append('mar')
            if month==4: self.months_mmm.append('apr')
            if month==5: self.months_mmm.append('may')
            if month==6: self.months_mmm.append('jun')
            if month==7: self.months_mmm.append('jul')
            if month==8: self.months_mmm.append('aug')
            if month==9: self.months_mmm.append('sep')
            if month==10: self.months_mmm.append('oct')
            if month==11: self.months_mmm.append('nov')
            if month==12: self.months_mmm.append('dec')

        self.standard_name='multi-season'

        if name == 'dec':  self.standard_name='winter'
        if name == 'dj':   self.standard_name='winter'
        if name == 'djf':  self.standard_name='winter'
        if name == 'djfm': self.standard_name='winter'
        if name == 'jf':   self.standard_name='winter'
        if name == 'jfm':  self.standard_name='winter'
        if name == 'jfma': self.standard_name='winter'
                       
        if name == 'ma':   self.standard_name='spring'
        if name == 'mam':  self.standard_name='spring'
        if name == 'mamj': self.standard_name='spring'
        if name == 'am':   self.standard_name='spring'
        if name == 'amj':  self.standard_name='spring'
        if name == 'amjj': self.standard_name='spring'

        if name == 'jj':   self.standard_name='summer'
        if name == 'jja':  self.standard_name='summer'
        if name == 'jjas': self.standard_name='summer'
        if name == 'ja':   self.standard_name='summer'
        if name == 'jas':  self.standard_name='summer'
        if name == 'jaso': self.standard_name='summer'
                
        if name == 'so':   self.standard_name='autumn'
        if name == 'son':  self.standard_name='autumn'
        if name == 'sond': self.standard_name='autumn'
        if name == 'on':   self.standard_name='autumn'
        if name == 'ond':  self.standard_name='autumn'
        if name == 'ondj': self.standard_name='autumn'

        if name == 'ondjfm': self.standard_name='winter'
        
        if name == 'djfmamjjason': self.standard_name='annual'
        if name == 'jfmamjjasond': self.standard_name='annual'

        self.standard_name2 = self.standard_name[0].upper()+self.standard_name[1:]
        

        # Set desired_forecast_reference_times for possible later use when loading in glosea5 data
        # in function g5andobs:
        
        # Scan over all possible lead times. Desirable forecast_reference_times are those between 
        # with a leadtime between minimum_forecast_leadtime_indays and maximum_forecast_leadtime_indays
        
        # Normally minimum_forecast_leadtime_in_days will be 21 days before the start of the forecast 
        # target period.
        
        # So a forecast target period of DJF  with a minimum_forecast_leadtime_in_days of 21 days
        # will have a latest desired_forecast_reference_times entry of November 9th.
        
        # Similarly, maximum_forecast_leadtime_in_days will normally be 35 days before the start of the 
        # forecast target period.
        
        # So a forecast target period of DJF  with a maximum_forecast_leadtime_in_days of 35 days
        # will have an EARLIEST desired_forecast_reference_times entry of October 25th.
        
        # maximum_forecast_leadtime_in_days can be set to a massive value to allow very early forecasts 
        # (ie with long lead time) to be loaded.
        
        # The format of the desired_forecast_reference_times will be of 6 digits: yymmdd
        # The mm and dd will be the month and day of the desired forecast reference times. The yy part however 
        # is slightly different. In cases where the year of the desire forecast reference time is numerically the 
        # same as the year of the start of the target yy is set to 00. If, however the year is one less, it is set to -1. 
        # As an example, if a forecast target period starts in February, the desired forecast reference times in January 
        # will have yy set to 00 but for desired forecast reference times of December, yy will be -1.  
        # The -1 will ensure that the correct forecasts or hindcasts will be picked up in function g5andobs (if called).


        forecastleadtime = 0
        self.desired_forecast_reference_times = []
        
        forecast_reference_time_mm = self.months[0]
        forecast_reference_time_dd = 1
        forecast_reference_time_yy = 0 
            
        while forecastleadtime<maximum_forecast_leadtime_in_days:
            #print 'forecastleadtime:',forecastleadtime
            # skip over leadtimes smaller than minimum_forecast_leadtime_in_days
            
            forecast_reference_time = "%02d" % (forecast_reference_time_yy) + "%02d" % (forecast_reference_time_mm) + "%02d" % (forecast_reference_time_dd)
            #print forecast_reference_time
            
            if forecastleadtime>minimum_forecast_leadtime_in_days:
                # append to desired_forecast_reference_times
                self.desired_forecast_reference_times.append(forecast_reference_time)
            
            # take a day off from forecast_reference_time_dd and by implication update forecast_reference_time_mm
            # and possibly even forecast_reference_time_yy:

            forecast_reference_time_dd += -1
            if forecast_reference_time_dd == 0:
                forecast_reference_time_mm -= 1
                if forecast_reference_time_mm == 0:
                    # forecast_reference_time_mm needs to be reset to December and the year (a relative number) to -1:
                    forecast_reference_time_mm = 12
                    forecast_reference_time_yy =- 1
                    
                # set forecast_reference_time_dd to be the last day of forecast_reference_time_mm
                if forecast_reference_time_mm==1:forecast_reference_time_dd=31
                if forecast_reference_time_mm==2:forecast_reference_time_dd=28
                if forecast_reference_time_mm==3:forecast_reference_time_dd=31
                if forecast_reference_time_mm==4:forecast_reference_time_dd=30
                if forecast_reference_time_mm==5:forecast_reference_time_dd=31
                if forecast_reference_time_mm==6:forecast_reference_time_dd=30
                if forecast_reference_time_mm==7:forecast_reference_time_dd=31
                if forecast_reference_time_mm==8:forecast_reference_time_dd=31
                if forecast_reference_time_mm==9:forecast_reference_time_dd=30
                if forecast_reference_time_mm==10:forecast_reference_time_dd=31
                if forecast_reference_time_mm==11:forecast_reference_time_dd=30
                if forecast_reference_time_mm==12:forecast_reference_time_dd=31                
            
            forecastleadtime = forecastleadtime+1

        if name == 'dec':   self.notmmm='jfmamjjason'
        if name == 'dj':    self.notmmm='fmamjjason'
        if name == 'djf':   self.notmmm='mamjjason'
        if name == 'djfm':  self.notmmm='amjjason'
        if name == 'jf':    self.notmmm='mamjjasond'
        if name == 'jfm':   self.notmmm='amjjasond'
        if name == 'jfma':  self.notmmm='mjjasond'

        if name == 'jfmam':        self.notmmm='jjasond'
        if name == 'jfmamj':       self.notmmm='jasond'
        if name == 'jfmamjj':      self.notmmm='asond'
        if name == 'jfmamjja':     self.notmmm='sond'
        if name == 'jfmamjjas':    self.notmmm='ond'
        if name == 'jfmamjjaso':   self.notmmm='nd'
        if name == 'jfmamjjason':  self.notmmm='d'
        if name == 'jfmamjjasond': self.notmmm=''
        if name == 'djfmamjjason': self.notmmm=''
        
        
        if name == 'ma':    self.notmmm='mjjasondjf'
        if name == 'mam':   self.notmmm='jjasondjf'
        if name == 'mamj':  self.notmmm='jasondjf'
        if name == 'am':    self.notmmm='jjasondjfm'
        if name == 'amj':   self.notmmm='jasondjfm'
        if name == 'amjj':  self.notmmm='asondjfm'
        if name == 'mjjas': self.notmmm='ondjfma'

        if name == 'jj':   self.notmmm='asondjfmam'
        if name == 'jja':  self.notmmm='sondjfmam'
        if name == 'jjas': self.notmmm='ondjfmam'
        if name == 'ja':   self.notmmm='sondjfmamj'
        if name == 'jas':  self.notmmm='ondjfmamj'
        if name == 'jaso': self.notmmm='ndjfmamj'
        

        if name == 'as':     self.notmmm='ondjfmamjj'
        if name == 'aso':    self.notmmm='ndjfmamjj'
        if name == 'ason':   self.notmmm='djfmamjj'
        if name == 'asond':  self.notmmm='jfmamjj'
        if name == 'asondj': self.notmmm='fmamjj'
        
        if name == 'so':     self.notmmm='ndjfmamjja'
        if name == 'son':    self.notmmm='djfmamjja'
        if name == 'sond':   self.notmmm='jfmamjja'
        if name == 'on':     self.notmmm='djfmamjjas'
        if name == 'ond':    self.notmmm='jfmamjjas'
        if name == 'ondj':   self.notmmm='fmamjjas'
        if name == 'ondjfm': self.notmmm='amjjas'
        if name == 'ndjfm':  self.notmmm='amjjaso'

