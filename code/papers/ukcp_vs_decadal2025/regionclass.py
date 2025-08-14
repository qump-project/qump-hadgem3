class InitRegion:
    def __init__(self, name):
        '''
        Function to set up regionclass. A class which holds a range of useful information about an individual region.
        Set it by: region=regionclass.InitRegion(individualseason). An example is:
            region = regionclass.InitRegion('regionfornao')
        
        The resulting class has attributes:
        region.coords: lat longitude coordinates of region in degrees form bottom left to top right.
        region.standard_name: Region name, eg 'North America'
        region.short_name: Shortened name of region, eg 'N.Amer' for North America (str)
        region.nameforfilenames: lowercase all spaces removed form of standard_name, eg 'northamerica' (str)
        region.nameincoords: String of coordinates with east and west added. eg '10.0E, 40.6N, 7E, 90N'

        REGION COORDINATES FROM regionclass.py WILL ALWAYS BE FROM -180 to +180
        '''

        degress_symbol = '$^\circ$'

        if name== 'None':
            self.coords='None '
            self.standard_name='None'
            self.short_name='None'
            self.nameincoords='None'
            self.nameforfilenames='globalukcp09'            
            return

        if name == 'None':      self.coords=[-180,-90,180,90]
        if name == 'Europe':    self.coords=[-30.0,30.0,40.0,75.0]
        if name == 'Nino3.4':   self.coords=[-170.0,-5.0,-120.0,5.0] # Nino 3.4 (5N-5S, 170W-120W)
        if name == 'NW Europe': self.coords=[-15.0,35.0,25.0,63.0]

        # then extended so JM can have all of Norway.    
        if name == 'NW Europe':   self.coords=[-10.0, 35.0, 40.0, 75.0]

        # region was then extended on April 7th so JM could include Iceland.
        if name == 'NW Europe':   self.coords=[-20.0, 35.0, 40.0, 75.0]

        # region was then extended on April 14th so JM could include a bit of Greenland.
        if name == 'NW Europe':   self.coords=[-25.0, 35.0, 40.0, 75.0]

        # region was then extended on April 14th so JM could include a bit of Greenland.
        # Extended more on May 4th for even more Greenland:
        if name == 'NW Europe':   self.coords=[-30.0, 35.0, 40.0, 75.0]
        if name == 'NW Europe':   self.coords=[-40.0, 35.0, 40.0, 75.0]


        if name == 'Extremely Wide Europe':   self.coords=[-100.0, -10.0, 100.0, 85.0]
        if name == 'Very Wide Europe':        self.coords=[-80.0, 0.0, 80.0, 85.0]
        if name == 'Wide Europe':             self.coords=[-50.0, 20.0, 60.0, 85.0]
        if name == 'North America':           self.coords=[-120.0, 30.0, -80.0, 80.0]
        if name == 'UK':                      self.coords=[-10.0, 50.0, 3.0, 60.0]

        if name == 'UKtwiceasbig':           self.coords=[-16.5,45.0,9.5,65.0]
        if name == 'UKshiftedeastbyhalf':    self.coords=[-3.5,50.0,9.5,60.0]
        if name == 'UKshiftedwestbyhalf':    self.coords=[-16.5,50.0,-3.0,60.0]
        if name == 'UKshiftednorthbyhalf':   self.coords=[-10.0,55.0,3.0,65.0]
        if name == 'UKshiftedsouthbyhalf':   self.coords=[-10.0,45.0,3.0,55.0]


        if name == 'England and Wales':             self.coords=[-10.0,50.0,3.0,55.0]
        if name == 'France':                        self.coords=[-5.0,43.0,7.0,50.0]
        if name == 'Southern UK Northern France':   self.coords=[-5.0,47.0,3.0,52.0]
        if name == 'UK and France':                 self.coords=[-10.0,43.0,7.0,60.0]
        if name == 'UK, France and Spain':          self.coords=[-10.0,37.0,5.0,60.0]

        if name == 'Eurasia':                              self.coords=[-30.0,0.0,150.0,70.0]
        if name == 'China':                                self.coords=[80.0,18.0,130.0,50.0]
        if name == 'China hydro region':                   self.coords=[90.0,18.0,130.0,45.0]
        if name == 'Wide China':                           self.coords=[65.0,5.0,130.0,50.0]
        if name == 'Wide China and into West Pacific':     self.coords=[65.0,5.0,145.0,50.0]
        if name == 'China and IndoChina':                  self.coords=[90.0,12.0,125.0,43.0]
        if name == 'Global':                               self.coords=[-170.0,-70.0,170.0,80.0]
        if name == 'Global':                               self.coords=[-170.0,-70.0,170.0,80.0]
        if name == 'Globalukcp09':                         self.coords=[-170.0,-70.0,170.0,80.0]
        if name == 'Globalukcp09':                         self.coords=[-180.0,-90.0,180.0,90.0]
        if name == 'Non-Polar Global':                     self.coords=[-175.0,-70.0,175.0,80.0]
        if name == 'Tropics':                              self.coords=[-180.0,-50.0,180.0,50.0]
        if name == 'Tropics':                              self.coords=[0.0,-50.0,359.0,50.0]
        if name == 'Tropical Pacific and Atlantic':        self.coords=[-220.0,-50.0,60.0,50.0]
        if name == 'Tropical Pacific and Atlantic':        self.coords=[90.0,-50.0,410.0,50.0]
        if name == 'Alaska to India':                      self.coords=[-140.0,-50.0,80.0,70.0]
        if name == 'Not so Tropical Pacific and Atlantic': self.coords=[100.0,-60.0,390.0,60.0]
        if name == 'Smallgridboxtestregion':               self.coords=[-10.0,50.0,-6.0,54.0]
        if name == 'Northern hemisphere':                  self.coords=[-180.0,0.0,179.0,80.0]
        if name == 'Southern hemisphere':                  self.coords=[-180.0,-80.0,179.0,0.0]

        if name == 'Tropical Atlantic and Africa':           self.coords=[-50.0,-10.0,10.0,40.0]
        if name == 'Wide Tropical Atlantic and Africa':      self.coords=[-80.0,-40.0,30.0,50.0]
        if name == 'Very Wide Tropical Atlantic and Africa': self.coords=[-120.0,-40.0,80.0,60.0]

        if name == 'North Atlantic sub-polar gyre region':     self.coords=[-60.0,50.0,-10.0,66.0]

        # 6 Chinese regions of same size:
        if name == 'NW China':          self.coords=[90.0,30.0,100.0,40.0]
        if name == 'Central N China':   self.coords=[100.0,30.0,110.0,40.0]
        if name == 'NE China':          self.coords=[110.0,30.0,120.0,40.0]
        if name == 'SW China':          self.coords=[90.0,20.0,100.0,30.0]
        if name == 'Central S China':   self.coords=[100.0,20.0,110.0,30.0]
        if name == 'SE China':          self.coords=[110.0,20.0,120.0,30.0]

        # Giorgi regions:
        if name == 'Antarctica':              self.coords=[-180, -90,  180,  -60 ]
        if name == 'North Australia':         self.coords=[110, -28,  155,  -11 ]
        if name == 'South Australia':         self.coords=[110, -48,  180,  -28 ]
        if name == 'Amazon Basin' :           self.coords=[ -82, -20,  -34,   12 ]
        if name == 'Southern South America':  self.coords=[-76, -56,  -40,  -20 ]
        if name == 'Central America':         self.coords=[-116,  10,  -83,   30 ]
        if name == 'Western North America':   self.coords=[-130,  30, -103,   60 ]
        if name == 'Central North America':   self.coords=[-103,  30,  -85,   50 ]
        if name == 'Eastern North America':   self.coords=[ -85,  25,  -60,   50 ]
        if name == 'Alaska':                  self.coords=[-170,  60, -103,   72 ]
        if name == 'Greenland':               self.coords=[-103,  50,  -10,   85 ]
        if name == 'Mediterranean Basin':     self.coords=[ -10,  30,   40,   48 ]
        if name == 'Northern Europe':         self.coords=[ -10,  48,   40,   75 ]
        if name == 'Western Africa':          self.coords=[-20, -12,   22,   18 ]
        if name == 'Eastern Africa':          self.coords=[ 22, -12,   52,   18 ]
        if name == 'Southern Africa':         self.coords=[-10, -35,   52,  -12 ]
        if name == 'Sahara':                  self.coords=[-20,  18,   65,   30 ]
        if name == 'Southeast Asia':          self.coords=[ 95, -11,   155,  20 ]
        if name == 'East Asia':               self.coords=[100,  20,   145,  50 ]
        if name == 'South Asia':              self.coords=[ 65,   5,   100,  30 ]
        if name == 'Central Asia':            self.coords=[40,  30,   75,   50 ]
        if name == 'Tibet':                   self.coords=[ 75,  30,   100,  50 ]
        if name == 'North Asia':              self.coords=[ 40,  50,   180,  70 ]
        # end of Giorgi regions

        if name == 'Australasia':   self.coords=[80, -50,  170,  10 ]

        if name == 'Wide East Asia':self.coords=[70,  10,   145,  60 ]

        # Ruth's regions for Caths hadgem3gc2 paper';
        if name == 'North Atlantic':   self.coords=[280,30,350,70]
        if name == 'Western Europe':   self.coords=[-30.0,35,30,80]
        if name == 'UK':               self.coords=[-10,48,2,62]
        if name == 'Mediterranean':    self.coords=[-12,25,42,47]
        
        if name == 'Africa':           self.coords=[-20, -35,   52,   35 ]


        # Other regions:
        if name == 'Pacific':  self.coords=[90.0, -50.0,  280.0,  60.0 ]
        #if name == 'Pacific': self.coords=[90.0, -50.0,  180.0,  40.0 ]
        #if name == 'Pacific': self.coords=[90.0, -50.0,  200.0,  40.0 ]
        #if name == 'Pacific': self.coords=[180.0, -50.0,  230.0,  40.0 ]

        # Chinese provinces:
        if name == 'Yunnan':   self.coords=[97.0,22.0,105.0,27.0]
        if name == 'Shandong': self.coords=[115.0,34.5,121.0,38.0]

        if name == 'Three-gorge-dam catchment':self.coords=[97.0,25.0,112,35.0]
        if name == 'SW China dam cluster':     self.coords=[97.0,24.0,108.0,33.0]
        if name == 'E China dam cluster':      self.coords=[115.0,28.0,121.0,33.0]
        if name == 'SE China dam cluster':     self.coords=[110.0,22.5,117.0,26.0]

        # New after map of rivers done:
        if name == 'Tibetan plateau': self.coords=[97.0,26.5,105.0,33.0]
        if name == 'Pearl river':     self.coords=[102.0,22.0,115.0,25.5]
        if name == 'Yellow river':    self.coords=[102.0,34.0,115.0,40.0]
        if name == 'Lower Yangtze':   self.coords=[106.0,26.5,120.0,33.0]
        
        # Used by nao:
        #box_Azores =  [ [-28.0, -20.0, 36.0, 40.0] ]  
        #box_Azores =  [ [332.0, 340.0, 36.0, 40.0] ]  
        #box_Iceland = [ [-25.0, -16.0, 63.0, 70.0] ]  
        #box_Iceland = [ [335.0, 344.0, 63.0, 70.0] ]  

        if name == 'Azores':    self.coords=[-28,36,-20,40]
        if name == 'Reykjavik': self.coords=[-25,63,-16,70]

        if name == '60degNto70degN': self.coords=[-180.0, 70.0,  180.0,  80.0 ]


        #if name == 'regionfornao':           self.coords=[-37.0,30.0,0.0,70.0]
        if name == 'regionforsam':            self.coords=[0.0,-74.0,360.0,-36.0]
        if name == 'regionfornpi':            self.coords=[-200.0,30.0,-140.0,65.0]
        if name == 'regionforsoi':            self.coords=[125.0,-18.0,212.0,-10.0]
        if name == 'regionforsoi':            self.coords=[120.0,-20.0,220.0,-8.0]
        if name == 'regionforamo':            self.coords=[-80.0,0.0,0.0,60.0]
        if name == 'subtractionregionforamo': self.coords=[-180.0,-60.0,180.0,60.0]
        if name == 'regionforksi':            self.coords=[-75.0,0.0,-7.5,60.0]
        if name == 'regionforqbo':            self.coords=[70.0,-1.0,105.0,1.0]

        if name == 'New Zealand': self.coords=[160.0,-50.0,170.0,-40.0]

        # Gill Martin's China region coordinates:
        if name == 'South China':       self.coords=[108,18,122,26]
        if name == 'Southwest China':   self.coords=[95,21,108,30]
        if name == 'North China LZ':    self.coords=[104,34,120,45]
        if name == 'North China all':   self.coords=[104,34,135,55]
        if name == 'Yangtze River':     self.coords=[108,26,122,34]
        if name == 'Northeast China':   self.coords=[120,43,135,55]
        if name == 'Northwest China':   self.coords=[76,37,104,55]
        if name == 'Korea':             self.coords=[120,34,135,43]
        if name == 'WNPSH':             self.coords=[115,15,150,25]
        if name == 'NPSH':              self.coords=[180,27.5,212,37.5]
        if name == 'Okhotsk':           self.coords=[120,50,150,60]
        if name == 'Okhotsk2':          self.coords=[140,50,160,60]
        if name == 'Siberia':           self.coords=[80,40,120,65]

        # Special types of region name:

        # Names created by function make_regular_regions (imported from make_regular_regions.py)
        if name[0:11]=='Grid region':
            print('Region name (',name,') has lat longs in it. Will try to extract and use these')
            self.coords=[]
            self.coords.append(float(name[11:17]))
            self.coords.append(float(name[22:28]))
            self.coords.append(float(name[35:41]))
            self.coords.append(float(name[46:52]))
            
        # At this point, correct entries [0] and [2] of self.coords so they 
        # are never outside of -180 to +180 to prevent later problems when plotting boxes on plots:
        if self.coords[0] > 180:   self.coords[0] = self.coords[0]-360.0
        if self.coords[2] > 180:   self.coords[2] = self.coords[2]-360.0

        self.short_name=name
        if name == 'Europe':        self.short_name='Europe'
        if name == 'Wide Europe':   self.short_name='Big Europe'
        if name == 'North America': self.short_name='N.Amer'
        if name == 'UK':            self.short_name='UK'
        if name == 'China':         self.short_name='China'
        if name == 'Global':        self.short_name='Global'
        
        if name == 'Smallgridboxtestregion':   self.short_name='Small reg.'

        # Gioigi regions:
        if name == 'Antarctica':               self.short_name='Antarctic'
        if name == 'North Australia':          self.short_name='N. Austr.'
        if name == 'South Australia':          self.short_name='S. Austr.'
        if name == 'Amazon Basin' :            self.short_name='Amazon'
        if name == 'Southern South America':   self.short_name='South\nAmerica'
        if name == 'Central America':          self.short_name='Central\nAmerica'
        if name == 'Western North America':    self.short_name='W.North\nAmerica'
        if name == 'Central North America':    self.short_name='C.North\nAmerica'
        if name == 'Eastern North America':    self.short_name='E.North\nAmerica'
        if name == 'Alaska':                   self.short_name='Alaska'
        if name == 'Greenland':                self.short_name='Green-\nland'
        if name == 'Mediterranean Basin':      self.short_name='Mediter\nranean'
        if name == 'Northern Europe':          self.short_name='Northern\nEurope'
        if name == 'Western Africa':           self.short_name='West\nAfrica'
        if name == 'Eastern Africa':           self.short_name='East\nAfrica'
        if name == 'Southern Africa':          self.short_name='South\nAfrica'
        if name == 'Sahara':                   self.short_name='Sahara'
        if name == 'Southeast Asia':           self.short_name='SE Asia'
        if name == 'East Asia':                self.short_name='East\nAsia'
        if name == 'South Asia':               self.short_name='South Asia'
        if name == 'Central Asia':             self.short_name='Central Asia'
        if name == 'Tibet':                    self.short_name='Tibet'
        if name == 'North Asia':               self.short_name='Northern Asia'
        
        if name == 'North Atlantic sub-polar gyre region':   self.short_name='Sub-polar gyre'

        if name == 'mlna':   self.short_name='North Atlantic'
        if name == 'mlweur': self.short_name='Western Europe'
        if name == 'ruthuk': self.short_name='UK'
        if name == 'mlmed':  self.short_name='Mediterranean'



        # Special types of region name:

        # Names created by function make_regular_regions (imported from make_regular_regions.py)
        if name[0:11] == 'Grid region':
            # Shorten by stripping out spaces, turning deg into degrees symbol and integerising:
            self.short_name = name[11:-1].strip().replace(' ','').replace('deg',degress_symbol).replace('.0','')
        
        
        self.standard_name = name
        # set self.nameforfilenames
        def getnameforfilenames(name):
            nameforfilenames = name.strip().lower().replace(' ','')
            return nameforfilenames
        self.nameforfilenames = getnameforfilenames(name)

        # string of coords:
        
        self.nameincoords=str(abs(self.coords[0]))
        if self.coords[0] >= 0:
            self.nameincoords = self.nameincoords+degress_symbol+'E'
        if self.coords[0] < 0:
            self.nameincoords = self.nameincoords+degress_symbol+'W'

        self.nameincoords = self.nameincoords+', '
            
        self.nameincoords = self.nameincoords+str(abs(self.coords[1]))
        if self.coords[1] >= 0:
            self.nameincoords = self.nameincoords+degress_symbol+'N'
        if self.coords[1] < 0:
            self.nameincoords = self.nameincoords+degress_symbol+'S'

        self.nameincoords = self.nameincoords+' to '
        
        self.nameincoords = self.nameincoords+str(abs(self.coords[2]))
        if self.coords[2] >= 0:
            self.nameincoords = self.nameincoords+degress_symbol+'E'
        if self.coords[2] < 0:
            self.nameincoords = self.nameincoords+degress_symbol+'W'

        self.nameincoords = self.nameincoords+', '

        self.nameincoords = self.nameincoords+str(abs(self.coords[3]))
        if self.coords[3] >= 0:
            self.nameincoords = self.nameincoords+degress_symbol+'N'
        if self.coords[3] < 0:
            self.nameincoords = self.nameincoords+degress_symbol+'S'

