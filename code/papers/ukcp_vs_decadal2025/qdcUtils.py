import os
import numpy
import scipy
import scipy.stats.mstats
import math

import matplotlib
import matplotlib.pyplot as plt

import iris
import iris.plot as iplt

import cf_units


############################################################  
def setcolours(choice=1):
    coldic={'BCC':'saddlebrown', 'BSC':'olive',             'CAFE':'purple',       'CMCC':'magenta',
            'CanESM5':'gold',    'Depresys4_gc3.1':'blue',  'IPSL':'darksalmon',   'MIROC6':'darkorange',
            'MPI':'red',         'NCAR40':'cyan',           'NorCPM':'darkgreen',  'NorCPMi2':'limegreen'}
    return coldic


############################################################  
def setmodelnames(choice=1):
    mdic={'BCC':'BCC',       'BSC':'BSC',                   'CAFE':'CAFE',     'CMCC':'CMCC',
          'CanESM5':'CCCma', 'Depresys4_gc3.1':'DePreSys4', 'IPSL':'IPSL',     'MIROC6':'MIROC', 
          'MPI':'MiKlip',    'NCAR40':'NCAR',               'NorCPM':'NCC-i1', 'NorCPMi2':'NCC-i2'}
    return mdic


############################################################  
def timefromsy(data, ssn, nlump):
    '''    
    This function returns a numpy array of times (reals) corresponding to the input data.
     
    Note that due to the need to equalize different calendars in the MMDPE, the input 
    cubes frequently do not have a time coord with a calendar. They do always have a 
    "season_year" coord though. Here we use the season_year coord of the input cube, 
    the input string variable in ssn, and the input integer nlump to estimate the output 
    array timereal. 
 
    Input    
    data: either an Iris cube with a season_year coordinate, or 
          a numpy array of season_year values.
        
    ssn:  one of ['djf', 'jja', 'mam', 'son' 'djfmamjjason'].

    nlump: integer specifying what lumping has been used for the input data.    
           nlump=1 implies no multi-year meaning of the data in the input cube.
           nlump=8 implies 8-year meaning has been performed the input cube.

    Here we assume a 360-day calendar of twelve 30-day months, and output times for the
    mid-point of the specified seasonal means. For 3-month seasons and nlump odd, this 
    corresponds to the 15th day of the middle month, eg for jja the mid-pint is 15 Jul 
    (the day of year for this is 195 for the 360-day calendar). For annual mean data 
    and nlump odd, the mid-point is the 1 Jun (since all annual means here are from 
    1 Dec to 30 Nov).
    
    Note that if nlump is even, the mid-point will not be the mid-point of the season,
    but midway between them, eg for ssn='djf' and nlump even, the mid-point will be 
    centred on 15 Jul for that season_year. For ssn='jja' and nlump even, the mid-point
    will be on 15 Jan, but in the year succeeding the specified season_year.
    '''

    if type(data) is iris.cube.Cube:
        syr = data.coord('season_year').points
    elif type(data) is numpy.ndarray:
        syr = data
    else:
        raise AssertionError('In function qdcUtils.timefromsy, type of input data not recognised')

    # Set the day of the month dd
    if ssn in ['djf', 'jja', 'mam', 'son']:
        dd = 15
    else:  # for 'djfmamjjason' 
        dd = 0

    # For 'jja' and 'son', set delta for season_year dsy=1
    dsy = 0
    if nlump%2 == 0 and ssn in ['jja', 'son']:
        dsy = 1

    # Set up dictionary of different mid-point months for odd and even nlump
    if nlump%2 == 1:
        mmdic = {'djf':1, 'mam':4,  'jja':7, 'son':10, 'djfmamjjason':6} 
    else:
        mmdic = {'djf':7, 'mam':10, 'jja':1, 'son':4,  'djfmamjjason':12} 

    mm = mmdic[ssn]

    # Iterate over season_year calculating timereal for output                
    timereal = []
    for sy in syr:
        # For the 360-day calendar and the definitions given above,
        # the day of year equals (mm-1)*30+dd
        timereal.append( (sy+dsy) + ((mm-1)*30+dd)/360 )
    return numpy.array(timereal)       
    
############################################################  
def hasCoord(cube,incoord):
   ctest = incoord in set([c.name() for c in cube.coords()])
   return ctest

############################################################  
def fixTime(cube, rm_fc_coords=True):
    """ A function which adds month and year to coordinates, and potentially  strips forecast coordinates """       
    if 'month' not in set([coord.name() for coord in cube.coords()]):       
        iris.coord_categorisation.add_month_number(cube, 'time', name='month')
    if 'year' not in set([coord.name() for coord in cube.coords()]):                
        iris.coord_categorisation.add_year(cube, 'time', name='year')
    if rm_fc_coords:
        # Remove forecast coordinates (if not relevant)    
        if 'forecast_period' in set([coord.name() for coord in cube.coords()]): 
            cube.remove_coord('forecast_period')
        if 'forecast_reference_time' in set([coord.name() for coord in cube.coords()]):         
            cube.remove_coord('forecast_reference_time')            
    return cube

############################################################
def time2real(incube):    
    '''
    This returns a numpy array of real times corresponding to those in the time coordinate 
    of the input cube or cubelist.
    NOTE - at the moment this is exact for 360_day and 365_day calendars. 
    Otherwise it returns a very close approximate value (midpoint of month).
    '''
    import calendar as cal
    
    #If input in a CubeList and note a cube, use first cube in list
    if isinstance(incube, iris.cube.CubeList): 
        cube=incube[0]
    else:
        cube=incube
        
    u = cube.coord('time').units
    if u.is_time_reference():
        timeCube = cube.coord('time').points
        calendar = u.calendar   #get calendar
        u0       = cf_units.Unit('days since 00-01-01', calendar=calendar)        
        if calendar == '360_day': 
            timeReal = u.convert(timeCube, u0)/360.        
        elif calendar == '365_day': 
            timeReal = u.convert(timeCube, u0)/365.
        else:
            #raise AssertionError('time2real error: calendar is %s, not 360_day or 365_day' %  calendar)
            print('>>> time2real message: calendar is %s, not 360_day or 365_day' %  calendar)
            print('>>> Use simple approach and set time to midpoint of each month')            
            cube = fixTime(cube) 
            year  = numpy.array([ int(y) for y in cube.coord('year').points ]) 
            if cube.coord('month').units == '1':
                month = cube.coord('month').points
            else:
                if not ('month_number' in set([c.name() for c in cube.coords()])):
                    iris.coord_categorisation.add_month_number(cube, 'time', name='month_number')
                month = cube.coord('month_number').points

            # Set real value to mid point of month
            timeReal = year + (month-0.5)*30./360.
            '''
            timeReal=[] 
            for y1,m1 in zip(year,month):            
                if cal.isleap(y1):
                    nyeardays=366.
                    nfebdays=29.
                else:
                    nyeardays=365.
                    nfebdays=28.
                if m1 in [1,3,5,7,8,10,12]:
                    t1 = y1 + (m1-0.5)*31./nyeardays 
                elif m1 in [4,6,9,11]:       
                    t1 = y1 + (m1-0.5)*30./nyeardays
                else: 
                    t1 = y1 + (m1-0.5)*nfebdays/nyeardays
                timeReal.append(t1)
            timeReal=numpy.array(timeReal)            
            ''' 
    else:
        raise AssertionError('Time units of cube not of form <time-unit> since <time-origin>')
    return timeReal

############################################################
def extractIndex(data, maxdata=None, mindata=None):
    '''
    Return the indices for which data in in the range [mindata, maxdata]
    '''
    if maxdata is None: maxdata=data[-1]
    if mindata is None: mindata=data[0]    
    maxdata1 = max( [data[1], min([data[-1],maxdata]) ] )
    mindata1 = min( [data[-2], max([data[0],mindata]) ] )    
    jout = [j for j,val in enumerate(data) if val >= mindata1 and val <= maxdata1]
    return jout

############################################################
def lump(data, nlump=1):
    if nlump <= 1:
        return data
    else:
        ans = []
        nt=data.shape[0]  
        for i1 in range(nt):
            dneg= ((nlump+1)//2-1)
            dpos= nlump//2+1        
            if i1-dneg >= 0 and i1+dpos <= nt:
                slc = slice(i1-dneg, i1+dpos)  # works for even and odd
                ans.append( numpy.mean(data[slc]) )                         
        return numpy.array(ans)

##################################################
def ci_func(xx, prob):
    qq = scipy.stats.mstats.mquantiles(xx[:].flatten(), prob=prob)
    return qq[1]-qq[0]

##################################################
def get_spread(p1, p2, prob, quant):
    prob   = list(prob)
    spread = quant[prob.index(p2)] - quant[prob.index(p1)] 
    return spread 

##################################################
def calcspread(data, prob=[0.1,0.9], ab=0.4, method='pool', poolav='mean'):
    ntim=data.shape[0]
    nrlz=data.shape[1]
    if method == 'mean':    
        spr=numpy.ma.zeros(ntim)
        for iii in range(ntim):
            x=data[iii,:]
            qq=scipy.stats.mstats.mquantiles(x,prob=prob,alphap=ab,betap=ab)
            spr[iii] = qq[1]-qq[0]
            #print(iii,spr)
        spread=numpy.ma.mean(spr)        
    else:
       if poolav in ['None', 'none']:
           x = data
       else:
           if poolav == 'median':
               dm=numpy.ma.median(data,axis=1)
           else:
               dm=numpy.ma.mean(data,axis=1)                  
           x = data-numpy.expand_dims(dm,-1)
       x = numpy.reshape(x, ntim*nrlz)
       qq=scipy.stats.mstats.mquantiles(x,prob=prob,alphap=ab,betap=ab)
       spread = qq[1]-qq[0]
    return spread


########################################################################
def rangefreq(data, dobs, prob=[0.1,0.9], ab=0.4):
    '''
    Estimate the hindcast range as a function of time for given probability range
    using the scipy mquantiles function, and then call freqratio to calculate  the 
    frequency of obs that are inside this range.
    Input
    data: 2D numpy array of shape (ntime,nrealization). This corresponds to hindcast 
          realizations 
    dobs: 1D numpy array of shape (ntime). This corresponds to corresponding observations.
    prob: 2-element list, default = [0.1,0.9]. Desired range for the hindcast data.
    ab:   float, default=0.4. Determines the interpolation method used to estimate
          quantiles in the mquantiles function. The default of 0.4 is the same as the 
          scipy.stats.mstats.mquantiles default (Cunnane, approximately quantile unbiased).
    '''   
    qq   = scipy.stats.mstats.mquantiles(data, prob=prob, alphap=ab, betap=ab, axis=1)
    freq = freqratio(dobs, qq[:,0], qq[:,1])
    return freq

########################################################################
def freqratio(dobs, qlo, qhi):
    ''' 
    Return the frequency of obs that are inside the forecast range [qlo,qhi].
    Input
    dobs: input 1D numpy array with shape=(ntime,). Observational data.
    qlo:  input 1D numpy array with shape=(ntime,). The lower value of the forecast range.
    qhi:  input 1D numpy array with shape=(ntime,). The upper value of the forecast range.
    Output
    freq: float, frequency of obs that are inside the input forecast range.
    '''
    ilo = numpy.where(dobs < qlo)[0]
    ihi = numpy.where(dobs > qhi)[0]
    # number of obs less than forecast range is ilo.shape[0]
    # number of obs greater than forecast range is ihi.shape[0]    
    freq = 1.-(ihi.shape[0]+ilo.shape[0])/dobs.shape[0]
    return freq

########################################################################
def noutside(data, qlo, qhi):
    '''
    Return number of data points outside range [qlo,qhi], and the total number of times.
    Also see the freqratio comments above (although note that freqratio does the freq 
    inside not outside).
    '''
    ilo = numpy.where(data < qlo)[0]
    ihi = numpy.where(data > qhi)[0]
    nout= ihi.shape[0]+ilo.shape[0]
    ntot= data.shape[0]
    return nout, ntot

########################################################################
def subset_ukcp(ukcp, dps, ssn, nlump, detrend=True):
    '''
    Return UKCP P50 data (as numpy array) from input ukcp cube for times 
    that match times in the input dps cube. Detrend the data if detrend is True.
    '''
    ukcptime = timefromsy(ukcp, ssn, nlump)
    dpstime  = timefromsy(dps, ssn, nlump)
    ukcp50   = ukcp.extract(iris.Constraint(percentile=0.5))        
    ukcpd=[]
    for tt in dpstime:
       mse = (ukcptime-tt)**2
       iok = numpy.where(mse == mse.min())[0][0]
       ukcpd.append(ukcp50.data[iok])
    if detrend:               
        ukcpdata = scipy.signal.detrend(numpy.array(ukcpd))                 
    else:
        ukcpdata = numpy.array(ukcpd)             
    return ukcpdata



########################################################################
# Some utilities for Butterworth filtering
########################################################################
def smooth(x, cutoff=30, padlen=None):
    if padlen is None:
        padlen = min([cutoff, int(x.shape[0]/2)])   
    out = butterworth(x, cutoff, padlen=padlen)
    return out

########################################################################
def predictLinearFit(y, newX):
    if y.ndim != 1:
        raise AssertionError('Y is %s-d but should be a vector' % y.ndim)
    ny = y.shape[0]
    x = numpy.array([numpy.ones(ny), numpy.arange(ny, dtype=float)])
    betas = numpy.linalg.lstsq(x.T, y, rcond=None)[0]
    return betas[0] + betas[1] * newX

########################################################################
def useLinearTrendToExtendEnds1d(y, m):
    if y.ndim != 1:
        raise AssertionError('Array must be 1d but is shape' % y)
    if m is None or m == 0:
        return y
    else:
        if 2*m > y.shape[0]:
            raise AssertionError('m = %s but this cannot be greater than half length of y which is %s' % (m, y.shape[0]/2.))
        y1 = predictLinearFit(y[0:m], numpy.linspace(-m, -1, m))
        y2 = predictLinearFit(y[-m:], numpy.linspace(m, 2*m-1, m))
        return numpy.concatenate((y1, y, y2))

########################################################################
def useLinearTrendToExtendEnds(y, m, axis=0):
    if y.ndim < 2:
        return useLinearTrendToExtendEnds1d(y, m)
    yy = y.reshape((y.shape[0], -1))
    ans = numpy.apply_along_axis(useLinearTrendToExtendEnds1d, axis, yy, m)
    return ans

########################################################################
def butterworth(x, period, axis=-1, order=4, padlen=3*(4+1), high_pass=False):
    '''
    High values for order (eg 8), give more end effects, while a default value of 4 seems ok. 
    Default padlen=3*(4+1) here, since we follow scipy.signal.filtfilt, which also implements padding. 
    This assumes default length of 3*max(len(a),len(b)) where b,a = scipy.signal.butter(order, wn). 
    Here one finds len(a)=len(b)=order+1, hence default padlen=3*(4+1) above.
    '''
    if period <= 1:
        ans = x
    else:
        n0 = x.shape[axis]
        padlen=int(round(padlen))
        x = useLinearTrendToExtendEnds(x, padlen, axis=axis)    
        n = x.shape[axis]
        nyquist = 0.5*n
        cutoff  = n/float(period)    
        wn      = cutoff/nyquist
        if wn > 1.0:
            padlen=0
            y=x
        else:
            b, a = scipy.signal.butter(order, wn)
            # Native padding in filtfilt just duplicates data, which is not very good.
            # Better to use linear extension above, and then set no padding (padlen=0) in filtfilt.
            y = scipy.signal.filtfilt(b, a, x, axis=axis, padlen=0)
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(padlen, padlen+n0)
        if high_pass:
            ans = x[tuple(slicer)] - y[tuple(slicer)]           
        else:
            ans = y[tuple(slicer)]            
    return ans

########################################################################
# Some utilities for estimation of ACC and MSSS skill scores
########################################################################
def ACC(model, obs, sstype='uncentred'):    
    if model.shape[0] != obs.shape[0]:
        raise AssertionError('Model and obs have different shapes, stop for debugging...')    
    mbar = numpy.mean(model)
    obar = numpy.mean(obs)
    if sstype.lower() in ['centred', 'centered']:
        mod_data = model - mbar
        obs_data = obs - obar        
    else:   # sstype.lower() in ['uncentred', 'uncentered']
        mod_data = model
        obs_data = obs
    numer = numpy.sum(mod_data*obs_data)   
    sq1   = numpy.sum(mod_data*mod_data) 
    sq2   = numpy.sum(obs_data*obs_data) 
    denom = (sq1*sq2)**0.5
    if denom == 0.0:
        ans = 1.0
    else:
        ans = numer/denom
    return ans

#########################################
def MSSS(model, obs, sstype='uncentred'):    
    if model.shape[0] != obs.shape[0]:
        raise AssertionError('Model and obs have different shapes, stop for debugging...')    
    def MSD(mod, obs):
        msd = numpy.mean( (mod-obs)*(mod-obs) )
        return msd
    mbar=numpy.mean(model)
    obar=numpy.mean(obs)
    if sstype.lower() in ['wmo']:
        numer = MSD(model, obs)    
        denom = MSD(obs, obar)
    elif sstype.lower() in ['centred', 'centered']:  
        cube1 = model - mbar
        cube2 = obs - obar
        numer = MSD(model-mbar, obs-obar)
        denom = MSD(obs, obar)    
    else:   # sstype.lower() in ['uncentred', 'uncentered']
        numer= MSD(model, obs)
        obar0 = 0.0
        denom = MSD(obs, obar0)
    ans = 1.- numer/denom
    return ans

########################################################################
def ACC_MSSS(model, obs, score='acc', sstype='uncentred'):
    if score.lower() == 'acc':
        ans = ACC(model, obs, sstype=sstype)
    elif score.lower() == 'msss':
        ans = MSSS(model, obs, sstype=sstype)
    else:
        raise AssertionError('Input score='+score+' not defined,stop for debugging...')    
    return ans

########################################################################
# Some utilities for estimation of uncertainty in linear trends   
########################################################################
def block_resamp_res(time, data, nsamp=5000, blocksize=5, seed=101, verbose=False):
    ntime=time.shape[0]
    ans = scipy.stats.linregress(time, data)
    # Attributes for ans: 
    #    ans.slope, ans.intercept, ans.rvalue, ans.pvalue, ans.stderr, ans.intercept_stderr        
    b  = ans.slope
    a  = ans.intercept
    se = ans.stderr
    pred = a+ b*time
    res  = data - pred
    dataresamp = numpy.zeros( (ntime,nsamp) )   
    tslice = slice(0,ntime)
    numpy.random.seed(seed)
    
    # Stack together random blocks of size blocksize, each with a randomly sampled start year
    # somewhere within all avaialble years. Keep doing this until we have at least ntime values 
    # in the stack, and then truncate. Actually need nsamp*nblock starting values in i0, 
    # where nblock = (ntime-1)//blocksize + 1. For example: ntime=55 => nblock=11, ntime=58 => nblock=12.
    # Here however, we simply create a much larger sample: nsamp*ntime (and discard unused).
    
    i0 = numpy.trunc( ntime*numpy.random.uniform(size=nsamp*ntime) ).astype(int) 
    jb = numpy.full(nsamp*ntime, blocksize)
    kk=0
    for i in range(nsamp):
        try:
            del isamp
        except:
            pass
        finished = False
        while not finished:
            isub1 = (i0[kk]+list(range(jb[kk]))) % ntime
            #print('kk=',kk, isub1)
            kk=kk+1
            try:
                isamp=numpy.hstack([isamp,isub1])
            except:
                isamp=isub1
            if isamp.shape[0] >= ntime:
                finished = True
        # Construct the resampled data
        if verbose and i < 50:
            print('isamp=',isamp[tslice])
        dataresamp[:,i] = pred + res[isamp[tslice]]            
    return dataresamp  

########################################################################
def tinv(pval, df):
    tinv = abs(scipy.stats.t.ppf(pval/2, df))
    return tinv

########################################################################
def trend_uncert_brr(time, data, resamp=True, prob=0.025, nsamp=5000, blocksize=5, seed=101):
    ''' 
    This function estimates the linear trend usin linear regresssion, and also estimates an
    associated uncertainty range for the linear trend using block resampling of the regression residuals.
    Note: "brr" in the name refers to the method used:  block resampling of residuals.

    Functional dependencies: 
        local function: block_resamp_res.
        local function: tinv (which in turn calls scipy.stats.t.ppf 
        scipy.stats.linregress
        scipy.stats.mstats.mquantiles
        
    Inputs:
        time:   1D numpy array of the independent variable (usually time).
        data:   1D numpy array of the dependent variable.

    Keyword inputs:
        resamp:    boolean, default = True. Switch to turn on block resampling of residuals.
                   if False, use scipy.stats.t.ppf method (see scipy.stats.linregress example).
        prob:      float, default = 0.025. Probability (one-sided) for uncertainty range. For example, 
                   a value of 0.025 will return the 95% confidence interval (from 0.025 to 0.975).
        nsamp:     integer, default = 5000. Number of bootstrap resamples to use.
        blocksize: integer, default = 5. Size of resampled blocks.
        seed:      integer, default = 101. Seed passed to numpy.random.seed in block_resamp_res call.  

    Outputs: A four element list of floats. In order they are:
               slope:     slope (or trend) for the input data estimated by scipy.stats.linregress. 
               intercept: intercept (constant of regression) for the regression.
               qq[0]:     lower value of the uncertainty range for the trend estimate.                   
               qq[1]:     upper value of the uncertainty range for the trend estimate.      
    '''      
        
    # prob is the one-sided probability, ie input of 0.025 will give the 95% range, from 0.025 to 0.975.

    ans   = scipy.stats.linregress(time, data)
    intercept = ans.intercept
    slope     = ans.slope
    # Attributes: ans.slope, ans.intercept, ans.rvalue, ans.pvalue, ans.stderr, ans.intercept_stderr

    if not resamp:
        tperc = tinv(prob, len(time)-2)   #scipy.stats.linregress example has 0.05/2
        cilo  = slope - tperc*ans.stderr
        cihi  = slope + tperc*ans.stderr    
        out   = [slope, intercept, cilo, cihi]

    else:         
        dataresamp = block_resamp_res(time, data, nsamp=nsamp, blocksize=blocksize, seed=seed)
        bb=[]
        for ii in range(nsamp):
            ans = scipy.stats.linregress(time, dataresamp[:,ii])
            bb.append(ans.slope)
        bb  = numpy.array(bb)
        qq  = scipy.stats.mstats.mquantiles(bb, prob=[prob, 1-prob])
        out = [slope, intercept, qq[0], qq[1]]

    return out

########################################################################
# Some plotting utilities for Iris cubes   
########################################################################

def plotCube(cube, nlev=21, cmap='jet', orientation='vertical', fulldomain=True, alpha=1.0,
             latlim=None, lonlim=None, showbar=True,
             title=None, levels=None, ticklabels=None, shrink=0.6, ifig=10, xsize=18., ysize=12.):

    # Does a quick contour plot of a cube    

    figsize=(xsize/2.54, ysize/2.54)   #convert input size (cm) to inches
    matplotlib.rcParams.update({'figure.figsize': figsize})
    nfig=plt.figure(ifig)
    nfig.patch.set_facecolor('white')    
    ax1=plt.subplot(1,1,1)
    if title is None: title=cube.name()
    #if orientation != 'vertical': shrink=0.8
    if levels is None: 
        levels = niceLevels(cube, n=nlev)                        

    #DEPRECATED
    #cmap_use = plt.cm.get_cmap(cmap, lut=levels.size)
    #NEW
    cmap_use = matplotlib.colormaps.get_cmap(cmap).resampled(levels.size) 
    
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap_use.N, clip=False)

    plt1 = iplt.pcolormesh(cube, cmap=cmap_use, norm=norm, facecolor='white', edgecolors='None',
                           antialiased=False, zorder=0, vmin=None, vmax=None, alpha=alpha)  

    if  fulldomain and iris.__version__ >= '2.2.0' and numpy.ma.is_masked(cube.data):
        dlat=abs(cube.coord('latitude').points[1]-cube.coord('latitude').points[0])
        dlon=abs(cube.coord('longitude').points[1]-cube.coord('longitude').points[0])
        latlo= -90.0 + dlat/2.   #cube.coord('latitude').points.min()+dlat/2.
        lathi=  90.0 - dlat/2.   #cube.coord('latitude').points.max()+dlat/2.
        lonp=cube.coord('longitude').points
        if lonp.max() > 180.0:  lonp = lonp -180.
        lonlo = max([lonp.min()+dlon/2., -180.+dlon/2.])
        lonhi = min([lonp.max()+dlon/2.,  180.-dlon/2.])
        plt1.axes.set_xlim([lonlo, lonhi])
        plt1.axes.set_ylim([latlo, lathi])

    if not lonlim is None:   plt1.axes.set_xlim(lonlim)
    if not latlim is None:   plt1.axes.set_ylim(latlim)
        
    plt.gca().coastlines(resolution='50m', zorder=1)
    plt.gca().set_title(title, fontsize=12)
            
    if showbar:
        cbar = plt.colorbar(plt1, orientation=orientation, shrink=shrink)
        if not ticklabels is None:  
            cbar.set_ticks(levels) 
            cbar.set_ticklabels(ticklabels) 
            cbar.update_ticks()       
        cbar.ax.tick_params(length=0)
        cbar.ax.tick_params(axis='x', labelsize=9)
        cbar.ax.tick_params(axis='y', labelsize=9)
    plt.show()
    #return levels in case they are needed elsewhere
    return levels 

########################################################################
def plotCubeSubplot(cube, nlev=21, cmap='jet', orientation='vertical', fulldomain=True, alpha=1.0,
              latlim=None, lonlim=None, showbar=True, fstit=11.0, subplot=(1,1,1),
              title=None, levels=None, ticklabels=None, shrink=0.6, barlabel='',
              fraction=0.15, aspect=20, padfrac=1.0):
    # Does a quick contour subplot of a cube
    
    if orientation == 'vertical':   pad=0.05*padfrac     #padfrac=1.0 gives pyplot defaults
    if orientation == 'horizontal': pad=0.15*padfrac
          
    ax1=plt.subplot(subplot[0],subplot[1],subplot[2])
    if title is None: title=cube.name()
    if levels is None: 
        levels = utils.niceLevels(cube, n=nlev)                        
    cmap_use = plt.cm.get_cmap(cmap, lut=levels.size)
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=cmap_use.N, clip=False)
    plt1 = iplt.pcolormesh(cube, cmap=cmap_use, norm=norm, facecolor='white', edgecolors='None',
                           antialiased=False, zorder=0, vmin=None, vmax=None, alpha=alpha)  
    if  fulldomain and iris.__version__ >= '2.2.0' and numpy.ma.is_masked(cube.data):
        dlat=abs(cube.coord('latitude').points[1]-cube.coord('latitude').points[0])
        dlon=abs(cube.coord('longitude').points[1]-cube.coord('longitude').points[0])
        latlo= -90.0 + dlat/2.   #cube.coord('latitude').points.min()+dlat/2.
        lathi=  90.0 - dlat/2.   #cube.coord('latitude').points.max()+dlat/2.
        lonp=cube.coord('longitude').points
        if lonp.max() > 180.0:  lonp = lonp -180.
        lonlo = max([lonp.min()+dlon/2., -180.+dlon/2.])
        lonhi = min([lonp.max()+dlon/2.,  180.-dlon/2.])
        plt1.axes.set_xlim([lonlo, lonhi])
        plt1.axes.set_ylim([latlo, lathi])

    if not lonlim is None:   plt1.axes.set_xlim(lonlim)
    if not latlim is None:   plt1.axes.set_ylim(latlim)
        
    plt.gca().coastlines(resolution='50m', zorder=1)
    plt.gca().set_title(title, fontsize=fstit)
            
    if showbar:
        cbar = plt.colorbar(plt1, orientation=orientation, shrink=shrink, fraction=fraction, 
                            aspect=aspect, pad=pad, label=barlabel)
        if not ticklabels is None:  
            cbar.set_ticks(levels) 
            cbar.set_ticklabels(ticklabels) 
            cbar.update_ticks()       
        cbar.ax.tick_params(length=2)   #0
        cbar.ax.tick_params(axis='x', labelsize=9)
        cbar.ax.tick_params(axis='y', labelsize=9)
    #return levels in case they are needed elsewhere
    return levels 

########################################################################
def setLevels(data, nlev=21, centre=False, inflate=0.0, var='tas', step=10.):
    vlo=data.min()
    vhi=data.max()
    vrange=vhi-vlo
    vhi=vhi+vrange*inflate
    vlo=vlo-vrange*inflate
    if var in ['pr', 'clt']:
        blo = 10.*round(vlo/10.-0.5)
        bhi = 10.*round(vhi/10.+0.5)
        if centre:
            bmx=numpy.max([abs(blo),abs(bhi)])
            levels = numpy.arange(-bmx, bmx+step, step)
        else:
            levels = numpy.arange(blo, bhi+step, step)
    else:
        levels = niceLevels(data, n=nlev, centre=centre, inflate=inflate) 
    return levels

########################################################################
def niceLevels(*args, **kwargs):
    '''
    Calculates array of nice contour levels based on list of arguments.
    Keywords:
    n:  guideline for how many intervals to make. Not guaranteed.
    inflate: range will be inflated by (1.0+inflate)
    '''
    n = kwargs.get('n', 9)
    inflate = kwargs.get('inflate', 0.0)
    centre = kwargs.get('centre', False)    
    prob=kwargs.get('prob', 0.05)    
    lo, hi = niceBounds(*args, prob=prob)
    if lo == hi:
        return numpy.array([lo, hi])    
    #if centre and numpy.sign(lo) != numpy.sign(hi):
    if centre:  
        temp = numpy.array([lo, hi, -lo, -hi])
        lo, hi = temp.min(), temp.max()
    mean = (lo + hi) * 0.5
    factor = 1.0 + inflate
    diff = hi - lo
    lo = mean - diff * 0.5 * factor
    hi = mean + diff * 0.5 * factor
    #
    locator = matplotlib.ticker.MaxNLocator(n) 
    locator.create_dummy_axis()
    #locator.set_bounds(lo, hi)
    locator.axis.set_view_interval(lo, hi)
    # numpy array returned
    return locator()

########################################################################
def niceBounds(*args, **kwargs):
    '''
    Return in a 2-tuple the lowest and highest values in the list of arguments, each
    of which must be acceptable as numpy.ma.max(arg) or numpy.ma.max(arg.data)
    '''
    prob=kwargs.get('prob', 0.05) 
    mx = []
    mn = []
    for i in args:
        try:
            mx.append(scipy.stats.mstats.mquantiles(i.ravel(), prob=1.-prob))
        except:
            mx.append(scipy.stats.mstats.mquantiles(i.data.ravel(), prob=1.-prob))
        try:
            mn.append(scipy.stats.mstats.mquantiles(i.ravel(), prob=prob))
        except:
            mn.append(scipy.stats.mstats.mquantiles(i.data.ravel(), prob=prob))
    # NEW: subtract off the mean from the bounds before finding niceMultiple,
    # then add back a rounded value of the mean.
    # Examples of logic here:
    #   rng in [0.5,    5]  => fac=0.1
    #   rng in [5,     50]  => fac=1
    #   rng in [50,   500]  => fac=10   etc
    # Old version was just:    niceMultiple(mn0, down=True), niceMultiple(mx0)
    mn0 = numpy.ma.min(mn)
    mx0 = numpy.ma.max(mx)
    avg =(mx0+mn0)/2.
    rng = mx0-mn0
    v=math.log10(2.*rng)
    m=int(v)
    if v < 0.0:  m=m-1
    fac = 10.0**(m-1)
    ravg=fac*round(avg/fac)
    return ravg+niceMultiple(mn0-avg, down=True),  ravg+niceMultiple(mx0-avg)

########################################################################
def niceMultiple(x, down=False):
    '''
    Find a number similar to x which is a nice multiple of a power of 10
    Returns a nice number, one of [1,2,2.5,5]*10^something
    down:   force returned value to be less than x. Default is greater than x.
    '''
    MULT = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
    if hasattr(x, '__iter__'):
        return [niceMultiple(i) for i in x]
    else:
        neg = False
        if x == 0.0:
            return x
        elif x < 0.0:
            xx = -x
            neg = True
        else:
            xx = x
        logX = math.log10(xx)
        px = math.floor(logX)
        px = 10.0 ** px
        ratio = xx / px
        if (down and not neg) or (not down and neg):
            mult = list(reversed(MULT)) + [1.0]
            m = [mu for mu in mult if mu <= ratio]
            if m:
                m = m[0]
            else:
                m = 1.0
        else:
            mult = MULT + [10.0]
            m = [mu for mu in mult if mu >= ratio]
            if m:
                m = m[0]
            else:
                m = 10.0
        if neg:
            m = -m
        return px * m



########################################################################
# Various cube manipution utilities (masking, meaning, regridding etc)  
########################################################################

def addRealization(cube, name):
    """ A function which adds an "Realization" coordinate which comes from name. """
    if not cube.coords('realization'):  
        exp_coord = iris.coords.AuxCoord(name, long_name='Experiment', units='no_unit', standard_name='realization')
        cube.add_aux_coord(exp_coord)
    return cube

########################################################################
def boxmean(cube, box, areaweight=True, ignore_bounds=True):    
    ce1 = iris.coords.CoordExtent('longitude', box[0],box[2])
    if ce1.minimum > ce1.maximum:
         ce1 = iris.coords.CoordExtent('longitude', box[0],box[2]+360.0)
    ce2 = iris.coords.CoordExtent('latitude', box[1], box[3])
    keep_all_longitudes=False
    if box[0] == 0    and box[2]==0:    keep_all_longitudes=True
    if box[0] == 0    and box[2]==360:  keep_all_longitudes=True
    if box[0] == -180 and box[2]==180:  keep_all_longitudes=True
    if keep_all_longitudes:
        # Want to just extract over latitudes:
        cube = cube.extract(iris.Constraint(latitude=lambda cell: box[1] <= cell <= box[3]))
    else:
        # Not that if we dont ignore_bounds, we usually get a larger area than desired.
        cube = cube.intersection(ce1, ce2, ignore_bounds=ignore_bounds)

    if areaweight:
        areas = iris.analysis.cartography.area_weights(cube)
        cube  = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=areas)
    else:
        cube  = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)    
    return cube

########################################################################
def landfrac_to_mask(landfrac, threshold=0.5):
    ans=landfrac.copy()
    try:
        dok = ~ans.data.mask
    except:
        m = numpy.full(ans.data.shape, False)  #no masked data, all False            
        ans.data = numpy.ma.array(ans.data, mask=m, keep_mask=True, shrink=False)    
    # Normalize, since often in range 0 to 100%
    dok = ~ans.data.mask        
    ans.data[dok] = landfrac.data[dok]/numpy.max(landfrac.data)
    # Create array with same shape as landfrac, with ones for fractions above threshold,
    # and zeros for fractions below threshold
    nok = landfrac.data[dok].shape
    msk = ans.data.copy()    
    msk.data[dok] = numpy.where(ans.data[dok] >= threshold, numpy.ones(nok), numpy.zeros(nok) )
    # Make numpy masked array where land frac is zero
    ans.data = numpy.ma.masked_values(msk, 0.0, shrink=False)
    return ans

########################################################################
def invertMask(a):
    return numpy.ma.masked_array(a.data, ~a.mask)

########################################################################
def applyMask(cube, mask):
    try:
        mask = iris.load_cube(mask)
    except:
        pass
    newMask = numpy.ma.getmaskarray(cube.data) | numpy.ma.getmaskarray(mask.data)
    return cube.copy(data=numpy.ma.array(cube.data, mask=newMask))

########################################################################
def regrid_separately(cube, lndfrac, gridcube, threshold=0.5, name='unknown'):
    lndmask = landfrac_to_mask(lndfrac, threshold=threshold)
    if cube.coord(axis='x') and not cube.coord(axis='x').circular:
        cube.coord(axis='x').circular=True   # Is this needed?
    cube.coord(axis='y').coord_system = None 
    cube.coord(axis='x').coord_system = None
    print('>>> Regridding ',cube.name(),' for ', name)
    newcube = regrid_land_ocean_separately(cube, lndmask, gridcube)
    return newcube

########################################################################
def regrid_land_ocean_separately(cube, cube_land_mask, new_grid_land_mask, mdtol=0.99):
    cube_land_mask.coord(axis='y').coord_system = new_grid_land_mask.coord(axis='y').coord_system
    cube_land_mask.coord(axis='x').coord_system = new_grid_land_mask.coord(axis='x').coord_system
    cube.coord(axis='y').coord_system           = new_grid_land_mask.coord(axis='y').coord_system
    cube.coord(axis='x').coord_system           = new_grid_land_mask.coord(axis='x').coord_system
    scheme1  = iris.analysis.AreaWeighted(mdtol=mdtol)
    new_cube = regrid_separate_scheme(cube, cube_land_mask, new_grid_land_mask, scheme=scheme1, mdtol=mdtol)
    if numpy.ma.getmaskarray(new_cube.data).any():
        missmask = numpy.ma.where(numpy.ma.getmaskarray(new_cube.data))
        # Cannot use AreaWeighted, since it can give missing column at 0 deg long (eg GFDL-ESM2M) 
        scheme2  = iris.analysis.Linear(extrapolation_mode='linear')
        lin_cube = cube.regrid(new_grid_land_mask, scheme2)
        new_cube.data[missmask] = lin_cube.data[missmask]
    return new_cube

########################################################################
def regrid_separate_scheme(cube, cube_land_mask, new_grid_land_mask, scheme=None, mdtol=0.99):
    if scheme is None:
        scheme = iris.analysis.AreaWeighted(mdtol=mdtol)
    new_grid_ocean_mask = new_grid_land_mask.copy(data=invertMask(new_grid_land_mask.data))
    cube_ocean_mask = cube_land_mask.copy(data=invertMask(cube_land_mask.data))
    new_land_cube   = applyMask(cube, cube_land_mask).regrid(new_grid_land_mask, scheme)
    new_land_cube   = applyMask(new_land_cube, new_grid_land_mask)
    new_ocean_cube  = applyMask(cube, cube_ocean_mask).regrid(new_grid_ocean_mask, scheme)
    new_ocean_cube  = applyMask(new_ocean_cube, new_grid_ocean_mask)
    new_cube        = new_land_cube.copy()
    ocnmask         = numpy.ma.where(~numpy.ma.getmaskarray(new_ocean_cube.data))
    new_cube.data[ocnmask] = new_ocean_cube.data[ocnmask]
    return new_cube

########################################################################
def tidy_cubes(cubes, guess_bounds=False, remove_coords=None, add_month_year=False, 
               set_var_name_None=None, match_coord_system=None):    
    '''
    Takes a list of cubes or a single cube and goes through each one either guessing 
    the bounds or removing coordinates.
    guess_bounds:   Default False for no guessing of bounds but if True, guesses bounds of axes 
                    that exist out of ['x', 'y', 'z', 't']. Can also be a list of a mixture of 
                    special axes ids ['x', 'y', 'z', 't'] or coordinate names.
    remove_coords:  A list of coordinate names to remove
    '''        
    # From qumpy.irislib.tidy_cubes.    
    SPECIAL_AXIS_IDS = ['x', 'y', 'z', 't']
    COORD_SHAPE_IF_LENGTH_IS_1 = (1,)
    if isinstance(cubes, iris.cube.Cube):
        cubes = [cubes]
    if guess_bounds is True:
        guess_bounds = SPECIAL_AXIS_IDS
    for c in cubes:
        print("Processing %s" % c.name())
        if remove_coords is not None:
            for rc in remove_coords:
                try:
                    c.coord(rc)
                except:
                    print("Does not have coord %s" % rc)
                else:
                    c.remove_coord(rc)
                    print("%s removed" % rc)
        if set_var_name_None is not None:
            if isinstance(set_var_name_None, bool):
                set_var_name_None = SPECIAL_AXIS_IDS
            for vn in set_var_name_None:
                if vn in SPECIAL_AXIS_IDS:
                    try:
                        coord = c.coord(axis=vn)
                    except:
                        print("Does not have coordinate for %s" % vn)
                        coord = False
                else:
                    try:
                        coord = c.coord(vn)
                    except:
                        print("Does not have coordinate for %s" % vn)
                        coord = False
                if coord:
                    coord.var_name = None
                    print("%s.var_name set to None" % vn)
        if guess_bounds:
            for gb in guess_bounds:
                if gb in SPECIAL_AXIS_IDS:
                    try:
                        coord = c.coord(axis=gb)
                    except:
                       print("Does not have coordinate for %s" % gb)
                       coord = False
                else:
                    try:
                        coord = c.coord(gb)
                    except:
                        print("Does not have coordinate for %s" % gb)
                        coord = False
                if coord:
                    if coord.has_bounds():
                        pass
                        #print("No need to guess bounds for %s" % gb)
                    elif coord.shape == COORD_SHAPE_IF_LENGTH_IS_1:
                        print("Cannot guess bounds for %s as length 1. Skipping" % gb)
                    else:
                        coord.guess_bounds()
                        print("Bounds guessed for %s" % gb)
        if add_month_year:
            try:
                c.coord('time')
            except:
                print("Does not have coordinate time so cannot make month, year, season_year")
            categories = [icat.add_month, icat.add_year, icat.add_season_year]
            for add_category in categories:
                try:
                    if add_category is icat.add_season_year:
                        add_category(c, 'time', name='season_year')
                    else:
                        add_category(c, 'time')
                except:
                    print("Failed to add category %s" % add_category)
                else:
                    print("Coordinate added using %s" % add_category)
        #print()
        if match_coord_system is not None:
            if isinstance(match_coord_system, iris.cube.Cube):
                for axis in ['x', 'y']:
                    coord = c.coords(axis=axis)
                    refcoord = match_coord_system.coords(axis=axis)
                    if coord and refcoord:
                        if coord[0].coord_system:
                            if coord[0].coord_system != refcoord[0].coord_system:
                                raise Exception('coord_systems %s and %s do not match' % (coord[0].coord_system, refcoord[0].coord_system))
                        else:
                            new_coord_system = match_coord_system.coord(axis=axis).coord_system
                            print("Setting coord_system to %s" % new_coord_system)
                            c.coord(axis=axis).coord_system = new_coord_system

