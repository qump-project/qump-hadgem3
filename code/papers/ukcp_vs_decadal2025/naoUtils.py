import os
import numpy
import iris

##############################################################
def rebase_obscube(northcube, southcube, base1, base2):
    ssnyear   = northcube.coord('season_year').points
    ibase1    = numpy.where(ssnyear == base1)[0][0]
    ibase2    = numpy.where(ssnyear == base2)[0][0]
    north_rebase = northcube - numpy.mean( northcube.data[ibase1:ibase2+1] )
    south_rebase = southcube - numpy.mean( southcube.data[ibase1:ibase2+1] )
    return north_rebase, south_rebase


##############################################################
def calc_anom(northarr, southarr, base1, base2, tinit0=1960, timeandmem=True, iyear=None, verbose=False):
    nyr = northarr.shape[0]
    nrlz= northarr.shape[1]
    nfp = northarr.shape[2]
    fcperiod = numpy.array(list(range(1,nfp+1)))    
    anomnorth=numpy.ma.zeros((nyr,nrlz,nfp))
    anomsouth=numpy.ma.zeros((nyr,nrlz,nfp))
    ibase = numpy.array(list(range(base1,base2+1)))    
    for ifp,fp in enumerate(fcperiod):
        idx = ibase-tinit0-fp
        if verbose and not iyear is None:
            print(ifp,fp,iyear[idx])
        if timeandmem:
            # Mean over baseperiod and realizations (default)
            basenorth = numpy.mean(northarr[idx,:,ifp])
            basesouth = numpy.mean(southarr[idx,:,ifp])
        else:
            # Only take mean over first time axis, ie not over realizations (used in testing).
            basenorth = numpy.mean(northarr[idx,:,ifp],axis=0)
            basesouth = numpy.mean(southarr[idx,:,ifp],axis=0)
        anomnorth[:,:,ifp] = northarr[:,:,ifp]-basenorth 
        anomsouth[:,:,ifp] = southarr[:,:,ifp]-basesouth          
    return anomnorth, anomsouth


#########################################
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


###################################################
def index_common(time1, time2, asint=True):
    if asint:
        time1 =  time1.astype('int')
        time2 =  time2.astype('int')        

    tlo = numpy.max([ time1.min(), time2.min()] )
    thi = numpy.min([ time1.max(), time2.max()] )

    i0for1 = numpy.where(time1 >= tlo)[0][0]
    i1for1 = numpy.where(time1 <= thi)[0][-1]
    ifor1  = numpy.arange(i0for1, i1for1+1)

    i0for2 = numpy.where(time2 >= tlo)[0][0]
    i1for2 = numpy.where(time2 <= thi)[0][-1]
    ifor2  = numpy.arange(i0for2, i1for2+1)
    #print('ifor1.shape=',ifor1.shape[0],' ifor2.shape=',ifor2.shape[0])
    return ifor1, ifor2


###################################################
def runningmean(cube, nlump, ssn='djf', verbose=False):
    if ssn == 'djf':
        delta = 15./360
    elif ssn == 'djfm':
        delta = 30./360
    ndata=cube.shape[0]
    i0=int(round(nlump/2 + 0.5))-1
    if nlump%2 == 1:
        imx=ndata-i0
    else:
        imx=ndata-i0-1
    dataarr=[]
    timearr=[]
    for ii in range(i0,imx):
        i1 = ii-i0
        i2 = i1+nlump
        sy = cube.coord('season_year').points[i1:i2]
        midyr = numpy.mean(sy) + delta
        dataarr.append(numpy.mean(cube.data[i1:i2]) )
        if verbose:
            print(ii,i1,i2,sy,midyr, numpy.mean(cube.data[i1:i2]))        
        timearr.append(midyr) 
    return numpy.array(timearr), numpy.array(dataarr) 


#############################################################
def loadNAO(indir, fp='2to9', region='Stephenson', ssn='djf', base1=1971, base2=2000, 
            tinit0=1960, central='mean', renorm=False, nsub=0, 
            ensemble=['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3.1','IPSL','MIROC6','MPI','NCAR','NorCPM','NorCPMi2']):

    if type(ensemble) == str:
        ensemble = [ensemble] 

    mdic={'BSC':             [10, '1960_2018'],
          'CAFE':            [10, '1960_2019'],
          'CanESM5':         [10, '1960_2016'],
          'CMCC':            [20, '1960_2019'],
          'Depresys4_gc3.1': [10, '1960_2022'],
          'IPSL':            [10, '1960_2016'],
          'MIROC6':          [10, '1960_2018'],
          'MPI':             [10, '1960_2017'],
          'NCAR':            [40, '1954_2017'],
          'NorCPM':          [10, '1960_2018'],
          'NorCPMi2':        [10, '1960_2018'],
          'CMCC-10':         [10, '1960_2019'],          
          'NCAR-10':         [10, '1954_2017'] }
            
    nlump=1
    if type(fp) == str:
        fpsplit= fp.split('to')
        t1    = int(fpsplit[0])
        t2    = int(fpsplit[1])
        nlump  = t2-t1+1
    else:
        print('Still need to code up for other fp=',fp)
        raise AssertionError('Stop for debugging...')
    islice = slice(t1-1, t2) 
        
    dlis=[]
    tlis=[]
    minlena=100000
    for kmod,model in enumerate(ensemble):
        if '-10' in model: 
            modelname = model.split('-')[0]
            ctlab= '_ct='+central
        else:
            modelname = model
            ctlab=''
        if nsub == 0:
           nrlz = mdic[model][0]
        else:
           nrlz = nsub
        name='NAO_'+modelname+'_nrlz='+str(nrlz)+'_region='+region+'_ssn='+ssn+'_aw=T_ib=T_north_south_'+mdic[model][1]+ctlab+'.npz'

        #file=os.path.join(indir, modelname, name)
        file=os.path.join(indir, name)
        
        print('Load data from:',file)
        x=numpy.load(file)
        
        iyear    = x['iyear']
        fcperiod = x['fcperiod']
        rlzarr   = x['rlzarr']
        northarr = x['northarr']     #shape=(nyear,nrlz,nfp)
        southarr = x['southarr']

        tinit0 = int( mdic[model][1].split('_')[0] )

        # NOTE - numpy.savez can lose the mask, so when we load we need to do numpy.ma.masked_invalid.
        # This works since we have set missing values equal to numpy.nan in make_decadal_nao.py.     
        northarr = numpy.ma.masked_invalid(northarr)
        southarr = numpy.ma.masked_invalid(southarr)
        
        anomnorth, anomsouth = calc_anom(northarr, southarr, base1, base2, timeandmem=True, tinit0=tinit0, iyear=iyear, verbose=False)
                              
        nyr = anomnorth.shape[0]
        nens= anomnorth.shape[1]
        if ssn == 'djf':
            deltassn = 15./360
        elif ssn == 'djfm':
            deltassn = 30./360
        delta = (1+islice.start+islice.stop)/2 + deltassn 
        timeout=numpy.zeros(nyr)
        for iy,inityear in enumerate(iyear):
            timeout[iy]= iyear[iy]+delta
            
        # This does: lump_diff on north and south anomalies. We do not normalise north/south anomalies separately.
        anomnorthlump = numpy.zeros((nyr,nens) )
        anomsouthlump = numpy.zeros((nyr,nens) )
        for iy,inityear in enumerate(iyear):
            anomnorthlump[iy,:] = numpy.mean(anomnorth[iy,:,islice],axis=1)
            anomsouthlump[iy,:] = numpy.mean(anomsouth[iy,:,islice],axis=1)            
        nao = anomsouthlump - anomnorthlump    #NAO for one model
               
        if kmod == 0:
            timeout_all = timeout
            nao_all     = nao
        else: 
            i1_all,i2_all = index_common(timeout_all, timeout)
            timeout_all   = timeout_all[i1_all] 
            nao_all       = numpy.ma.concatenate((nao_all[i1_all,:], nao[i2_all,:]), axis=1)
        dlis.append(nao)
        tlis.append(timeout)
    # End loop over ensemble

    if central == 'median': 
       nao_allm = numpy.median(nao_all,axis=1)
    else:
       nao_allm = numpy.mean(nao_all,axis=1)

    if renorm:
       nao_allm=nao_allm/numpy.std(nao_allm, ddof=1)

    return timeout_all, nao_allm, nao_all, tlis, dlis  


##############################################################
def obs_smoothdiff(northcube, southcube, nlump, ssn='djf', order='lump_diff', renorm=False):
    '''
    Estimate NAO from input cubes for southern and northern NAO region anomalies.
    Do not separately normalize/standardize the north/south anomalies.
    Can eith smooth the two regions first, and then estimate difference (default),
    or else calculate difference first, and then smooth.
    One can also renormalize the final index by its standard deviation if required.
    '''
    if order == 'lump_diff':
        # Smooth cubes and calculate difference    
        timeobs, naonorthlump = runningmean(northcube, nlump, ssn=ssn)
        timeobs, naosouthlump = runningmean(southcube, nlump, ssn=ssn)
        naoout = naosouthlump - naonorthlump         
    if order == 'diff_lump':    
        # Calculate difference then smooth    
        naocube = southcube - northcube 
        timeobs, naoout = runningmean(naocube, nlump, ssn=ssn)

    if renorm:
        naoout = naoout/numpy.std(naoout,ddof=1) 
        
    return timeobs, naoout











