import os
import pickle
import copy
import numpy
import scipy
import iris

#########################################
def loadDPS(longname, indir, fp=1, fac=1.0, yearmax=2018, verbose=True,
            ensembles=['BSC','CAFE','CanESM5','CMCC','Depresys4_gc3.1','IPSL','MIROC6','MPI','NorCPM','NorCPMi2','NCAR40']):

    longname=longname.replace(' ', '_')
    nlump=1
    if type(fp) == str:
        fpsplit= fp.split('to')
        fp1    = int(fpsplit[0])
        fp2    = int(fpsplit[1])
        nlump  = fp2-fp1+1

    dlis=[]
    minlena=100000
    for model in ensembles:
        pklfile = os.path.join(indir, model+'_'+longname+'.pkl')
        a     = pickle.load(open(pklfile, 'rb'))
        if fp == 1:        
            alast=a[-1].extract(iris.Constraint(forecast_period=fp))
            yrlast=alast.coord('season_year').points[-1]        
            print('Loading from:',pklfile,'len(a)=',len(a),'yrlast=',yrlast)
        else:
            print('Loading from:',pklfile,'len(a)=',len(a))      
        syarr=[]
        dataarr=[]
        lena=len(a)
        if verbose:
            print('MODEL, numpy.mean(numpy.abs(a[-1].data))=',numpy.mean(numpy.abs(a[-1].data)) )
        for ii,a1 in enumerate(a):
            if nlump == 1:
                fpcon=iris.Constraint(forecast_period=fp)
                a2 = a1.extract(fpcon)
                syarr.append(a2.coord('season_year').points[0])
                if numpy.ma.isMaskedArray(a2.data):
                    if a2.data.mask[0]:
                        if ii == 0:   print('WARNING, mask=True for',model)
                        dval=numpy.nan*numpy.ones(a2.shape[0])
                    else:
                        dval=a2.data.data
                else:
                    dval=a2.data.data
                if ii == lena-1:
                    if lena < minlena:
                        minlena = lena  
                        syr     = syarr                                                  
                if fac != 1.0: 
                    dval=fac*dval
                dataarr.append(dval) 
            else:
                slc=slice(fp1-1,fp2)
                syslice=a1.coord('season_year').points[slc]           
                sy=int(numpy.mean(syslice))
                syarr.append(sy)
                # Lump over slc of season_year
                dval=numpy.mean(a1[:,slc].data, axis=1)   # (realization, season_year)
                #raise AssertionError('Stop for debugging...')
                if ii == lena-1: 
                    if lena < minlena:
                        minlena = lena 
                        syr     = syarr      
                if fac != 1.0: 
                    dval=fac*dval
                dataarr.append(dval) 
        dataarr=numpy.array(dataarr)   #shape is (ntim,nrlz)
        dlis.append(dataarr)
    # end loop over ensembles
    
    syr = numpy.array(syr)
    iok = numpy.where(syr <= yearmax)[0]
    syrout=syr[iok]    
    ntim_min = iok.shape[0]
    modlist=[]
    for ii,model in enumerate(ensembles):
        if numpy.isnan(dlis[ii]).any(axis=(0,1)):
            pass
        else:
            modlist.append(model) 

    dlistok = []
    kount=-1
    dall=None
    for ii,model in enumerate(ensembles):
        if model in modlist:
            dlistok.append( dlis[ii][0:ntim_min,:] )
            kount = kount+1
            if kount == 0:
                dall = copy.copy(dlis[ii][0:ntim_min,:]) 
            else:       
                dall = numpy.concatenate((dall, dlis[ii][0:ntim_min,:]), axis=1)

    modlist=numpy.array(modlist)
    return dall, dlistok, modlist, syrout, nlump
  
#########################################
def loadOBS(longname, indir, syr, nlump): 
    lname=longname.replace(' ', '_')    
    ofile  = os.path.join(indir, 'OBS_'+lname+'.nc')
    obs    = iris.load_cube(ofile)
    syobs  = obs.coord('season_year').points
    print('Loading from:',ofile,'syobs[-1]=',syobs[-1])    
    dataobs= obs.data
    ddd = []    
    for ii,yr in enumerate(syr):
        if yr <= syobs.max():
            i1=numpy.where(syobs == yr)[0][0]
            if nlump == 1:
                ddd.append(dataobs[i1])
            else:
                dneg= ((nlump+1)//2-1)
                dpos= nlump//2+1
                slc = slice(i1-dneg, i1+dpos)  # this works for even and odd
                #print(ii,dataobs[slc])
                if dataobs[slc].shape[0] == nlump:
                   ddd.append( numpy.mean(dataobs[slc]) )               
    dobs=numpy.array(ddd)       
    return dobs


#########################################
def loadOBSb(longname, indir, syr, nlump, baselim=None): 
    lname=longname.replace(' ', '_')    
    ofile  = os.path.join(indir, 'OBS_'+lname+'.nc')
    obs    = iris.load_cube(ofile)
    syobs  = obs.coord('season_year').points
    print('Loading from:',ofile,'syobs[-1]=',syobs[-1])    
    dataobs= obs.data
    if not baselim is None:
        #baselim typically [1981, 1981+30]    
        idx1 = list(syobs).index(baselim[0])
        idx2 = list(syobs).index(baselim[1])
        bval = numpy.mean(dataobs[idx1:idx2])
        print(longname, ', Obs_baseval:', bval)
        dataobs = dataobs - bval
    ddd = []    
    for ii,yr in enumerate(syr):
        if yr <= syobs.max():
            i1=numpy.where(syobs == yr)[0][0]
            if nlump == 1:
                ddd.append(dataobs[i1])
            else:
                dneg= ((nlump+1)//2-1)
                dpos= nlump//2+1
                slc = slice(i1-dneg, i1+dpos)  # this works for even and odd
                #print(ii,dataobs[slc])
                if dataobs[slc].shape[0] == nlump:
                   ddd.append( numpy.mean(dataobs[slc]) )               
    dobs=numpy.array(ddd)       
    return dobs




#########################################
def loadUKCP(longname, indir,  syr, nlump, prob=0.5, ukwil=''):   
    lname=longname.replace(' ', '')
    names=lname.split('_')
    if len(names) == 4:
        vname=names[0]+'_'+names[1]
        sname=names[2]
        fname=names[3]       
    else:
        vname=names[0]
        sname=names[1]
        fname=names[2] 

    cukwil = ''        
    if len(ukwil) != 0:
       if ukwil[0] != '_':
           cukwil = '_'+ ukwil
       else:
           cukwil = ukwil
           
    fdic={'tas':'air_temperature', 'pr':'precipitation_flux', 'psl':'psl'}
    if nlump == 1:
        uname = 'ukcp09_T+1yr_'+vname+'_'+fdic[fname]+'_'+sname+cukwil+'.nc'
    else:
        uname = 'ukcp09_T+2toT+9yr_'+vname+'_'+fdic[fname]+'_'+sname+cukwil+'.nc'
    ufile = os.path.join(indir, uname)

    print('Loading from:',ufile)
    ukcp  = iris.load_cube(ufile)
    syukcp  = ukcp.coord('season_year').points
    u1 = ukcp.extract(iris.Constraint(percentile=prob))
    dataukcp = u1.data 
    uuu=[]
    for ii,yr in enumerate(syr):
        if yr <= syukcp.max():
            i1=numpy.where(syukcp == yr)[0][0]
            uuu.append(dataukcp[i1])
    dukcp=numpy.array(uuu)  
    return dukcp


#########################################
def trimdata(dobs, syr, dall, dlistok):
    nobs=dobs.shape[0]
    if syr.shape[0] != nobs:
        syr=syr[0:nobs]
        dall=dall[0:nobs,:]
        dlistok2=[] 
        for jj in range(len(dlistok)):
             dlistok2.append(dlistok[jj][0:nobs,:])
        dlistok=dlistok2         
    return syr,dall,dlistok


#########################################
def scores(dallcent, dobs, dukcp, dlistok, sstype='uncentred', central='median'):

    # Scores for DPS mean
    acc_mean              = ACC(dallcent, dobs, sstype=sstype)
    msss_mean, mbar, obar = MSSS(dallcent, dobs, sstype=sstype)

    # Scores for individual DPS 
    acc_arr=[]
    msss_arr=[]
    for ii,data in enumerate(dlistok):
        if central == 'median':
            datam=numpy.median(data, axis=1)   # eg data.shape=(54,8), (ntime,nrlz)  
        else:
            datam=numpy.mean(data, axis=1)    
        acc = ACC(datam, dobs, sstype=sstype)
        acc_arr.append(acc)    
        msss, mbar, obar= MSSS(datam, dobs, sstype=sstype)
        msss_arr.append(msss)
    acc_arr  = numpy.array(acc_arr)
    msss_arr = numpy.array(msss_arr)

    # Scores for UKCP 
    acc_ukcp                = ACC(dukcp, dobs, sstype=sstype)
    msss_ukcp, mbaru, obaru = MSSS(dukcp, dobs, sstype=sstype)

    return acc_mean,msss_mean, acc_arr,msss_arr, acc_ukcp,msss_ukcp

#########################################
def scores_detrend(dallcent, dobs, dukcp, dlistok, sstype='uncentred', central='median'):

    dallcent_dtr = scipy.signal.detrend(dallcent)
    dobs_dtr     = scipy.signal.detrend(dobs)
    dukcp_dtr    = scipy.signal.detrend(dukcp)

    # Scores for DPS mean
    acc_mean              = ACC(dallcent_dtr, dobs_dtr, sstype=sstype)
    msss_mean, mbar, obar = MSSS(dallcent_dtr, dobs_dtr, sstype=sstype)

    # Scores for individual DPS 
    acc_arr=[]
    msss_arr=[]
    for ii,data in enumerate(dlistok):
        if central == 'median':
            datam=numpy.median(data, axis=1)   # eg data.shape=(54,8), (ntime,nrlz)  
        else:
            datam=numpy.mean(data, axis=1)
            
        datam_dtr     = scipy.signal.detrend(datam)
               
        acc = ACC(datam_dtr, dobs_dtr, sstype=sstype)
        acc_arr.append(acc)    
        msss, mbar, obar= MSSS(datam_dtr, dobs_dtr, sstype=sstype)
        msss_arr.append(msss)
    acc_arr  = numpy.array(acc_arr)
    msss_arr = numpy.array(msss_arr)

    # Scores for UKCP 
    acc_ukcp                = ACC(dukcp_dtr, dobs_dtr, sstype=sstype)
    msss_ukcp, mbaru, obaru = MSSS(dukcp_dtr, dobs_dtr, sstype=sstype)

    return acc_mean,msss_mean, acc_arr,msss_arr, acc_ukcp,msss_ukcp

#########################################
def ACC(model, obs, sstype='uncentred'):
    mbar = numpy.mean(model)
    obar = numpy.mean(obs)
    if sstype.lower() in ['centred', 'centered', 'wmo']:
        var1 = model - mbar
        var2 = obs - obar        
    else:   # sstype.lower() in ['uncentred', 'uncentered']
        var1 = model
        var2 = obs
    numer = numpy.sum(var1*var2)
    sq1   = numpy.sum(var1*var1)
    sq2   = numpy.sum(var2*var2)
    denom = (sq1*sq2)**0.5
    acc   = numer/denom
    return acc

#########################################
def MSSS(model, obs, sstype='uncentred'):
    def MSD(var1, var2):
        # Mean Square Diff
        diff    = var1-var2
        msd = numpy.mean(diff*diff)
        return msd
    mbar = numpy.mean(model)
    obar = numpy.mean(obs)
    if sstype.lower() in ['wmo']:
        numer = MSD(model, obs)    
        denom = MSD(obs, obar)
    elif sstype.lower() in ['centred', 'centered']:    
        var1 = model - mbar
        var2 = obs - obar
        numer = MSD(var1, var2)
        denom = MSD(obs, obar)
    else:   # sstype.lower() in ['uncentred', 'uncentered']
        numer= MSD(model, obs)
        obar0 = 0.0
        denom = MSD(obs, obar0)
    msss = 1.0-numer/denom
    return msss, mbar, obar





