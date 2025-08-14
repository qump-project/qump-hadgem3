####################
def block_bootstrap(data, nsamp, obs=None, blocksize=5, method='fixed', seed=11, verbose=False, shape3d=True):
    '''
    Create a bootstrap sample with specified or varying blocksize (Politis & Romano: PR).
    Set method='fixed' and blocksize>1 to use block bootstrap with fixed blocksize.
    Set method='fixed' and blocksize=1 to use simple bootstrap with NO blocks.    
    Set method='stationary' to use Politis and Romano variable blocklength.
    Set blocksize='Wilks' to use Wilks expression (Eq 5.36, page 178, Wilks 3rd ed) to estimate the optimum block length.    
    Inputs:
       data:   numpy 2D array of size (ntime,nens), or numpy 1D array of size (ntime).
       obs:    numpy 1D array of size (ntime).        
       nsamp:  integer scaler, specifying the desired bootstrap sample size.
       method: string, one of ('fixed', 'stationary').
               if 'fixed', use a constant block length.
               if 'stationary', use the Politis & Romano stationary bootstrap method, where blocklength
               is sampled fromn the Geometric distribution, with mean blocklength equal to blocksize. 
       blocksize: integer scaler, or one of ('Wilks', 'wilks', None).
                  if integer and method='fixed', this specifies FIXED block size.
                  if integer and method='stationary', this specifies AVERAGE block size.
                  if equal to 'wilks', estimate using Wilks formula (Eq 5.36, page 178, Wilks 3rd ed)
       seed:    integer scaler, specifying seed to initialize sampling.
       verbose: boolean, to turn on/off additional output.
    Returns:
       ans:  numpy 3D array of size (ntime,nens,nsamp) if a 2D array is input, or:
             numpy 2D array of size (ntime,nsamp) if 1D array is input.
             This array is of one dimension greater than the dimension of the input data,
             with the shape of this final added dimension equal to nsamp.
       obs_boot:  numpy 2D array of size (ntime,nsamp), or None if input obs is None.
       #       
       blocksize,idx1,idx2: Only output if verbose is True              
       blocksize:  block length - constant or average depending on method choice. This will be the same 
                   as input, except if 'Wilks' is input, in which case the value estimated is output.
       idx1: numpy 2D integer array of size (ntime, nsamp). 
             This contains the sampled time indexes for each of the bootstrap samples.
       idx2: numpy 3D integer array of size (ntime, nens, nsamp). 
             This contains the sampled member indexes for each of the bootstrap samples. 
    '''
    import numpy.random
    
    numpy.random.seed(seed)
    nsample=max([1,nsamp])
    if data.ndim == 1:
        data=numpy.expand_dims(data, -1)
    ntim=data.shape[0]
    nens=data.shape[1]
    
    # Estimate optimum blocksize Using Wilks formula (Eq 5.36, page 178, Wilks 3rd ed).
    # Iterate over ensemble and take the mean.
    bsize  = [WilksBlockLength(data[:,kens], nitmax=10) for kens in range(nens)]        
    wilksBS= max([1, numpy.mean(bsize)])    
   
    # Only use Wilks value if requested, otherwise use input blocksize.
    if blocksize in ['wilks', 'Wilks', None]:
        if method == 'stationary':
            blocksize=wilksBS
        else:   #'fixed'
            blocksize=int(round(wilksBS))

    # Create random sample of ensemble members.    

    if type(data) is numpy.ma.core.MaskedArray:
        ans = numpy.ma.zeros( (ntim,nens,nsample) )
    else:
        ans = numpy.zeros( (ntim,nens,nsample) )
    
    if obs is None:
        obs_boot = None
    else:      
        obs_boot = numpy.zeros( (ntim,nsample) )      

    # Special simple case for blocksize equal to one           
    if blocksize == 1:
        for i in range(nsample):        
            isamp = numpy.trunc( ntim*numpy.random.uniform(size=ntim) ).astype(int)
            jsamp = numpy.trunc( nens*numpy.random.uniform(size=(ntim,nens)) ).astype(int)            
            for iens in range(nens):
                ans[:,iens,i] = data[isamp, jsamp[:,iens] ]
            if obs is not None:
                obs_boot[:,i] = obs[isamp]

    # General case for blocksize > 1.
    else:
        # Sampling of the ensemble index with kens   
        kens= numpy.trunc( nens*numpy.random.uniform(size=(nens,nsample*ntim)) ).astype(int)
        # Sampling of the block start time with i0.            
        i0  = numpy.trunc( ntim*numpy.random.uniform(size=nsample*ntim) ).astype(int)         
        if method ==  'stationary':
            # Variable blocksize, sampling a geometric dist (Politis & Romano).
            prob = 1./float(blocksize)        
            jb = numpy.random.geometric(prob, size=nsample*ntim)
        else:
            # Fixed block size, all blocksizes identical
            jb = numpy.full(nsample*ntim, blocksize)
        kk=0
        tslice=slice(0,ntim)
        if verbose:
            idx1=numpy.full( (ntim,nsample), 0)
            idx2=numpy.full( (ntim,nens,nsample), 0)
        # Loop over each bootstrap sample        
        for i in range(nsample):
            try:
                del isamp
                del jsamp
            except:
                pass
            # Combine the blocks over the time period.   
            finished = False
            while not finished:
                # isub1 is the array of indexes for each block.
                # isub2 is sample of the ensemble, duplicated for the block length.
                isub1 = (i0[kk]+list(range(jb[kk]))) % ntim
                jsub1 = numpy.tile(kens[:,kk], (jb[kk],1) ) 
                #print 'kk=',kk, isub1
                kk=kk+1
                try:
                    isamp=numpy.hstack([isamp,isub1])
                    jsamp=numpy.vstack([jsamp,jsub1])
                except:
                    isamp=isub1
                    jsamp=jsub1
                if isamp.shape[0] >= ntim:
                    finished = True
            # Construct the sample over the ensemble for one bootstrap sample.
            for iens in range(nens):
                ans[:,iens,i] = data[isamp[tslice], jsamp[tslice,iens] ]
            # Construct the resampled obs data
            if obs is not None:
                obs_boot[:,i] = obs[isamp[tslice]]            
            # Save extra index arrays for verbose output
            if verbose:
                idx1[:,i]=isamp[tslice]
                for iens in range(nens):
                    idx2[:,iens,i]=jsamp[tslice,iens]                            
    if not shape3d:
        ans=numpy.squeeze(ans)
    if verbose:
        return ans, obs_boot, blocksize,idx1,idx2 
    else:
        return ans, obs_boot


####################
def WilksBlockLength(data, nitmax=10):
    # Estimate optimum blocksize Using Wilks formula (Eq 5.36, page 178, Wilks 3rd ed) 
    # Block length depends only on lag-1 autocorrelation and sample size 

    import scipy.stats 
    
    N = data.shape[0]
    rval,pval = scipy.stats.pearsonr(data[:-1], data[1:])
    nu        = (4./3.)*rval/(1.+rval)
    Lold      = 1./(nu/(N+1.) + 1./(N+1.)**nu)
    finished  = False
    kount     = 0
    while not finished:
        Lnew = (N-Lold+1.)**nu
        kount= kount+1
        #print kount,Lold,Lnew,Lnew-Lold
        if kount == nitmax or abs(Lnew-Lold) < 1e-4:
            finished=True 
        else:
            Lold=Lnew             
    if kount == nitmax:
         print('WilksBlockLength warning: kount exceeds '+str(nitmax)+', not yet converged')
         print('Lold=',Lold,' Lnew=',Lnew,' Lnew-Lold=',Lnew-Lold) 
    return Lnew



