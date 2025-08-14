# "Comparing near-term information from national climate scenarios and initialised decadal predictions"
James M. Murphy, Glen R. Harris, Robin T. Clark (2025), Climate Dynamics (accepted).

## 1. Code Repository
https://github.com/qump-project/qump-hadgem3/tree/master/code/papers/ukcp_vs_decadal2025

contains code to reproduce all the Figures in this paper (both main text and Supplementary Material).
This code and all necessary input data is also available as a unix gzipped tarfile at  
https://doi.org/10.5281/zenodo.14655585  

Using the shell variable 'basedir' to label the local directory of the downloaded repository, the sub-directory  
'''
    $basedir/code
'''
contains the python scripts to reproduce all the Figures in Murphy et al. (2025). Before running any code the file  
'''
    pathnames_v1.py
'''
will need to be edited, potentially replacing the definition of basedir with the local directory name to which the 
repository has been installed. Currently basedir defaults to  
'''
    basedir = os.path.join(os.getenv('HOME'), 'QDC')
'''

Note: the acronym QDC here refers to "QUMP Decadal Comparison", while QUMP stands for 
"Quantifying Uncertainty in Model Predictions", the orginial title for the research theme that developed 
the probabilistic projections analysed here.

## 2. Python environment and creating figures
Some of the input iris cube data is in binary pickle format, and requires an iris version number of 3.7.0 (or less)
to load. A suitable python environment can be created and activated using the  yaml configuration file iris370t.yaml:  
'''
    conda env create -f iris370t.yaml
    conda activate iris370t
'''

Figures can then be created directly using python or ipython. For example, for Figure 7:  
'''
    ipython --matplotlib  
    %run $basedir/code/plot_fig7.py
'''

## 3. Figures directory
The location for all Figures produced by the scripts in $basedir/code is  
'''
    $basedir/Figures
'''

## 4. Input Data
Both data and code is available from https://doi.org/10.5281/zenodo.14655585  
as a Unix gzipped tar file:  
'''
    murphy2025_qdc.tar.gz
'''

The code alone is also available on GitHub:  
https://github.com/qump-project/qump-hadgem3/tree/master/code/papers/ukcp_vs_decadal2025  

The data in murphy2025_qdc.tar.gz should be downloaded to the desired top-level directory  
'''
    $basedir
'''  
and then gunzip-ed and extracted with tar:  
'''
    gunzip murphy2025_qdc.tar.gz  
    tar -xvf murphy2025_qdc.tar
'''

There are seven sub-directories in Data:  

    ## $basedir/Data/Obs  
    Observational data, in netcdf (.nc) and text (.txt) format.

    ## $basedir/Data/DPS  
    Decadal Prediction System data: netcdf, text and pickled format (.pkl, .pickle, .pickled),

    ## $basedir/Data/UKCP  
    Uninitialised projection data from the UKCP national climate scenarios, in netcdf, text and 
    numpy.savez (.npz) format

    ## $basedir/Data/PPE  
    Annual surface temperature projections for the 57 member Earth System Perturbed Parameter Ensemble 
    (ESPPE), in netcdf format. 
    Some mask files are also here, in netcdf and pp (Met Office post-processed) format. 

    ## $basedir/Data/CMIP5  
    Annual surface temperature projections for 12 CMIP5 ESMs, and land-fraction files for the 
    same models, in netcdf format.

    ## $basedir/Data/NAO  
    Different NAO indices calculated for the Decadal Prediction System models, in netcdf and
    numpy.savez (.npz) format. 

    ## $basedir/Data/Scores  
    Skill scores (both original and detrended) for different variables and periods, for both DPS and 
    UKCP projections, in numpy.savez (.npz) format. For example, skill scores for original projections 
    (with trend) for GMST T+1 are in 'tas_ann_glb_T1_uncentred_orig.npz'.
    Median values for Obs, DPS and UKCP data are contained in files named like: *_median_data_orig.npz.

