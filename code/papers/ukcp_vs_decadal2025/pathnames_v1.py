import os

# Replace this with suitable local directory 
basedir = os.path.join(os.getenv('HOME'), 'QDC')

plotdir = os.path.join(basedir, 'Figures')

datadir = os.path.join(basedir, 'Data')

ukcpdir = os.path.join(basedir, 'Data', 'UKCP')

obsdir  = os.path.join(basedir, 'Data', 'Obs')

dpsdir  = os.path.join(basedir, 'Data', 'DPS')

ppedir  = os.path.join(basedir, 'Data', 'PPE')

cmip5dir= os.path.join(basedir, 'Data', 'CMIP5')

naodir  = os.path.join(basedir, 'Data', 'NAO')

scoredir= os.path.join(basedir, 'Data', 'Scores')
