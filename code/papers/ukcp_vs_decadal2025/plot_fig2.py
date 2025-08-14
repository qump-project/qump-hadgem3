##################################################
import os
import numpy
import scipy
import math
import matplotlib
import matplotlib.pyplot as plt

import iris
import iris.plot as iplt

import qdcUtils as utils

from pathnames_v1 import *

############################################################

namefig= 'fig2'
ifig   = 2

#dpiarr=[100]
dpiarr=[150]

saveplot   = True
#saveplot   = False

mask1 = iris.load_cube(os.path.join(ppedir, 'EnglandWales_n216e.pp') )
mask2 = iris.load_cube(os.path.join(ppedir, 'eng_wales.pp') )
mask3 = iris.load_cube(os.path.join(ppedir, 'hadgem3_mask_n216e.pp') )

d3=mask3.data.copy()    #all land
de=mask1.data.copy()    #eaw land

# With cmap='jet' and levels from [-8,-7...7,8], a value of -4.5 gives mid-blue color, and +4.5 give orange
cmap  = 'jet'
levels= numpy.array([-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8])

# Fill in Swedish/Russian lakes, will reset data to a value (4.5) below
d3.mask[270,37:39] = False
d3.mask[271,36:39] = False
d3.mask[272,36:38] = False
d3.mask[272,42:44] = False
d3.mask[273,41:43] = False
d3.mask[267,15]    = False
d3.mask[268,16]    = False

d3ok = ~d3.mask
d3.data[d3ok] = 4.5    # +4.5 gives orange for land points

deok = ~de.mask
d3.data[deok]= -4.5    # -4.5 gives blue for EAW points
mask4 = mask3.copy(data=d3)

lat=mask4.coord('latitude').points
lon=mask4.coord('longitude').points

# Pull out NEU region from land mask 
# (S, N, W, E) = (48.0, 75.0, -10.0, 40.0)

mm=mask4.data.copy()
ilo=numpy.where(lat < 48.0)
ihi=numpy.where(lat > 75.0)
mm[ilo,:] = numpy.ma.masked
mm[ihi,:] = numpy.ma.masked

j1=numpy.where(40.0 < lon)
j2=numpy.where(lon[j1] < 350.0)
j3=(j2[0]+j1[0][0])
mm[:,j3] = numpy.ma.masked

mask5 = mask4.copy(data=mm)
mask5.coord('latitude').guess_bounds() 
mask5.coord('longitude').guess_bounds() 

fs      = 9.0
alpha   = 0.35
latlim  = [40, 74]
lonlim  = [-19.5, 43.5]
xsize   = 16.
ysize   = 8.

lev=utils.plotCube(mask5, levels=levels, cmap=cmap, orientation='vertical', ifig=ifig, 
                   xsize=xsize, ysize=ysize, latlim=latlim, lonlim=lonlim, title='', 
                   alpha=alpha, showbar=False)

plt.text(-5.7, 49.4, 'England-Wales',  size=fs,color='dodgerblue',ha='right')
plt.text(12.2, 66.0, 'Northern Europe',size=fs,color='darkorange',ha='right')

for dpi in dpiarr:           
    cdpi  = str(int(dpi))+'dpi'
    oname = namefig+'_'+cdpi+'.png'
    outfile=os.path.join(plotdir, oname)
    if saveplot:
        plt.savefig(outfile, format='png', dpi=dpi)
        print('Saved ',outfile)
    else:
        print('NOT saved:',outfile)
