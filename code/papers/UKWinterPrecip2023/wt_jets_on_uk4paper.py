# (C) Crown Copyright, Met Office. All rights reserved.
# 
# This file is released under the BSD 3-Clause License.
# See LICENSE.txt in this directory for full licensing details. 

import os, glob, sys, re
from copy import deepcopy
import pdb
import datetime
import operator
import numpy
import iris
import iris.coord_categorisation
# import sklearn.isotonic
import scipy.stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import scipy.stats


import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.backends.backend_pdf import PdfPages
import cf_units

# import qumpy_lite as qumpy
from qumpy_lite import *


# import path names from parent directory. This is done to hide full pathnames for
# our institute. Set to False if using paths.py in current directory

if True:
	current = os.path.dirname(os.path.realpath(__file__))
	parent = os.path.dirname(current)
	sys.path.insert(0, parent)
	from paths import *
else:
	from paths import *

if len(sys.argv) < 3:
    sname = 'DJF'
    var = 'pr'
else:
    sname = sys.argv[1]
    var = sys.argv[2]
    
sname = sname.upper()    
print("Season: ", sname)
print("Variable: ", var)
loopover = ['scotland', 'south_east_england', 'england', 'wales', 'northern_ireland']
loopover = ['scotland', 'south_east_england', 'england']
loopover = ['wales']
loopover = ['scotland', 'wales', 'south_east_england']
loopover = ['uk', 'scotland', 'northern_ireland', 'wales', 'england', 'south_east_england']
loopover = ['uk', 'scotland', 'northern_ireland', 'wales', 'england']
# loopover = ['uk']
nrow, ncol = 5, 1

#===============================================================================

months_dict = dict(DJF=(12, 1, 2), MAM=(3, 4, 5), JJA=(6, 7, 8), SON=(9, 10, 11))


searchstr = os.path.join(coupleddir, 'UK', 'histrcp8p5_%s_uk.nc')
cachedir = os.path.join(tempdir, 'cache', 'wt_jets')
figpath = os.path.join(plotdir, 'paper_ukwinter_precip')
jets_dir = os.path.join(coupleddir, 'jets', 'gcm_jets')



# KEEP = [1100000, 1100605, 1100834, 1101113, 1101554, 1101649,
#         1101843, 1101935, 1102123, 1102242, 1102305, 1102335,
#         1102491, 1102832, 1102868]
# DROP = [1100090, 1102089, 1102753, 1102884, 1102914]
KEEP = [1100000, 1100605, 1100834, 1101113, 1101554, 1101649,
        1101843, 1101935, 1102123, 1102242,
        1102305, 1102335, 1102491, 1102832, 1102868]
DROP = [1100090, 1102089, 1102753, 1102884, 1102914]
WEATHER_TYPES = [1, 2, 3, 4, 5, 6, 7, 8]

degree_sign= '\N{DEGREE SIGN}'
units_dict = dict(tas=degree_sign + 'C', pr='mm/day')
cmaps_dict = dict(tas='YlOrRd', pr='GnBu')
cmap_change_dict = dict(tas='RdBu_r', pr='RdBu')
varname_dict = dict(tas='Temperature', pr='Precipitation')


#===============================================================================

units = units_dict[var]
cmap = cmaps_dict[var]
cmap_change = cmap_change_dict[var]
varname = varname_dict[var]


if not os.path.exists(figpath):
    os.makedirs(figpath)

#===============================================================================

def make_grid_match_if_close(grid, target):
    match = True
    for ax in ['x', 'y']:
        gcoord = grid.coord(axis=ax)
        tcoord = target.coord(axis=ax)
        if gcoord.shape != tcoord.shape:
            print('%s-axes are different shape' % ax)
            match = False
        else:
            if not numpy.allclose(gcoord.points, tcoord.points, atol=1e-6):
                print('%s-axes have different points' % ax)
                match = False
            if tcoord.has_bounds() and gcoord.has_bounds():
                if not numpy.allclose(gcoord.bounds, tcoord.bounds, atol=1e-6):
                    print('%s-axes have different bounds' % ax)
                    print(gcoord.bounds, tcoord.bounds)
                    print(gcoord.bounds - tcoord.bounds)
                    match = False    
                        
        if gcoord.name() != tcoord.name():
            print('%s-axes have different names')
            match = False
        if gcoord.units != tcoord.units:
            print('%s-axes have different units')
            match = False

    if not match:
        raise AssertionError('Coordinates do not match')
    else:
        for ax in ['x', 'y']:
            tcoord = target.coord(axis=ax)
            grid.coord(axis=ax).var_name = tcoord.var_name
            grid.coord(axis=ax).coord_system = tcoord.coord_system
            grid.coord(axis=ax).points = tcoord.points.copy()
            if tcoord.has_bounds():
                grid.coord(axis=ax).bounds = tcoord.bounds.copy()
            else:
                grid.coord(axis=ax).bounds = None
            
def anom(x):
    if x.ndim == 1:
        return x - x[:50].mean()
    else:
        return (x.T - x[:,:50].mean(1)).T
    
def anomc(x):
    return x.copy(data=anom(x.data))

def stacker(ix, x, labels=None, colors=None, verbose=False,
            width=0.3, total_color='b', total=True, linewidth=0.5):
    nx = x.size
    isrt = numpy.argsort(x)
    xsrt = numpy.sort(x)
    if x.min() < 0.0:
        xbot = x[x < 0.0].sum()
    else:
        xbot = 0.0
    delta = numpy.abs(x)[isrt]
    stack = xbot + numpy.cumsum(numpy.concatenate([[0], delta]))
    if verbose:
        print(stack)
        print(delta)
        print()
#     pdb.set_trace()
    if labels is None:
        labels = [None] * nx
    else:
        if len(labels) != nx:
            raise ValueError('Labels %s is not of length %s like x' % (labels, nx))

    if colors is None:
        colors = [None] * nx
    else:
        if len(colors) != nx:
            raise ValueError('Colors %s is not of length %s like x' % (colors, nx))

            
    labels = numpy.array(labels)[isrt]
    colors = numpy.array(colors)[isrt]
            
    for st, d, lab, col in zip(stack, delta, labels, colors):
        plt.bar(ix, d, bottom=st, color=col, label=lab, edgecolor='k',
                width=width, linewidth=linewidth)
    if total:
        plt.bar(ix - width, x.sum(), color=total_color, edgecolor='k',
                width=width, linewidth=linewidth, 
                label=None if labels[0] is None else 'Total' )
        
        
def make_mask(region, model_grid):
    fmask = os.path.join(coupleddir, 'masks/UKCP18/N216_UK/%s.nc' % region.lower())
    print(region, fmask)
    mask_name = os.path.basename(fmask).replace('.nc', '').capitalize().replace('_', ' ')
    mask = iris.load_cube(fmask)
    mask = mask.copy(data=numpy.ma.masked_less(mask.data, 0.5))
    invert_wrap_lons(mask)
    
    model_grid = getSpatialGrid(model_grid)
    for ax in ['x', 'y']:
        model_grid.coord(axis=ax).bounds = None
        mask.coord(axis=ax).bounds = None

    mask = mask.subset(model_grid.coord('longitude')).subset(model_grid.coord('latitude'))
    mask_mean = MaskMean(mask, name=mask_name)
    return mask_mean
    
def make_daily_cube(var, f, sname, region):
        daily_uk = iris.load_cube(f)
        mask_mean = make_mask(region, daily_uk)
        if not daily_uk.coords('month_number'):
            iris.coord_categorisation.add_month_number(daily_uk, 'time')
        for tc in daily_uk.coords(axis='t'):
            tc.bounds = None
        daily_uk = daily_uk.extract(iris.Constraint(month_number=months_dict[sname]))
        make_grid_match_if_close(daily_uk, mask_mean._mask)
        
        if var.startswith('pr'):
            daily_uk.data *= 86400.0
            daily_uk.units = cf_units.Unit('mm day-1')
        elif var.startswith('ta'):
            daily_uk.data -= 273.15
            daily_uk.units = cf_units.Unit('deg_c')

        return mask_mean(daily_uk)




def make_intensities(wtypes, wt_ann, var, sname, region, jet_str):
    wt_intensities = iris.cube.CubeList()
    daily_uk_cubes = iris.cube.CubeList()
    
    for realn in wtypes.coord('realization').points:
        rip = 'r001i1p%.5i' % (realn - 1100000)
        print("Processing WT intensities for %s" % rip)
        fuk_pr = os.path.join(coupleddir, 'daily/uk_%s/%s/UK/histrcp8p5_daily_%s_%s_UK.nc' % (var, rip, rip, var))
        daily_uk = make_daily_cube(var, fuk_pr, sname, region)
        print(daily_uk.shape, wtypes.shape)
        daily_uk_cubes.append(daily_uk)
        
        common_yyyymmdd = set(jet_str.coord('yyyymmdd').points.tolist())
        common_yyyymmdd = common_yyyymmdd.intersection(daily_uk.coord('yyyymmdd').points)
        common_yyyymmdd = common_yyyymmdd.intersection(wtypes.coord('yyyymmdd').points)
        common_yyyymmdd = sorted(common_yyyymmdd)
        yyyymmdd_crit = iris.Constraint(yyyymmdd=common_yyyymmdd)
        if wtypes.coord('yyyymmdd').points.tolist() != common_yyyymmdd:
            print("Extracting WTs")
            i0 = wtypes.coord('yyyymmdd').points.tolist().index(common_yyyymmdd[0])
            i1 = wtypes.coord('yyyymmdd').points.tolist().index(common_yyyymmdd[-1]) + 1
#             wtypes = wtypes.extract(yyyymmdd_crit)
            wtypes = wtypes[:,i0:i1]
        if daily_uk.coord('yyyymmdd').points.tolist() != common_yyyymmdd:
            print("Extracting var")
            i0 = daily_uk.coord('yyyymmdd').points.tolist().index(common_yyyymmdd[0])
            i1 = daily_uk.coord('yyyymmdd').points.tolist().index(common_yyyymmdd[-1]) + 1
            daily_uk = daily_uk[i0:i1]
#             daily_uk = daily_uk.extract(yyyymmdd_crit)


        for iwt in WEATHER_TYPES:
            wt_str = 'wt%.1i' % iwt
            wcrit = iris.Constraint(realization=realn, month=months_dict[sname])
            wty = wtypes.extract(wcrit)
            wta = wt_ann[wt_str].extract(iris.Constraint(realization=realn))
            var_wt = wty.copy(data=(wty.data == iwt) * daily_uk.data)
            wti = var_wt.aggregated_by ('season_year', iris.analysis.MEAN)
            wti.data = numpy.ma.masked_where(wta.data == 0.0, wti.data / wta.data)
            if wti.coords('forecast_reference_time'):
                wti.remove_coord('forecast_reference_time')
            wtcoord = iris.coords.DimCoord([iwt], long_name='Weather type')
            if not wti.coords('Weather type'):
                wti.add_aux_coord(wtcoord)
            wti.data = wti.data.filled(0.0)


            wt_intensities.append(wti)
            
    return daily_uk_cubes.merge_cube(), wt_intensities.merge_cube()


def extract_rip(f):
    pattern = re.compile('r[0-9][0-9][0-9]i[0-9]p[0-9][0-9][0-9][0-9][0-9]')
    return pattern.search(f).group(0)

def central_value(string, sep='|'):
    values = string.split(sep)
    if len(values) % 2 == 1:
        return values[len(values) // 2]
    else:
        raise ValueError('No central value as even number of objects')

def is_continuous(x):
    x = numpy.array(x)
    diff = numpy.unique(x[1:] - x[:-1])
    return diff.size == 1 and diff[0] != 0.0

def make_common(coord, *cubelists):
    print(cubelists)
    ccoord = cubelists[0][0].coord(coord).copy()
    common = set(ccoord.points)
    for cl in cubelists:
        for c in cl:
            print(list(common)[:5])
            print(len(common))
            common.intersection_update(c.coord(coord).points)
    common = numpy.array(sorted(common))
    
    print("Made overlap")
    
#     for cl in cubelists:
#         ans.append([c.extract(iris.Constraint(coord_values={coord:common})) for c in cl])
    all_ans = iris.cube.CubeList()
    for cl in cubelists:
# ans.append([c.subset(ccoord) for c in cl])
        ans = iris.cube.CubeList()
        for c in cl:
            _ = c.coord(coord)
            _.bounds = None
            index = [_.points.tolist().index(pt) for pt in common]
            if is_continuous(index):
                ilo, ihi = min(index), max(index) + 1
            else:
                raise IndexError("Common times are not continuous")

            dim = c.coord_dims(coord)[0]
            if dim == 0:
                cc = c[ilo:ihi]
            elif dim == 1:
                cc = c[:,ilo:ihi]
            elif dim == 2:
                cc = c[:,:,ilo,ihi]
            else:
                raise NotImplementedYetError()
            ans.append(cc)
#             print ans[-1]
        all_ans.append(ans)
        
    if len(all_ans) == 1:
        return all_ans[0]
    else:
        return all_ans

class Binner2D(object):
    '''
    Bins a third variable, V, in context of two other explanatory variables, X and Y, recording the mean and
    frequency as a function of the X and Y. Also provides tools to plot and compare binned objects.

    X and Y are cubes with 1-d time series with a time coordinate.

    V is a 2-d cube with second coordinate being a time coordinate.

    '''

    def __init__(self, x, y, bins=15, xedges=None, yedges=None):
        '''
        Sets up a bin object using two 1-d cubes, X and Y, with matching time coordinates.
        
        X-length of 2-d bin is determined by xedges if specified else bins.
        Y-length of 2-d bin is determined by yedges if specified else bins.
        
        '''
                
        self._check_has_time_cube(x)
        self._check_has_time_cube(y)
        if x.coord('time') != y.coord('time'):
            raise ValueError('X and Y do not have matching time coordinates - need to subset')
        
        self._x = x.data
        self._y = y.data
        self._t = x.coord('time')
        self._nt = self._t.shape[0]
        self._xlabel = '%s (%s)' % (x.name().capitalize(), x.units)
        self._ylabel = '%s (%s)' % (y.name().capitalize(), y.units)


# strip bounds to aid easier subsetting        
        self._t.bounds = None
        
# if xedges or yedges None then dummy call to set up xedges and yedges for the one or two that need replacing
        if xedges is None or yedges is None:
            dummy_data = numpy.zeros(self._nt)
            _0, _xedges, _yedges, _1 = scipy.stats.binned_statistic_2d(self._x, self._y, dummy_data, bins=bins)
            self._xedges = _xedges if xedges is None else xedges
            self._yedges = _yedges if yedges is None else yedges
        else:
            self._xedges = xedges
            self._yedges = yedges
            
        self._bin4call = (self._xedges, self._yedges)
        self._nx = self._xedges.shape[0] - 1
        self._ny = self._yedges.shape[0] - 1
        self._fits = dict()                                 # stores results from fits

    def _check_has_time_cube(self, cube):
        if not isinstance(cube, iris.cube.Cube):
            raise TypeError('Input is not a cube')

        if not cube.coords('time'):
            raise ValueError('This is not a cube with a time coordinate')

#         if cube.ndim != 1:
#             raise ValueError('Cube is not 1-d')


       
    def fit(self, title, x, y, cube, xstr=None):
        '''
        Bin the data in 2-d cube and labelled with _title_ which has one time coordinate that overlaps
        completely with time coordinate of X and Y cubes.
        
        '''
        
        if not isinstance(cube, iris.cube.Cube):
            raise TypeError('Input is not a cube')

        if cube.ndim != 2:
            raise ValueError('Cube is not 2-d')
            
        if title in self._fits:
            raise ValueError('The title = %s has already been used.' % title)
            
# check time coordinates
        self._check_has_time_cube(x)
        self._check_has_time_cube(y)
        self._check_has_time_cube(cube)

        x.coord('time').bounds = None
        y.coord('time').bounds = None
        cube.coord('time').bounds = None
#         if x.coord('time') != self._t:
#             raise ValueError('X does not have matching time coordinates')
        if y.coord('time') != x.coord('time'):
            raise ValueError('X and Y do not have matching time coordinates')
#         if x.coord('time') != self._t:
#             raise ValueError('X does not have matching time coordinates')
        if cube.coord('time') != x.coord('time'):
            raise ValueError('Cube does not have matching time coordinates')



#         try:
#             cube = cube.subset(self._t)
#         except:
#             raise ValueError('cube could not be subsetted with time coordinate of X and Y')
            
            
# check coordinate of first axis and make bin arrays of correct shape
        xcoord0 = x.coord(dim_coords=True, dimensions=[0])
        ycoord0 = y.coord(dim_coords=True, dimensions=[0])
        ccoord0 = cube.coord(dim_coords=True, dimensions=[0])

        if xcoord0 != ycoord0:
            raise ValueError('X and Y coordinates in first dimension are different.')
                
        if ccoord0 != xcoord0:
            common_coord0 = xcoord0.intersect(ccoord0)
            x = x.subset(common_coord0)
            y = y.subset(common_coord0)
            cube = cube.subset(common_coord0)

        shape = (cube.shape[0], self._nx, self._ny)
        mean = numpy.zeros(shape)
        istd = numpy.zeros(shape)
        freq = numpy.zeros(shape)
        
            
        for i in range(cube.shape[0]):
            mean[i,:,:], _x, _y, self._binnumber = scipy.stats.binned_statistic_2d(x[i].data, y[i].data, cube[i].data,
                                                                                   bins=self._bin4call,
                                                                                   statistic='mean')
        
            istd[i,:,:], _x, _y, self._binnumber = scipy.stats.binned_statistic_2d(x[i].data, y[i].data, cube[i].data,
                                                                                   bins=self._bin4call,
                                                                                   statistic=numpy.std)

            freq[i,:,:], _x, _y, self._binnumber = scipy.stats.binned_statistic_2d(x[i].data, y[i].data, cube[i].data,
                                                                                   bins=self._bin4call,
                                                                                   statistic='count')
            if freq[i,:,:].sum() != x[i].data.size:
                print((freq[i,:,:].sum(), x[i].data.size))
                raise Exception('Mismatch in frequency sum')
            
            
        mean = numpy.ma.masked_invalid(mean)
        istd = numpy.ma.masked_invalid(istd)
        pdf = (freq.T / freq.T.sum((0, 1))).T
        name = '%s (%s)' % (cube.name(), cube.units)

        fit = dict(mean=mean, istd=istd, freq=freq, x=x.data, y=y.data, pdf=pdf, name=name, xstr=xstr)
        self._fits[title] = fit
        
    def process_and_fit(self, title, iterator, func):
        pass
        
    def diagnose_comparison(self, title1, title2):
        pass
    
    def _plot1panel(self, data, title, colorbar=True, xlabel=True, ylabel=True, orientation='horizontal', **kwargs):
        print(('In plot1panel: xlabel=%s, colorbar=%s, ylabel=%s' % (xlabel, colorbar, ylabel)))
        print(kwargs)
        xx, yy = numpy.meshgrid(self._xedges, self._yedges)
        p = plt.pcolormesh(xx, yy, numpy.ma.masked_invalid(data).T, **kwargs)
        if colorbar:
            print('Making colorbar')
            cbar = plt.colorbar(p, orientation=orientation)
        else:
            cbar = None
        plt.xlim(self._xedges.min(), self._xedges.max())
        plt.ylim(self._yedges.min(), self._yedges.max())
        if xlabel:
            plt.xlabel(self._xlabel)
        if ylabel:
            plt.ylabel(self._ylabel)
        plt.title(title)
        print('colorbar', cbar)
        return p, cbar
    
    def _plot_ratio(self, data, titl, **kwargs):
        if 'norm' in kwargs:
            del kwargs['norm']
        if 'tickv' not in kwargs:
            tickv = [.33, .5, 1., 2, 3]
        else:
            tickv = kwargs['tickv']
            del kwargs['tickv']
        p, cbar = self._plot1panel(data, titl,
                                   norm=matplotlib.colors.LogNorm(vmin=tickv[0],
                                   vmax=tickv[-1]), **kwargs)
                                   
        print(dir(p))

        ticks = p.norm(tickv)
#         ticks[ticks == 0.0] = 1e-8
#         ticks = tickv
        print(tickv, ticks)
        if cbar.orientation == 'horizontal':
            cbar.ax.xaxis.set_ticks(ticks, minor=False)
            cbar.ax.xaxis.set_ticklabels(tickv)
        else:
#             cbar.set_ticks([])
#             print(dir(cbar.ax))
#             cbar.ax.yaxis.set_ticks(tickv, minor=False)
#             cbar.ax.yaxis.set_ticklabels(tickv)
            print(cbar.ax.get_yticks())
            print(cbar.get_ticks())
            cbar.set_ticks([0.5,1,2], minor=False)
            cbar.minorticks_off()
            print(cbar.ax.get_yticks(), cbar.ax.get_yticklabels())
            cbar.set_ticklabels(tickv)
            
        
        return p, cbar

        
    def plot(self, title, index, vmax=None, cmap='GnBu'):
        fit = self._fits[title]
        mn = fit['mean'][index]
        sd = fit['mean'].std(0, ddof=1)
        fq = fit['freq'][index] / fit['freq'][index].mean()
        ii = fit['istd'][index]
        member = index + 1
        
        if vmax is None:
            vmax = mn.max()
        
        plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title), fontsize=14)
        
        plt.subplot(221)
        p, cbar = self._plot1panel(mn, 'Mean Member #%s' % member, vmin=0, vmax=vmax, cmap=cmap)
        
        plt.subplot(222)
        p, cbar = self._plot1panel(sd, 'Ensemble st dev of mean', vmin=0, vmax=vmax, cmap=cmap)

        plt.subplot(223)
        p, cbar = self._plot1panel(fq, 'Frequency Member #%s\n(normalised by mean frequency)' % member, vmin=0, vmax=8, cmap=cmap)

        plt.subplot(224)
        p, cbar = self._plot1panel(ii, 'Within bin spread Member #%s' % member, vmin=0, vmax=vmax, cmap=cmap)

        
        
    def plot_ensemble(self, title, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
                        
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'GnBu'
            
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
        fit = self._fits[title]
        mn = fit['mean'][slicer].mean(0)
        sd = fit['mean'][slicer].std(0, ddof=1)
        fq = fit['freq'][slicer].mean(0) / fit['freq'].mean()
        ii = fit['istd'][slicer].mean(0)
        
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mn.max()
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n = mn.shape[0]
                dof = n - 1
                sig = mn / (sd / numpy.sqrt(n))
                mn = numpy.ma.masked_where(sig < scipy.stats.t.ppf(.95, dof), mn)
#                 pdb.set_trace()

        pdf_kwargs = kwargs.copy()
        pdf_kwargs['vmax'] = 8.0
                  
        plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title), fontsize=14)
        
        plt.subplot(221)
        p, cbar = self._plot1panel(mn, '%s mean' % prefix, **kwargs)
        
        plt.subplot(222)
        p, cbar = self._plot1panel(sd, '%s st dev of mean' % prefix, **kwargs)

        plt.subplot(223)
        p, cbar = self._plot1panel(fq, '%s Mean frequency' % prefix, **pdf_kwargs)

        plt.subplot(224)
        p, cbar = self._plot1panel(ii, '%s mean within bin spread' % prefix, **kwargs)



    def plot_ensemble_change(self, title_cntl, title_futr, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
            

        fit_cntl = self._fits[title_cntl]
        fit_futr = self._fits[title_futr]
        mn = fit_futr['mean'][slicer].mean(0) - fit_cntl['mean'][slicer].mean(0)
        sd = fit_futr['mean'][slicer].std(0) / fit_cntl['mean'][slicer].std(0)
#         fq = (fit_futr['freq'].mean(0) / fit_futr['freq'].sum()) - (fit_cntl['freq'].mean(0) / fit_cntl['freq'].sum())
        fq = numpy.ma.masked_invalid(fit_futr['pdf'][slicer].mean(0) / fit_cntl['pdf'][slicer].mean(0))
        ii = fit_futr['istd'][slicer].mean(0) / fit_cntl['istd'][slicer].mean(0)
        
        sd = numpy.ma.masked_invalid(sd)
        fq = numpy.ma.masked_invalid(fq)
        ii = numpy.ma.masked_invalid(ii)
        
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n1 = fit_cntl['mean'][slicer].shape[0]
                n2 = fit_futr['mean'][slicer].shape[0]
                sd1 = fit_cntl['mean'][slicer].std(0)
                sd2 = fit_futr['mean'][slicer].std(0)
                denom = numpy.sqrt(sd1**2 / n1 + sd2**2 / n2)
                dof = denom**4 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1))
#                 print "dof = ", dof
                sig = mn / denom
                pvalue = scipy.stats.mstats.ttest_ind(fit_cntl['mean'][slicer], fit_futr['mean'][slicer])
                mn = numpy.ma.masked_where(pvalue < 0.05, mn)
                
#                 raise Exception('stop')

        
#         plt.suptitle('%s minus %s' % (title_futr, title_cntl), fontsize=14)
        suptitl_params = (fit_cntl['name'], self._xlabel, self._ylabel, title_futr, title_cntl)
        suptitl = 'Change in %s dependence on %s and %s\n%s minus %s' % suptitl_params
        plt.suptitle(suptitl, fontsize=14)

        lo = min([mn.min(), mn.max(), -mn.min(), -mn.max()])
        hi = max([mn.min(), mn.max(), -mn.min(), -mn.max()])
        plt.subplot(221)
        p, cbar = self._plot1panel(mn, 'Ensemble mean change', vmin=lo, vmax=hi, cmap='RdBu')
        
        plt.subplot(222)
        p, cbar = self._plot_ratio(sd, 'Ratio Ensemble st dev of mean', cmap='PuOr')

        plt.subplot(223)
        p, cbar = self._plot_ratio(fq, 'Ratio Ensemble Mean frequency', cmap='PuOr')

        plt.subplot(224)
        p, cbar = self._plot_ratio(ii, 'Ratio Ensemble mean within bin spread', cmap='PuOr')
        

    def plot_ensemble_base_and_change(self, title_cntl, title_futr, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
                       
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'GnBu'
            
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
# baseline first column
        fit = self._fits[title_cntl]
        mn = fit['mean'][slicer].mean(0)
        sd = fit['mean'][slicer].std(0, ddof=1)
        fq = fit['freq'][slicer].mean(0) / fit['freq'].mean()
        ii = fit['istd'][slicer].mean(0)
        
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mn.max()
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n = mn.shape[0]
                dof = n - 1
                sig = mn / (sd / numpy.sqrt(n))
                mn = numpy.ma.masked_where(sig < scipy.stats.t.ppf(.95, dof), mn)
#                 pdb.set_trace()

        pdf_kwargs = kwargs.copy()
        pdf_kwargs['vmax'] = 8.0
                  
#         plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title_cntl), fontsize=14)
        
        plt.subplot(231)
        p, cbar = self._plot1panel(mn, '%s mean\n%s' % (prefix, title_cntl), **kwargs)
        

        plt.subplot(234)
        pdf_kwargs2 = pdf_kwargs.copy()
        pdf_kwargs2['cmap'] = 'GnBu'
        p, cbar = self._plot1panel(fq, '%s Mean frequency\n%s' % (prefix, title_cntl), **pdf_kwargs2)




# now do             

        fit_cntl = self._fits[title_cntl]
        fit_futr = self._fits[title_futr]
        mn = fit_futr['mean'][slicer].mean(0) - fit_cntl['mean'][slicer].mean(0)
        sd = fit_futr['mean'][slicer].std(0) / fit_cntl['mean'][slicer].std(0)
#         fq = (fit_futr['freq'].mean(0) / fit_futr['freq'].sum()) - (fit_cntl['freq'].mean(0) / fit_cntl['freq'].sum())
        fq = numpy.ma.masked_invalid(fit_futr['pdf'][slicer].mean(0) / fit_cntl['pdf'][slicer].mean(0))
        ii = fit_futr['istd'][slicer].mean(0) / fit_cntl['istd'][slicer].mean(0)
        
        sd = numpy.ma.masked_invalid(sd)
        fq = numpy.ma.masked_invalid(fq)
        ii = numpy.ma.masked_invalid(ii)
        mn_fut = fit_futr['mean'][slicer].mean(0)
        fq_fut = fit_futr['freq'][slicer].mean(0) / fit_futr['freq'].mean()
        
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n1 = fit_cntl['mean'][slicer].shape[0]
                n2 = fit_futr['mean'][slicer].shape[0]
                sd1 = fit_cntl['mean'][slicer].std(0)
                sd2 = fit_futr['mean'][slicer].std(0)
                denom = numpy.sqrt(sd1**2 / n1 + sd2**2 / n2)
                dof = denom**4 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1))
#                 print "dof = ", dof
                sig = mn / denom
                pvalue = scipy.stats.mstats.ttest_ind(fit_cntl['mean'][slicer], fit_futr['mean'][slicer])
                mn = numpy.ma.masked_where(pvalue < 0.05, mn)
                
#                 raise Exception('stop')

        
#         plt.suptitle('%s minus %s' % (title_futr, title_cntl), fontsize=14)
        suptitl_params = (fit_cntl['name'], self._xlabel, self._ylabel, title_futr, title_cntl)
        suptitl = 'Change in %s dependence on %s and %s\n%s minus %s' % suptitl_params
        plt.suptitle(suptitl, fontsize=14)

        lo = min([mn.min(), mn.max(), -mn.min(), -mn.max()])
        hi = max([mn.min(), mn.max(), -mn.min(), -mn.max()])
        plt.subplot(233)
        p, cbar = self._plot1panel(mn, 'Ensemble mean change', vmin=lo, vmax=hi, cmap=kwargs['cmap'])
        

        plt.subplot(236)
        p, cbar = self._plot_ratio(fq, 'Ratio Ensemble Mean frequency', cmap='PuOr')


        plt.subplot(232)
        p, cbar = self._plot1panel(mn_fut, '%s mean\n%s' % (prefix, title_futr), **kwargs)
        

        plt.subplot(235)
        p, cbar = self._plot1panel(fq_fut, '%s Mean frequency\n%s' % (prefix, title_futr), **pdf_kwargs2)
        

    def plot_ensemble_intensity_base_and_change(self, title_cntl, title_futr, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
                       
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'GnBu'
            
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
        if 'pdf_threshold' in kwargs:
            pdf_threshold = kwargs['pdf_threshold']
            del kwargs['pdf_threshold']
        else:
            pdf_threshold = 0.0
            
        lo = kwargs.get('lo', None)
        hi = kwargs.get('hi', None)
            
        ipanel = kwargs.get('ipanel', 0)
        ncol = kwargs.get('ncol', 3)
        nrow = kwargs.get('nrow', 1)
        
        colorbar = kwargs.get('colorbar', True)
        xlabel = kwargs.get('xlabel', True)
        ylabel = kwargs.get('ylabel', True)
        
        titl = kwargs.get('title', '')
        full_title = kwargs.get('full_title', True)
        
        cmap_change = kwargs.get('cmap_change', 'PuOr')
        
        for lose in ['ipanel', 'ncol', 'nrow', 'lo', 'hi', 'colorbar', 'xlabel', 'ylabel', 'title', 'full_title', 'cmap_change']:
            if lose in kwargs:
                del kwargs[lose]
                
                
        return_val = list()
        
            
# baseline first column
        fit = self._fits[title_cntl]
        mn = fit['mean'][slicer].mean(0)
        sd = fit['mean'][slicer].std(0, ddof=1)
        fq = fit['freq'][slicer].mean(0) / fit['freq'].mean()
        ii = fit['istd'][slicer].mean(0)
        
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mn.max()
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n = mn.shape[0]
                dof = n - 1
                sig = mn / (sd / numpy.sqrt(n))
                mn = numpy.ma.masked_where(sig < scipy.stats.t.ppf(.95, dof), mn)
#                 pdb.set_trace()

        pdf_kwargs = kwargs.copy()
        pdf_kwargs['vmax'] = 8.0
                  
#         plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title_cntl), fontsize=14)
        
        plt.subplot(nrow, ncol, ipanel)
        print((nrow, ncol, ipanel))
        if full_title:
            ftitle = '%s mean\n%s' % (prefix, title_cntl)
            if titl:
                ftitle += '\n' + titl
        else:
            ftitle = titl
        p, cbar = self._plot1panel(mn, ftitle, 
                                   ylabel=True, colorbar=colorbar, xlabel=xlabel,
                                   **kwargs)
        return_val.append((p, cbar))
        
        fit_cntl = self._fits[title_cntl]
        fit_futr = self._fits[title_futr]
        mn = fit_futr['mean'][slicer].mean(0) - fit_cntl['mean'][slicer].mean(0)
        sd = fit_futr['mean'][slicer].std(0) / fit_cntl['mean'][slicer].std(0)
#         fq = (fit_futr['freq'].mean(0) / fit_futr['freq'].sum()) - (fit_cntl['freq'].mean(0) / fit_cntl['freq'].sum())
        fq = numpy.ma.masked_invalid(fit_futr['pdf'][slicer].mean(0) / fit_cntl['pdf'][slicer].mean(0))
        ii = fit_futr['istd'][slicer].mean(0) / fit_cntl['istd'][slicer].mean(0)
        
        sd = numpy.ma.masked_invalid(sd)
        fq = numpy.ma.masked_invalid(fq)
        ii = numpy.ma.masked_invalid(ii)
        mn_fut = fit_futr['mean'][slicer].mean(0)
        fq_fut = fit_futr['freq'][slicer].mean(0) / fit_futr['freq'].mean()
        
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n1 = fit_cntl['mean'][slicer].shape[0]
                n2 = fit_futr['mean'][slicer].shape[0]
                sd1 = fit_cntl['mean'][slicer].std(0)
                sd2 = fit_futr['mean'][slicer].std(0)
                denom = numpy.sqrt(sd1**2 / n1 + sd2**2 / n2)
                dof = denom**4 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1))
#                 print "dof = ", dof
                sig = mn / denom
                pvalue = scipy.stats.mstats.ttest_ind(fit_cntl['mean'][slicer], fit_futr['mean'][slicer])
                mn = numpy.ma.masked_where(pvalue < 0.05, mn)
                
#                 raise Exception('stop')

        
#         plt.suptitle('%s minus %s' % (title_futr, title_cntl), fontsize=14)
#         suptitl_params = (fit_cntl['name'], self._xlabel, self._ylabel, title_futr, title_cntl)
#         suptitl = 'Change in %s dependence on %s and %s\n%s minus %s' % suptitl_params
#         plt.suptitle(suptitl, fontsize=matplotlib.rcParams['font.size'] + 3

        plt.subplot(nrow, ncol, ipanel + 1)
        print((nrow, ncol, ipanel + 1))

        if full_title:
            ftitle = '%s mean\n%s' % (prefix, title_futr)
            if titl:
                ftitle += '\n' + titl
        else:
            ftitle = titl
        p, cbar = self._plot1panel(mn_fut, ftitle,
                                   xlabel=xlabel, colorbar=colorbar, ylabel=False, 
                                   **kwargs)
        return_val.append((p, cbar))

        if lo is None:
            lo = min([mn.min(), mn.max(), -mn.min(), -mn.max()])
        if hi is None:
            hi = max([mn.min(), mn.max(), -mn.min(), -mn.max()])
        print((nrow, ncol, ipanel + 2))
        plt.subplot(nrow, ncol, ipanel + 2)
        if full_title:
            ftitle = 'Ensemble mean change'
            if titl:
                ftitle += '\n' + titl
        else:
            ftitle = titl
        not_enough = (fit_futr['pdf'][slicer].mean(0) < pdf_threshold) | (fit_cntl['pdf'][slicer].mean(0) < pdf_threshold)
        mn = numpy.ma.array(mn, mask=numpy.ma.getmaskarray(mn) | not_enough)
        p, cbar = self._plot1panel(mn, ftitle,
                                   vmin=lo, vmax=hi, cmap=cmap_change, 
                                   xlabel=xlabel, colorbar=colorbar, ylabel=False)
        
        return_val.append((p, cbar)) 

        return(return_val)

    def plot_ensemble_frequency_base_and_change2(self, title_cntl, title_futr, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
                       
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'GnBu'
            
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
        if 'orientation' not in kwargs:
            orientation = 'horizontal'
            
        if 'diff_levels' in kwargs:
            levels = kwargs['diff_levels']
            del kwargs['diff_levels']
        else:
            levels = None
            
# baseline first column
        fit = self._fits[title_cntl]
        mn = fit['mean'][slicer].mean(0)
        sd = fit['mean'][slicer].std(0, ddof=1)
        fq = fit['freq'][slicer].mean(0) / fit['freq'].mean()
        ii = fit['istd'][slicer].mean(0)
        
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mn.max()
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n = mn.shape[0]
                dof = n - 1
                sig = mn / (sd / numpy.sqrt(n))
                mn = numpy.ma.masked_where(sig < scipy.stats.t.ppf(.95, dof), mn)
#                 pdb.set_trace()

        pdf_kwargs = kwargs.copy()
        pdf_kwargs['vmax'] = 8.0
                  
#         plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title_cntl), fontsize=14)
        

        plt.subplot(221)
        pdf_kwargs2 = pdf_kwargs.copy()
        pdf_kwargs2['cmap'] = 'GnBu'
        p1, cbar1 = self._plot1panel(fq, '%s Mean frequency\n%s' % (prefix, title_cntl), **pdf_kwargs2)
            

        fit_cntl = self._fits[title_cntl]
        fit_futr = self._fits[title_futr]
        mn = fit_futr['mean'][slicer].mean(0) - fit_cntl['mean'][slicer].mean(0)
        sd = fit_futr['mean'][slicer].std(0) / fit_cntl['mean'][slicer].std(0)
#         fq = (fit_futr['freq'].mean(0) / fit_futr['freq'].sum()) - (fit_cntl['freq'].mean(0) / fit_cntl['freq'].sum())
        fq = numpy.ma.masked_invalid(fit_futr['pdf'][slicer].mean(0) / fit_cntl['pdf'][slicer].mean(0))
        ii = fit_futr['istd'][slicer].mean(0) / fit_cntl['istd'][slicer].mean(0)
        
        sd = numpy.ma.masked_invalid(sd)
        fq = numpy.ma.masked_invalid(fq)
        ii = numpy.ma.masked_invalid(ii)
        mn_fut = fit_futr['mean'][slicer].mean(0)
        fq_fut = fit_futr['freq'][slicer].mean(0) / fit_futr['freq'].mean()
        
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n1 = fit_cntl['mean'][slicer].shape[0]
                n2 = fit_futr['mean'][slicer].shape[0]
                sd1 = fit_cntl['mean'][slicer].std(0)
                sd2 = fit_futr['mean'][slicer].std(0)
                denom = numpy.sqrt(sd1**2 / n1 + sd2**2 / n2)
                dof = denom**4 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1))
#                 print "dof = ", dof
                sig = mn / denom
                pvalue = scipy.stats.mstats.ttest_ind(fit_cntl['mean'][slicer], fit_futr['mean'][slicer])
                mn = numpy.ma.masked_where(pvalue < 0.05, mn)
                
#                 raise Exception('stop')

        
#         plt.suptitle('%s minus %s' % (title_futr, title_cntl), fontsize=14)
        suptitl_params = (self._xlabel, self._ylabel, title_futr, title_cntl)
        suptitl = 'Change in frequency on %s and %s\n%s minus %s' % suptitl_params
        plt.suptitle(suptitl, fontsize=matplotlib.rcParams['font.size'] + 3)

        lo = min([mn.min(), mn.max(), -mn.min(), -mn.max()])
        hi = max([mn.min(), mn.max(), -mn.min(), -mn.max()])
        

        plt.subplot(223)
        p3, cbar3 = self._plot_ratio(fq, 'Ratio Ensemble Mean frequency', tickv=[.33, .5, 1., 2, 3], 
                                     cmap='PuOr', orientation=kwargs['orientation'])        

        plt.subplot(222)
        p2, cbar2 = self._plot1panel(fq_fut, '%s Mean frequency\n%s' % (prefix, title_futr), **pdf_kwargs2)
        
        plt.subplot(224)
        print(('freq check', fq_fut.sum(), fq.sum()))
        diff = fit_futr['pdf'][slicer].mean(0) - fit_cntl['pdf'][slicer].mean(0)
        if levels is None:
            levels = niceLevels3(diff.ravel(), 11, centre=True)
        p4, cbar4 = self._plot1panel(diff, 'Difference Ensemble Mean frequency',
                                     cmap='PuOr', orientation=kwargs['orientation'],
                                     vmin=levels.min(), vmax=levels.max())
        
        return [p1, p2, p3, p4], [cbar1, cbar2, cbar3, cbar4]


    def plot_ensemble_pdf_base_and_change2(self, title_cntl, title_futr, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
                       
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'GnBu'
            
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
        if 'orientation' not in kwargs:
            orientation = 'horizontal'
            
        if 'diff_levels' in kwargs:
            levels = kwargs['diff_levels']
            del kwargs['diff_levels']
        else:
            levels = None
            
# baseline first column
        fit = self._fits[title_cntl]
        mn = fit['mean'][slicer].mean(0)
        sd = fit['mean'][slicer].std(0, ddof=1)
        fq = fit['freq'][slicer].mean(0) / fit['freq'].mean()
        ii = fit['istd'][slicer].mean(0)
        
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mn.max()
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n = mn.shape[0]
                dof = n - 1
                sig = mn / (sd / numpy.sqrt(n))
                mn = numpy.ma.masked_where(sig < scipy.stats.t.ppf(.95, dof), mn)
#                 pdb.set_trace()

        pdf_kwargs = kwargs.copy()
        pdf_kwargs['vmax'] = 0.025
                  
#         plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title_cntl), fontsize=14)
        
        fit_cntl = self._fits[title_cntl]
        fit_futr = self._fits[title_futr]

        plt.subplot(221)
        pdf_ctl = fit_cntl['pdf'][slicer].mean(0)
        pdf_kwargs2 = pdf_kwargs.copy()
        pdf_kwargs2['cmap'] = 'GnBu'
        p1, cbar1 = self._plot1panel(pdf_ctl, '%s Mean PDF\n%s' % (prefix, title_cntl), **pdf_kwargs2)
            

        mn = fit_futr['mean'][slicer].mean(0) - fit_cntl['mean'][slicer].mean(0)
        sd = fit_futr['mean'][slicer].std(0) / fit_cntl['mean'][slicer].std(0)
#         fq = (fit_futr['freq'].mean(0) / fit_futr['freq'].sum()) - (fit_cntl['freq'].mean(0) / fit_cntl['freq'].sum())
        pdf_ratio = numpy.ma.masked_invalid(fit_futr['pdf'][slicer].mean(0) / fit_cntl['pdf'][slicer].mean(0))
        pdf_diff = fit_futr['pdf'][slicer].mean(0) - fit_cntl['pdf'][slicer].mean(0)
        ii = fit_futr['istd'][slicer].mean(0) / fit_cntl['istd'][slicer].mean(0)
        
        sd = numpy.ma.masked_invalid(sd)
        fq = numpy.ma.masked_invalid(fq)
        ii = numpy.ma.masked_invalid(ii)
        mn_fut = fit_futr['mean'][slicer].mean(0)
        pdf_fut = fit_futr['pdf'][slicer].mean(0) 
        
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n1 = fit_cntl['mean'][slicer].shape[0]
                n2 = fit_futr['mean'][slicer].shape[0]
                sd1 = fit_cntl['mean'][slicer].std(0)
                sd2 = fit_futr['mean'][slicer].std(0)
                denom = numpy.sqrt(sd1**2 / n1 + sd2**2 / n2)
                dof = denom**4 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1))
#                 print "dof = ", dof
                sig = mn / denom
                pvalue = scipy.stats.mstats.ttest_ind(fit_cntl['mean'][slicer], fit_futr['mean'][slicer])
                mn = numpy.ma.masked_where(pvalue < 0.05, mn)
                
#                 raise Exception('stop')

        
#         plt.suptitle('%s minus %s' % (title_futr, title_cntl), fontsize=14)
        suptitl_params = (self._xlabel, self._ylabel, title_futr, title_cntl)
        suptitl = 'Change in PDF on %s and %s\n%s minus %s' % suptitl_params
        plt.suptitle(suptitl, fontsize=matplotlib.rcParams['font.size'] + 3)

        lo = min([mn.min(), mn.max(), -mn.min(), -mn.max()])
        hi = max([mn.min(), mn.max(), -mn.min(), -mn.max()])
        

        plt.subplot(223)
#         pdb.set_trace()
#         p3, cbar3 = self._plot_ratio(pdf_ratio, 'Ensemble Mean PDF\nRatio', tickv=[.33, .5, 1., 2, 3], 
#                                      cmap='PuOr_r', orientation=kwargs['orientation'])        
        print('Plot ratio')
        pdf_ratio = numpy.ma.masked_equal(pdf_ratio, 0.0)
        print(pdf_ratio.min())
        p3, cbar3 = self._plot_ratio(pdf_ratio, 'Ensemble Mean PDF\nRatio', tickv=[.5, 1., 2], 
                                     cmap='PuOr_r', orientation=kwargs['orientation']) 
        print('Plot ratio done')       

        plt.subplot(222)
        p2, cbar2 = self._plot1panel(pdf_fut, '%s Mean PDF\n%s' % (prefix, title_futr), **pdf_kwargs2)
        
        plt.subplot(224)
        print(('pdf check', pdf_ctl.sum(), pdf_fut.sum()))
        if levels is None:
            levels = niceLevels3(pdf_diff.ravel() * 1000, 11, centre=True) / 1000
        p4, cbar4 = self._plot1panel(pdf_diff, 'Ensemble Mean PDF\nDifference',
                                     cmap='PuOr_r', orientation=kwargs['orientation'],
                                     vmin=levels.min(), vmax=levels.max())
        
        return [p1, p2, p3, p4], [cbar1, cbar2, cbar3, cbar4]



    def plot_ensemble_frequency_base_and_change(self, title_cntl, title_futr, **kwargs):
        if 'index' in kwargs:
            slicer = kwargs['index']
            del kwargs['index']
            prefix = 'Sub-ensemble'
        else:
            slicer = slice(None)
            prefix = 'Ensemble'
                       
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'GnBu'
            
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
            
# baseline first column
        fit = self._fits[title_cntl]
        mn = fit['mean'][slicer].mean(0)
        sd = fit['mean'][slicer].std(0, ddof=1)
        fq = fit['freq'][slicer].mean(0) / fit['freq'].mean()
        ii = fit['istd'][slicer].mean(0)
        
        if 'vmax' not in kwargs:
            kwargs['vmax'] = mn.max()
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n = mn.shape[0]
                dof = n - 1
                sig = mn / (sd / numpy.sqrt(n))
                mn = numpy.ma.masked_where(sig < scipy.stats.t.ppf(.95, dof), mn)
#                 pdb.set_trace()

        pdf_kwargs = kwargs.copy()
        pdf_kwargs['vmax'] = 8.0
                  
#         plt.suptitle('%s dependence on %s and %s\n%s' % (fit['name'], self._xlabel, self._ylabel, title_cntl), fontsize=14)
        

        plt.subplot(131)
        pdf_kwargs2 = pdf_kwargs.copy()
        pdf_kwargs2['cmap'] = 'GnBu'
        p, cbar = self._plot1panel(fq, '%s Mean frequency\n%s' % (prefix, title_cntl), **pdf_kwargs2)
            

        fit_cntl = self._fits[title_cntl]
        fit_futr = self._fits[title_futr]
        mn = fit_futr['mean'][slicer].mean(0) - fit_cntl['mean'][slicer].mean(0)
        sd = fit_futr['mean'][slicer].std(0) / fit_cntl['mean'][slicer].std(0)
#         fq = (fit_futr['freq'].mean(0) / fit_futr['freq'].sum()) - (fit_cntl['freq'].mean(0) / fit_cntl['freq'].sum())
        fq = numpy.ma.masked_invalid(fit_futr['pdf'][slicer].mean(0) / fit_cntl['pdf'][slicer].mean(0))
        ii = fit_futr['istd'][slicer].mean(0) / fit_cntl['istd'][slicer].mean(0)
        
        sd = numpy.ma.masked_invalid(sd)
        fq = numpy.ma.masked_invalid(fq)
        ii = numpy.ma.masked_invalid(ii)
        mn_fut = fit_futr['mean'][slicer].mean(0)
        fq_fut = fit_futr['freq'][slicer].mean(0) / fit_futr['freq'].mean()
        
        
        if 'mask_nonsignificant' in kwargs:
            if kwargs['mask_nonsignificant']:
                del kwargs['mask_nonsignificant']
                n1 = fit_cntl['mean'][slicer].shape[0]
                n2 = fit_futr['mean'][slicer].shape[0]
                sd1 = fit_cntl['mean'][slicer].std(0)
                sd2 = fit_futr['mean'][slicer].std(0)
                denom = numpy.sqrt(sd1**2 / n1 + sd2**2 / n2)
                dof = denom**4 / ((sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1))
#                 print "dof = ", dof
                sig = mn / denom
                pvalue = scipy.stats.mstats.ttest_ind(fit_cntl['mean'][slicer], fit_futr['mean'][slicer])
                mn = numpy.ma.masked_where(pvalue < 0.05, mn)
                
#                 raise Exception('stop')

        
#         plt.suptitle('%s minus %s' % (title_futr, title_cntl), fontsize=14)
        suptitl_params = (self._xlabel, self._ylabel, title_futr, title_cntl)
        suptitl = 'Change in frequency on %s and %s\n%s minus %s' % suptitl_params
        plt.suptitle(suptitl, fontsize=matplotlib.rcParams['font.size'] + 3)

        lo = min([mn.min(), mn.max(), -mn.min(), -mn.max()])
        hi = max([mn.min(), mn.max(), -mn.min(), -mn.max()])
        

        plt.subplot(133)
        p, cbar = self._plot_ratio(fq, 'Ratio Ensemble Mean frequency', cmap='PuOr')        

        plt.subplot(132)
        p, cbar = self._plot1panel(fq_fut, '%s Mean frequency\n%s' % (prefix, title_futr), **pdf_kwargs2)







    def diagnose_change(self, title_cntl, title_futr):
        def weighted_mean(data, wts):
            return (data * wts).reshape((data.shape[0], -1)).sum(-1)

        fitc = self._fits[title_cntl]
        fitf = self._fits[title_futr]
        data_cntl, wts_cntl, data_futr, wts_futr = fitc['mean'], fitc['pdf'], fitf['mean'], fitf['pdf']
        
        dc = data_cntl.filled(0.0)
        df = data_futr.filled(0.0)
        cntl = weighted_mean(dc, wts_cntl)
        futr = weighted_mean(df, wts_futr)
        delta = futr - cntl
        wts_delta = wts_futr - wts_cntl
        cntl_times_deltaw = weighted_mean(dc, wts_delta)
        delta_times_cntlw = weighted_mean(df - dc, wts_cntl)
        cross_term = weighted_mean(df - dc, wts_delta)
        return delta, cntl_times_deltaw, delta_times_cntlw, cross_term

    def plot_diagnosed_change(self, title_cntl, title_futr):
        delta, cntl_times_deltaw, delta_times_cntlw, cross_term = self.diagnose_change(title_cntl, title_futr)
        srt = numpy.argsort(delta)

        p1 = plt.plot(delta[srt], label='$\Delta$')
        plt.plot(cntl_times_deltaw[srt], label='$I.{\Delta}w$')
        plt.plot(delta_times_cntlw[srt], label='${\Delta}I.w$')
        plt.plot(cross_term[srt], label='${\Delta}I.{\Delta}w$')
        plt.axhline(y=0, linestyle='--', linewidth=2, color='k')
        plt.legend(loc='best')
        xstr = self._fits[title_cntl]['xstr']
        if xstr is not None:
            plt.xticks(numpy.arange(len(xstr)), xstr, rotation=30)


        return p1
    
    def barplot_diagnosed_change(self, title_cntl, title_futr):
        delta, cntl_times_deltaw, delta_times_cntlw, cross_term = bin2d.diagnose_change(base_str, anom_str)
        srt = numpy.argsort(delta)

        ind = numpy.arange(1, delta.shape[0] + 1)
        plt.bar(ind - 0.4, delta[srt], width=0.15, color='b', label='${\Delta}$')
        plt.bar(ind - 0.2, cntl_times_deltaw[srt], width=0.15, color='g', label='$I.{\Delta}w$')

        plt.bar(ind, delta_times_cntlw[srt], width=0.15, color='r', label='${\Delta}I.w$')
        plt.bar(ind + 0.2, cross_term[srt], width=0.15, color='cyan', label='${\Delta}I.{\Delta}w$')

        plt.legend(loc='best')
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k')
        plt.axhline(y=delta.mean(), linewidth=2, linestyle='--', color='LightGray')

        # plt.axvline(x=0, linewidth=2, linestyle='--', color='k')
        plt.xlabel('Ensemble member ranked by delta')
        plt.ylabel('Component (mm day-1)')
        xstr = self._fits[title_cntl]['xstr']
        if xstr is not None:
            xt = plt.xticks(1 + numpy.arange(len(xstr)), numpy.array(xstr)[srt], rotation=30)

        return plt.gca()

        
    def get_ensemble_freq_spread(self, title):
        pass
        
    def compare_freq_spread(self, title1, title2, reference_spread):
        pass
        
    def plot_diagnosed_change_by_bin(self, title_cntl, title_futr,
                                     cmap='RdBu', vmax=None, orientation='horizontal'):
        fitc = self._fits[title_cntl]
        fitf = self._fits[title_futr]

        delta_grid = (fitf['mean'].mean(0) * fitf['pdf'].mean(0)) - (fitc['mean'].mean(0) * fitc['pdf'].mean(0))
        cntl_times_deltaw_grid = fitc['mean'].mean(0) * (fitf['pdf'].mean(0) - fitc['pdf'].mean(0))
        delta_times_cntlw_grid = (fitf['mean'].mean(0) - fitc['mean'].mean(0)) * fitc['pdf'].mean(0)
        cross_terms_grid = (fitf['mean'].mean(0) - fitc['mean'].mean(0)) * (fitf['pdf'].mean(0) - fitc['pdf'].mean(0))
        check = delta_grid - cntl_times_deltaw_grid - delta_times_cntlw_grid - cross_terms_grid
    
        xx, yy = numpy.meshgrid(self._xedges, self._yedges)

#         xlabel = sst4k_jets.extract_strict('jet strength')
#         ylabel = sst4k_jets.extract_strict('jet latitude')

        if vmax is None:
            vmax = numpy.abs(numpy.concatenate([delta_grid.ravel(), cntl_times_deltaw_grid.ravel(), delta_times_cntlw_grid.ravel()])).max()
    #         print vmax
            vmax = numpy.max(numpy.abs([delta_grid.min(), delta_grid.max(), \
                                        cntl_times_deltaw_grid.min(), cntl_times_deltaw_grid.max(), \
                                        delta_times_cntlw_grid.min(), delta_times_cntlw_grid.max()]))
            vdata = numpy.ma.abs(numpy.ma.concatenate([delta_grid.ravel(), cntl_times_deltaw_grid.ravel(), delta_times_cntlw_grid.ravel()]))
    #         pdb.set_trace()
            vdata = vdata[vdata > (vdata.max() / 1000.0)]
            vmax = scipy.stats.mstats.mquantiles(vdata, prob=[0.9, 0.95, 0.975, 0.99, 0.995])
            print(vdata.max(), vmax)
            vmax = vmax[-2]
#         print numpy.abs(numpy.concatenate([delta_grid.ravel(), cntl_times_deltaw_grid.ravel(), delta_times_cntlw_grid.ravel()]))


        plt.subplot(221)
        pl1 = plt.pcolormesh(xx, yy, numpy.ma.masked_invalid(delta_grid).T, cmap=cmap, vmin=-vmax, vmax=vmax)
        cb1 = plt.colorbar(orientation=orientation)
        plt.xlim(self._xedges.min(), self._xedges.max())
        plt.ylim(self._yedges.min(), self._yedges.max())
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title('Change ($\Delta$)\nSum = %.3f' % numpy.ma.masked_invalid(delta_grid).sum())

        plt.subplot(222)
        pl2 = plt.pcolormesh(xx, yy, numpy.ma.masked_invalid(cntl_times_deltaw_grid).T, cmap=cmap, vmin=-vmax, vmax=vmax)
        cb2 = plt.colorbar(orientation=orientation)
        plt.xlim(self._xedges.min(), self._xedges.max())
        plt.ylim(self._yedges.min(), self._yedges.max())
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title('Delta from change in frequency (I.${\Delta}w$)\nSum = %.3f' % numpy.ma.masked_invalid(cntl_times_deltaw_grid).sum())

        plt.subplot(223)
        pl3 = plt.pcolormesh(xx, yy, numpy.ma.masked_invalid(delta_times_cntlw_grid).T, cmap=cmap, vmin=-vmax, vmax=vmax)
        cb3 = plt.colorbar(orientation=orientation)
        plt.xlim(self._xedges.min(), self._xedges.max())
        plt.ylim(self._yedges.min(), self._yedges.max())
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title('Delta from change in intensity (${\Delta}I$.w)\nSum = %.3f' % numpy.ma.masked_invalid(delta_times_cntlw_grid).sum())


        plt.subplot(224)
        pl4 = plt.pcolormesh(xx, yy, numpy.ma.masked_invalid(cross_terms_grid).T, cmap=cmap, vmin=-vmax, vmax=vmax)
        cb4 = plt.colorbar(orientation=orientation)
        plt.xlim(self._xedges.min(), self._xedges.max())
        plt.ylim(self._yedges.min(), self._yedges.max())
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.title('Change from cross term (${\Delta}I$.${\Delta}w$)\nSum = %.3f' % numpy.ma.masked_invalid(cross_terms_grid).sum())
        return [pl1, pl2, pl3, pl4], [cb1, cb2, cb3, cb4]

    def compare_anomalies(self, anomalies, base_str):
        
        delta_all = list()
        ctd_all = list()
        dtc_all = list()
        ct_all = list()
        
        for anom_str in anomalies:
            delta, cntl_times_deltaw, delta_times_cntlw, cross_term = bin2d.diagnose_change(base_str, anom_str)
            delta = delta.mean()
            cntl_times_deltaw = cntl_times_deltaw.mean()
            delta_times_cntlw = delta_times_cntlw.mean()
            cross_term = cross_term.mean()
            delta_all.append(delta)
            ctd_all.append(cntl_times_deltaw)
            dtc_all.append(delta_times_cntlw)
            ct_all.append(cross_term)
            
        ind = numpy.arange(1, len(anomalies) + 1)
        plt.bar(ind - 0.4, delta_all, width=0.15, color='b', label='${\Delta}$')
        plt.bar(ind - 0.2, ctd_all, width=0.15, color='g', label='$I.{\Delta}w$')

        plt.bar(ind, dtc_all, width=0.15, color='r', label='${\Delta}I.w$')
        plt.bar(ind + 0.2, ct_all, width=0.15, color='cyan', label='${\Delta}I.{\Delta}w$')

        plt.legend(loc='best')
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k')

        # plt.axvline(x=0, linewidth=2, linestyle='--', color='k')
#         plt.xlabel('Ensemble member ranked by delta')
        plt.ylabel('Component (mm day-1)')
        xt = plt.xticks(ind, anomalies)
        
        return plt.gca()





def capitalize(s):
    return ' '.join([i.capitalize() for i in s.split('_')]).replace('Uk', 'UK')
    
def gp_smoother(y, max_length=10.0): 
    
    x = numpy.arange(y.size)
    xx = ((x - x[0]) / float(x.ptp())).reshape((-1, 1))
    ydash = butterworth(y, 10.0, high_pass=True)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xx.ravel(), y)
    residual = y - intercept - slope * xx.ravel()
    yvar = ydash.var()
    kernel = Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(0.01, max_length)) + \
             WhiteKernel(noise_level=yvar, noise_level_bounds=(yvar * 1e-5, yvar * 2))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=False)
    gp.fit(xx, residual)
    ypred = gp.predict(xx, return_std=False) + intercept + slope * xx.ravel()
#     print(gp.kernel_)
    return ypred


#===============================================================================


fcache_wts = os.path.join(cachedir, 'wts_%s.dat' % sname.lower())
fs = matplotlib.rcParams['font.size']


if not os.path.exists(fcache_wts):

    fwt = sorted(glob.glob(os.path.join(wt_dir, '*.nc')))  
    wtypes = iris.load_cube(fwt)
    add_yyyymmdd(wtypes, 'time')
    iris.coord_categorisation.add_season_year(wtypes, 'time')

    for tc in wtypes.coords(axis='t'):
        tc.bounds = None
    wtypes = wtypes.extract(iris.Constraint(month=months_dict[sname]))

    wt_ann = dict()
    for iwt in WEATHER_TYPES:
        wt_ann['wt%.1i' % iwt] = wtypes.aggregated_by('season_year', iris.analysis.COUNT, function=lambda values: values == iwt)
    
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    cacheSave([wtypes, wt_ann], fcache_wts)
else:
    wtypes, wt_ann = cacheLoad(fcache_wts)


#===============================================================================

fcache_jets = os.path.join(cachedir, 'jets_%s.nc' % sname.lower())

if not os.path.exists(fcache_jets):

    ALLOWED_RIPS = ['r001i1p00000', 'r001i1p00090', 'r001i1p00605', 'r001i1p00834',
                    'r001i1p01113', 'r001i1p01554', 'r001i1p01649', 'r001i1p01843',
                    'r001i1p01935', 'r001i1p02089', 'r001i1p02123', 'r001i1p02242',
                    'r001i1p02305', 'r001i1p02335', 'r001i1p02491', 'r001i1p02753',
                    'r001i1p02832', 'r001i1p02868', 'r001i1p02884', 'r001i1p02914']

    jet_files = sorted(glob.glob(os.path.join(jets_dir, '*.nc')))
    jet_files = [jf for jf in jet_files if extract_rip(jf) in ALLOWED_RIPS]
    jet_rips = sorted(set([extract_rip(jf) for jf in jet_files]))

    jets = iris.load_raw(jet_files)
   
    for c in jets:
        if not c.coords('month_number'):
            iris.coord_categorisation.add_month_number(c, 'time')
        add_yyyymmdd(c, 'time')


    for j in jets:
        for tc in j.coords(axis='t'):
            tc.bounds = None
    
    jets = jets.extract(iris.Constraint(month_number=months_dict[sname]))
    for c in jets:
        c.coord('yyyymmdd').points = [central_value(s, sep='|') for s in c.coord('yyyymmdd').points]

    jet_str = jets.extract('jet strength').merge_cube()
    jet_lat = jets.extract('jet latitude').merge_cube()

    jet_lat.coord('time').bounds = None
    jet_str.coord('time').bounds = None

    tidy_cubes([jet_str, jet_lat], set_var_name_None=True)

    iris.save(iris.cube.CubeList([jet_str, jet_lat]), fcache_jets)

else:
    print(('Restoring jets data from %s' % fcache_jets))
    jet_str = iris.load_cube(fcache_jets, 'jet strength')
    jet_lat = iris.load_cube(fcache_jets, 'jet latitude')
    

    


#===============================================================================

base_str = 'Baseline (1899-1950)'
# anom_str = 'Anomaly period (1960-1989)'
# anom_str = 'Anomaly period (1951-2000)'
anom_str = 'Anomaly period (2050-2099)'
jet_str_edges = numpy.linspace(0, 28, num=15)
jet_lat_edges = numpy.linspace(16, 70, num=28)


# set to True to run for first time and save output in cache. Can be set to False
# once cached.
if True:

    deltas_per_wt_dict = dict()
    deltas_per_wt_fixed_freq_dict = dict()
    deltas_per_wt_fixed_intensity_dict = dict()
    deltas_per_wt_cross_terms_dict = dict()
    deltas_per_wt_means_dict = dict()
    yr2yr_residuals_dict = dict()


    wt_intensities_dict = dict()
    pr_monthly_dict = dict()
    daily_uk_dict = dict()
    bin2d_dict = dict()

    for region in loopover:
        print("Region: %s" % region)
        f = searchstr % var
        months = months_dict[sname]
    #     cube = iris.load_cube(f, iris.Constraint(month_number=months))
    #     d = cube.data
    # 
    #     season_averager = MonthMean(sname)
    #     cube = season_averager(cube)
    #     
    #     if var.startswith('pr'):
    #         cube.data *= 86400.0
    #         cube.units = cf_units.Unit('mm day-1')
    #     elif var.startswith('ta'):
    #         cube.data -= 273.15
    #         cube.units = cf_units.Unit('deg_c')
        
    #     mask_mean = make_mask(region, cube)
    #     make_grid_match_if_close(cube, mask_mean._mask)
    #     cube = mask_mean(cube)
    
    
    #     print cube

        tags = (var, sname.lower(), region)
        fintens = os.path.join(cachedir, '%s_%s_%s_wt_intensities.nc' % tags)
        fuk_var = os.path.join(cachedir, '%s_%s_%s.nc' % tags)
        if not os.path.exists(fintens) or not os.path.exists(fuk_var):
            print("Making %s %s %s and its intensities" % (sname, region, var))
            daily_uk, wt_intensities = make_intensities(wtypes, wt_ann, var, sname, region, jet_str)
            iris.save(daily_uk, fuk_var)
            iris.save(wt_intensities, fintens)
        else:
            daily_uk = iris.load_cube(fuk_var)
            wt_intensities = iris.load_cube(fintens)
        
        common_yyyymmdd = set(jet_str.coord('yyyymmdd').points.tolist())
        common_yyyymmdd = common_yyyymmdd.intersection(daily_uk.coord('yyyymmdd').points)
        common_yyyymmdd = common_yyyymmdd.intersection(wtypes.coord('yyyymmdd').points)
        common_yyyymmdd = sorted(common_yyyymmdd)
        yyyymmdd_crit = iris.Constraint(yyyymmdd=common_yyyymmdd)
        if jet_str.coord('yyyymmdd').points.tolist() != common_yyyymmdd:
            print("Extracting jets")
            i0 = jet_str.coord('yyyymmdd').points.tolist().index(common_yyyymmdd[0])
            i1 = jet_str.coord('yyyymmdd').points.tolist().index(common_yyyymmdd[-1])
            jet_str = jet_str[i0:i1]
            jet_lat = jet_lat[i0:i1]
    #         jet_str = jet_str.extract(yyyymmdd_crit)
    #         jet_lat = jet_lat.extract(yyyymmdd_crit)
    


        pr_wt_x_intens = iris.cube.CubeList()
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            intensity = wt_intensities[ii]
            wt_ = wt_ann['wt%.1i' % iwt]
            prod = intensity.copy(data=wt_.data * intensity.data)
            pr_wt_x_intens.append(prod)
        pr_wt_x_intens = pr_wt_x_intens.merge_cube()
        pr_monthly = pr_wt_x_intens.collapsed('Weather type', iris.analysis.SUM)
    
    

    # plot to explain changes in terms of weather types
        uyears = pr_wt_x_intens.coord('season_year').points
        ind_base = numpy.where((uyears < 1951) & (uyears > 1900))[0]
        ind_anom = numpy.where(uyears >= 2050)[0]
        print(len(ind_base), len(ind_anom), pr_wt_x_intens.coord('time')[-1])

        deltas_per_wt = pr_wt_x_intens.data[...,ind_anom].mean(-1) - pr_wt_x_intens.data[...,ind_base].mean(-1)
        deltas_per_wt.shape
        iord = numpy.argsort(deltas_per_wt.sum(0))
        print(wt_intensities.coord('realization').points[iord])
        print(deltas_per_wt.sum(0)[iord])
        print(repr(deltas_per_wt.sum(0).tolist()))
        
        
        deltas_per_wt_fixed_freq = numpy.zeros_like(deltas_per_wt)
        deltas_per_wt_fixed_intensity = numpy.zeros_like(deltas_per_wt)
        deltas_per_wt_cross_terms = numpy.zeros_like(deltas_per_wt)
        deltas_per_wt_means = numpy.zeros_like(deltas_per_wt)
        yr2yr_residuals = numpy.zeros_like(deltas_per_wt)


        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            intensity = wt_intensities[ii].data #.filled(0.0)
            wt_ = wt_ann['wt%.1i' % iwt]
            baseline_freq = wt_.data[...,ind_base].mean(-1)
            future_freq = wt_.data[...,ind_anom].mean(-1)
            delta_frequency = future_freq - baseline_freq
            baseline_intensity = intensity[...,ind_base].mean(-1)
            future_intensity = intensity[...,ind_anom].mean(-1)
            delta_intensity = future_intensity - baseline_intensity
            deltas_per_wt_fixed_freq[ii] = baseline_freq * delta_intensity
            deltas_per_wt_fixed_intensity[ii] = delta_frequency * baseline_intensity
            deltas_per_wt_cross_terms[ii] = delta_frequency * delta_intensity
            deltas_per_wt_means[ii] = wt_.data[...,ind_anom].mean(-1) * intensity[...,ind_anom].mean(-1) - \
                                      wt_.data[...,ind_base].mean(-1) * intensity[...,ind_base].mean(-1)
        
            baseline_freq_anom = (wt_.data[...,ind_base].T - baseline_freq).T
            baseline_inty_anom = (intensity[...,ind_base].T - baseline_intensity).T
            future_freq_anom = (wt_.data[...,ind_anom].T - future_freq).T
            future_inty_anom = (intensity[...,ind_anom].T - future_intensity).T
    
            baseline_yr2yr = (baseline_freq_anom * baseline_inty_anom).mean(-1)
            future_yr2yr = (future_freq_anom * future_inty_anom).mean(-1)
            yr2yr_residuals[ii] = future_yr2yr - baseline_yr2yr
    
    
        print((numpy.ptp(deltas_per_wt_means - deltas_per_wt_fixed_freq - deltas_per_wt_fixed_intensity - deltas_per_wt_cross_terms)))
        print((numpy.ptp(deltas_per_wt - deltas_per_wt_fixed_freq - deltas_per_wt_fixed_intensity - deltas_per_wt_cross_terms)))
        print((numpy.ptp(deltas_per_wt - deltas_per_wt_fixed_freq - deltas_per_wt_fixed_intensity - deltas_per_wt_cross_terms - yr2yr_residuals)))

        numpy.set_printoptions(precision=4, suppress=True)


        print((numpy.array(deltas_per_wt[ii] - deltas_per_wt_means[ii]),
              numpy.array(yr2yr_residuals[ii]),
              numpy.array(deltas_per_wt[ii] - deltas_per_wt_means[ii] - yr2yr_residuals[ii])))

    
        #========================================

        new_name = '%s %s %s' % (capitalize(region), sname, var)
        daily_uk.rename(new_name)
        iris.coord_categorisation.add_season_year(daily_uk, 'time')


        bin2d = Binner2D(jet_str,
                         jet_lat,
                         xedges=jet_str_edges,
                         yedges=jet_lat_edges)
                 
        base_crit = iris.Constraint(coord_values={'season_year': lambda sy: (sy < 1951) & (sy > 1900)})
        # anom_crit = iris.Constraint(coord_values={'season_year': lambda sy: (sy >= 1951) & (sy <= 2000)})
        anom_crit2 = iris.Constraint(coord_values={'season_year': lambda sy: (sy >= 2050) & (sy < 2100)})



        # 
        # 
        pairs = [(base_str, base_crit),
                 (anom_str, anom_crit2)]

        xstr = ['%.5i' % pt for pt in daily_uk.coord('realization').points - 1100000]

        for s, crit in pairs:
            bin2d.fit(s, jet_str.extract(crit), jet_lat.extract(crit), daily_uk.extract(crit),  xstr=xstr)
    
    
        bin2d_dict[region] = deepcopy(bin2d)
        deltas_per_wt_dict[region] = deltas_per_wt.copy()
        deltas_per_wt_fixed_freq_dict[region] = deltas_per_wt_fixed_freq.copy()
        deltas_per_wt_fixed_intensity_dict[region] = deltas_per_wt_fixed_intensity.copy()
        deltas_per_wt_cross_terms_dict[region] = deltas_per_wt_cross_terms.copy()
        deltas_per_wt_means_dict[region] = deltas_per_wt_means.copy()
        yr2yr_residuals_dict[region] = yr2yr_residuals.copy()
        wt_intensities_dict[region] = wt_intensities.copy()
        pr_monthly_dict[region] = pr_monthly.copy()
        daily_uk_dict[region] = daily_uk.copy()
        
    results = [bin2d_dict, deltas_per_wt_dict, deltas_per_wt_fixed_freq_dict,
               deltas_per_wt_fixed_intensity_dict, deltas_per_wt_cross_terms_dict,
               deltas_per_wt_means_dict, wt_intensities_dict, wt_intensities_dict,
               pr_monthly_dict, daily_uk_dict, yr2yr_residuals_dict]
                   
    cacheSave(results, os.path.join(cachedir, 'wt_jets_%s_results.dat' % sname.lower()))
    
    
results = cacheLoad(os.path.join(cachedir, 'wt_jets_%s_results.dat' % sname.lower()))
bin2d_dict = results[0]
deltas_per_wt_dict = results[1]
deltas_per_wt_fixed_freq_dict = results[2]
deltas_per_wt_fixed_intensity_dict = results[3]
deltas_per_wt_cross_terms_dict = results[4]
deltas_per_wt_means_dict = results[5]
wt_intensities_dict = results[6]
wt_intensities_dict = results[7]
pr_monthly_dict = results[8]
daily_uk_dict = results[9]
yr2yr_residuals_dict = results[10]
wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
wt_labs = ['WT%.1i' % il for il in range(1, 9)]
width = 0.3


if False:
    for region in loopover:

        realns = jet_str.coord('realization').points.tolist()
        jkeep = [realns.index(k) for k in KEEP]
        
        jet_str = jet_str[jkeep]
        jet_lat = jet_lat[jkeep]

        bin2d = Binner2D(jet_str,
                         jet_lat,
                         xedges=jet_str_edges,
                         yedges=jet_lat_edges)

        base_crit = iris.Constraint(coord_values={'season_year': lambda sy: (sy < 1951) & (sy > 1900)})
        # anom_crit = iris.Constraint(coord_values={'season_year': lambda sy: (sy >= 1951) & (sy <= 2000)})
        anom_crit2 = iris.Constraint(coord_values={'season_year': lambda sy: (sy >= 2050) & (sy < 2100)})

        pairs = [(base_str, base_crit),
                 (anom_str, anom_crit2)]

        xstr = ['%.5i' % pt for pt in daily_uk_dict[region].coord('realization').points - 1100000]

        for s, crit in pairs:
            bin2d.fit(s, jet_str.extract(crit), jet_lat.extract(crit), daily_uk_dict[region].extract(crit),  xstr=xstr)
    
        bin2d_dict[region] = deepcopy(bin2d)


#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================

# Plot for RSS talk
KEEP2 = KEEP
DROP2 = DROP
realns = wt_intensities_dict[loopover[0]].coord('realization').points.tolist()
ikeep = [realns.index(k) for k in KEEP2]

# UK is Figure 10
if False:
    for region in ['uk']: #loopover:
        wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
        wt_labs = ['WT%.1i' % il for il in range(1, 9)]

        numpy.set_printoptions(precision=3, suppress=True)

        nrow, ncol=4, 1
        matplotlib.rcParams['font.size'] = 8
        fig = plt.figure(figsize=(174/25.4, 7))
        plt.subplots_adjust(bottom=0.12, hspace=0.5, top=0.85, right=0.8)
        plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)
        first = True
        ipanel = 1
        plt.subplot(nrow, ncol, ipanel)
        deltas_per_wt = deltas_per_wt_dict[region][:,ikeep]
        wt_intensities = wt_intensities_dict[region][:,ikeep,:]
        xx = numpy.arange(wt_intensities.coord('realization').shape[0])
        stacks = numpy.zeros_like(deltas_per_wt)
        if first:
            iord = numpy.argsort(deltas_per_wt.sum(0))


        for ii, io in zip(xx, iord):
            stacker(ii, deltas_per_wt[:,io], colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)
    
        ordered = wt_intensities.coord('realization').points[iord]
        if False:
            tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
            plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
            for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                ticklabel.set_color('r' if rn in DROP else 'k')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
        
        
        print('%Mean per WType')
        print((region, deltas_per_wt.mean(-1).sum(), deltas_per_wt.mean(-1)))
        print('')

        print('%variance per WType')
        print((region, numpy.cov(deltas_per_wt, ddof=0).diagonal() / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100.0))
#         print((region, numpy.cov(deltas_per_wt [:,ikeep], ddof=0).diagonal() / numpy.cov(deltas_per_wt[:,ikeep], ddof=0).diagonal().sum() * 100.0))
        print('')

    
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
        plt.title(capitalize(region).replace('Uk', 'UK'))
        plt.text(-0.75, 1.7, 'a', weight='bold')
        if region in loopover:
            plt.ylabel('%s\nchange\n[%s]' % (varname, units))
        plt.ylim(-1.05, 2.1)

        handles, labels = plt.gca().get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: t[0])))
#         if region == loopover[-1]:
#             plt.xlabel('Ensemble member')
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9
        first = False


        first = True
        ipanel = 2
        plt.subplot(nrow, ncol, ipanel)
        deltas_per_wt = deltas_per_wt_dict[region][:,ikeep]
        deltas_per_wt_fixed_freq = deltas_per_wt_fixed_freq_dict[region][:,ikeep]
        deltas_per_wt_fixed_intensity = deltas_per_wt_fixed_intensity_dict[region][:,ikeep]
        deltas_per_wt_cross_terms = deltas_per_wt_cross_terms_dict[region][:,ikeep]
        deltas_per_wt_means = deltas_per_wt_means_dict[region][:,ikeep]
        yr2yr_residuals = yr2yr_residuals_dict[region][:,ikeep]
        wt_intensities = wt_intensities_dict[region][:,ikeep,:]
        xx = numpy.arange(wt_intensities.coord('realization').shape[0])
        stacks = numpy.zeros_like(deltas_per_wt)
        wt_cols = numpy.array(['cyan', 'LightGray', 'pink', 'gold'])
        wt_labs = ['Intensity', 'Frequency', 'Cross terms', 'Residuals']



        for ii, io in zip(xx, iord):
            terms = numpy.array([deltas_per_wt_fixed_freq[:,io].sum(0), deltas_per_wt_fixed_intensity[:,io].sum(0),
                                 deltas_per_wt_cross_terms[:,io].sum(0), yr2yr_residuals[:,io].sum(0)])
            stacker(ii, terms, colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)

        ordered = wt_intensities.coord('realization').points[iord]
        if False:
            tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
            plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
            for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                ticklabel.set_color('r' if rn in DROP else 'k')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
    
        plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)

        plt.title(capitalize(region).replace('Uk', 'UK'))
        plt.text(-0.75, 1.7, 'b', weight='bold')
        if region in loopover:
            plt.ylabel('%s\nchange\n[%s]' % (varname, units))
        plt.ylim(-1.05, 2.1)

        handles, labels = plt.gca().get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: t[0])))
#         if region == loopover[-1]:
#             plt.xlabel('Ensemble member')
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), fontsize='x-small') #, ncol=9
        first = False
    
        terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                               [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                               [deltas_per_wt_cross_terms_dict[region].sum(0)],
                               [yr2yr_residuals_dict[region].sum(0)]])
        print('Four components')
        print((region, terms.mean(-1).sum(), terms.mean(-1)))
        print('')


        iteration = [(deltas_per_wt_fixed_freq_dict[region][:,ikeep], 'Change in intensity, fixed frequency'),
                     (deltas_per_wt_fixed_intensity_dict[region][:,ikeep], 'Change in frequency, fixed intensity'),
                     (deltas_per_wt_cross_terms_dict[region][:,ikeep], 'Cross terms'),
                     (yr2yr_residuals_dict[region][:,ikeep], 'Contribution from residuals')][:2]
        for iplot, ((delta_dict, titl), letter, tc) in enumerate(zip(iteration, ['c', 'd'], ['cyan', 'LightGray']), 3):
            plt.subplot(nrow, ncol, iplot)
            deltas_per_wt = delta_dict
            wt_intensities = wt_intensities_dict[region][:,ikeep,:]
            xx = numpy.arange(wt_intensities.coord('realization').shape[0])
            stacks = numpy.zeros_like(deltas_per_wt)
    #             iord = numpy.argsort(deltas_per_wt_fixed_freq_dict[region].sum(0))
            wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
            wt_labs = ['WT%.1i' % il for il in range(1, 9)]

            print('Mean')
            print((region,  titl, delta_dict.mean(-1).sum(), delta_dict.mean(-1)))
#             print((region,  titl, delta_dict[:,ikeep].mean(-1).sum(), delta_dict[:,ikeep].mean(-1)))
            print('')

            print('%variance per WType')
            print((region,  titl,numpy.cov(delta_dict, ddof=0).diagonal() / numpy.cov(delta_dict, ddof=0).diagonal().sum() * 100.0))
#             print((region,  titl,numpy.cov(delta_dict[:,ikeep], ddof=0).diagonal() / numpy.cov(delta_dict[:,ikeep], ddof=0).diagonal().sum() * 100.0))
            print('')


            for ii, io in zip(xx, iord):
                stacker(ii, deltas_per_wt[:,io], colors=wt_cols, total_color=tc, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)

            ordered = wt_intensities.coord('realization').points[iord]
            if iplot == 4:
                plt.xlabel('Ensemble member')
                tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
                plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
                for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                    ticklabel.set_color('r' if rn in DROP else 'k')
            else:
                plt.gca().xaxis.set_major_locator(plt.NullLocator())

            plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)
            plt.title(titl)
            plt.ylabel('%s\nchange\n[%s]' % (varname, units))
            plt.ylim(-1.05, 2.1)
            plt.text(-0.75, 1.7, letter, weight='bold')

            
                
    
        if loopover == 'uk':
            fig.savefig('rss_talk.png', dpi=300)
        fig.savefig(os.path.join(figpath, 'Figure_%s_%s_%s_summary.png' % (sname, var, region)), dpi=300)

#             if iplot == len(iteration):
#                 handles, labels = plt.gca().get_legend_handles_labels()
#                 # sort both labels and handles by labels
#                 labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#             if iplot == 4:
#                 plt.xlabel('Ensemble member')
#                 plt.legend(handles, labels, loc='best', ncol=5, fontsize='x-small')


#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================


if False:
# Old figure 12
# Plot
# 1. Ensemble mean contribution from WTypes and 4 components
# 2. Ensemble mean contribution from Wtypes to first 2 components
# 3. %variance explained by each weather type

    numpy.set_printoptions(precision=3, suppress=True)
    
    c4_cols = numpy.array(['cyan', 'LightGray', 'pink', 'gold'])
    c4_labs = ['Intensity', 'Frequency', 'Cross terms', 'Residuals']
    wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    wt_labs = ['WT%.1i' % il for il in range(1, 9)]
    WIDTH = 0.3
    all_labs = [str(s) for s in wt_labs + ['Total'] + c4_labs]
    YLIM = -0.5, 1.5

    nrow, ncol= 3, 2
    matplotlib.rcParams['font.size'] = 8
    fig = plt.figure(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.2, hspace=0.5, top=0.85, right=0.8, wspace=0.4)
    plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)


    ipanel = 1
    plt.subplot(nrow, ncol, ipanel)
    
    for xreg, region in enumerate(loopover, 1):
        deltas_per_wt = deltas_per_wt_dict[region]
        wt_intensities = wt_intensities_dict[region]
        terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                           [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                           [deltas_per_wt_cross_terms_dict[region].sum(0)],
                           [yr2yr_residuals_dict[region].sum(0)]])
        wt_contrib = deltas_per_wt.mean(-1)
        
        WIDTH = 0.15
        stacker(xreg - WIDTH * 0.5, deltas_per_wt.mean(-1), width=WIDTH, colors=wt_cols, labels=wt_labs if xreg == 1 else None, verbose=False, total=True)
        stacker(xreg - WIDTH * 0.5 + WIDTH, terms.mean(-1), width=WIDTH, colors=c4_cols, labels=c4_labs if xreg == 1 else None, verbose=False, total=False)

        
    xx = list(range(1, len(loopover) + 1))
    loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
#     plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
    plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
    plt.title('Ensemble mean contribution\nfrom weather types')
    plt.ylim(YLIM[0], YLIM[1])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

#     handles, labels = plt.gca().get_legend_handles_labels()
#     sort both labels and handles by labels
#     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#     plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9


    ipanel = 2
    plt.subplot(nrow, ncol, ipanel)
    
    for xreg, region in enumerate(loopover, 1):
        deltas_per_wt = deltas_per_wt_dict[region]
        wt_intensities = wt_intensities_dict[region]
        terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                           [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                           [deltas_per_wt_cross_terms_dict[region].sum(0)],
                           [yr2yr_residuals_dict[region].sum(0)]])
        wt_contrib = deltas_per_wt.mean(-1)
        
        wt_var = numpy.cov(deltas_per_wt, ddof=0).diagonal() / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
        c4_var = numpy.cov(terms, ddof=0).diagonal() / numpy.cov(terms, ddof=0).diagonal().sum() * 100
        
        WIDTH = 0.15
        stacker(xreg - WIDTH * 0.5, wt_var, width=WIDTH, colors=wt_cols, labels=wt_labs if xreg == 1 else None, verbose=False, total=True)
        stacker(xreg - WIDTH * 0.5 + WIDTH, c4_var, width=WIDTH, colors=c4_cols, labels=c4_labs if xreg == 1 else None, verbose=False, total=False)

        
    xx = list(range(1, len(loopover) + 1))
    loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
#     plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
    plt.axhline(y=100, linewidth=2, linestyle='--', color='k', zorder=-1)
    plt.title('% variance explained')
    plt.ylim(0, 110)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    handles, labels = plt.gca().get_legend_handles_labels()
#     sort both labels and handles by labels
    labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t:all_labs.index(t[0]))))
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9



    iteration = [(deltas_per_wt_fixed_freq_dict, 'Change in intensity, fixed frequency'),
                 (deltas_per_wt_fixed_intensity_dict, 'Change in frequency, fixed intensity'),
                 (deltas_per_wt_cross_terms_dict[region], 'Cross terms'),
                 (yr2yr_residuals_dict[region], 'Contribution from residuals')][:2]


    for ipanel, (delta_dict, titl), tc in zip([3, 5], iteration, ['cyan', 'LightGray']):
        plt.subplot(nrow, ncol, ipanel)
        for xreg, region in enumerate(loopover, 1):
            deltas_per_wt = delta_dict[region]
            WIDTH = 0.15
            stacker(xreg - WIDTH * 0.5, deltas_per_wt.mean(-1), width=WIDTH, 
                    colors=wt_cols, labels=wt_labs if xreg == 1 else None,
                    total_color=tc, verbose=False, total=True)

        
        xx = list(range(1, len(loopover) + 1))
        loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
        if ipanel == 5:
            plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
        plt.title('Ensemble mean contribution to\n%s' % titl)
        plt.ylim(YLIM[0], YLIM[1])
      

    for ipanel, (delta_dict, titl), tc in zip([4, 6], iteration, ['cyan', 'LightGray']):
        plt.subplot(nrow, ncol, ipanel)
        for xreg, region in enumerate(loopover, 1):
            deltas_per_wt = delta_dict[region]
            wt_var = numpy.cov(deltas_per_wt, ddof=0).diagonal() / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
            WIDTH = 0.15
            stacker(xreg - WIDTH * 0.5, wt_var, width=WIDTH, 
                    colors=wt_cols, labels=wt_labs if xreg == 1 else None,
                    total_color=tc, verbose=False, total=True)

        
        xx = list(range(1, len(loopover) + 1))
        loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
        if ipanel == 5:
            plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.axhline(y=100, linewidth=2, linestyle='--', color='k', zorder=-1)
        plt.title('%variance explained')
        plt.ylim(0, 110)
        if ipanel == 6:
            plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            
    fig.savefig(os.path.join(figpath, 'Figure_%s_%s_contributions.png' % (sname, var)), dpi=300)


          
            
#         wt_intensities = wt_intensities_dict[region]
#         xx = numpy.arange(wt_intensities.coord('realization').shape[0])
#         stacks = numpy.zeros_like(deltas_per_wt)
# #             iord = numpy.argsort(deltas_per_wt_fixed_freq_dict[region].sum(0))
#         wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
#         wt_labs = ['WT%.1i' % il for il in xrange(1, 9)]
# 
#         print('Mean')
#         print(region,  titl, delta_dict.mean(-1).sum(), delta_dict.mean(-1))
#         print(region,  titl, delta_dict[:,ikeep].mean(-1).sum(), delta_dict[:,ikeep].mean(-1))



#     first = True
#     ipanel = 2
#     plt.subplot(nrow, ncol, ipanel)
#     deltas_per_wt = deltas_per_wt_dict[region]
#     deltas_per_wt_fixed_freq = deltas_per_wt_fixed_freq_dict[region]
#     deltas_per_wt_fixed_intensity = deltas_per_wt_fixed_intensity_dict[region]
#     deltas_per_wt_cross_terms = deltas_per_wt_cross_terms_dict[region]
#     deltas_per_wt_means = deltas_per_wt_means_dict[region]
#     yr2yr_residuals = yr2yr_residuals_dict[region]
#     wt_intensities = wt_intensities_dict[region]
#     xx = numpy.arange(wt_intensities.coord('realization').shape[0])
#     stacks = numpy.zeros_like(deltas_per_wt)
#     wt_cols = numpy.array(['cyan', 'LightGray', 'pink', 'gold'])
#     wt_labs = ['Intensity', 'Frequency', 'Cross terms', 'Residuals']
# 
# 
# 
#     for ii, io in zip(xx, iord):
#         terms = numpy.array([deltas_per_wt_fixed_freq[:,io].sum(0), deltas_per_wt_fixed_intensity[:,io].sum(0),
#                              deltas_per_wt_cross_terms[:,io].sum(0), yr2yr_residuals[:,io].sum(0)])
#         stacker(ii, terms, colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)
# 
#     ordered = wt_intensities.coord('realization').points[iord]
#     if False:
#         tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
#         plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
#         for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
#             ticklabel.set_color('r' if rn in DROP else 'k')
#     else:
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
# 
#     plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)
# 
#     plt.title(capitalize(region).replace('Uk', 'UK'))
#     if region in loopover:
#         plt.ylabel('%s\nchange\n[%s]' % (varname, units))
#     plt.ylim(-1.05, 2.1)
# 
#     handles, labels = plt.gca().get_legend_handles_labels()
#     # sort both labels and handles by labels
#     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# #         if region == loopover[-1]:
# #             plt.xlabel('Ensemble member')
#     plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), fontsize='x-small') #, ncol=9
#     first = False
# 
#     terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
#                            [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
#                            [deltas_per_wt_cross_terms_dict[region].sum(0)],
#                            [yr2yr_residuals_dict[region].sum(0)]])
#     print('Four components')
#     print(region, terms.mean(-1).sum(), terms.mean(-1))
#     print('')
# 
# 
#     iteration = [(deltas_per_wt_fixed_freq_dict[region], 'Change in intensity, fixed frequency'),
#                  (deltas_per_wt_fixed_intensity_dict[region], 'Change in frequency, fixed intensity'),
#                  (deltas_per_wt_cross_terms_dict[region], 'Cross terms'),
#                  (yr2yr_residuals_dict[region], 'Contribution from residuals')][:2]
#     for iplot, ((delta_dict, titl), tc) in enumerate(zip(iteration, ['cyan', 'LightGray']), 3):
#         plt.subplot(nrow, ncol, iplot)
#         deltas_per_wt = delta_dict
#         wt_intensities = wt_intensities_dict[region]
#         xx = numpy.arange(wt_intensities.coord('realization').shape[0])
#         stacks = numpy.zeros_like(deltas_per_wt)
# #             iord = numpy.argsort(deltas_per_wt_fixed_freq_dict[region].sum(0))
#         wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
#         wt_labs = ['WT%.1i' % il for il in xrange(1, 9)]
# 
#         print('Mean')
#         print(region,  titl, delta_dict.mean(-1).sum(), delta_dict.mean(-1))
#         print(region,  titl, delta_dict[:,ikeep].mean(-1).sum(), delta_dict[:,ikeep].mean(-1))
#         print('')
# 
#         print('%variance per WType')
#         print(region,  titl,numpy.cov(delta_dict, ddof=0).diagonal() / numpy.cov(delta_dict, ddof=0).diagonal().sum() * 100.0)
#         print(region,  titl,numpy.cov(delta_dict[:,ikeep], ddof=0).diagonal() / numpy.cov(delta_dict[:,ikeep], ddof=0).diagonal().sum() * 100.0)
#         print('')
# 
# 
#         for ii, io in zip(xx, iord):
#             stacker(ii, deltas_per_wt[:,io], colors=wt_cols, total_color=tc, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)
# 
#         ordered = wt_intensities.coord('realization').points[iord]
#         if iplot == 4:
#             plt.xlabel('Ensemble member')
#             tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
#             plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
#             for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
#                 ticklabel.set_color('r' if rn in DROP else 'k')
#         else:
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
# 
#         plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)
#         plt.title(titl)
#         plt.ylabel('%s\nchange\n[%s]' % (varname, units))
#         plt.ylim(-1.05, 2.1)
        
            

#             if iplot == len(iteration):
#                 handles, labels = plt.gca().get_legend_handles_labels()
#                 # sort both labels and handles by labels
#                 labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#             if iplot == 4:
#                 plt.xlabel('Ensemble member')
#                 plt.legend(handles, labels, loc='best', ncol=5, fontsize='x-small')



if True:
# Figure 4
# Plot
# 1. Ensemble mean contribution from WTypes and 4 components
# 2. Ensemble mean contribution from Wtypes to first 2 components
# 3. %variance explained by each weather type

    numpy.set_printoptions(precision=3, suppress=True)
    
    c4_cols = numpy.array(['cyan', 'LightGray', 'pink', 'gold'])
    c4_labs = ['Intensity', 'Frequency', 'Cross terms', 'Residuals']
    wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    wt_labs = ['WT%.1i' % il for il in range(1, 9)]
    WIDTH = 0.3
    all_labs = [str(s) for s in wt_labs + ['Total'] + c4_labs]
    YLIM = -0.5, 1.5

    nrow, ncol= 3, 2
    matplotlib.rcParams['font.size'] = 7
    fig = plt.figure(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.2, hspace=0.5, top=0.85, right=0.75, wspace=0.6, left=0.07)
    plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)


    ipanel = 1
    plt.subplot(nrow, ncol, ipanel)
    
    for xreg, region in enumerate(loopover, 1):
        deltas_per_wt = deltas_per_wt_dict[region][:,ikeep]
        wt_intensities = wt_intensities_dict[region][:,ikeep]
        terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region][:,ikeep].sum(0)],
                           [deltas_per_wt_fixed_intensity_dict[region][:,ikeep].sum(0)],
                           [deltas_per_wt_cross_terms_dict[region][:,ikeep].sum(0)],
                           [yr2yr_residuals_dict[region][:,ikeep].sum(0)]])
        wt_contrib = deltas_per_wt.mean(-1)
        
        WIDTH = 0.15
        stacker(xreg - WIDTH * 0.5, deltas_per_wt.mean(-1), width=WIDTH, colors=wt_cols, labels=wt_labs if xreg == 1 else None, verbose=False, total=True)
        stacker(xreg - WIDTH * 0.5 + WIDTH, terms.mean(-1), width=WIDTH, colors=c4_cols, labels=c4_labs if xreg == 1 else None, verbose=False, total=False)

        
    xx = list(range(1, len(loopover) + 1))
    loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
#     plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
    plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
    plt.title('Contributions to ensemble mean\nTotal precipitation change')
    plt.ylabel('[mm/day]')
    plt.ylim(YLIM[0], YLIM[1])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

#     handles, labels = plt.gca().get_legend_handles_labels()
#     sort both labels and handles by labels
#     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
#     plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9


    ipanel = 2
    plt.subplot(nrow, ncol, ipanel)
    
    for xreg, region in enumerate(loopover, 1):
        deltas_per_wt = deltas_per_wt_dict[region][:,ikeep]
        wt_intensities = wt_intensities_dict[region][:,ikeep]
        terms_ = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region][:,ikeep].sum(0)],
                           [deltas_per_wt_fixed_intensity_dict[region][:,ikeep].sum(0)],
                           [deltas_per_wt_cross_terms_dict[region][:,ikeep].sum(0)],
                           [yr2yr_residuals_dict[region][:,ikeep].sum(0)]])
        wt_contrib = deltas_per_wt.mean(-1)
        
#         wt_var = numpy.cov(deltas_per_wt, ddof=0).diagonal() # / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
#         c4_var = numpy.cov(terms_, ddof=0).diagonal() #/ numpy.cov(terms_, ddof=0).diagonal().sum() * 100

        wt_var = numpy.cov(deltas_per_wt, ddof=0).sum(0) # / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
        c4_var = numpy.cov(terms_, ddof=0).sum(0) #/ numpy.cov(terms_, ddof=0).diagonal().sum() * 100
        
        print(region, c4_var, c4_var / c4_var.sum() * 100)

        if region.lower() == 'scotland':
            print((region, wt_var, c4_var))
            print((numpy.cov(deltas_per_wt, ddof=0)))
            print((numpy.corrcoef(deltas_per_wt, ddof=0)))
            print()
            print((numpy.cov(terms_, ddof=0)))
        
#         raise Exception('stop here')
        WIDTH = 0.15
        stacker(xreg - WIDTH * 0.5, wt_var, width=WIDTH, colors=wt_cols, labels=wt_labs if xreg == 1 else None, verbose=False, total=True)
        stacker(xreg - WIDTH * 0.5 + WIDTH, c4_var, width=WIDTH, colors=c4_cols, labels=c4_labs if xreg == 1 else None, verbose=False, total=False)

        
    xx = list(range(1, len(loopover) + 1))
    loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
#     plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
#     plt.axhline(y=100, linewidth=2, linestyle='--', color='k', zorder=-1)
    plt.title('Contributions to variance\nTotal precipitation change')
    plt.ylabel('Variance')
#     plt.ylim(0, 110)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    
    yl = plt.gca().get_ylim()
    yl = (yl[0] * 1.2, yl[1] * 1.05)
    plt.ylim(yl[0], yl[1])


    handles, labels = plt.gca().get_legend_handles_labels()
#     sort both labels and handles by labels
    labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t:all_labs.index(t[0]))))
#     plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9
    plt.legend(handles, labels, bbox_to_anchor=(1.8, 1), ncol=1, fontsize='x-small') #, ncol=9

    ax2 = plt.gca().twinx()
    ax2.set_ylim(yl[0], yl[1])
    ax2ticks = numpy.array([.1, .2, .3, .4, .5])
    ax2ticks = ax2ticks[ax2ticks < yl[1]**0.5]
    ax2.set_yticks(ax2ticks**2)
    ax2.set_yticklabels(['%.2f' % xt for xt in ax2ticks])
    ax2.set_ylabel('St dev [mm/day]')


    iteration = [(deltas_per_wt_fixed_freq_dict, 'Change in intensity, fixed frequency'),
                 (deltas_per_wt_fixed_intensity_dict, 'Change in frequency, fixed intensity'),
                 (deltas_per_wt_cross_terms_dict[region][:,ikeep], 'Cross terms'),
                 (yr2yr_residuals_dict[region][:,ikeep], 'Contribution from residuals')][:2]


    for ipanel, (delta_dict, titl), tc in zip([3, 5], iteration, ['cyan', 'LightGray']):
        plt.subplot(nrow, ncol, ipanel)
        for xreg, region in enumerate(loopover, 1):
            deltas_per_wt = delta_dict[region][:,ikeep]
            WIDTH = 0.15
            stacker(xreg - WIDTH * 0.5, deltas_per_wt.mean(-1), width=WIDTH, 
                    colors=wt_cols, labels=wt_labs if xreg == 1 else None,
                    total_color=tc, verbose=False, total=True)

        
        xx = list(range(1, len(loopover) + 1))
        loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
        if ipanel == 5:
            plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
        plt.title('WT contributions to ensemble mean\n%s' % titl)
        plt.ylim(YLIM[0], YLIM[1])
        plt.ylabel('[mm/day]')
      

    for ipanel, (delta_dict, titl), tc in zip([4, 6], iteration, ['cyan', 'LightGray']):
        plt.subplot(nrow, ncol, ipanel)
        for xreg, region in enumerate(loopover, 1):
            deltas_per_wt = delta_dict[region][:,ikeep]
            wt_var = numpy.cov(deltas_per_wt, ddof=0).diagonal() #/ numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
            wt_var = numpy.cov(deltas_per_wt, ddof=0).sum(0) #/ numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
            WIDTH = 0.15
            stacker(xreg - WIDTH * 0.5, wt_var, width=WIDTH, 
                    colors=wt_cols, labels=wt_labs if xreg == 1 else None,
                    total_color=tc, verbose=False, total=True)

        
        xx = list(range(1, len(loopover) + 1))
        loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
        if ipanel == 5:
            plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.axhline(y=100, linewidth=2, linestyle='--', color='k', zorder=-1)
        plt.title('WT contributions to variance\n%s' % titl)
#         plt.ylim(0, 110)
        if ipanel == 6:
            plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylabel('Variance')

#         yl = plt.gca().get_ylim()
        plt.ylim(yl[0], yl[1] - 0.02)
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
        
            
        ax2 = plt.gca().twinx()
        ax2.set_ylim(yl[0], yl[1])
        ax2ticks = numpy.array([.1, .2, .3, .4, .5])
        ax2ticks = ax2ticks[ax2ticks < yl[1]**0.5]
        ax2.set_yticks(ax2ticks**2)
        ax2.set_yticklabels(['%.2f' % xt for xt in ax2ticks])
        ax2.set_ylabel('St dev [mm/day]')
            
    fig.savefig(os.path.join(figpath, 'Figure_%s_%s_contributions_variance.png' % (sname, var)), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig4.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig4.tiff'), dpi=300)


if False:
# Slide based on top row of Figure 12
# Plot
# 1. Ensemble mean contribution from WTypes and 4 components

    numpy.set_printoptions(precision=3, suppress=True)
    
    c4_cols = numpy.array(['cyan', 'LightGray', 'pink', 'gold'])
    c4_labs = ['Intensity', 'Frequency', 'Cross terms', 'Residuals']
    wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    wt_labs = ['WT%.1i' % il for il in range(1, 9)]
    WIDTH = 0.3
    all_labs = [str(s) for s in wt_labs + ['Total'] + c4_labs]
    YLIM = -0.5, 1.5

    nrow, ncol= 1, 1
    matplotlib.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(7, 6))
    plt.subplots_adjust(bottom=0.2, right=0.6)
#     plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)


#     ipanel = 1
#     plt.subplot(nrow, ncol, ipanel)
#     
#     for xreg, region in enumerate(loopover, 1):
#         deltas_per_wt = deltas_per_wt_dict[region]
#         wt_intensities = wt_intensities_dict[region]
#         terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
#                            [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
#                            [deltas_per_wt_cross_terms_dict[region].sum(0)],
#                            [yr2yr_residuals_dict[region].sum(0)]])
#         wt_contrib = deltas_per_wt.mean(-1)
#         
#         WIDTH = 0.15
#         stacker(xreg - WIDTH * 0.5, deltas_per_wt.mean(-1), width=WIDTH, colors=wt_cols, labels=wt_labs if xreg == 1 else None, verbose=False, total=True)
#         stacker(xreg - WIDTH * 0.5 + WIDTH, terms.mean(-1), width=WIDTH, colors=c4_cols, labels=c4_labs if xreg == 1 else None, verbose=False, total=False)
# 
#         
#     xx = range(1, len(loopover) + 1)
#     loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
# #     plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
#     plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
#     plt.title('Contributions to ensemble mean\nTotal precipitation change')
#     plt.ylabel('[mm/day]')
#     plt.ylim(YLIM[0], YLIM[1])
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
# 
# #     handles, labels = plt.gca().get_legend_handles_labels()
# #     sort both labels and handles by labels
# #     labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# #     plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9


    ipanel = 1
    plt.subplot(nrow, ncol, ipanel)
    plt.axhline(y=0, color='k', linestyle=':')
    loopover2 = loopover[1:5]
    
    for xreg, region in enumerate(loopover2, 1):
        deltas_per_wt = deltas_per_wt_dict[region]
        wt_intensities = wt_intensities_dict[region]
        terms_ = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                           [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                           [deltas_per_wt_cross_terms_dict[region].sum(0)],
                           [yr2yr_residuals_dict[region].sum(0)]])
        wt_contrib = deltas_per_wt.mean(-1)
        
#         wt_var = numpy.cov(deltas_per_wt, ddof=0).diagonal() # / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
#         c4_var = numpy.cov(terms_, ddof=0).diagonal() #/ numpy.cov(terms_, ddof=0).diagonal().sum() * 100

        wt_var = numpy.cov(deltas_per_wt, ddof=0).sum(0) # / numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
        c4_var = numpy.cov(terms_, ddof=0).sum(0) #/ numpy.cov(terms_, ddof=0).diagonal().sum() * 100

#         if region.lower() == 'scotland':
#             print(region, wt_var, c4_var)
#             print(numpy.cov(deltas_per_wt, ddof=0))
#             print(numpy.corrcoef(deltas_per_wt, ddof=0))
#             print()
#             print(numpy.cov(terms_, ddof=0))
        
#         raise Exception('stop here')

        WT_names_dict_slide = {'WT1':'Northerly/Easterly (NAO-)',
                         'WT2':'Westerly',
                         'WT3':'Northwesterly',
                         'WT4':'Southwesterly',
                         'WT5':'Scandinavian High',
                         'WT6':'UK High',
                         'WT7':'UK Low',
                         'WT8':'Azores High'}
        wt_labs2 = [WT_names_dict_slide[iwt] for iwt in wt_labs]
        c4_labs2 = c4_labs[:]
        c4_labs2[0] = 'Rainfall per event'
        all_labs2 = [str(s) for s in wt_labs2 + ['Total'] + c4_labs2]


        WIDTH = 0.15
        stacker(xreg - WIDTH * 0.5, wt_var, width=WIDTH, colors=wt_cols, labels=wt_labs2 if xreg == 1 else None, verbose=False, total=True)
        stacker(xreg - WIDTH * 0.5 + WIDTH, c4_var, width=WIDTH, colors=c4_cols, labels=c4_labs2 if xreg == 1 else None, verbose=False, total=False)

        
    xx = list(range(1, len(loopover2) + 1))
    loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover2]
#     plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
#     plt.axhline(y=100, linewidth=2, linestyle='--', color='k', zorder=-1)
    plt.title('Contributions to spread\nWinter precipitation change 2050-2099')
    plt.ylabel('Variance')
#     plt.ylim(0, 110)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
    
    yl = plt.gca().get_ylim()
    yl = (yl[0] - 0.005, yl[1])
    plt.gca().set_ylim(yl[0], yl[1])


    handles, labels = plt.gca().get_legend_handles_labels()
#     sort both labels and handles by labels
    labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t:all_labs2.index(t[0]))))
#     plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), ncol=2, fontsize='xx-small') #, ncol=9
    plt.legend(handles, labels, bbox_to_anchor=(1.7, 1), ncol=1, fontsize='xx-small') #, ncol=9

    ax2 = plt.gca().twinx()
    ax2.set_ylim(yl[0], yl[1])
    ax2ticks = numpy.array([.1, .2, .3, .4, .5])
    ax2ticks = ax2ticks[ax2ticks < yl[1]**0.5]
    ax2.set_yticks(ax2ticks**2)
    ax2.set_yticklabels(['%.2f' % xt for xt in ax2ticks])
    ax2.set_ylabel('St dev [mm/day]')


#     iteration = [(deltas_per_wt_fixed_freq_dict, 'Change in intensity, fixed frequency'),
#                  (deltas_per_wt_fixed_intensity_dict, 'Change in frequency, fixed intensity'),
#                  (deltas_per_wt_cross_terms_dict[region], 'Cross terms'),
#                  (yr2yr_residuals_dict[region], 'Contribution from residuals')][:2]
# 
# 
#     for ipanel, (delta_dict, titl), tc in zip([3, 5], iteration, ['cyan', 'LightGray']):
#         plt.subplot(nrow, ncol, ipanel)
#         for xreg, region in enumerate(loopover, 1):
#             deltas_per_wt = delta_dict[region]
#             WIDTH = 0.15
#             stacker(xreg - WIDTH * 0.5, deltas_per_wt.mean(-1), width=WIDTH, 
#                     colors=wt_cols, labels=wt_labs if xreg == 1 else None,
#                     total_color=tc, verbose=False, total=True)
# 
#         
#         xx = range(1, len(loopover) + 1)
#         loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
#         if ipanel == 5:
#             plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
#         else:
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
#         plt.title('WT contributions to ensemble mean\n%s' % titl)
#         plt.ylim(YLIM[0], YLIM[1])
#         plt.ylabel('[mm/day]')
#       
# 
#     for ipanel, (delta_dict, titl), tc in zip([4, 6], iteration, ['cyan', 'LightGray']):
#         plt.subplot(nrow, ncol, ipanel)
#         for xreg, region in enumerate(loopover, 1):
#             deltas_per_wt = delta_dict[region]
#             wt_var = numpy.cov(deltas_per_wt, ddof=0).diagonal() #/ numpy.cov(deltas_per_wt, ddof=0).diagonal().sum() * 100
#             WIDTH = 0.15
#             stacker(xreg - WIDTH * 0.5, wt_var, width=WIDTH, 
#                     colors=wt_cols, labels=wt_labs if xreg == 1 else None,
#                     total_color=tc, verbose=False, total=True)
# 
#         
#         xx = range(1, len(loopover) + 1)
#         loop_str = [capitalize(reg).replace('Uk', 'UK') for reg in loopover]
#         if ipanel == 5:
#             plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
#         else:
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
# #         plt.axhline(y=100, linewidth=2, linestyle='--', color='k', zorder=-1)
#         plt.title('WT contributions to variance\n%s' % titl)
# #         plt.ylim(0, 110)
#         if ipanel == 6:
#             plt.xticks(xx, loop_str, rotation=60, horizontalalignment='right')
#         else:
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.ylabel('Variance')
# 
# #         yl = plt.gca().get_ylim()
#         plt.ylim(yl[0], yl[1])
#         plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
#         
#             
#         ax2 = plt.gca().twinx()
#         ax2.set_ylim(yl[0], yl[1])
#         ax2ticks = numpy.array([.1, .2, .3, .4, .5])
#         ax2ticks = ax2ticks[ax2ticks < yl[1]**0.5]
#         ax2.set_yticks(ax2ticks**2)
#         ax2.set_yticklabels(['%.2f' % xt for xt in ax2ticks])
#         ax2.set_ylabel('St dev [mm/day]')
            
    fig.savefig(os.path.join(figpath, 'Figure_%s_%s_contributions_variance4slides.png' % (sname, var)), dpi=300)



#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================
#==================================================================================



# plot of breakdown across WTs in the changes for each member 
if False:
    nrow, ncol=6, 1
    fig = plt.figure(figsize=(7, 7.7))
    plt.subplots_adjust(bottom=0.1, hspace=0.5, top=0.9, right=0.8)
    plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)
    first = True
    for ipanel, region in enumerate(loopover, 1):
        plt.subplot(nrow, ncol, ipanel)
        deltas_per_wt = deltas_per_wt_dict[region]
        wt_intensities = wt_intensities_dict[region]
        xx = numpy.arange(wt_intensities.coord('realization').shape[0])
        stacks = numpy.zeros_like(deltas_per_wt)
        if first:
            iord = numpy.argsort(deltas_per_wt.sum(0))


        for ii, io in zip(xx, iord):
            stacker(ii, deltas_per_wt[:,io], colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)
        
        ordered = wt_intensities.coord('realization').points[iord]
        if region == loopover[-1]:
            tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
            plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
            for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                ticklabel.set_color('r' if rn in DROP else 'k')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())


        
        plt.axhline(y=0, linewidth=2, linestyle='--', color='k', zorder=-1)
        plt.title(capitalize(region).replace('Uk', 'UK'))
        if region == loopover[2]:
            plt.ylabel('%s change\n[%s]' % (varname, units))
        plt.ylim(-1.05, 2.1)

        handles, labels = plt.gca().get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: t[0])))
        if region == loopover[-1]:
            plt.xlabel('Ensemble member')
        if region == loopover[0]:
            plt.legend(handles, labels, bbox_to_anchor=(1.1, 1)) #, fontsize='x-small') #, ncol=9
        first = False


        # plt.legend(loc='upper left', ncol=2)

    fig.savefig(os.path.join(figpath, 'Figure_%s_%s_by_wt.png' % (sname, var)), dpi=300)


    
# plot of breakdown across freq, intensity in the changes for each member 
if False:
    nrow, ncol=6, 1
    fig = plt.figure(figsize=(7, 7.7))
    plt.subplots_adjust(bottom=0.1, hspace=0.5, top=0.9, right=0.8)
    plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)
    first = True
    for ipanel, region in enumerate(loopover, 1):
        plt.subplot(nrow, ncol, ipanel)
        deltas_per_wt = deltas_per_wt_dict[region]
        deltas_per_wt_fixed_freq = deltas_per_wt_fixed_freq_dict[region]
        deltas_per_wt_fixed_intensity = deltas_per_wt_fixed_intensity_dict[region]
        deltas_per_wt_cross_terms = deltas_per_wt_cross_terms_dict[region]
        deltas_per_wt_means = deltas_per_wt_means_dict[region]
        yr2yr_residuals = yr2yr_residuals_dict[region]
        wt_intensities = wt_intensities_dict[region]
        xx = numpy.arange(wt_intensities.coord('realization').shape[0])
        stacks = numpy.zeros_like(deltas_per_wt)
        if first:
            iord = numpy.argsort(deltas_per_wt.sum(0))
        wt_cols = numpy.array(['white', 'r', 'pink', 'gold'])
        wt_labs = ['Intensity', 'Frequency', 'Cross terms', 'Residuals']



        for ii, io in zip(xx, iord):
            terms = numpy.array([deltas_per_wt_fixed_freq[:,io].sum(0), deltas_per_wt_fixed_intensity[:,io].sum(0),
                                 deltas_per_wt_cross_terms[:,io].sum(0), yr2yr_residuals[:,io].sum(0)])
            stacker(ii, terms, colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)

        ordered = wt_intensities.coord('realization').points[iord]
        if region == loopover[-1]:
            tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
            plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
            for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                ticklabel.set_color('r' if rn in DROP else 'k')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
        
        plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)

        plt.title(capitalize(region).replace('Uk', 'UK'))
        if region == loopover[2]:
            plt.ylabel('%s change\n[%s]' % (varname, units))
        plt.ylim(-1.05, 2.1)

        handles, labels = plt.gca().get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: t[0])))
        if region == loopover[-1]:
            plt.xlabel('Ensemble member')
        if region == loopover[0]:
            plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), fontsize='small') #, ncol=9
        first = False

        # plt.legend(loc='upper left', ncol=2)

    fig.savefig(os.path.join(figpath, 'Figure_%s_%s_by_4terms.png' % (sname, var)), dpi=300)
    
print('Region, Means')
for region in loopover:
    terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                               [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                               [deltas_per_wt_cross_terms_dict[region].sum(0)],
                               [yr2yr_residuals_dict[region].sum(0)]])
    print((region, terms.mean(), terms.mean(-1)))
    
print('Region, %intensity, %frequency')
for region in loopover:
    terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                               [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                               [deltas_per_wt_cross_terms_dict[region].sum(0)],
                               [yr2yr_residuals_dict[region].sum(0)]])
    print((region, numpy.cov(terms, ddof=0).diagonal() / numpy.cov(terms, ddof=0).diagonal().sum() * 100.0))

print()
print('NO AMOC spindowns')
print('Region, Means')
for region in loopover:
    terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                               [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                               [deltas_per_wt_cross_terms_dict[region].sum(0)],
                               [yr2yr_residuals_dict[region].sum(0)]])
    terms = terms[:,ikeep]
    print((region, terms.mean(), terms.mean(-1)))

print('Region, %intensity, %frequency')
for region in loopover:
    terms = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)],
                               [deltas_per_wt_fixed_intensity_dict[region].sum(0)],
                               [deltas_per_wt_cross_terms_dict[region].sum(0)],
                               [yr2yr_residuals_dict[region].sum(0)]])
    terms = terms[:,ikeep]
    print((region, numpy.cov(terms, ddof=0).diagonal() / numpy.cov(terms, ddof=0).diagonal().sum() * 100.0))

deltas_fixed_freq_all = numpy.concatenate([[deltas_per_wt_fixed_freq_dict[region].sum(0)] for region in loopover])
deltas_fixed_intensity_all = numpy.concatenate([[deltas_per_wt_fixed_intensity_dict[region].sum(0)] for region in loopover])
print(loopover)
print('Consistency across regions of intensity')
print((numpy.corrcoef(deltas_fixed_freq_all)))
print('Consistency across regions of frequency')
print((numpy.corrcoef(deltas_fixed_intensity_all)))
cacheSave([deltas_fixed_freq_all, deltas_fixed_intensity_all, loopover], 'deltas.dat')




if False:
    for region in loopover:
        fig = plt.figure(figsize=(7, 5.5))
        plt.subplots_adjust(bottom=0.15, hspace=0.3, top=0.84, right=0.8)
        plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950\n%s' % (sname, varname, units, capitalize(region).replace('Uk', 'UK')), fontsize=matplotlib.rcParams['font.size'] + 2)
        iteration = [(deltas_per_wt_fixed_freq_dict[region], 'Change in intensity, fixed frequency'),
                     (deltas_per_wt_fixed_intensity_dict[region], 'Change in frequency, fixed intensity'),
                     (deltas_per_wt_cross_terms_dict[region], 'Cross terms'),
                     (yr2yr_residuals_dict[region], 'Contribution from residuals')]
        for iplot, (delta_dict, titl) in enumerate(iteration, 1):
            plt.subplot(4, 1, iplot)
            deltas_per_wt = delta_dict
            wt_intensities = wt_intensities_dict[region]
            xx = numpy.arange(wt_intensities.coord('realization').shape[0])
            stacks = numpy.zeros_like(deltas_per_wt)
            iord = numpy.argsort(deltas_per_wt_fixed_freq_dict[region].sum(0))
            wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
            wt_labs = ['WT%.1i' % il for il in range(1, 9)]



            for ii, io in zip(xx, iord):
                stacker(ii, deltas_per_wt[:,io], colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)


    
            ordered = wt_intensities.coord('realization').points[iord]
            if iplot == len(iteration):
                tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
                plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
                for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                    ticklabel.set_color('r' if rn in DROP else 'k')
            else:
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
    
            plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)
            plt.title(titl)
            plt.ylabel('%s change\n[%s]' % (varname, units))
            plt.ylim(-1.05, 2.1)

            if iplot == len(iteration):
                handles, labels = plt.gca().get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: t[0])))
            if iplot == 4:
                plt.xlabel('Ensemble member')
                plt.legend(handles, labels, loc='best', ncol=5, fontsize='x-small')

        # plt.legend(loc='upper left', ncol=2)

        fig.savefig(os.path.join(figpath, 'Figure_%s_%s_%s_by_wt_4terms.png' % (sname, var, region)), dpi=300)
    
if False:
    for region in loopover:
        fig = plt.figure(figsize=(7, 5.))
        plt.subplots_adjust(bottom=0.15, hspace=0.3, top=0.84, right=0.8)
        plt.suptitle('%s %s change [%s]\n2050-2100 minus 1900-1950\n%s' % (sname, varname, units, capitalize(region).replace('Uk', 'UK')), fontsize=matplotlib.rcParams['font.size'] + 2)
        iteration = [(deltas_per_wt_fixed_freq_dict[region], 'Change in intensity, fixed frequency'),
                     (deltas_per_wt_fixed_intensity_dict[region], 'Change in frequency, fixed intensity'),
                     (deltas_per_wt_cross_terms_dict[region], 'Cross terms'),
                     (yr2yr_residuals_dict[region], 'Contribution from residuals')][:2]
        for iplot, (delta_dict, titl) in enumerate(iteration, 1):
            plt.subplot(2, 1, iplot)
            deltas_per_wt = delta_dict
            wt_intensities = wt_intensities_dict[region]
            xx = numpy.arange(wt_intensities.coord('realization').shape[0])
            stacks = numpy.zeros_like(deltas_per_wt)
            iord = numpy.argsort(deltas_per_wt_fixed_freq_dict[region].sum(0))
            wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
            wt_labs = ['WT%.1i' % il for il in range(1, 9)]



            for ii, io in zip(xx, iord):
                stacker(ii, deltas_per_wt[:,io], colors=wt_cols, labels=wt_labs if xx[0] == ii else None, verbose=xx[0] == ii)
    
            ordered = wt_intensities.coord('realization').points[iord]
            if iplot == len(iteration):
                tick_labels = ['%.5i' % (rn - 1100000) for rn in ordered]
                plt.xticks(xx, tick_labels, rotation=60, horizontalalignment='right')
                for ticklabel, rn in zip(plt.gca().get_xticklabels(), ordered):
                    ticklabel.set_color('r' if rn in DROP else 'k')
            else:
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
    
            plt.axhline(y=0, linewidth=1, linestyle='--', color='k', zorder=-1)
            plt.title(titl)
            plt.ylabel('%s change\n[%s]' % (varname, units))
            plt.ylim(-1.05, 2.1)

            if iplot == len(iteration):
                handles, labels = plt.gca().get_legend_handles_labels()
                # sort both labels and handles by labels
                labels, handles = list(zip(*sorted(zip(labels, handles), key=lambda t: t[0])))
            if iplot == 4:
                plt.xlabel('Ensemble member')
                plt.legend(handles, labels, loc='best', ncol=5, fontsize='x-small')

        # plt.legend(loc='upper left', ncol=2)

        fig.savefig(os.path.join(figpath, 'Figure_%s_%s_%s_by_wt_4terms_reduced.png' % (sname, var, region)), dpi=300)
    
    
# precip time series for each local area  
# Figure 7 for supp  
if False:
    fig = plt.figure(figsize=(7, 8))
    plt.subplots_adjust(wspace=0.4, hspace=1, top=0.9, bottom=0.08)
    plt.suptitle('%s %s  [%s]' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)
    matplotlib.rcParams['font.size'] = 5
    for ipanel, region in enumerate(loopover, 1):
        regcap = capitalize(region)
        cube = pr_monthly_dict[region]
        wt_intensities = wt_intensities_dict[region]
        cube5 = cube.extract(iris.Constraint(realization=DROP))
        cube15 = cube.extract(iris.Constraint(realization=KEEP))

    # fix intensities and convolve with weather types
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            baseline_intensity = wt_intensities.data[ii][:,:50].mean(-1)
            wt_ = wt_ann['wt%.1i' % iwt]
            if ii == 0:
                pr_fix_intens = wt_.copy(data=(wt_.data.T * baseline_intensity).T)
            else:
                pr_fix_intens += wt_.copy(data=(wt_.data.T * baseline_intensity).T)

    # fix wtype frequency and vary intensity
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            wt_ = wt_ann['wt%.1i' % iwt]
            base = wt_.data[:,:50].mean(-1)
            nmem, nt = wt_.shape
            baseline_wt_ = wt_.copy(data=numpy.repeat(base, nt).reshape((nmem, nt)))
            intensity = wt_intensities[ii]
            if ii == 0:
                pr_fix_wtfreq = wt_.copy(data=(intensity.data.T * base).T)
            else:
                pr_fix_wtfreq += wt_.copy(data=(intensity.data.T * base).T)

        thin = 0.3
        thk = 1.5
        years = cube.coord('season_year').points
        plt.subplot(len(loopover), 2, ipanel * 2 - 1)
        plta = plt.plot(years, cube.data[ikeep].T, color='DodgerBlue', linewidth=thin)
#         plt.plot(years, gp_smoother(cube5.data.mean(0)), linewidth=thk, color='lime')
#         plt.plot(years, gp_smoother(cube.data.mean(0)), linewidth=thk, color='b')
        plt.plot(years, cube.data[ikeep].mean(0), linewidth=thk, color='b')
        plt.title('%s\nChange relative to 1900-1950' % regcap)
        plt.ylabel('$\Delta$%s [%s]' % (varname, units))
        plt.xlabel('Year')
        plt.ylim(0, 9.7)

        plt.subplot(len(loopover), 2, ipanel * 2)
        yr2yr_var = cube.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var15 = cube15.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var5 = cube5.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_years = yr2yr_var.coord('season_year').points
        # print yr2yr_years
        plta = plt.plot(yr2yr_years, yr2yr_var15.data.T, color='DodgerBlue', linewidth=thin)
        # plt.plot(yr2yr_years, yr2yr_var5.data.mean(0), linewidth=3, color='w')
        plt.plot(yr2yr_years, yr2yr_var15.data.mean(0), linewidth=thk, color='b')
        plt.title('%s\n30-year rolling st dev' % regcap)
        plt.ylabel('St dev [%s]' % units)
        plt.xlabel('Year')
        plt.ylim(0.35, 1.62)
        
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_variability.png' % (sname, var)), dpi=300)
    matplotlib.rcParams['font.size'] = 7
    

#  Figure 2
if True:
    fig = plt.figure(figsize=(7, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.3, top=0.9, bottom=0.08)
#     plt.suptitle('%s %s  [%s]' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 2)
    matplotlib.rcParams['font.size'] = 8
    region_colors = ['k', 'b', 'cyan', 'r', 'gray']
    plt.subplot(211)
    for ipanel, region in enumerate(loopover, 1):
        regcap = capitalize(region)
        cube = pr_monthly_dict[region]
        wt_intensities = wt_intensities_dict[region]
        cube5 = cube.extract(iris.Constraint(realization=DROP))
        cube15 = cube.extract(iris.Constraint(realization=KEEP))

    # fix intensities and convolve with weather types
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            baseline_intensity = wt_intensities.data[ii][:,:50].mean(-1)
            wt_ = wt_ann['wt%.1i' % iwt]
            if ii == 0:
                pr_fix_intens = wt_.copy(data=(wt_.data.T * baseline_intensity).T)
            else:
                pr_fix_intens += wt_.copy(data=(wt_.data.T * baseline_intensity).T)

    # fix wtype frequency and vary intensity
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            wt_ = wt_ann['wt%.1i' % iwt]
            base = wt_.data[:,:50].mean(-1)
            nmem, nt = wt_.shape
            baseline_wt_ = wt_.copy(data=numpy.repeat(base, nt).reshape((nmem, nt)))
            intensity = wt_intensities[ii]
            if ii == 0:
                pr_fix_wtfreq = wt_.copy(data=(intensity.data.T * base).T)
            else:
                pr_fix_wtfreq += wt_.copy(data=(intensity.data.T * base).T)

        thin = 0.3
        thk = 1.5
        years = cube.coord('season_year').points
#         plta = plt.plot(years, cube.data[ikeep].T, color='DodgerBlue', linewidth=thin)
#         plt.plot(years, gp_smoother(cube5.data.mean(0)), linewidth=thk, color='lime')
#         plt.plot(years, gp_smoother(cube.data.mean(0)), linewidth=thk, color='b')
        plt.plot(years, cube.data[ikeep].mean(0), linewidth=thk, color=region_colors[ipanel-1])
    plt.title('%s %s \nChange relative to 1900-1950' % (sname, varname))
    plt.ylabel('$\Delta$%s [%s]' % (varname, units))
    plt.xlabel('Year')
    plt.ylim(0, 7.5)
    plt.grid(linestyle=':', color='LightGray')
        
        
        

    plt.subplot(212)
    for ipanel, region in enumerate(loopover, 1):
        regcap = capitalize(region)
        cube = pr_monthly_dict[region]
        wt_intensities = wt_intensities_dict[region]
        cube5 = cube.extract(iris.Constraint(realization=DROP))
        cube15 = cube.extract(iris.Constraint(realization=KEEP))
    
        yr2yr_var = cube.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var15 = cube15.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var5 = cube5.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_years = yr2yr_var.coord('season_year').points
        # print yr2yr_years
#         plta = plt.plot(yr2yr_years, yr2yr_var15.data.T, color='DodgerBlue', linewidth=thin)
        # plt.plot(yr2yr_years, yr2yr_var5.data.mean(0), linewidth=3, color='w')
        plt.plot(yr2yr_years, yr2yr_var15.data.mean(0), linewidth=thk, color=region_colors[ipanel-1], label=regcap)
    plt.title('30-year rolling st dev')
    plt.ylabel('St dev [%s]' % units)
    plt.xlabel('Central year of 30-year rolling window')
    plt.grid(linestyle=':', color='LightGray')
    plt.ylim(0.45, 1.3)
    plt.legend(loc='upper left', fontsize='small')
        
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_variability_summary.png' % (sname, var)), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig2.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig2.tiff'), dpi=300)

    

# breakdown of variability for each local area
# Figure 5
if True:
    matplotlib.rcParams['font.size'] = 7
    fig = plt.figure(figsize=(7, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.9, top=0.85, bottom=0.08)
#     plt.suptitle('Components of 30-year rolling st dev\n%s %s [%s]' % (sname, varname, units), fontsize=matplotlib.rcParams['font.size'] + 4)
    for ipanel, region in enumerate(loopover, 1):
        regcap = capitalize(region)
        cube = pr_monthly_dict[region]
        wt_intensities = wt_intensities_dict[region]
        cube5 = cube.extract(iris.Constraint(realization=DROP))
        cube15 = cube.extract(iris.Constraint(realization=KEEP))
        yr2yr_var = cube.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var15 = cube15.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var5 = cube5.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_years = yr2yr_var.coord('season_year').points


    # fix intensities and convolve with weather types
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            baseline_intensity = wt_intensities.data[ii][:,:50].mean(-1)
            wt_ = wt_ann['wt%.1i' % iwt]
            if ii == 0:
                pr_fix_intens = wt_.copy(data=(wt_.data.T * baseline_intensity).T)
            else:
                pr_fix_intens += wt_.copy(data=(wt_.data.T * baseline_intensity).T)

    # fix wtype frequency and vary intensity
        for ii, iwt in enumerate(wt_intensities.coord('Weather type').points):
            wt_ = wt_ann['wt%.1i' % iwt]
            base = wt_.data[:,:50].mean(-1)
            nmem, nt = wt_.shape
            baseline_wt_ = wt_.copy(data=numpy.repeat(base, nt).reshape((nmem, nt)))
            intensity = wt_intensities[ii]
            if ii == 0:
                pr_fix_wtfreq = wt_.copy(data=(intensity.data.T * base).T)
            else:
                pr_fix_wtfreq += wt_.copy(data=(intensity.data.T * base).T)

        thin = 0.3
        thk = 1.5
        plt.subplot(3, 2, ipanel)
        yr2yr_var15_fix_intens = pr_fix_intens.rolling_window('time', iris.analysis.STD_DEV, 30)
        yr2yr_var15_fix_wtfreq = pr_fix_wtfreq.rolling_window('time', iris.analysis.STD_DEV, 30)


        yr2yr_years = yr2yr_var15.coord('season_year').points
        yr2yr_years2 = yr2yr_var15_fix_intens.coord('season_year').points
        plt.plot(yr2yr_years, yr2yr_var15.data.mean(0), linewidth=2, color='b', label='Original')
        plt.plot(yr2yr_years2, yr2yr_var15_fix_intens.data.mean(0), linewidth=2, color='orange', label='Fix intensity, vary WT frequency')
        plt.plot(yr2yr_years2, yr2yr_var15_fix_wtfreq.data.mean(0), linewidth=2, color='r', label='Fix WT frequency, vary intensity')
        plt.plot(yr2yr_years2, (yr2yr_var15_fix_intens.data.mean(0)**2 + yr2yr_var15_fix_wtfreq.data.mean(0)**2)**0.5,
                 linewidth=2, linestyle='--', color='k', label='Sum of fixed terms')
        plt.title(regcap)
        plt.ylabel('St dev [%s]' % units)
        plt.xlabel('Central year of 30-year rolling window')


        if region == loopover[-1]:
            plt.legend(loc='lower right', bbox_to_anchor=(2.25, 0), handlelength=4,
                       fancybox=True, shadow=True, ncol=1)

        plt.ylim(ymin=0, ymax=1.25)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        plt.grid(linestyle=':')
    
        fig.savefig(os.path.join(figpath, 'Figures_%s_%s_variability_decomposition.png' % (sname, var)), dpi=300)
        fig.savefig(os.path.join(figpath, 'Fig5.png'), dpi=300)
        fig.savefig(os.path.join(figpath, 'Fig5.tiff'), dpi=300)


# plot change in frequency of WT per year
if False:
    wtfmax = numpy.max([iwt.data.max() for iwt in list(wt_ann.values())])
    fig = plt.figure(figsize=(7, 3))
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams['font.size'] = 5
    plt.subplots_adjust(wspace=0.4, hspace=0.7, top=0.8, bottom=0.1)
    for iwt in WEATHER_TYPES:
        wt_str = 'wt%.1i' % iwt
        plt.subplot(2, 4, iwt)
        wt_ = wt_ann[wt_str]
        plt1 = plt.plot(wt_.coord('season_year').points, wt_.data.T,
                        linewidth=0.2, color='DodgerBlue')
        plt.plot(wt_.coord('season_year').points, gp_smoother(wt_.extract(iris.Constraint(realization=DROP)).data.mean(0), max_length=3), color='lime', linewidth=1.5)

        plt.plot(wt_.coord('season_year').points, gp_smoother(wt_.data.mean(0), max_length=3), color='b', linewidth=1.)
        plt.title('WT %.1i' % iwt)
        plt.ylim(0, wtfmax * 1.02)
        if iwt % 4 == 1:
            plt.ylabel('Frequency\n[days per season]')
        if iwt > 4:
            plt.xlabel('Year')
    
    plt.suptitle('Change in frequency of each weather type for %s' % sname, fontsize=matplotlib.rcParams['font.size'] + 4)

    fig.savefig(os.path.join(figpath, 'Figures_%s_change_freq_per_wt.png' % sname), dpi=300)
    matplotlib.rcParams['font.size'] = fs

    

    
# plot change in average daily amount per WT per year for each local area
if False:
    for region in loopover:
        regcap = capitalize(region)
        wt_intensities = wt_intensities_dict[region]
        fig = plt.figure(figsize=(7, 3))
        plt.subplots_adjust(wspace=0.4, hspace=0.7, top=0.8, bottom=0.1)
        fs = matplotlib.rcParams['font.size']
        matplotlib.rcParams['font.size'] = 5

        for iwt in WEATHER_TYPES:
            plt.subplot(2, 4, iwt)
            wt_ = wt_intensities[iwt-1]
            plt1 = plt.plot(wt_.coord('season_year').points, wt_.data.T,
                            linewidth=0.2, color='DodgerBlue')
            plt.plot(wt_.coord('season_year').points, gp_smoother(wt_.extract(iris.Constraint(realization=DROP)).data.mean(0), max_length=3), color='lime', linewidth=2)

            plt.plot(wt_.coord('season_year').points, 
                     gp_smoother(wt_.data.mean(0), max_length=3), 
                     color='b', linewidth=1.)
            plt.title('WT %.1i' % iwt)
            if wt_intensities.data.min() > 0.0:
                plt.ylim(0, wt_intensities.data.max() * 1.02)
            if iwt % 4 == 1:
                plt.ylabel('%s per day\n[%s]' % (varname, units))
            if iwt > 4:
                plt.xlabel('Year')
    
        plt.suptitle('Average daily %s %s %s per day\nfor each weather type' % (capitalize(region), sname, varname), fontsize=matplotlib.rcParams['font.size'] + 4)

        fig.savefig(os.path.join(figpath, 'Figures_%s_%s_%s_amount_per_wt.png' % (sname, region, var)), dpi=300)
        matplotlib.rcParams['font.size'] = fs
    
    


#=====================================================================
# Figure 7
if True:
    fig = plt.figure(figsize=(174/25.4, 6))
    plt.subplots_adjust(wspace=0.6, top=0.8, hspace=0.6)
    matplotlib.rcParams['font.size'] = 8
    for region in loopover[1:2]:
        bin2d = bin2d_dict[region]
        plots, cbars = bin2d.plot_ensemble_pdf_base_and_change2(base_str, anom_str,
                                                      mask_nonsignificant=True,
                                                      cmap=cmap, orientation='vertical') #,       diff_levels=numpy.linspace(-6, 6, num=11))
        for pl in plots:
            pl.axes.axhline(y=42, color='r', linestyle=':')
            pl.axes.axhline(y=52, color='r', linestyle=':')
        
        for cbar in cbars[-2:]:
            cbar.orientation = 'vertical'            
        
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_jets_pdf_bin2d.png' % (sname, var)), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig7.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig7.tiff'), dpi=300)
    matplotlib.rcParams['font.size'] = 10
    
    
# relationship to jets of each local area's rainfall    
if False:
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(wspace=0.6, top=0.9, hspace=0.9, bottom=0.18)
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams['font.size'] = 6
    nrow, ncol = 3, 3
    for irow, region in enumerate(loopover):
        bin2d = bin2d_dict[region]
        bin2d._ylabel = bin2d._ylabel.replace(' ', '\n')
        regcap = capitalize(region)
        print(("ipanel = ", irow * ncol + 1,))
        plts = bin2d.plot_ensemble_intensity_base_and_change(base_str, anom_str,
                                                      mask_nonsignificant=True,
                                                      cmap=cmap, 
                                                      nrow=len(loopover), ncol=3,
                                                      ipanel=irow * ncol + 1,
                                                      xlabel=region == loopover[-1],
                                                      colorbar=False,
                                                      title=regcap,
                                                      full_title=region == loopover[0],
                                                      vmax=10, lo=-2.5, hi=2.5,
                                                      pdf_threshold=1e-3)
        (plt1, cbar1), (plt2, cbar2), (plt3, cbar3) = plts
        cbar_size = .2, .02
        cbar_base = .05
        xaxes = [.1, .41, .71]
        for xax, pp in zip(xaxes, [plt1, plt2, plt3]):
            axes = ([xax, cbar_base, cbar_size[0], cbar_size[1]])
            colorbar_axes = plt.gcf().add_axes(axes)
            cbar = plt.colorbar(pp, colorbar_axes, orientation='horizontal',
                                extend='both' if xax == xaxes[-1] else 'max')
            cbar.ax.set_title('Precipitation [mm/day]')
            pp.axes.axhline(y=52, color='r', linestyle=':', linewidth=0.5)
            pp.axes.axhline(y=42, color='r', linestyle=':', linewidth=0.5)
            pp.axes.set_yticks([20, 30, 40, 50, 60, 70])

        
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_jets_intensity_bin2d.png' % (sname, var)), dpi=300)
    matplotlib.rcParams['font.size'] = fs

    
# breakdown of jet comntribution to ensemble mean for each local area
if False:
    for region in loopover:
        fig = plt.figure(figsize=(7, 6))
        matplotlib.rcParams['font.size'] = 7
        bin2d = bin2d_dict[region]
        plots, cbars = bin2d.plot_diagnosed_change_by_bin(base_str, anom_str,
                                                          cmap=cmap_change, vmax=0.07,
                                                          orientation='vertical')
        plt.subplots_adjust(top=0.8, hspace=0.5, wspace=0.5)
        plt.suptitle('Change per 2-d bin\n%s %s %s' % (capitalize(region), sname, varname), fontsize=12)
        fig.savefig(os.path.join(figpath, 'Figures_%s_%s_%s_jets_ens_freq_v_intens.png' % (sname, region, var)), dpi=300)
    



# jet effect on local rainfall changes across runs
if False:
    nrow, ncol = 6, 1
    fig = plt.figure(figsize=(7, 8))
    matplotlib.rcParams['font.size'] = 7
    plt.subplots_adjust(hspace=0.8, bottom=0.15, top=0.95)
    first = True
    for irow, region in enumerate(loopover, 1):
        regcap = capitalize(region)
        plt.subplot(nrow, ncol, irow)
        bin2d = bin2d_dict[region]
        daily_uk = daily_uk_dict[region]
        realn = daily_uk.coord('realization').points
        xstr = ['%.5i' % pt for pt in  realn - 1100000]

        delta, cntl_times_deltaw, delta_times_cntlw, cross_term = bin2d.diagnose_change(base_str, anom_str)
        if first:
            srt = numpy.argsort(delta)
        ind = numpy.arange(1, delta.shape[0] + 1)
        plt.bar(ind - 0.4, delta[srt], width=0.15, color='b', label='${\Delta}$')
        plt.bar(ind - 0.2, cntl_times_deltaw[srt], width=0.15, color='g', label='$I.{\Delta}w$')

        plt.bar(ind, delta_times_cntlw[srt], width=0.15, color='r', label='${\Delta}I.w$')
        plt.bar(ind + 0.2, cross_term[srt], width=0.15, color='cyan', label='${\Delta}I.{\Delta}w$')

        if region == loopover[-1]:
            plt.legend(bbox_to_anchor=(0.9, -1), ncol=4) #loc='upper left')
            plt.xlabel('Ensemble member ranked by delta')

        plt.axhline(y=0, linewidth=2, linestyle='--', color='k')
        plt.axhline(y=delta.mean(), linewidth=2, linestyle='--', color='LightGray')
        plt.ylim(-0.55, 1.6)
        plt.title('%s %s %s' % (regcap, sname, varname))
        
        # plt.axvline(x=0, linewidth=2, linestyle='--', color='k')
        plt.ylabel('Component\n[%s]' % units)
        if region == loopover[-1]:
            xt = plt.xticks(1 + numpy.arange(len(xstr)), numpy.array(xstr)[srt], rotation=45, horizontalalignment='right')
            for ticklabel, rn in zip(plt.gca().get_xticklabels(), realn[srt]):
                ticklabel.set_color('r' if rn in DROP else 'k')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())

        first = False

        
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_jets_freq_v_intens_by_member.png' % (sname, var)), dpi=300)
    
    
    
# jet effect on local rainfall changes across runs
if False:
    nrow, ncol = 6, 1
    fig = plt.figure(figsize=(7, 8))
    matplotlib.rcParams['font.size'] = 7
    plt.subplots_adjust(hspace=0.8, bottom=0.15, top=0.95)
    first = True
    for irow, region in enumerate(loopover, 1):
        regcap = capitalize(region)
        plt.subplot(nrow, ncol, irow)
        bin2d = bin2d_dict[region]
        daily_uk = daily_uk_dict[region]
        realn = daily_uk.coord('realization').points
        xstr = ['%.5i' % pt for pt in  realn - 1100000]

        delta, cntl_times_deltaw, delta_times_cntlw, cross_term = bin2d.diagnose_change(base_str, anom_str)
        if first:
            srt = numpy.argsort(delta)
        ind = numpy.arange(1, delta.shape[0] + 1)
        wt_labs = ['$I.{\Delta}w$', '${\Delta}I.w$', '${\Delta}I.{\Delta}w$']
        wt_cols = ['g', 'r', 'cyan']
        
        terms = numpy.concatenate([[cntl_times_deltaw], [delta_times_cntlw], [cross_term]])
        iord = numpy.argsort(deltas_per_wt_dict['uk'].sum(0))

        for ii, io in zip(ind, iord):
            stacker(ii, terms[:,io], colors=wt_cols, labels=wt_labs if ind[0] == ii else None, verbose=ind[0] == ii)


        if region == loopover[-1]:
            plt.legend(bbox_to_anchor=(0.9, -1), ncol=4) #loc='upper left')
            plt.xlabel('Ensemble member ranked by delta')

        plt.axhline(y=0, linewidth=2, linestyle='--', color='k')
        plt.axhline(y=delta.mean(), linewidth=2, linestyle='--', color='LightGray')
        plt.ylim(-0.55, 1.6)
        plt.title('%s %s %s' % (regcap, sname, varname))
        
        # plt.axvline(x=0, linewidth=2, linestyle='--', color='k')
        plt.ylabel('Component\n[%s]' % units)
        if region == loopover[-1]:
            xt = plt.xticks(1 + numpy.arange(len(xstr)), numpy.array(xstr)[srt], rotation=45, horizontalalignment='right')
            for ticklabel, rn in zip(plt.gca().get_xticklabels(), realn[srt]):
                ticklabel.set_color('r' if rn in DROP else 'k')
        else:
            plt.gca().xaxis.set_major_locator(plt.NullLocator())

        first = False

        
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_jets_freq_v_intens_by_member_stacked.png' % (sname, var)), dpi=300)


# Figure 8 jet trends
if False:
    period = 40.0
    interannual_jet_posn = jet_lat.aggregated_by('season_year', iris.analysis.MEAN)
    notok = [interannual_jet_posn.coord('realization').points.tolist().index(ip) for ip in interannual_jet_posn.coord('realization').points if ip  in DROP]
    interannual_jet_str = jet_str.aggregated_by('season_year', iris.analysis.MEAN)
    print(notok)
    quantity = interannual_jet_posn.copy(data=butterworth(interannual_jet_posn.data, period, m=20))                               
    quantity2 = interannual_jet_str.copy(data=butterworth(interannual_jet_str.data, period, m=20))




    fig = plt.figure(figsize=(174/25.4, 3))
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams['font.size'] = 7


    plt.subplots_adjust(wspace=0.3, bottom=0.2)
    plt.subplot(121)
    xx = interannual_jet_posn.coord('year').points
    plt1 = plt.plot(xx, interannual_jet_posn.data[ikeep].T, color='DodgerBlue', linewidth=0.2)
#     #     plt.plot(xx, gp_smoother(interannual_jet_posn.data[ikeep].mean(0), max_length=0.5), color='lime', linewidth=2)
#     # plt.plot(xx, gp_smoother(interannual_jet_posn.data.mean(0), max_length=0.5), color='b', linewidth=1.5)
#     plt.plot(xx, quantity.data.mean(0), color='b', linewidth=1.5)
#     plt.plot(xx, butterworth(scipy.stats.mstats.mquantiles(interannual_jet_posn.data, axis=0, prob=0.1).ravel(), period, m=20), color='b', linestyle=':')
#     plt.plot(xx, butterworth(scipy.stats.mstats.mquantiles(interannual_jet_posn.data, axis=0, prob=0.9).ravel(), period, m=20), color='b', linestyle=':')
#     # plt.plot(xx, scipy.stats.mstats.mquantiles(quantity.data, axis=0, prob=0.95).ravel(), color='b', linestyle=':')
    # plt.plot(xx, scipy.stats.mstats.mquantiles(quantity.data, axis=0, prob=0.05).ravel(), color='b', linestyle=':')
    plt.plot(xx, interannual_jet_posn.data[ikeep].mean(0), color='b', linewidth=1.5)
#     plt.plot(xx, scipy.stats.mstats.mquantiles(interannual_jet_posn.data, axis=0, prob=0.1).ravel(), color='b', linestyle=':')
#     plt.plot(xx, scipy.stats.mstats.mquantiles(interannual_jet_posn.data, axis=0, prob=0.9).ravel(), color='b', linestyle=':')
    plt.plot(xx, gp_smoother(scipy.stats.mstats.mquantiles(interannual_jet_posn.data[ikeep], axis=0, prob=0.1).ravel(), max_length=0.1), color='b', linestyle=':')
    plt.plot(xx, gp_smoother(scipy.stats.mstats.mquantiles(interannual_jet_posn.data[ikeep], axis=0, prob=0.9).ravel(), max_length=0.1), color='b', linestyle=':')




    #     plt.plot(xx, interannual_jet_posn.data[notok].mean(0), color='lime', linewidth=2)
    #     plt.plot(xx, interannual_jet_posn.data.mean(0), color='b', linewidth=1.5)

    plt.title('Mean jet latitude')
    plt.ylabel(u'Latitude [\xb0]')
    plt.xlabel('Year')


    plt.subplot(122)
    xx = interannual_jet_posn.coord('year').points
    plt1 = plt.plot(xx, interannual_jet_str.data[ikeep].T, color='DodgerBlue', linewidth=0.2)
#     #     plt.plot(xx, gp_smoother(interannual_jet_str.data[ikeep].mean(0), max_length=1.5), color='lime', linewidth=2)
#     # plt.plot(xx, gp_smoother(interannual_jet_str.data.mean(0), max_length=1.5), color='b', linewidth=1.5)
#     plt.plot(xx, quantity2.data.mean(0), color='b', linewidth=1.5)
#     plt.plot(xx, butterworth(scipy.stats.mstats.mquantiles(interannual_jet_str.data, axis=0, prob=0.1).ravel(), period, m=20), color='b', linestyle=':')
#     plt.plot(xx, butterworth(scipy.stats.mstats.mquantiles(interannual_jet_str.data, axis=0, prob=0.9).ravel(), period, m=20), color='b', linestyle=':')
    plt.plot(xx, interannual_jet_str.data[ikeep].mean(0), color='b', linewidth=1.5)
#     plt.plot(xx, scipy.stats.mstats.mquantiles(interannual_jet_str.data, axis=0, prob=0.1).ravel(), color='b', linestyle=':')
#     plt.plot(xx, scipy.stats.mstats.mquantiles(interannual_jet_str.data, axis=0, prob=0.9).ravel(), color='b', linestyle=':')
    plt.plot(xx, gp_smoother(scipy.stats.mstats.mquantiles(interannual_jet_str.data[ikeep], axis=0, prob=0.1).ravel(), max_length=0.1), color='b', linestyle=':')
    plt.plot(xx, gp_smoother(scipy.stats.mstats.mquantiles(interannual_jet_str.data[ikeep], axis=0, prob=0.9).ravel(), max_length=0.1), color='b', linestyle=':')


    plt.title('Mean jet strength')
    plt.ylabel('Strength [m/s]')
    plt.xlabel('Year')

    sname, var = 'DJF', 'pr'
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_jet_trends_butterworth.png' % (sname, var)), dpi=300)
    matplotlib.rcParams['font.size'] = fs

# Figure 8
if True:
    interannual_jetlat_north = jet_lat.copy(data=jet_lat.data > 52).aggregated_by('season_year', iris.analysis.MEAN)
    interannual_jetlat_central = jet_lat.copy(data=(jet_lat.data > 42) & (jet_lat.data < 52)).aggregated_by('season_year', iris.analysis.MEAN)
    interannual_jetlat_south = jet_lat.copy(data=jet_lat.data < 42).aggregated_by('season_year', iris.analysis.MEAN)
    interannual_jetstr_north = jet_str.copy(data=jet_str.data * (jet_lat.data > 52)).aggregated_by('season_year', iris.analysis.MEAN)
    interannual_jetstr_central = jet_str.copy(data=jet_str.data * ((jet_lat.data > 42) & (jet_lat.data < 52))).aggregated_by('season_year', iris.analysis.MEAN)
    interannual_jetstr_south = jet_str.copy(data=jet_str.data * (jet_lat.data < 42)).aggregated_by('season_year', iris.analysis.MEAN)

    interannual_jetstr_north.data = interannual_jetstr_north.data / interannual_jetlat_north.data
    interannual_jetstr_central.data = interannual_jetstr_central.data / interannual_jetlat_central.data
    interannual_jetstr_south.data = interannual_jetstr_south.data / interannual_jetlat_south.data
    

    fig = plt.figure(figsize=(174/25.4, 6))
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams['font.size'] = 7
    plt.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.15)

    xx = interannual_jetlat_north.coord('year').points


    plt.subplot(221)
    plt.plot(xx, interannual_jetlat_north.data.mean(0), label='Northern jet', color='b')
    plt.plot(xx, interannual_jetlat_central.data.mean(0), label='Central jet', color='orange')
    plt.plot(xx, interannual_jetlat_south.data.mean(0), label='Southern jet', color='r')
    plt.title('Ensemble mean jet position')
    plt.ylabel('Frequency')
    plt.grid(linestyle=':')

    plt.subplot(223)
    plt.plot(xx, interannual_jetstr_north.data.mean(0), label='Northern jet', color='b')
    plt.plot(xx, interannual_jetstr_central.data.mean(0), label='Central jet', color='orange')
    plt.plot(xx, interannual_jetstr_south.data.mean(0), label='Southern jet', color='r')
    plt.ylabel('Jet strength [m/s]')
    plt.title('Ensemble mean jet strength')
    plt.grid(linestyle=':')
    plt.legend(bbox_to_anchor=(1.5,-0.3), ncol=4)
    
    def plot_filter(x, y, **kwargs):
        yy = butterworth(y, 30.0, m=20)
        if y.ndim == 1:
            plt.plot(x, yy, **kwargs)
        else:
            pp = plt.plot(x, yy.T, **kwargs)
    
    
    plt.subplot(222)
    plot_filter(xx, interannual_jetlat_north.data, color='b', linewidth=0.5)
    plot_filter(xx, interannual_jetlat_central.data, color='orange', linewidth=0.5)
    plot_filter(xx, interannual_jetlat_south.data, color='r', linewidth=0.5)
    plot_filter(xx, interannual_jetlat_north.data.mean(0), linewidth=3, color='b')
    plot_filter(xx, interannual_jetlat_central.data.mean(0), linewidth=3, color='orange')
    plot_filter(xx, interannual_jetlat_south.data.mean(0), linewidth=3, color='r')
    plt.title('Filtered jet position')
    plt.ylabel('Frequency')
    plt.grid(linestyle=':')

    plt.subplot(224)
    plot_filter(xx, interannual_jetstr_north.data, color='b', linewidth=0.5)
    plot_filter(xx, interannual_jetstr_central.data, color='orange', linewidth=0.5)
    plot_filter(xx, interannual_jetstr_south.data, color='r', linewidth=0.5)
    plot_filter(xx, interannual_jetstr_north.data.mean(0), linewidth=3, color='b')
    plot_filter(xx, interannual_jetstr_central.data.mean(0), linewidth=3, color='orange')
    plot_filter(xx, interannual_jetstr_south.data.mean(0), linewidth=3, color='r')
    plt.ylabel('Jet strength [m/s]')
    plt.title('Filtered jet strength')
    plt.grid(linestyle=':')

    
    fig.savefig(os.path.join(figpath, 'Figures_%s_%s_jet_trends_butterworth_alt.png' % (sname, var)), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig8.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig8.tiff'), dpi=300)
    matplotlib.rcParams['font.size'] = fs

# jet v wtype  - old stlye
if False:
    ymd0, ymd1 = jet_lat.coord('yyyymmdd').points[[0, -1]]
    ind0 = wtypes.coord('yyyymmdd').points.tolist().index(ymd0)
    ind1 = wtypes.coord('yyyymmdd').points.tolist().index(ymd1)
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams['font.size'] = 8
    fig = plt.figure(figsize=(7.5, 2.5))
    plt.subplots_adjust(wspace=0.8, top=0.8, bottom=0.2)
    xbin = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    ybin = numpy.linspace(10, 70, num=25)
    nperiod = 5400

    bin2d_early = scipy.stats.binned_statistic_2d(wtypes.data[:,ind0:ind1][:,:nperiod].ravel(), jet_lat.data[:,:nperiod].ravel(), None, statistic='count',
                                                  bins=(xbin, ybin))

    xbin = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    ybin = numpy.linspace(10, 70, num=25)

    bin2d_late = scipy.stats.binned_statistic_2d(wtypes[:,ind0:ind1].data[:,-nperiod:].ravel(), jet_lat.data[:,-nperiod:].ravel(), None, statistic='count',
                                                bins=(xbin, ybin))


    bin2d, x_edge, y_edge, binnumber = bin2d_early
    print(bin2d.shape, x_edge.shape, y_edge.shape)

    x0 = (x_edge[1:] + x_edge[:-1]) * 0.5
    y0 = (y_edge[1:] + y_edge[:-1]) * 0.5

    vmax = numpy.concatenate((bin2d.ravel(), bin2d_late[0].ravel())).max()

    plt.subplot(131)
    plt.pcolormesh(x_edge, y_edge, numpy.ma.masked_equal(bin2d.T, 0.0), vmin=0, vmax=vmax, cmap='YlOrBr')
    plt.axhline(y=42, color='r', linestyle=':', linewidth=1)
    plt.axhline(y=52, color='r', linestyle=':', linewidth=1)
    plt.xlabel('Weather Type')
    plt.ylabel('Jet latitude')
    plt.xticks([1,2,3,4,5,6,7,8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.title('1900-1960')
    plt.colorbar()

    plt.subplot(132)
    plt.pcolormesh(x_edge, y_edge, numpy.ma.masked_equal(bin2d_late[0].T, 0.0), vmin=0, vmax=vmax, cmap='YlOrBr')
    plt.colorbar()
    plt.axhline(y=42, color='r', linestyle=':', linewidth=1)
    plt.axhline(y=52, color='r', linestyle=':', linewidth=1)
    plt.xticks([1,2,3,4,5,6,7,8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Weather Type')
    plt.ylabel('Jet latitude')
    plt.title('2040-2100')


    plt.subplot(133)
    plt.pcolormesh(x_edge, y_edge, numpy.ma.masked_invalid(bin2d_late[0].T / bin2d.T),
                   norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=5.0),
                   cmap='RdBu_r')
    yticks = [0.2, 1, 5]
    cbar = plt.colorbar(ticks=yticks, extend='both')
    cbar.set_ticklabels(['%s' % yt for yt in yticks])
    plt.axhline(y=42, color='r', linestyle=':', linewidth=1)
    plt.axhline(y=52, color='r', linestyle=':', linewidth=1)
    plt.xticks([1,2,3,4,5,6,7,8], ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Weather Type')
    plt.ylabel('Jet latitude')
    plt.title('Relative frequency')


    plt.suptitle('2-d histogram daily jet latitude and Weather Type. DJF', fontsize=14)

    fig.savefig(os.path.join(figpath, 'Figures_WT_v_jetlat_DJF.png'), dpi=300)
    matplotlib.rcParams['font.size'] = 8

# wtype freq time series
if False:
    wtfmax = numpy.max([iwt.data.max() for iwt in list(wt_ann.values())])
    fig = plt.figure(figsize=(7, 7.7))
    fs = matplotlib.rcParams['font.size']
    matplotlib.rcParams['font.size'] = 5
    plt.subplots_adjust(wspace=0.4, hspace=0.7, top=0.95, bottom=0.1)
    for iwt in WEATHER_TYPES:
        wt_str = 'wt%.1i' % iwt
        plt.subplot(8, 3, (iwt - 1) * 3 + 1)
        wt_ = wt_ann[wt_str]
        plt1 = plt.plot(wt_.coord('season_year').points, wt_.data.T,
                        linewidth=0.2, color='DodgerBlue')
        plt.plot(wt_.coord('season_year').points, gp_smoother(wt_.extract(iris.Constraint(realization=DROP)).data.mean(0), max_length=1.5), color='lime', linewidth=1.5)

        ensmean = gp_smoother(wt_.data.mean(0), max_length=1.5)
        plt.plot(wt_.coord('season_year').points, ensmean, color='b', linewidth=1.)
        if iwt == 1:
            plt.title('Frequency of weather type\nWT %.1i' % iwt)
        else:
            plt.title('WT %.1i' % iwt)
        plt.ylim(0, wtfmax * 1.02)
        plt.ylabel('Frequency\n[days per season]')
        if iwt == WEATHER_TYPES[-1]:
            plt.xlabel('Year')
        else:
            plt.gca().axes.get_xaxis().set_visible(False)
            
        
        wt_filt = numpy.apply_along_axis(gp_smoother, -1, wt_.data, max_length=1.5)
        wt_devn = wt_filt - wt_filt.mean(0)
        plt.subplot(8, 3, (iwt - 1) * 3 + 2)
        plt1 = plt.plot(wt_.coord('season_year').points, wt_filt.T,
                        linewidth=0.2, color='DodgerBlue')
        plt.plot(wt_.coord('season_year').points, ensmean,
                        linewidth=1.5, color='b')
        plt.ylim(0, wtfmax * 1.02 * 0.5)
        if iwt == 1:
            plt.title('Smoothed frequency\nWT %.1i' % iwt)
        else:
            plt.title('WT %.1i' % iwt)
        if iwt == WEATHER_TYPES[-1]:
            plt.xlabel('Year')
        else:
            plt.gca().axes.get_xaxis().set_visible(False)

        
        plt.subplot(8, 3, (iwt - 1) * 3 + 3)
        plt.axhline(y=0, linestyle='--', color='k')
        plt1 = plt.plot(wt_.coord('season_year').points, wt_devn.T,
                        linewidth=0.2, color='DodgerBlue')
        plt.plot(wt_.coord('season_year').points, ensmean - ensmean[:50].mean(),
                        linewidth=1.5, color='b')
        if iwt == 1:
            plt.title('Smoothed frequency\nEnsemble mean v Deviations\nWT %.1i' % iwt)
        else:
            plt.title('WT %.1i' % iwt)
        if iwt == WEATHER_TYPES[-1]:
            plt.xlabel('Year')
        else:
            plt.gca().axes.get_xaxis().set_visible(False)
        plt.ylim(-10, 10)
        

#         plt.ylim(0, wtfmax * 1.02 * 0.5)
#     plt.suptitle('Change in frequency of each weather type for %s' % sname, fontsize=matplotlib.rcParams['font.size'] + 4)

    fig.savefig(os.path.join(figpath, 'Figures_%s_change_freq_per_wt_big.png' % sname), dpi=300)
    matplotlib.rcParams['font.size'] = fs

if True:
# Figure 1
    WT_names_dict = {'WT1':'Northerly',
                     'WT2':'Westerly',
                     'WT3':'Northwesterly',
                     'WT4':'Southwesterly',
                     'WT5':'Scandinavian High',
                     'WT6':'UK High',
                     'WT7':'UK Low',
                     'WT8':'Azores High'}
    fig = plt.figure(figsize=(174/25.4, 7.5))  
    plt.subplots_adjust(wspace=0.5, hspace=0.8, top=0.95, bottom=0.08)
    matplotlib.rcParams['font.size'] = 6
    for iwt in [1, 2, 3, 4, 5, 6, 7, 8]:
        wt_ = wt_ann['wt%.1i' % iwt]
        plt.subplot(8, 2, (iwt - 1) * 2 + 1)
        plt1 = plt.plot(wt_.coord('season_year').points, wt_.data.T, color='DodgerBlue', linewidth=0.2)
#         plt.plot(wt_.coord('season_year').points, wt_.data[ikeep].mean(0), color='lime', linewidth=2)
        plt.plot(wt_.coord('season_year').points, wt_.data.mean(0), color='b', linewidth=2)
        wt_titl='WT%.1i' % iwt
        plt.title('%s (%s)' % (wt_titl, WT_names_dict[wt_titl]))
        plt.ylabel('Frequency')
        if iwt == 8:
            plt.xlabel('Year')
        plt.ylim(0, 70)
    
        plt.subplot(8, 2, (iwt - 1) * 2 + 2)
        yr2yr = wt_.rolling_window('time', iris.analysis.STD_DEV, 30)
#         plt.plot(yr2yr.coord('season_year').points, yr2yr.data[ikeep].mean(0), color='lime')
        plt.plot(yr2yr.coord('season_year').points, yr2yr.data.mean(0), color='DodgerBlue')
        plt.title(wt_titl)
        plt.ylabel('Stdev')
        if iwt == 8:
            plt.xlabel('Year')
#         plt.ylim(0, 12)

        
    fig.savefig(os.path.join(figpath, 'Figures_%s_change_freq_per_wt_plus_variability.png' % sname), dpi=300)
    matplotlib.rcParams['font.size'] = fs


    wt_cols = numpy.array(['k', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    WTYPES = [1, 2, 3, 4, 5, 6, 7, 8]
    wt_thk = [1,1,1,1,1,2,1,1]
    
    fig = plt.figure(figsize=(174/25.4, 8))  
    plt.subplots_adjust(wspace=0.5, hspace=0.4, top=0.95, bottom=0.08, left=0.15)
    matplotlib.rcParams['font.size'] = 8
    
    plt.subplot(311)
    for iwt, col, thk in zip(WTYPES, wt_cols, wt_thk):
        wt_ = wt_ann['wt%.1i' % iwt]

#         plt1 = plt.plot(wt_.coord('season_year').points, wt_.data.T, color=col, linewidth=0.2)
#         plt1 = plt.plot(wt_.coord('season_year').points, gp_smoother(scipy.stats.mstats.mquantiles(wt_.data, axis=0, prob=0.9).ravel(), max_length=0.1), color=col, linestyle=':')
#         plt.plot(wt_.coord('season_year').points, wt_.data[ikeep].mean(0), color='lime', linewidth=2)
        plt.plot(wt_.coord('season_year').points, wt_.data[ikeep].mean(0), color=col, linewidth=thk, label='WT%i' % iwt)
        wt_titl='WT%.1i' % iwt
#         plt.title('%s (%s)' % (wt_titl, WT_names_dict[wt_titl]))
    plt.ylabel('Frequency [days per winter]')
    plt.xlabel('Year')
    plt.title('Ensemble mean change in frequency of weather types')
    plt.ylim(0, 40)
    plt.legend(loc='upper left', ncol=4)
    plt.grid(linestyle=':')

#     plt.subplot(312)
#     for iwt, col in zip(WTYPES[:2], ['LightGray', 'LightGreen']):
#         wt_ = wt_ann['wt%.1i' % iwt]
# 
# #         plt1 = plt.plot(wt_.coord('season_year').points, wt_.data.T, color=col, linewidth=0.2)
# #         plt1 = plt.plot(wt_.coord('season_year').points, gp_smoother(scipy.stats.mstats.mquantiles(wt_.data, axis=0, prob=0.1).ravel(), max_length=0.02), color=col, linestyle=':')
# #         plt1 = plt.plot(wt_.coord('season_year').points, gp_smoother(scipy.stats.mstats.mquantiles(wt_.data, axis=0, prob=0.9).ravel(), max_length=0.02), color=col, linestyle=':')
# #         plt.plot(wt_.coord('season_year').points, wt_.data[ikeep].mean(0), color='lime', linewidth=2)
# #         plt.plot(wt_.coord('season_year').points, wt_.data.mean(0), color=col, linewidth=2)
#         plt.plot(wt_.coord('season_year').points, wt_.data.T, color=col, linewidth=2)
#         wt_titl='WT%.1i' % iwt
# #         plt.title('%s (%s)' % (wt_titl, WT_names_dict[wt_titl]))
#     plt.ylabel('Frequency')
#     plt.xlabel('Year')
#     plt.ylim(0, 50)
#     plt.title('Change in 10th and 90th percentile of weather types')

    
    plt.subplot(313)
    for iwt, col in zip(WTYPES, wt_cols):
        wt_ = wt_ann['wt%.1i' % iwt]
        yr2yr = wt_.rolling_window('time', iris.analysis.STD_DEV, 30)
#         plt.plot(yr2yr.coord('season_year').points, yr2yr.data[ikeep].mean(0), color='lime')
        plt.plot(yr2yr.coord('season_year').points, yr2yr.data[ikeep].mean(0), color=col)
#     plt.title(wt_titl)
    plt.ylabel('Stdev [days per winter]')
    plt.xlabel('Central year of 30-year rolling window')
    plt.ylim(0, 13)
    plt.title('Ensemble mean change in 30-year standard deviation')
    plt.grid(linestyle=':')
    
    
    plt.subplot(312)
    wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    WTYPES = [1, 2, 3, 4, 5, 6, 7, 8]

    plt.axhline(y=0, linestyle=':', color='k', linewidth=2)
    for iwt, col in zip(WTYPES, wt_cols):
        wt_ = wt_ann['wt%.1i' % iwt]
        plt.plot([iwt] * wt_.data.shape[0], wt_.data[:,-50:].mean(-1) - wt_.data[:,:50].mean(-1), 'o',
                 color=col, markeredgecolor='k' if col == 'white' else None, markersize=3)
    plt.xticks(WTYPES, WTYPES)
    if region == loopover[-1]:
        plt.xlabel('Weather type')
    plt.ylabel('%change in frequency of weather types\nfor 2050-2099 relative to 1900-1949 [%]')
#     plt.ylim(-0, 13)

        
    fig.savefig(os.path.join(figpath, 'Figures_%s_change_freq_per_wt_plus_variability_new.png' % sname), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig1.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig1.tiff'), dpi=300)
    matplotlib.rcParams['font.size'] = fs



if False:
    wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    WTYPES = [1, 2, 3, 4, 5, 6, 7, 8]

    fig = plt.figure(figsize=(174/25.4, 8))
    plt.subplots_adjust(hspace=0.9, wspace=0.5, bottom=0.02, top=0.95)
    matplotlib.rcParams['font.size'] = 6

    for irow, region in enumerate(loopover):
        regcap = capitalize(region)
        wt_intensities = wt_intensities_dict[region]


        plt.subplot(7, 2, irow * 2 + 1)
        plt.axhline(y=0, linestyle=':', color='k', linewidth=2)
        for iwt, col in zip(WTYPES, wt_cols):
            wt_ = wt_intensities[iwt-1]
            plt.plot(wt_.coord('season_year').points, butterworth(wt_.data.mean(0) / wt_.data[:,:50].mean(), 20, m=20) * 100 - 100, label='WT %.1i' % iwt)
        if region == loopover[-1]:
            plt.xlabel('Year')
            plt.legend(bbox_to_anchor=(0.9, -1), ncol=4) #, fontsize='x-small')
        if irow in [3]:
            plt.ylabel('              %change in average daily precipitation per event relative to 1900-1949')
        plt.ylim(-100, 100)
        plt.title(regcap)

        plt.subplot(7, 2, irow * 2 + 2)
        plt.axhline(y=0, linestyle=':', color='k', linewidth=2)
        for iwt, col in zip(WTYPES, wt_cols):
            wt_ = wt_intensities[iwt-1]
            plt.plot([iwt] * wt_.data.shape[0], (wt_.data[:,-50:].mean(-1) / wt_.data[:,:50].mean(-1)) * 100 - 100, 'o',
                     color=col, markeredgecolor='k' if col == 'white' else None, markersize=3)
        plt.xticks(WTYPES, WTYPES)
        if region == loopover[-1]:
            plt.xlabel('Weather type')
        if irow in [3]:
            plt.ylabel('%change in average daily precipitation per event for 2050-2099 relative to 1900-1949 [%]')
        plt.ylim(-100, 100)
        plt.title(regcap)

    fig.savefig(os.path.join(figpath, 'WT_intensities_change.png'), dpi=300)

    matplotlib.rcParams['font.size'] = 10



# Figure 3
if True:
    matplotlib.rcParams['font.size'] = 10
    wt_cols = numpy.array(['white', 'g', 'r', 'magenta', 'orange', 'pink', 'gold', 'gray'])
    WTYPES = [1, 2, 3, 4, 5, 6, 7, 8]

    fig = plt.figure(figsize=(174/25.4, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.1, top=0.9)
    matplotlib.rcParams['font.size'] = 8

    for irow, region in enumerate(loopover):
        regcap = capitalize(region)
        wt_intensities = wt_intensities_dict[region]


        plt.subplot(3, 2, irow + 1)
        plt.axhline(y=0, linestyle=':', color='k', linewidth=2)
        for iwt, col in zip(WTYPES, wt_cols):
            wt_ = wt_intensities[iwt-1][ikeep]
            plt.plot([iwt] * wt_.data.shape[0], (wt_.data[:,-50:].mean(-1) / wt_.data[:,:50].mean(-1)) * 100 - 100, 'o',
                     color=col, markeredgecolor='k' if col == 'white' else None, markersize=3)
        plt.xticks(WTYPES, WTYPES)
        if region in loopover[-2:]:
            plt.xlabel('Weather type')
        if irow in [2]:
            plt.ylabel('%change in average daily precipitation per event for 2050-2099 relative to 1900-1949')
        plt.ylim(-100, 100)
        plt.title(regcap)

    fig.savefig(os.path.join(figpath, 'WT_intensities_change_perc.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig3.png'), dpi=300)
    fig.savefig(os.path.join(figpath, 'Fig3.tiff'), dpi=300)
    matplotlib.rcParams['font.size'] = 10

#     fig = plt.figure(figsize=(174/25.4, 6))
#     plt.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.1, top=0.9)
#     matplotlib.rcParams['font.size'] = 8
# 
#     for irow, region in enumerate(loopover):
#         regcap = capitalize(region)
#         wt_intensities = wt_intensities_dict[region]
# 
# 
#         plt.subplot(3, 2, irow + 1)
#         plt.axhline(y=0, linestyle=':', color='k', linewidth=2)
#         for iwt, col in zip(WTYPES, wt_cols):
#             wt_ = wt_intensities[iwt-1]
#             plt.plot([iwt] * wt_.data.shape[0], (wt_.data[:,-50:].mean(-1) - wt_.data[:,:50].mean(-1)) , 'o',
#                      color=col, markeredgecolor='k' if col == 'white' else None, markersize=3)
#         plt.xticks(WTYPES, WTYPES)
#         if region in loopover[-2:]:
#             plt.xlabel('Weather type')
#         if irow in [2, 3]:
#             plt.ylabel('%change in average daily precipitation per event for 2050-2099 relative to 1900-1949 [%]')
#         plt.ylim(-2, 2)
#         plt.title(regcap)
# 
#     fig.savefig(os.path.join(figpath, 'WT_intensities_change.png'), dpi=300)
#     matplotlib.rcParams['font.size'] = 10



def plot_scatter(ax, realns, xx, yy, titl, xlabel, ylabel):
    p1 = ax.scatter(xx, yy, s=8)
    plt.ylim(-0.2, 1)
    ax.set_title('%s\nCorr=%.2f' % (titl, numpy.corrcoef(xx, yy)[0,1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(y=0, linestyle=':', color='k')
    ax.axvline(x=0, linestyle=':', color='k')
    for ix, iy, rn in zip(xx, yy, realns):
        if numpy.abs(ix - iy) > 0.2:
            ax.text(ix, iy, rn)


if False:
    
    realns = ['%.4i' % (rn - 1100000) for rn in jet_lat.coord('realization').points]
    for region in loopover:
        deltas_per_wt = deltas_per_wt_dict[region]
        deltas_per_wt_fixed_freq = deltas_per_wt_fixed_freq_dict[region]
        deltas_per_wt_fixed_intensity = deltas_per_wt_fixed_intensity_dict[region]
        deltas_per_wt_cross_terms = deltas_per_wt_cross_terms_dict[region]
        deltas_per_wt_means = deltas_per_wt_means_dict[region]
        yr2yr_residuals = yr2yr_residuals_dict[region]
        wt_intensities = wt_intensities_dict[region]

        wt_total = deltas_per_wt_fixed_freq + deltas_per_wt_fixed_intensity + \
                   deltas_per_wt_cross_terms + yr2yr_residuals
                   
        bin2d = bin2d_dict[region]
        daily_uk = daily_uk_dict[region]
        delta, cntl_times_deltaw, delta_times_cntlw, cross_term = bin2d.diagnose_change(base_str, anom_str)
#         delta = delta.mean()
#         cntl_times_deltaw = cntl_times_deltaw.mean()
#         delta_times_cntlw = delta_times_cntlw.mean()
#         cross_term = cross_term.mean()



        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        matplotlib.rcParams['font.size'] = 8
        plt.subplots_adjust(hspace=0.7, wspace=0.4, top=0.84)
        plt.suptitle(capitalize(region))
        # plt.subplot(221)
        # plt.scatter(wt_total.sum(0), delta)
        print((delta, wt_total.sum(0), delta - wt_total.sum(0)))

        plot_scatter(axes[0,0], realns, deltas_per_wt_fixed_freq.sum(0), delta_times_cntlw, 'Fixed frequency', 'WT estimate', 'Jet estimate')

        plot_scatter(axes[0,1], realns, deltas_per_wt_fixed_intensity.sum(0), cntl_times_deltaw, 'Fixed intensity', 'WT estimate', 'Jet estimate')

        plot_scatter(axes[1,0], realns, deltas_per_wt_cross_terms.sum(0), cross_term, 'Cross terms', 'WT estimate', 'Jet estimate')

        plot_scatter(axes[1,1], realns, (deltas_per_wt_cross_terms + yr2yr_residuals).sum(0), cross_term, 'Cross terms + residals', 'WT estimate', 'Jet estimate')


        x1, x2 = axes[1,1].get_xlim()
        y1, y2 = axes[1,1].get_ylim()
        xy = [numpy.min([x1, y1]), numpy.max([x2, y2])]
        for ax in axes.ravel():
            ax.plot(xy, xy, scalex=False, scaley=False)
        fig.savefig(os.path.join(figpath, 'jet_v_wt_estimate_%s.png' % region), dpi=300)

matplotlib.rcParams['font.size'] = 10


plt.show()

