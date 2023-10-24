# (C) Crown Copyright, Met Office. All rights reserved.
# 
# This file is released under the BSD 3-Clause License.
# See LICENSE.txt in this directory for full licensing details. 

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import time
import sys
import os
from collections import OrderedDict
import operator
import pickle
import math

import numpy
import scipy.stats
import scipy.signal

import iris
import iris.coord_categorisation


def cacheLoad(cacheFile, verbose=True):
    '''
    Wrapper function for loading result from cacheFile using cPickle.
    '''
    if not os.path.isfile(cacheFile):
        raise ValueError('File %s does not exist' % cacheFile)
    if verbose:
        print("%s is up to date and cache being restored" % cacheFile)
    cf = open(cacheFile, 'rb')
    result = pickle.load(cf, encoding='latin1')
    cf.close()

    return result

def cacheSave(result, cacheFile):
    '''
    Wrapper function for saving result using cPickle.
    '''
    print("Writing to file %s with mode wb " % cacheFile)
    cf = open(cacheFile, 'wb')
    pickle.dump(result, cf)
    cf.close()


def add_yyyymmdd(cube, coord, name='yyyymmdd'):
    """Add a categorical calendar-year coordinate."""
    if not cube.coords(name):
        _pt_date = iris.coord_categorisation._pt_date
        iris.coord_categorisation.add_categorised_coord(
            cube, name, coord,
            lambda coord, x: '%s%2.2i%2.2i' % (_pt_date(coord, x).year, _pt_date(coord, x).month, _pt_date(coord, x).day))

def niceMultiple(x, down=False):
    '''
    Find a number similar to x which is a nice multiple of a power of 10
    Returns a nice number, one of [1,2,2.5,5]*10^something

    down:   force returned value to be less than x. Default is greater than x.

    '''
    MULT = [1.0, 2.0, 2.5, 5.0]
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
        ratio = old_div(xx, px)
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



def niceRange(*args):
    '''
    Return in a 2-tuple the lowest and highest values in the list of arguments, each
    of which must be acceptable as numpy.ma.max(arg) or numpy.ma.nax(arg.data)

    '''
    mx = []
    mn = []
    for i in args:
        try:
            mx.append(numpy.ma.max(i))
        except:
            mx.append(numpy.ma.max(i.data))
        try:
            mn.append(numpy.ma.min(i))
        except:
            mn.append(numpy.ma.min(i.data))
    return niceMultiple(numpy.ma.min(mn), down=True), niceMultiple(numpy.ma.max(mx))

def niceRange2(*args, **kwargs):
    '''
    Return in a 2-tuple the lowest and highest values in the list of arguments, each
    of which must be acceptable as numpy.ma.max(arg) or numpy.ma.nax(arg.data)

    '''
    mx = []
    mn = []
    prob1 = kwargs.get('prob', 0.99)
    prob0 = 1.0 - prob1
    for i in args:
        if isinstance(i, iris.cube.Cube):
            mx.append(scipy.stats.mstats.mquantiles(i.data, prob=prob1))
            mn.append(scipy.stats.mstats.mquantiles(i.data, prob=prob0))
        else:
            mx.append(scipy.stats.mstats.mquantiles(i, prob=prob1))
            mn.append(scipy.stats.mstats.mquantiles(i, prob=prob0))
    return niceMultiple(numpy.ma.min(mn), down=True), niceMultiple(numpy.ma.max(mx))


scipy.stats.mstats.mquantiles

def actualRange(*args):
    '''
    Return in a 2-tuple the lowest and highest values in the list of arguments, each
    of which must be acceptable as numpy.ma.max(arg) or numpy.ma.nax(arg.data)

    '''
    mx = []
    mn = []
    for i in args:
        try:
            mx.append(numpy.ma.max(i))
        except:
            mx.append(numpy.ma.max(i.data))
        try:
            mn.append(numpy.ma.min(i))
        except:
            mn.append(numpy.ma.min(i.data))
    return numpy.ma.min(mn), numpy.ma.max(mx)


def niceLevels(*args, **kwargs):
    '''
    Calculates array of nice contour levels based on list of arguments.

    Keywords:
    n:  guideline for how many intervals to make. Not guaranteed.
    inflate: range will be inflated by (1.0+inflate)
    positive: if True, return only positive values

    '''

    n = kwargs.get('n', 9)
    inflate = kwargs.get('inflate', 0.0)
    positive = kwargs.get('positive', False)
    centre = kwargs.get('centre', False)
    delta = 1e-3

# first pass can sometimes produce wide ranges so clip this and do it again
    alo, ahi = actualRange(*args)
    lo, hi = niceRange(*args)
    if centre:
        temp = numpy.array([-lo, lo, -hi, hi])
        lo, hi = temp.min(), temp.max()
    mean = (lo + hi) * 0.5
    factor = 1.0 + inflate
    diff = hi - lo
    lo = mean - diff * 0.5 * factor
    hi = mean + diff * 0.5 * factor
    diff = hi - lo
    step = niceMultiple(old_div(diff, n), down=False)
    try:
        lo, hi = (round(old_div(lo, step)) - 1) * step, (int(old_div(hi, step)) + 1) * step
    except:
        print(args)
        raise ValueError('Problem with lo, hi, step = %s, %s, %s' % (lo, hi, step))
    if centre:
        temp = numpy.array([-lo, lo, -hi, hi])
        lo, hi = temp.min(), temp.max()
    ans = numpy.arange(lo, hi + delta * diff, step)

# clipping to get better ranges
    try:
        lo2 = ans[numpy.where(ans < alo)[0].max()]
    except:
        lo2 = ans[0]

    try:
        hi2 = ans[numpy.where(ans > ahi)[0].min()]
    except:
        hi2 = ans[-1]

    if centre:
        temp = numpy.array([-lo2, lo2, -hi2, hi2])
        lo2, hi2 = temp.min(), temp.max()
    mean = (lo2 + hi2) * 0.5
    diff = hi2 - lo2
    lo2 = mean - diff * 0.5 * factor
    hi2 = mean + diff * 0.5 * factor
    diff = hi2 - lo2
    step = niceMultiple(old_div(diff, n), down=False)
    try:
        lo2, hi2 = (round(old_div(lo2, step)) - 1) * step, (int(old_div(hi2, step)) + 1) * step
    except:
        print(args)
        raise ValueError('Problem with lo, hi, step = %s, %s, %s' % (lo, hi, step))
    print(lo2, hi2)
    if centre:
        temp = numpy.array([-lo2, lo2, -hi2, hi2])
        lo2, hi2 = temp.min(), temp.max()
    print(lo2, hi2)

    ans = numpy.arange(lo2, hi2 + delta * diff, step)

    ans[numpy.abs(old_div(ans, diff)) < 1e-6] = 0.0

    if positive:
        ans = ans[ans > 0.0]
    return ans

def niceLevels2(*args, **kwargs):
    '''
    Calculates array of nice contour levels based on list of arguments.

    Keywords:
    n:  guideline for how many intervals to make. Not guaranteed.
    inflate: range will be inflated by (1.0+inflate)
    positive: if True, return only positive values

    '''

    n = kwargs.get('n', 9)
    inflate = kwargs.get('inflate', 0.0)
    positive = kwargs.get('positive', False)
    centre = kwargs.get('centre', False)
    down = kwargs.get('down', False)
    delta = 1e-3
    prob1 = kwargs.get('prob', 0.99)
    prob0 = 1.0 - prob1

# first pass can sometimes produce wide ranges so clip this and do it again
    alo, ahi = actualRange(*args)
    lo, hi = niceRange2(*args, **kwargs)
    if centre:
        temp = numpy.array([-lo, lo, -hi, hi])
        lo, hi = temp.min(), temp.max()
    mean = (lo + hi) * 0.5
    factor = 1.0 + inflate
    diff = hi - lo
    lo = mean - diff * 0.5 * factor
    hi = mean + diff * 0.5 * factor
    diff = hi - lo
    step = niceMultiple(old_div(diff, n), down=down)
    try:
        lo, hi = (round(old_div(lo, step)) - 1) * step, (int(old_div(hi, step)) + 1) * step
    except:
        print(args)
        raise ValueError('Problem with lo, hi, step = %s, %s, %s' % (lo, hi, step))
    if centre:
        temp = numpy.array([-lo, lo, -hi, hi])
        lo, hi = temp.min(), temp.max()
    ans = numpy.arange(lo, hi + delta * diff, step)

# clipping to get better ranges
    try:
        lo2 = ans[numpy.where(ans < alo)[0].max()]
    except:
        lo2 = ans[0]

    try:
        hi2 = ans[numpy.where(ans > ahi)[0].min()]
    except:
        hi2 = ans[-1]

    if centre:
        temp = numpy.array([-lo2, lo2, -hi2, hi2])
        lo2, hi2 = temp.min(), temp.max()
    mean = (lo2 + hi2) * 0.5
    diff = hi2 - lo2
    lo2 = mean - diff * 0.5 * factor
    hi2 = mean + diff * 0.5 * factor
    diff = hi2 - lo2
    step = niceMultiple(old_div(diff, n), down=False)
    try:
        lo2, hi2 = (round(old_div(lo2, step)) - 1) * step, (int(old_div(hi2, step)) + 1) * step
    except:
        print(args)
        raise ValueError('Problem with lo, hi, step = %s, %s, %s' % (lo, hi, step))
    if centre:
        temp = numpy.array([-lo2, lo2, -hi2, hi2])
        lo2, hi2 = temp.min(), temp.max()
    ans = numpy.arange(lo2, hi2 + delta * diff, step)

    ans[numpy.abs(old_div(ans, diff)) < 1e-6] = 0.0

    if positive:
        ans = ans[ans >= 0.0]
    return ans

def niceLevels3(*args, **kwargs):
    if kwargs.get('centre', False):
        kwargs['centre'] = False
        a = niceLevels2(*args, **kwargs)
        a = numpy.around(a, decimals=kwargs.get('decimals', 5))
        l = numpy.unique(numpy.concatenate([-a, a]))
        l[l == 0.0] = 0.0
        return l
    else:
        return niceLevels2(*args, **kwargs)
        
        

def buildCubeFromAnother(oldCube, aliases, newAxes, data=None, standard_name=None, long_name=None, attributes=None, cell_methods=None, units=None, verbose=True, **kwargs):
    '''
    Function which builds a new cube using coordinates from oldCube in newAxes order. New coordinates can be added by including
    axis tags not in aliases but there must be a keyword with name of new axis tag pointing to the new coordinate. Standard name,
    long_name, cell_methods, attributes  and units can be adopted from oldCube or can be overriden.

    Data is a new masked array or if None, is an array of zeros of the correct shape inferred from the aliases.

    '''

    if oldCube.ndim != len(aliases):
        raise AssertionError('Aliases is length %s but should be length %s' % (len(aliases), oldCube.ndim))
    if len(set(aliases)) != len(aliases):
        raise AssertionError('Aliases needs to have unique values')
    for k in kwargs:
        if k not in aliases and k not in newAxes:
            raise ValueError('Keyword %s is not in aliases or newAxes' % k)
    for c in list(kwargs.values()):
        if not isinstance(c, iris.coords.Coord):
            raise TypeError('%s is not an iris coordinate type' % c)

    if attributes is None:
        attributes = oldCube.attributes
    if cell_methods is None:
        cell_methods = oldCube.cell_methods
    if units is None:
        units = oldCube.units
    if standard_name is None:
        standard_name = oldCube.standard_name
    if long_name is None:
        long_name = oldCube.long_name

    dcad = []
    acad = []
    newShape = []
    scalarCoords = oldCube.coords(dimensions=())
    for axis in newAxes:
        if axis in aliases and axis not in kwargs:
            dim = aliases.index(axis)
            cs = oldCube.coords(contains_dimension=dim)
        else:
            cs = [kwargs[axis]]
        sizes = [c.shape[0] for c in cs if len(c.shape) == 1]
        if len(set(sizes)) > 1:
            raise AssertionError('Coordinates %s for axis %s are different sizes' % (cs, axis))
        if sizes[0] > 1:
            newShape.append(sizes[0])


        for c in cs:
            iaxis = newAxes.index(axis)
#             print newAxes, axis, iaxis
# some cubes have auxCoords that apply over more than 1 dimension. If the newAxes does not
# contain both of these dimensions omit the auxCoord.
            if axis in aliases:
                cd = oldCube.coord_dims(c)
                if len(cd) > 1:
                    if all([aliases[dd] in newAxes for dd in cd]):
                        iaxis = tuple([newAxes.index(aliases[dd]) for dd in cd])
                    else:
                        break

            for sc in scalarCoords:
                if sc.is_compatible(c):
                    if verbose:
                        print("Going to be a clash with %s and %s" % (sc.name(), c.name()))
                        print("Removing the scalar coordinate")
                    scalarCoords.remove(sc)


#            print c.name(), c.shape
            if c.shape == (1,):
                scalarCoords.append(c)
            else:
                toadd = (c, iaxis)
                #if isinstance(c, iris.coords.DimCoord):
##                    if c in oldCube.aux_coords:
##                        print "Are you sure? %s" % c
                    #dcad.append(toadd)
                #else:
                    #acad.append(toadd)

#                 if c in oldCube.dim_coords:
                if any([c.is_compatible(dc) for dc in oldCube.dim_coords]):
#                    if c in oldCube.aux_coords:
#                        print "Are you sure? %s" % c
                    dcad.append(toadd)
                elif isinstance(c, iris.coords.DimCoord):
#                    print "Made it in"
                    dcad.append(toadd)
                else:
                    acad.append(toadd)
#                else:
#                    raise Exception('Should not have got here with %s' % c)

    dcad = list(set(dcad))
    acad = list(set(acad))
    newShape = tuple(newShape)
#     print 'dcad ', [(c[0].name(), c[1]) for c in dcad]
#     print 'acad ', [(c[0].name(), c[1]) for c in acad]
#
#     print 'shap ', newShape
#     print 'sc C', scalarCoords

    if data is None:
        data = numpy.ma.zeros(newShape)
    else:
#         print newShape, data.shape
        data.shape = newShape

    try:
        ans = iris.cube.Cube(data, standard_name=oldCube.standard_name, long_name=oldCube.long_name, units=units, attributes=attributes, cell_methods=cell_methods, dim_coords_and_dims=dcad, aux_coords_and_dims=acad)
    except:
        print(data.shape)
        print("newShape", newShape)
        print("dcad", dcad)
        print("acad", acad)
        print('sc C', scalarCoords)
        raise
    for sc in set(scalarCoords):
        ans.add_aux_coord(sc)
#    print ans
    return ans



def getSpatialGrid(cube, spatial_axes=('x', 'y', 'z')):
    '''
    Reduce a cube to a sub-cube with just spatial (axis = x, y, z) dimensions.
    Side-effect: data and its mask is set to be the same as the first sample of the sub-cube
    that can be retrieved by cube.slices.

    '''
    AXES = spatial_axes
    axes = ['a%s' % i for i in range(cube.ndim)]
    spatial_axes = []
    for dim in range(cube.ndim):
#        print
#        print dim
        dc = cube.coords(contains_dimension=dim, dim_coords=True)
#        print [dn.name() for dn in dc]
        if dc:
            for axis in AXES:
                try:
                    coord = cube.coords(axis=axis, dim_coords=True)
                except:
                    pass
                else:
#                    print axis, [c.name() for c in coord]
                    if dc == coord:
#                        print True, [dn.name() for dn in dc], [c.name() for c in coord]
                        axes[dim] = axis
                        spatial_axes.append(axis)
#    print axes, spatial_axes
    ans = buildCubeFromAnother(cube, axes, spatial_axes)
# now sort out masking
    slicer = cube.slices(ans.dim_coords)
    sample_data = next(slicer)
    ans.data = sample_data.data
    return ans


def invert_wrap_lons(cube):
    """Convert longitudes to the range [-180, 180)."""
    lon = cube.coord(axis="x")
    dim = cube.coord_dims(lon)[0]
    cube.remove_coord(lon)
    points = numpy.where(lon.points >= 180, lon.points - 360, lon.points)
    order =numpy.argsort(points).tolist()
    lon.points = points[order]
    lon.bounds = None
    slicer = [slice(None)] * cube.ndim
    slicer[dim] = order
    cube.data = cube.data[slicer]
    lon = iris.coords.DimCoord.from_coord(lon)
    lon.guess_bounds()
    cube.add_dim_coord(lon, dim)
    
def tidy_cubes(cubes, guess_bounds=False, remove_coords=None, add_month_year=False, set_var_name_None=None, match_coord_system=None):
    '''
    Takes a list of cubes or a single cube and goes through each one either guessing the bounds or removing coordinates.
    guess_bounds:   Default False for no guessing of bounds but if True, guesses bounds of axes that exist out of
                    ['x', 'y', 'z', 't']. Can also be a list of a mixture of special axes ids ['x', 'y', 'z', 't'] or
                    coordinate names.
    remove_coords:  A list of coordinate names to remove

    '''
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
                        print("No need to guess bounds for %s" % gb)
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
        print()
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

def mdtolAverage(a, weights, axes, mdtol=0.3, return_sum=False):
    '''
    Returns weighted average of array a, averaged over axes using
    weights allowing for a certain amount of missing data according to mdtol.

    mdtol: tolerance of missing data. A value will be returned if the
    fraction of missing data is less than mdtol. mdtol=0 means no missing
    data is tolerated.

    '''
    try:
        axes = list(axes)
    except:
        axes = [axes]

    if len(a.shape) < (max(axes) + 1) or min(axes) < 0:
        raise AssertionError('Axes must be in range 0-%s' % (len(a.shape) - 1) )

    if weights.shape != tuple(numpy.array(a.shape)[axes]):
        raise AssertionError('Weights %s is different shape to data %s for axes %s' %
                             (weights.shape, a.shape, axes) )

    if not return_sum:
        w = old_div(weights, numpy.sum(weights))
    else:
        w = weights
    new_order = [i for i in range(a.ndim) if i not in axes] + axes
    b = a.copy().transpose(new_order)
    newBshape = list( b.shape[:-len(w.shape)] ) + [w.size]
    b.shape = newBshape

    if hasattr(a, 'mask'):
        result = numpy.ma.average(b, axis=-1, weights=w.flat, returned=True)
# if b is all masked then result is masked and cannot be unpacked
        try:
            avg, weights = result
        except:
            return result
# sometimes ma.average returns numpy arrays, sometimes masked arrays so...
        if hasattr(weights, 'mask'):
            weights = weights.filled(0.0)
        if return_sum:
#            print 'mask ', weights.shape, avg.shape, weights[0]
            avg *= weights
        ans = numpy.ma.array(avg, mask=numpy.less(weights, 1.0 - mdtol)|numpy.isnan(avg)|numpy.isinf(avg))
#         if weights < 0.99:
#             print mdtol, weights, avg, ans.data, ans.mask
    else:
# masked array average always returns float64 so force numpy.average to do this
        ans = numpy.array(numpy.average(b, axis=-1, weights=w.flat), dtype=numpy.float64)
        if return_sum:
#            print 'no mask', w.shape, ans.shape, w[0], w.sum()
            ans *= w.sum()

#    print type(b), type(ans), hasattr(a, 'mask'), type(w), type(weights)
#    print b.dtype, ans.dtype, w.dtype, weights.dtype
    return ans

def mdtolSum(a, weights, axes, mdtol=0.3):
    '''
    Returns weighted average of array a, averaged over axes using
    weights allowing for a certain amount of missing data according to mdtol.

    mdtol: tolerance of missing data. A value will be returned if the
    fraction of missing data is less than mdtol. mdtol=0 means no missing
    data is tolerated.

    '''
    return mdtolAverage(a, weights, axes, mdtol=0.3, return_sum=True)


def areaAverage(cube, mask=None, mdtol=1.0):
    'Returns area average using mask'

    lats = cube.coord(axis="y")
    lons = cube.coord(axis="x")
    latAxis = cube.coord_dims(lats)[0]
    lonAxis = cube.coord_dims(lons)[0]

    if mask is None:
        if latAxis < lonAxis:
            dcad = ((lats, 0), (lons, 1))
            data = numpy.ma.ones((lats.shape[0], lons.shape[0]))
        else:
            dcad = ((lons, 0), (lats, 1))
            data = numpy.ma.ones((lons.shape[0], lats.shape[0]))
        mask = iris.cube.Cube(data=data, dim_coords_and_dims=dcad)

    for axis in ["x", "y"]:
        if not mask.coord(axis=axis).has_bounds():
            mask.coord(axis=axis).guess_bounds()
    weights = mask.data.filled(0.0) * iris.analysis.cartography.area_weights(mask)
# for our case where sea is masked out, mdtol needs to be bigger than fraction of
# land points in the entire grid. Use mdtol=1.0 here to be sure.
    newData = mdtolAverage(cube.data, weights, [latAxis, lonAxis], mdtol=mdtol)

    aliases = ['Axis%s' % i for i in range(cube.ndim)]
    aliases[latAxis] = 'y'
    aliases[lonAxis] = 'x'
    newAxes = aliases[:]
    newAxes.remove('x')
    newAxes.remove('y')
    newCube = buildCubeFromAnother(cube, aliases, newAxes, data=newData)

    return newCube




def predictLinearFit(y, newX):
    if y.ndim != 1:
        raise AssertionError('Y is %s-d but should be a vector' % y.ndim)
    l = y.shape[0]
    x = numpy.array([numpy.ones(l), numpy.arange(l, dtype=numpy.float)])
    betas = numpy.linalg.lstsq(x.T, y)[0]
    return betas[0] + betas[1] * newX


def useLinearTrendToExtendEnds1d(y, m):
    if y.ndim != 1:
        raise AssertionError('Array must be 1d but is shape' % y)
    if m is None or m == 0:
        return y
    else:
        if 2*m > y.shape[0]:
            raise AssertionError('m = %s but this cannot be greater than half length of y which is %s' % (m, old_div(y.shape[0],2)))
        y1 = predictLinearFit(y[0:m], numpy.linspace(-m, -1, m))
        y2 = predictLinearFit(y[-m:], numpy.linspace(m, 2*m-1, m))
        return numpy.concatenate((y1, y, y2))

def useLinearTrendToExtendEnds(y, m, axis):
    if y.ndim < 2:
        return useLinearTrendToExtendEnds1d(y, m)
    yy = y.reshape((y.shape[0], -1))
#     axis = 0
    ans = numpy.apply_along_axis(useLinearTrendToExtendEnds1d, axis, yy, m)
#     ans.shape = [ans.shape[0]] + list(y.shape[1:])
    return ans

def butterworth(x, period, axis=-1, order=4, m=0, high_pass=False):
    n0 = x.shape[axis]

    x = useLinearTrendToExtendEnds(x, m, axis)
    n = x.shape[axis]
    nyquist = 0.5 * n
    cutoff = old_div(n, float(period))
    wn = old_div(cutoff, nyquist)
    b, a = scipy.signal.butter(order, wn)
    y = scipy.signal.filtfilt(b, a, x, axis=axis)
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(m, m+n0)
    if high_pass:
        return x[slicer] - y[slicer]
    else:
        return y[slicer]
        
        
class MaskMean(object):
    '''
    Class which defines a operation to mean over x and y given a mask.

    '''
    def __init__(self, fmask, name=None, mdtol=1.0):
        '''
        Constructs class MaskMean with fmask, a file of a cube or a cube with x and y axes
        defined only.

        name: Name string used for printing.

        '''

        if isinstance(fmask, str):
            if os.path.exists(fmask):
                mask = iris.load_cube(fmask)
                self._defaultName = 'MaskMean(%s)' % repr(os.path.abspath(fmask))
            else:
                raise ValueError('%s is not a valid file name' % fmask)
        else:
            mask = fmask
            if not isinstance(mask, iris.cube.Cube):
                raise ValueError('Mask needs to be a cube')
            self._defaultName = 'MaskMean(%s)' % mask.name()
            fmask = mask.name()

#        info = iris.analysis.coord_comparison(mask)
#        print info
        if mask.ndim != 2:
            raise ValueError('mask can only have 2 dimensions. This has %s' % mask.ndim)
#        ydim, xdim = info['dimensioned']
#        dimAxes = ydim + xdim

        #GH addition to guess bounds and remove var_name from mask
        tidy_cubes(mask, guess_bounds=['x','y'] )        
        tidy_cubes(mask, set_var_name_None=True )

        xcoord = mask.coords(axis='x', dim_coords=True)
        ycoord = mask.coords(axis='y', dim_coords=True)
        if not xcoord:
            raise AssertionError('Xcoord %s not in set of dimensioned coords %s' % (xcoord, dimAxes))
        if not ycoord:
            raise AssertionError('Ycoord %s not in set of dimensioned coords %s' % (ycoord, dimAxes))
        mask.data = numpy.ma.asarray(mask.data)


        self._axes = ['y', 'x']
        self._mask = mask
        self._fmask = os.path.abspath(fmask)
        self._history = 'Mean of {standard_name:s} over {mask:s}'
        self._cell_method = 'mask mean'
        self._coord_names = [xcoord[0].name(), ycoord[0].name()]
        self._mdtol = mdtol

        if name is None:
            self._name = self._defaultName
        else:
            self._name = name

    name = property(lambda self: self._name, doc='Name')

    def __repr__(self):
        return '''Maskmean(%s, name='%s')''' % (repr(self._fmask), self._name)

    def __eq__(self, other):
        try:
            return self.name == other.name
        except:
            return False
            
    def __ne__(self, other):
        return not self == other        

    def __str__(self):
        if self._name == self._defaultName:
            return self._name
        else:
            return '%s = %s' % (self._name, self._defaultName)

    def __call__(self, cube):
        for axis in ['x', 'y']:
            if self._mask.coord(axis=axis, dim_coords=True) not in cube.coords(axis=axis):
                raise AssertionError('Mask must have same %s axis as cube' % axis)

        avg = areaAverage(cube, mask=self._mask, mdtol=self._mdtol)
#            avg.data = iris.util.ensure_array(data_result)

# Add standard items to the history dictionary.
        history_dict = {}
        history_dict['standard_name'] = cube.name()
        history_dict['mask'] = self._name
        # Add history in-place.
#        avg.add_history(self._history.format(**history_dict))
        # Add a cell method.
        cell_method = iris.coords.CellMethod(repr(self), self._coord_names)
        avg.add_cell_method(cell_method)

#        print "in mask ", type(avg)
        return avg


class MonthMean(object):
    def __init__(self, start, *length):
        MNAMES = ' JFMAMJJASONDJFMAMJJASOND'
        MNUMS = [None] + list(range(1,13)) + list(range(1, 13))
        MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        ALLOWED_ANNUAL_NAMES = ['annual', 'ann']

        if len(MNAMES) != len(MNUMS):
            raise AssertionError('MNAMES and MNUMS must be same length')

        if length == ():
            if not isinstance(start, str):
                raise TypeError('When 1 argument parsed, it must be a string e.g. "JJA"')
            if start.lower() in ALLOWED_ANNUAL_NAMES:
                start = 'DJFMAMJJASON'
            if start.lower() in MONTHS:
                self._name = start.lower()[0:3].capitalize()
                start, length = MONTHS.index(start.lower()) + 1, 1
            else:
                length = len(start)
                try:
                    start = MNAMES.lower().index(start.lower())
                except:
                    raise ValueError('%s is not a contiguous set of months' % start)
                self._name = MNAMES[start:start+length]
        else:
            length = length[0]
            if not 1 <= length <= 12:
                raise ValueError('Length %s must be between 1 and 12 inclusive' % length)
            self._name = MNAMES[start:start+length]
        self._months = MNUMS[start:start+length]
        self._axes = ['t']


    name = property(lambda self: self._name, doc='Name')
    months = property(lambda self: self._months, doc='Months to average over')

    def __eq__(self, other):
        return self._months == other._months
            
    def __ne__(self, other):
        return not self == other        

    def __hash__(self):
# needed so set(self) works as expected
        return hash(tuple(self._months))

    def __str__(self):
        return self._name

    def __repr__(self):
        return 'MonthMean(%s, %s)' % (self._months[0], len(self._months))

    def __call__(self, cube, mdtol=0.66):
        _MONTHS = 'jfmamjjasondjfmamjjasond'
        _CUSTOM_SEASON_NAME = 'season_year'
        
# first see if a month_number coordinate exists.
        need2add_month_coord = True
        if cube.coords('month_number'):
            mname = 'month_number'
            need2add_month_coord = False
        elif cube.coords('month'):
            if isinstance(cube.coord('month').points[0], int):
                mname = 'month'
                need2add_month_coord = False
        else:
            pass
        
# next find tcoord name. If more than try for 'time' as highest priority
        tnames = [tc.name() for tc in cube.coords(axis='t', dim_coords=True)]
        if tnames:
            if len(tnames) > 1:
                if 'time' in tnames:
                    tname = 'time'
                else:
                    raise AssertionError('More than one time coordinate %s exists' % tnames)
            else:
                tname = tnames[0]            
            
        if need2add_month_coord:
            mname = 'month_number'
            iris.coord_categorisation.add_month_number(cube, tname, name=mname)                       
        
# is1month is True if month is its own axis
        is1month = len(set(cube.coord(mname).points)) == cube.coord(mname).shape[0]
        if is1month:
            constraint = iris.Constraint(month=self._months)
            ans = cube.extract(constraint)
            if ans.coord(mname).shape[0] > 1:
                ans = ans.collapsed(mname, iris.analysis.MEAN, mdtol=mdtol)
            return ans
# deal when there is just one month
        elif len(self._months) == 1 and cube.coords(mname):
            constraint = iris.Constraint(coord_values={mname:self._months})
            ans = cube.extract(constraint)
            if not tname:
                raise AssertionError('There is no suitable time coordinate')
            tcoord = ans.coords(tname, dim_coords=True)
            if tcoord:
                try:
                    iris.coord_categorisation.add_year(ans, tname)
                except:
                    pass
                if not ans.coords(_CUSTOM_SEASON_NAME):
                    season_year = ans.coord('year').copy()
                    season_year.rename(_CUSTOM_SEASON_NAME)
                    dim = ans.coord_dims(tcoord[0])[0]
                    ans.add_aux_coord(season_year, dim)
            return ans
        else:
            if not tname:
                raise AssertionError('There is no suitable time coordinate')
            tcoord = cube.coord(tname)
            months = [dt.month for dt in tcoord.units.num2date(tcoord.points)]
            ix = months.index(self.months[0])
            iy = max(i for i, mth in enumerate(months) if mth == self.months[-1]) + 1
            constraint = iris.Constraint(coord_values={mname:self._months})
            if tcoord[ix:iy] == tcoord:
                x = cube.extract(constraint)
            elif cube.ndim == 1:
                x = cube[ix:iy].extract(constraint)
            else:        
                x = cube.subset(tcoord[ix:iy]).extract(constraint)
            sname = self.name.lower()
            isn = _MONTHS.index(sname)
            isn2 = isn + len(sname)
            if len(sname) == 12:
                seasons2avg = [sname]
            elif isn2 > 12:
                seasons2avg = [sname] + [s for s in _MONTHS[isn2-12:isn].split(sname) if s != '']
            else:
                seasons2avg = [sname] + [s for s in _MONTHS[:12].split(sname) if s != '']
#             iris.coord_categorisation.add_custom_season_year(x, tcoord.name(), seasons2avg, name=_CUSTOM_SEASON_NAME)
            if not x.coords(_CUSTOM_SEASON_NAME):
                iris.coord_categorisation.add_season_year(x, tcoord.name(),
                                                          seasons=seasons2avg,
                                                          name=_CUSTOM_SEASON_NAME)
            
            syr = x.coord(_CUSTOM_SEASON_NAME).points.tolist()
#             ok = [i for i, ss in enumerate(syr) if syr.count(ss) == len(sname)]
            slen = len(sname)
            scnt = dict([(ss, syr.count(ss)) for ss in set(syr)])
#             ok = [syr.count(ss) == slen for i, ss in enumerate(syr)]
            ok = [scnt[ss] == slen for ss in syr]
            
            if not all(ok):
                if not tname:
                    raise AssertionError('There is no time coordinate in this cube')
                old_tcoord = x.coord(tname)
                iok = numpy.where(ok)
                new_tcoord = old_tcoord.copy()[iok]
                if new_tcoord != old_tcoord:
                    if x.ndim == 1:
                        x = x[iok]
                    else:    
                        x = x.subset(new_tcoord)
                
            ans = x.aggregated_by(_CUSTOM_SEASON_NAME, iris.analysis.MEAN, mdtol=mdtol)

            return ans



