#(C) Crown Copyright, 2024, Met Office. All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE


import os, glob, sys


import warnings
import numpy
import iris
#import qumpy
import pandas as pd
import itertools
from past.utils import old_div

from scipy import stats
import scipy.signal
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


import dowhy
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
import pydot
#import graphviz
import pygraphviz
from dowhy import CausalModel


import matplotlib
import matplotlib.pyplot as plt

#from qumpy.irislib.downscaler import butterworth


import statsmodels.api as sm
from scipy import signal
from scipy import stats
from scipy.spatial import ConvexHull
import pdb
from scipy.signal import detrend


def predictLinearFit(y, newX):
    if y.ndim != 1:
        raise AssertionError('Y is %s-d but should be a vector' % y.ndim)
    l = y.shape[0]
    x = numpy.array([numpy.ones(l), numpy.arange(l, dtype=numpy.float)])
    betas = numpy.linalg.lstsq(x.T, y, rcond=None)[0]
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
    slicer = tuple(slicer)
    if high_pass:
        return x[slicer] - y[slicer]
    else:
        return y[slicer]

def filter(x):
    return butterworth(x, 10, m=20)

def detrend2d(x):
    if x.ndim == 1:
        return x - butterworth(x, 100, m=20)
    else:
        return x - butterworth(x, 100, axis=-1, m=20)

def detrend_cube(cube):
    return cube.copy(data=detrend2d(cube.data))

def pearsonr2d(x, y):
    '''
    Function that takes 2-d data X and Y applies pearsonr looping over axis 0.
    '''
    return numpy.array([stats.pearsonr(x[i], y[i])[0] for i in range(x.shape[0])])

def conditionOnZ(x, z):
    zz = sm.tools.tools.add_constant(z)
    sm_model = sm.OLS(x, zz)
    results = sm_model.fit()
#     print(x.shape, zz.shape, results.resid.shape)
    return results.resid

def estimate_slope2d(x, y, condition=None):
    '''
    Return slope of 
    '''
    xx = sm.tools.tools.add_constant(x, prepend=True)
    if xx.shape[-1] != 2:
        raise ValueError('X needed to be a vector or a 2-column array with an intercept')
    if condition is not None:
        xx = numpy.concatenate([xx, numpy.reshape(condition, (xx.shape[0], -1))], axis=1)
    sm_model = sm.OLS(y, xx)
    results = sm_model.fit()
    return results.params[1]

def slope2d(x, y, detrend=False, condition=None):
    '''
    Function that takes 2-d data X and Y and estimates slope using regression possibly conditioned on
    condition, looping over axis 0.
    
    '''
    if detrend:
        x, y = detrend2d(x), detrend2d(y)
        if condition is not None:
            for i in range(condition.shape[-1]):
                condition[:,:,i] = detrend2d(condition[:,:,i])
    if condition is None:
        return numpy.array([estimate_slope2d(x[i], y[i], condition=None) for i in range(x.shape[0])])
    else:
        return numpy.array([estimate_slope2d(x[i], y[i], condition=condition[i]) for i in range(x.shape[0])])
        
    


def partialcorr2d(x, y, z, detrend=False):
    '''
    Function that takes 2-d data X and Y applies partial correlation conditioned on z looping over axis 0.
    '''
    if detrend:
        x, y, z = detrend2d(x), detrend2d(y), detrend2d(z)
    return numpy.array([stats.pearsonr(conditionOnZ(x[i], z[i]), conditionOnZ(y[i], z[i]))[0] for i in range(x.shape[0])])
    
# points = np.random.randint(0, 10, size=(15, 2))  # Random points in 2-D

def plot_convex_hull(x, y, label=None, **kwargs):
    points = numpy.concatenate([[x], [y]]).T
    hull = ConvexHull(points)
#     simplex1, simplex2 = zip(*hull.simplices)
#     pdb.set_trace()
#     pch = plt.plot(points[simplex1, 0], points[simplex2, 1], **kwargs)
#     print(hull.vertices)
    hv = hull.vertices
    hv = numpy.append(hv, hull.vertices[0])
    pch = plt.plot(points[hv,0], points[hv,1], label=label, **kwargs)
    return pch




def cdf2d(x, y, percentiles=(0.33, 0.66, 0.9), bins=None, fill=False, alpha_fill=0.3,
          cmap='jet', extend='max', plot_label=False, **kwargs):
    '''
    Produces a contour plot with a line for each percentile which describes the smallest region (i.e. the
    one with greatest density of points) which contains 100*percentile % of the (X, Y)-points.

    It does this by first binning X, Y points into a mesh determined by bins. See
    scipy.stats.binned_statistic_2d help.

    If bins is None, it is set to the nearest integer to sqrt(x.size) or 11 if that is bigger.

    If fill is True, a contourf is done too using keywords alpha_fill, cmap and extend

    '''
    
    print('kwargs', kwargs)

    from past.utils import old_div
    
    if x.ndim != 1:
        raise ValueError('X should be a 1-d array')
    if y.ndim != 1:
        raise ValueError('Y should be a 1-d array')
    if x.shape != y.shape:
        raise AssertionError('X and Y should be identical size but are shapes %s and %s' % (x.shape, y.shape))

    if bins is None:
        bins = min([11, int(numpy.rint(x.size**0.5))])

    count, xedges, yedges, index = stats.binned_statistic_2d(x, y,
                                                                   numpy.ones_like(x),
                                                                   statistic='count',
                                                                   bins=bins)
    count = old_div(count, float(count.sum()))
    count1d = count.ravel()
    order = numpy.argsort(count1d)
    scost = count1d[order]
    ncost = (count.shape[0] + 1) * (count.shape[1] + 1)
    zfnc = 1.0 - scost.cumsum()
    uscost = numpy.unique(scost)
    uzfnc = numpy.array([zfnc[scost == usc].mean() for usc in uscost])
    uzfnc_r = uzfnc[::-1]
    uscost_r = uscost[::-1]
    lev = numpy.interp(percentiles, uzfnc_r, uscost_r)
    cdf = numpy.interp(count1d, uscost, uzfnc).reshape(count.shape)

    xmid = (xedges[:-1] + xedges[1:]) * 0.5
    ymid = (yedges[:-1] + yedges[1:]) * 0.5
    X, Y = numpy.meshgrid(xmid, ymid)
    clabel_fmt = dict(list(zip(lev, [str(p) for p in percentiles])))
    pcont = plt.contour(X, Y, count, lev, **kwargs)
    if fill:
        for k in ['colors', 'linewidths', 'linestyles']:
            if k in kwargs:
                del kwargs[k]
        kwargs['alpha'] = alpha_fill
        pcontf = plt.contourf(pcont, cmap=cmap, extend=extend, **kwargs)
    if plot_label:
        plt.clabel(pcont, lev, inline=1, fontsize=10, fmt=clabel_fmt)

    return pcont


def unify_time_units(cubes, base_coord=None):
    start = 0
    if isinstance(cubes, iris.cube.Cube):
        cubes = [cubes]
    if base_coord is None:
        base_coord = cubes[0].coord(axis='t', dim_coords=True)
        start = 1
    elif isinstance(base_coord, iris.cube.Cube):
        base_coord = base_coord.coord(axis='t', dim_coords=True)
    for cube in cubes[start:]:
        tcoord = cube.coord(axis='t', dim_coords=True)
#        except:             # might not be a dim coord but a scalar coord instead
#            tcoord = cube.coord(base_coord.name())
        dim = cube.coord_dims(tcoord)[0]
        new_points = base_coord.units.date2num(tcoord.units.num2date(tcoord.points))
        if tcoord.has_bounds():
            new_bounds = base_coord.units.date2num(tcoord.units.num2date(tcoord.bounds))
        else:
            new_bounds = None
        tcoord.units = base_coord.units
        new_coord = base_coord.copy(points=new_points, bounds=new_bounds)
        cube.remove_coord(tcoord)
        cube.add_dim_coord(new_coord, dim)

def make_cube_2d(cube):
    if cube.ndim > 1:
        return cube
    elif len(cube.coords('realization')) > 0:
        return iris.util.new_axis(cube, scalar_coord='realization')
    else:
        realn_coord = iris.coords.DimCoord([1], long_name='realization')
        c = cube.copy()
        c.add_aux_coord(realn_coord)
        return iris.util.new_axis(c, scalar_coord='realization')



def match_times(cubes, names, target):

# skip trivial case for AMIP where time points size == 1
    if all([c.coord('time').points.size == 1 for c in cubes]):
        return cubes

    for nm, c in zip(names, cubes):
        print(nm, c.coord('time')[0])

# find first point of target variable    
    iref = names.index(target)
    unify_time_units(cubes, base_coord=cubes[iref].coord('time'))
    tref0 = cubes[iref].coord('time').points[0]

    
# are the start points of the other variables later than this point. If so pick earliest of these.
    t0 = numpy.array([c.coord('time').points[0] for c in cubes])
    is_later = [c.coord('time').points[0] > tref0 for c in cubes]
#    print(is_later)
    if any(is_later):
        tmax = numpy.min(t0[is_later])
        i0 = numpy.where(t0 == tmax)[0][0]
    else:
        i0 = iref 

    tref = t0[i0]
#    print(i0, tref)

    cubes_ = iris.cube.CubeList()
    for c in cubes:
        times = c.coord('time').points
        tok = times[times >= tref]
#        print(numpy.where(times >= tref)[0])
        if times.min() >= tref:
            cubes_.append(c)
        else:
            tsub = c.coord('time')[numpy.where(times >= tref)[0]]
            cubes_.append(c.subset(tsub))

    print(cubes_)
    tlens = [c.coord('time').points.size for c in cubes_]
    print(tlens)
    if len(set(tlens)) == 1:
        return cubes_

    new_list = iris.cube.CubeList()
    tsize = [c.coord('time').shape[0] for c in cubes_]
    imin = numpy.argmin(tsize)
    nmin = numpy.min(tsize) 

    tbase = cubes_[imin].coord('time').points
    for i, c in enumerate(cubes_):
        tcoord = c.coord('time').points
        if c.coord('time').shape[0] > tsize[imin]:
#            ind = [numpy.argmin(numpy.abs(tcoord - tb)) for tb in tbase]
            tsub = c.coord('time')[:nmin]
            new_list.append(c.subset(tsub))
            print('Reducing size of cube %s' % i)
        else:
            new_list.append(c)

    tref1 = new_list[iref].coord('time').points[0]
    print([c.coord('time').points[0] - tref1 for c in new_list])

    return new_list

def match_tcoord(cubes):
    for c in cubes:
        c.coord('time').var_name = None
    tcoord = cubes[0].coord('time')
    for c in cubes:
        tcoord = tcoord.intersect(c.coord('time'))
    return iris.cube.CubeList([c.subset(tcoord) for c in cubes])

def match_members(cubes):
    if cubes[0].coords('realization') and len(cubes) > 1:
        common_coord = cubes[0].coord('realization')
        for c in cubes[1:]:
            if common_coord.shape[0] > 1:
                common_coord = common_coord.intersect(c.coord('realization'))
        if common_coord.shape[0] > 1:
            return iris.cube.CubeList([c.subset(common_coord) for c in cubes])
        else:
            return cubes
    else:
        return cubes
    
def plot_causal_strength(causal_strength, label, imem, span=False, tstart=None, **kwargs):
    if 'causal_strength' in causal_strength:
        cs = causal_strength['causal_strength']
    else:
        cs = causal_strength
        
    times, strengths = cs[label]
    if tstart is not None:
        times = times - times[0] + tstart
    if span:
        plt.axhspan(strengths[imem].min(), strengths[imem].max(), **kwargs)
    else:
        if strengths.shape[-1] == 1:
            plt.plot(times, strengths[imem], 'o', **kwargs)
        else:
            plt.plot(times, strengths[imem], **kwargs)


#########################################################
def print_list(l):
    for i in l:
        print(i)
        
def get_unique_perms(*lists):
    perms = [tuple(sorted(list(ii))) for ii in itertools.product(*lists)]
    perms = sorted(set(perms))
    return perms

def is_not_superset(cond, conditions):
    return not any([set(cond).issuperset(cc) for cc in conditions if cc != cond])
    
def remove_redundant_tests(test):
    pairs = set([links for links, conditions in test])
    reduced_set = list()
    for pair in sorted(pairs):
        subset = [vals for vals in test if vals[0] == pair]
        if len(subset) == 1:
            reduced_set.append(subset[0])
        else:
            conditions = [vals[1] for vals in subset]
            keep = [cc for cc in conditions if is_not_superset(cc, conditions)]
            for k in keep:
                reduced_set.append((pair, k))
    return reduced_set




def get_all_conditional_independent_sets(gr, verbose=False):
    '''
    Based on a graph object set up by networkx or call to Dowhy or LinearStructuralCausalModel,
    identify all the unique conditional independent sets that are implicated by the causal
    graph and can be tested to refute the model.
    
    Return value:
    List of tuples each of form (2-tuple, list) where 2-tuple identifies the pair of nodes
    that are implicated to be conditionally independent conditioned on the nodes in the 
    list (which can be empty).
    
    '''


    nodes = gr.get_all_nodes()
    exogenous = [node for node in nodes if not gr.get_parents(node)]

    pairs = [[p1, p2] for p1, p2 in itertools.product(nodes, nodes) if p2 not in gr.get_parents(p1) and p1 not in gr.get_parents(p2) and p1 != p2]
    pairs = [tuple(sorted(pair)) for pair in pairs]
    pairs = set(pairs)

    all_conditional_independent_sets = list()
#    print(len(pairs), pairs)
    for pair in pairs:
        rest = [node for node in set(gr.get_ancestors(pair[0])).union(gr.get_ancestors(pair[1])) if node not in pair]
        to_consider = [[]] + [[node] for node in rest]
        if len(rest) > 1:
            to_consider += get_unique_perms(rest, rest)
        if len(rest) > 2:
            to_consider += get_unique_perms(rest, rest, rest)
        if len(rest) > 3:
            to_consider += get_unique_perms(rest, rest, rest, rest)
        to_consider = [tc for tc in to_consider if len(set(tc)) == len(tc)]
        if verbose:
            print('For %s, consider %s' % (pair, to_consider))
        conditional_independent_sets = [(pair, tc) for tc in to_consider if gr.check_dseparation([pair[0]], [pair[1]], tc)]
#        print('Found %s' % conditional_independent_sets)
        all_conditional_independent_sets.extend(conditional_independent_sets)

    all_conditional_independent_sets = remove_redundant_tests(all_conditional_independent_sets)

    return all_conditional_independent_sets

def df_detrend(df):
    ans = df.copy()
    
    for k in df.columns:
        ans[k] = detrend(df[k])
        
    return ans

def make_mappers(dot_string):
    mapper = dict()
    reverse_mapper = dict()
    sep = '[label='
    for s in dot_string.split('\n'):
        if sep in s:
            s = s.replace(' ', '')
            before, after = s.split(sep)
            if '->' not in before:
                after = after.replace('];', '')
                mapper[before] = eval(after)
                reverse_mapper[eval(after)] = before
    return mapper, reverse_mapper


#def add_variable_based_on_residuals(df, process_info):
#    new_df = df.copy()
#    use_df = df.copy()
#    
#    for new_var, formula, high_pass_filter in process_info:
#        base_var = formula.split('~')[0].replace(' ', '')
#        if high_pass_filter:
#            use_df[base_var] =  butterworth(df[base_var], 30, m=15, high_pass=True)
#            result = OLS.from_formula(formula, use_df).fit()
#            new_df[new_var] = df[base_var] - result.fittedvalues
#        else:
#            result = OLS.from_formula(formula, use_df).fit()
#            new_df[new_var] = result.resid
#            
#    
#    return new_df

def add_variable_based_on_residuals(df, process_info, use_df=None):
    '''
    Make a set of new columns in dataframe df using rules listed in process_info. The rules are
    then used to estimate regression coefficients from df and return residuals in the new column.
    It is possible to apply the regression to high-pass filtered data if trying to avoid aliasing
    the effect onto a trend. Sometimes the regression coefficients need to be based on a dataframe
    in use_df.
    
    Keywords:
    process_info: list of 3-tuples(new_column_name, regression_formula, high_pass_boolean)
    use_df:       base regression fit on a second dataframe use_df
    
    Example of process_info:    
    
    process_info = [('SST50S50NnoENSO', "SST50S50N ~ 1 + ENSO", True),
                    ('LowerGradientNoBKSIC', 'LowerGradient ~ 1 + BKSIC + ENSO', False)]

    
    '''
    new_df = df.copy()
    if use_df is None:
        use_df = df.copy()
    
    for new_var, formula, high_pass_filter in process_info:
        base_var = formula.split('~')[0].replace(' ', '')
        if high_pass_filter:
            use_df[base_var] =  butterworth(use_df[base_var], 30, m=15, high_pass=True)
            result = OLS.from_formula(formula, use_df).fit()
        else:
            result = OLS.from_formula(formula, use_df).fit()
            
        new_df[new_var] = df[base_var] - result.predict(exog=df)
    return new_df



def get_link_from_strengths(causal_strength, ilink):
    ans = causal_strength.copy()
    causal_dict = dict()
    for k, (t, v, f, n) in causal_strength['causal_strength'].items():
        causal_dict[k] = (t[t != 0.0], v[:,:,ilink])
    ans['causal_strength'] = causal_dict
    ans['xvar'] = ans['xvar'][ilink]
    ans['yvar'] = ans['yvar'][ilink]
    ans['names'] = ans['names'][ilink]
    ans['tcoord'] = ans['tcoord'][ans['tcoord'] != 0.0]

    return ans


def nice_format_graph(graph, scmfit, processor=None, omit_strengths=False):
    dstring2 = graph
    if processor is not None:
        for ul in scmfit.used_labels:
            if ul in processor and ul in dstring2:
                sname = processor[ul].name.replace('DJFMAMJJASON', 'Annual')
                new_label = '%s %s' % (sname, ul)
                dstring2 = dstring2.replace(ul, new_label)
    if scmfit._links is not None and not omit_strengths:
        for (parent, child), coefft in scmfit._links.items():
            pattern = '%s -> %s;' % (parent, child)
            if scmfit._specified_links is not None:
                if (parent, child) in scmfit._specified_links:
                    fontcolor = '#0000FF'
                else:
                    fontcolor = '#FF0000'
            else:
                fontcolor = '#FF0000'
            new_pattern = '''%s -> %s[label="%.2f" fontcolor="%s"];''' % (parent, child, coefft, fontcolor)
            dstring2 = dstring2.replace(pattern, new_pattern)
    return dstring2


def get_link(scmfit, parent, child):
    parent_ = scmfit._labels2nodes[parent]
    child_ = scmfit._labels2nodes[child]
    return scmfit._links[(parent_, child_)]

#===========================================================
#===========================================================
#===========================================================

class RidgeRegressionEstimator(LinearRegressionEstimator):
    """Compute effect of treatment using ridge regression.

    Fits ridge regression models for estimating the outcome using treatment(s) and confounders by varying alpha and choosing the value of alpha that optimises BIC. For a univariate treatment, the treatment effect is equivalent to the coefficient of the treatment variable.

    Simple method to show the implementation of a causal inference method that can handle multiple treatments and heterogeneity in treatment. Requires a strong assumption that all relationships from (T, W) to Y are linear.

    """

    def __init__(self, *args, **kwargs):
        """For a list of args and kwargs, see documentation for
        :class:`~dowhy.causal_estimator.CausalEstimator`.

        """
        # Required to ensure that self.method_params contains all the
        # parameters to create an object of this class
#        args_dict = {k: v for k, v in locals().items()
#                     if k not in type(self)._STD_INIT_ARGS}
#        args_dict.update(kwargs)
#        print(args_dict)
        super().__init__(*args, **kwargs)
        self.logger.info("INFO: Using Ridge Regression Estimator")

    def _build_model(self):
# this code works and shows the effect of varying alpha on the ridge regression. BUT ridge regression method
# does not have an implemented summary() method and that crashes subsequent code. Anyway, this is enough to see 
# if ridge can help. 
        ALPHAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1, 10, 100]
        features = self._build_features()
        for alpha in ALPHAS:
            model = sm.OLS(self._outcome, features).fit_regularized(method='elastic_net', alpha=alpha, L1_wt=0.0, refit=False)
            score = numpy.mean((numpy.array(self._outcome) - model.fittedvalues)**2)
            print(alpha, score)
        return (features, model)



class LinearStructuralCausalModel(object):
    '''
    A class to either specify a linear causal network/graph or fit one to data and 
    then simulate from it.
    
    
    '''
    
    def __init__(self, graph, df=None, path_coefficients=None, effect_modifiers=None, detrend=True, verbose=False,
                 specified_links=None, method="ols"):
        '''
        Set up the causal network possibly with specified causal strengths
        
        Inputs:
        graph: path to DOT file containing a DAG or a string containing
               a DAG specification in DOT format. Format has to consist of lines
               with single-letter nodes plus a label. The links use the single 
               letters. e.g.

                dot_string = """digraph {bgcolor="#DDDDDD"
                bb="0,0,1,1";
                A[label="NAOlate"];
                B[label="ENSO"];
                C[label="IOD"];
                B -> A;
                B -> C;
                }""" 

        df:    Pandas DataFrame with a column for each node in nodes. The columns 
               can use either the single letter node convention or the labels 
               assigned in the DOT format.
        
        Keywords:
        path_coefficients:  a dictionary with keys (node1, node2) and value which 
                            stores the path coefficient between node1 causing node2.
        effect_modifiers:   a dictionary with keys (node1, node2) and value which 
                            stores the effect modifiers needed to estimate effect of
                            node1 causing node2.
        detrend:            detrend each column of the dataframe prior to estimation.
                            Doing so might mean that there are less problems caused
                            by non-stationarity.
        method:             regression method to use. Currently only "ols" works. 
                            Trying to implement "ridge".
        
        '''
        
        self._graph = graph
        self._links = path_coefficients
        self._effect_modifiers = effect_modifiers
        self._method = method
        self._nodes2labels, self._labels2nodes = make_mappers(graph.replace('style=invis', ''))
        self._nodes = sorted(self._nodes2labels.keys())
        self._labels = [self._nodes2labels[k] for k in self._nodes]
        self._noise = None
        if len(self._nodes) < 2:
            raise ValueError('Nodes %s must be a list of at least length 2.' % self._nodes)
        

        if df is None:
            self._df = None
# set up df as a dummy dataframe for call to CausalModel to work
            dummy_dict = dict([(node, [0, 1]) for node in self._nodes])
            df4model = pd.DataFrame(data=dummy_dict)
            need_fit = False
        else:
            if any([nn not in df.columns and self._nodes2labels[nn] not in df.columns for nn in self._nodes]):
                print([self._nodes2labels[nn] for nn in self._nodes if nn not in df.columns and self._nodes2labels[nn] not in df.columns])
                raise ValueError('Some nodes not used in DataFrame')

            self._df = df.rename(columns=self._labels2nodes)
            keep = [col for col in self._df.columns if col in self._nodes2labels]
            self._df = self._df[keep]
            if detrend:
                print('Detrending data')
                self._orig_df = self._df.copy()
                self._df = df_detrend(self._df)
            else:
                print('Not detrending data')
            df4model = self._df
            need_fit = True
        
        self._model = CausalModel(data=df4model,
                                  treatment=self._nodes[0],
                                  outcome=self._nodes[1],
                                  graph=graph
                                 )
        
        nodes_in_graph = list(self._model._graph.get_all_nodes())
        if sorted(nodes_in_graph) != sorted(self._nodes):
            print(list(set(nodes_in_graph).difference(self._nodes)))
            raise ValueError('Nodes %s does not match nodes in the graph %s' % (nodes, nodes_in_graph))
            
        self._causal_links = list()
        for node in nodes_in_graph:
            ancestors = self._model._graph.get_parents(node)
            for anc in ancestors:
                link = [anc, node]
                self._causal_links.append(link)

        if specified_links is not None:
            for k in specified_links:
                if list(k) not in self._causal_links:
                    raise KeyError('Specified link %s does not exist in SCM' % k)
        self._specified_links = specified_links

        if verbose:
            print('Causal links (to be used as keys for path coefficients):')
            for cl in self._causal_links:
                print(self._as_labels(cl))
#        print(self._causal_links)
        
        self._ordered_nodes = self._order_links()
        self._exogenous = [node for node in self._nodes if not self._model._graph.get_parents(node)]
        self._exogenous_labels = [self._nodes2labels[node] for node in self._exogenous]
        self._parents = sorted(set([cl[0] for cl in self._causal_links]))
        self._children = sorted(set([cl[1] for cl in self._causal_links]))
        self._used_nodes = self.get_all_used_nodes()
        self._used_labels = [self._nodes2labels[un] for un in self._used_nodes]

        
        if need_fit:
            print('Fitting model')
            self._links, self._effect_modifiers = self.fit_path_coefficients(self._df, specified_links=specified_links, method=self._method)
            self._noise = self.retrieve_noise(self._df)
            print('Fitted')
            self.validate()
            print('Validated that simulate() method is invertible.')
            self._path_contribution = dict()
            for path in self.get_all_paths():
                last_node = path[-1]
                first_node = path[0]
                self._path_contribution[tuple(path)] = numpy.prod([self._links[tuple(link)] for link in zip(path[:-1], path[1:])])

        

    @property
    def causal_links(self):
        return self._causal_links
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def ordered_nodes(self):
        return self._ordered_nodes

    @property
    def used_nodes(self):
        return self._used_nodes

    @property
    def used_labels(self):
        return self._used_labels

    @property
    def path_names(self):
        path_names = ['->'.join([self._nodes2labels[node] for node in path]) for path in self.get_all_paths()]
        return path_names

    def _as_labels(self, l):
        return [self._nodes2labels.get(v, v) for v in l]

    def _as_nodes(self, l):
        return [self._labels2nodes.get(v, v) for v in l]
        
    def _are_links_ok(self, path_dict):
        return sorted(path_dict.keys()) == sorted(self._causal_links)
    
    def _order_links(self):
        ordered = list()
        ancestors_dict =   dict((node, self._model._graph.get_ancestors(node)) for node in self.nodes)
        descendants_dict = dict((node, self._model._graph.get_descendants(node)) for node in self.nodes)
        nodes2use = self.nodes[:]
        
        while len(nodes2use) > 0:
            added = list()
            for k in nodes2use:
                v = ancestors_dict[k]
                if len(v) == 0:
                    ordered.append(k)
                    added.append(k)
                    nodes2use.remove(k)
                    
            if len(ordered) == 0:
                raise AssertionError('No exogenous variables found. Cyclic causal networks not allowed.')
                
            for i in added:
                for k in ancestors_dict:
                    if i in ancestors_dict[k]:
                        ancestors_dict[k].remove(i)
                        
        if len(ordered) != len(self.nodes):
            print(ordered)
            print(self.nodes)
            raise Exception('Something has gone wrong')
                    
        return ordered  

    def get_all_used_nodes(self):
        used = list()
        for node in self._model._graph.get_all_nodes():
            used += list(self._model._graph.get_parents(node)) + list(self._model._graph.get_descendants(node))
        used = list(set(used))
        return used

    def get_link(self, parent, child):
        if parent in self._labels:
            parent = self._labels2nodes[parent]
        if child in self._labels:
            child = self._labels2nodes[child]
        return scmfit._links[(parent, child)]

    def print_dataframe(self, topn=10):
        print(self._df[:topn])

    def specify_links(self, path_dict):
        '''
        Specify a set of path coefficients, one value for each causal link.
        
        Inputs:
        path_dict:  a dict with keys (node1, node2) and value which 
                    stores the path coefficient between node1 causing node2.
        
        '''
        
        if not self._are_links_ok(path_dict):
            raise KeyError('Causal links are not specified correctly')
            
        for k in path_dict:
            path_dict[k] = float(path_dict[k])
            
        self._links = path_dict
        
    def fit(self, df, specified_links=None, method="ols"):
        '''
        Fit a new SCM based on this one fitted to data.
        
        Inputs:
        df:   pandas.DataFrame that has a column for each of the nodes
        
        Return value:
        A new instance of LinearStructuralCausalModel with path coefficients
        fitted to the data.
        
        '''
        
        path_dict, effect_modifier_dict = self.fit_path_coefficients(df.rename(columns=self._labels2nodes),
                                               specified_links=specified_links,
                                               method=method)
        return self.__class__(self._graph, path_coefficients=path_dict, effect_modifiers=effect_modifier_dict)
        
    def fit_path_coefficients(self, df, specified_links=None, method="ols"):
        '''
        Fit a new SCM based on this one fitted to data.
        
        Inputs:
        df:   pandas.DataFrame that has a column for each of the nodes
        
        
        Return value:
        Dictionary of path coefficients fitted to the data.

        '''
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas dataframe')

        df = df.rename(columns=self._labels2nodes)
            
        for node in self.nodes:
            if node not in df:
                raise KeyError('%s not in the data' % node)
                
        path_dict, effect_modifier_dict = self._fit_causal_model(df, specified_links=specified_links, method=method)
        return path_dict, effect_modifier_dict
    
    
    def _fit_causal_model(self, df, specified_links=None, method="ols"):
        '''Fit the causal model to data'''

        def nice_print_estimand(ide, estimate_effect):
            instrument_labels = self._as_labels(ide.__dict__['instrumental_variables'])
            effect_modifiers = []
            if instrument_labels or estimate_effect.__dict__['params']:
                print('Link: %s -> %s' % (self._as_labels(ide.__dict__['treatment_variable']), self._as_labels(ide.__dict__['outcome_variable'])))
                if instrument_labels:
                    print('Instrumental variables:' % instrument_labels)
    #            print(estimate_effect.__dict__['params'])
                if estimate_effect.__dict__['params']:
                    effect_modifiers = self._as_labels(estimate_effect.__dict__['params']['effect_modifiers'])
                    print('Effect modifiers: %s' % effect_modifiers )
            return effect_modifiers

        if specified_links is None:
            specified_links = dict()

        if method == "ols":
            estimator = dowhy.causal_estimators.linear_regression_estimator.LinearRegressionEstimator
        else:
            estimator = RidgeRegressionEstimator
    
        path_dict = dict()
        effect_modifier_dict = dict()
        eps = 1e-8
        df = df.rename(columns=self._labels2nodes)
        for treatment, outcome in self._causal_links:
            key = (treatment, outcome)
            effect_key = '%s -> %s' % (self._nodes2labels[treatment], self._nodes2labels[outcome])
            if key in specified_links:
                print('Specified link %s: %s' % (effect_key, specified_links[key]))
                print()
                path_dict[key] = specified_links[key]
                effect_modifier_dict[effect_key] = 'This was specified'
            else:

                print('Estimating %s' % effect_key)
    # estimate total and then direct effects
                model=CausalModel(data=df, treatment=treatment, outcome=outcome, graph=self._graph)
                print('For total effect')
                identified_estimand = model.identify_effect()
    #            print(identified_estimand)
                estimate_total_effect = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
                effect_modifiers_total = nice_print_estimand(identified_estimand, estimate_total_effect)
                used_nodes = [treatment, outcome] + estimate_total_effect.__dict__['params']['effect_modifiers']
                print('Determinant: %.3f' % numpy.linalg.det(df[used_nodes].corr()))
                estimate_total_effect = estimate_total_effect.value

                print('For direct effect')            
                identified_estimand = model.identify_effect(estimand_type='nonparametric-nde')
    #            print(identified_estimand)
                estimate_direct_effect = model.estimate_effect(identified_estimand,
                                                               method_name="mediation.two_stage_regression",
                                                               method_params = {'first_stage_model':estimator,
                                                                                 'second_stage_model':estimator })
                try:
                    used_nodes = [treatment, outcome] + estimate_direct_effect.__dict__['params']['effect_modifiers']
                    print('Determinant: %.3f' % numpy.linalg.det(df[used_nodes].corr()))
                except:
                    pass
                effect_modifiers_direct = nice_print_estimand(identified_estimand, estimate_direct_effect)
    #            pdb.set_trace()
    # if direct is None or zero, use total effect, otherwise use direct effect
                print('Effects. Total=%s,   direct=%s' % (estimate_total_effect, estimate_direct_effect.value))
                if estimate_direct_effect.value is None or numpy.abs(estimate_direct_effect.value) < eps:
                    path_dict[key] = estimate_total_effect
                    effect_modifier_dict[effect_key] = effect_modifiers_total
                    print('Picking total effect')
                else:
                    path_dict[key] = estimate_direct_effect.value
                    effect_modifier_dict[effect_key] = effect_modifiers_direct
                    print('Picking direct effect')
                print()
        return path_dict, effect_modifier_dict

    def get_all_paths(self, child=None):
        all_paths = list()

        if child is None:
            for child in self._children:
                all_paths.extend(self.get_all_paths(child=child))
        else:
            for node in self._nodes:
                all_paths.extend(self._model._graph.get_all_directed_paths(node, child))
        return all_paths

    def get_full_paths(self):
        g = self._model._graph

        all_paths = list()
        for node in self._nodes:
            all_paths.extend(g.get_all_directed_paths(node, 'A'))
        print(all_paths)

        def is_path1_a_subpath(path1, path2):
            if path1 == path2:
                return False
            else:
                return path2[-len(path1):] == path1

        def path1_is_a_subpath_at_least_once(path, all_paths):
            return any(is_path1_a_subpath(path, p_) for p_ in all_paths)

        full_paths = [path for path in all_paths if not path1_is_a_subpath_at_least_once(path, all_paths)]
        return full_paths


    def get_all_contributions(self, child=None, verbose=True, noise_df=None):
        if child in self._labels:
            child = self._labels2nodes[child]
        all_paths = self.get_all_paths(child=child)

        preds = dict()
        total_effect = dict()
        if noise_df is None:
            noise_df = self.retrieve_noise()

        for path in all_paths:
            prod = self._path_contribution[tuple(path)]
            last_node = path[-1]
            first_node = path[0]
#            prod = numpy.prod([self._links[link] for link in zip(path[:-1], path[1:])])
            contribution = prod * noise_df[first_node]
            if verbose:
                print('->'.join([self._nodes2labels[node] for node in path]), prod) #, contribution)
            total_effect[last_node] = total_effect.get(last_node, 0.0) + contribution
            preds[tuple(path)] = contribution

        if verbose:
            print('Total effects:')
            for k, v in total_effect.items():
                print('%s: %s' % (self._nodes2labels[k], numpy.array(v)))
        return preds, total_effect

        
    def simulate(self, noise_df=None, new_links=None):
        '''
        Simulate data from the nodes using noise that has already been generated and stored
        in a pandas DataFrame with columns for each node.
        
        '''
        
        if noise_df is None:
            effect = self._noise.rename(columns=self._labels2nodes)
        else:
            effect = noise_df.rename(columns=self._labels2nodes)

        links = self._links.copy()
        if new_links is not None:
            for newlink, newval in new_links.items():
                if newlink not in self._links:
                    raise KeyError('Link %s -> %s not one of the causal links' % newlink)
                else:
                    links[newlink] = newval

        for o in self._ordered_nodes:
            parents = [k[0] for k in self._causal_links if k[1] == o]
            for parent in parents:
                key = (parent, o,)
                effect[o] += effect[parent] * links[key]
        return effect
        
    
    def compare(self, other, noise_dict):
        '''
        Compare mean and variances from simulations with this SCM and another one.
        
        '''
        simulations1 = self.simulate(noise_dict)
        simulations2 = other.simulate(noise_dict)
        print('Comparison of means')
        for node in self.nodes:
            print('%s    %.4f   %.4f' % (node, simulations1[node].mean(), simulations2[node].mean()))
            
        print()
        print('Comparison of variances')
        for node in self.nodes:
            print('%s    %.4f   %.4f' % (node, simulations1[node].var(), simulations2[node].var()))
    
    def view_model(self, processor=None, omit_strengths=False, 
                   layout='dot', size=(8, 6), filename='causal_model', display=True, width=600, height=400):
        dstring2 = nice_format_graph(self._graph, self, processor=processor, omit_strengths=omit_strengths)
# make a dummy scm object for plotting
        scm4plot = LinearStructuralCausalModel(dstring2, df=None)

# view_model adds PNG so remove this if present in keyword.
        file_name = filename.replace('.png', '')
        scm4plot._model.view_model(layout=layout, size=size, file_name=file_name)
        if display:
# capitalise first two letters of Ipython - see https://stackoverflow.com/questions/45179915/importerror-no-module-named-ipython. Thanks!
            from IPython.display import Image, display
            display(Image(filename=file_name + ".png", width=width, height=height))

        
    def refute_estimate(self, df=None, pval_thresh=0.1, verbose=10, ignore_unused=False):
        '''
        Test all the conditional independences implicated by the causal graph based
        on data in DataFrame df.

        Keywords:
        pval_thresh: p-value threshold of significance test, below which a test is failed signifying
                     conditional independence.
        verbose:     If the number of implicated tests 

        
        Return Value: Tuple (refute, list_of_failed_tests)
        refute: True if any tests failed
        list_of_failed_tests: list of tuples(failed_set, pvalue) - empty if refute is False.
        
        
        '''

        def all_nodes_are_used(tuple_of_tuples, used):
            '''Return True if all elements of tuple_of_tuples in used.

            '''
            full_set = list()
            for tt in tuple_of_tuples:
                full_set.extend(list(tt))
            return all([fs in used for fs in full_set])

        if df is None:
            df = self._df
        else:
            df = df.rename(columns=self._labels2nodes)
        
        refute = False
        nrefute = 0
        all_failed_tests = list()
        all_conditional_independent_sets = get_all_conditional_independent_sets(self._model._graph, verbose=verbose > 999)
        
        if ignore_unused:
            all_conditional_independent_sets = [acis for acis in all_conditional_independent_sets if all_nodes_are_used(acis,self._used_nodes)]
            print('Found %s conditional independent sets' % len(all_conditional_independent_sets) )     


        if len(all_conditional_independent_sets) <= verbose:
            print('Implicated tests:')
            print_list([(tuple(self._as_labels(pair)), self._as_labels(cond)) for pair, cond in all_conditional_independent_sets])

        for cis in all_conditional_independent_sets:
            formula = '%s ~ %s' % cis[0]
            if cis[1]:
                formula += ' + ' + ' + '.join(cis[1])
        #     print(formula, df.columns)
            result = sm.OLS.from_formula(formula=formula, data=df).fit()
        #     result.summary()
        #     dir(result)
            pval = result.pvalues[cis[0][1]]
            if pval < pval_thresh:
#                if len(all_conditional_independent_sets) <= verbose:
                cis1 = self._as_labels(cis[0])
                cis2 = self._as_labels(cis[1])
                print('%s and %s not independent conditional on %s' % (cis1[0], cis1[1], cis2))
                print(pval)
                print()
                refute = True
                nrefute += 1
                all_failed_tests.append((cis, pval))
            else:
                pass
#         print('%s and %s independent conditional on %s' % (cis[0][0], cis[0][1], cis[1]))
    
        if refute:
            print()
            print('%s implied tests out of %s failed at pval=%s' % (nrefute, len(all_conditional_independent_sets), pval_thresh))
        else:
            print('All %s implicated tests on conditional indepences all passed' % len(all_conditional_independent_sets))
        return refute, all_failed_tests
    
    def summary(self):
        self._model.summary(print_to_stdout=True)
        
    def retrieve_noise(self, df=None, new_links=None):
        '''
        Inverse problem where noise needed to generate dataframe df is inferred. Useful for testing
        counterfactuals for a given value of noise. If no dataframe parsed, use the one used to 
        estimate the Structural Causal Model.
        
        Return value:
        noise: dataframe with nodes "u" + node in df 
        
        
        '''

        links = self._links.copy()
        if new_links is not None:
            for newlink, newval in new_links.items():
                if newlink not in self._links:
                    raise KeyError('Link %s -> %s not one of the causal links' % newlink)
                else:
                    links[newlink] = newval

        
        if df is None:
            df = self._df
        df = df.rename(columns=self._labels2nodes)[self._nodes]
        noise = df.copy()
        for (parent, child), coefft in links.items():
            noise[child] = noise[child] - df[parent] * coefft
        return noise[self._nodes]

    def diagnostic_noise_plot(self, x=None, noise_df=None):
        if noise_df is None:
            noise_df = self._noise
        if x is None:
            x = numpy.arange(len(noise_df))
        noise_df = noise_df.rename(columns=self._labels2nodes)
        keys = [node for node in self._nodes if node not in self._exogenous]
        nkey = len(keys)
        for ii, k in enumerate(keys, 1):
            plt.subplot(nkey, 2, ii * 2 - 1)
            plt.plot(noise_df[k])
            plt.title(self._nodes2labels.get(k, k))
            plt.subplot(nkey, 2, ii * 2)
            plt.hist(noise_df[k])
            plt.axvline(x=0, linestyle=':', color='k')
            plt.title(self._nodes2labels[k])


    def print_explained_variance(self):
        if self._noise is  None:
            print('Model has not been fit yet')
        else:
            var = self._noise.rename(columns=self._nodes2labels).var()
            for node in self._nodes:
                if node not in self._exogenous:
                    label = self._nodes2labels[node]
                    print('%s: %.2f' % (label, var[label]))

    def validate(self, nt=30, tol=1e-10):
        '''
        Test to check that the retrieve_noise() returns something very close to the
        random noise to simulate a random time series.

        '''

        noise_dict = dict([(nn, numpy.random.randn(nt)) for nn in self.nodes])
        noise_df = pd.DataFrame(noise_dict)
        rand = self.simulate(noise_df)
        noise2_df = self.retrieve_noise(rand)

        max_diff = numpy.abs(numpy.array((noise2_df - noise_df))).max()
        if max_diff > tol:
            raise ValueError('''
                             In this causal model, the simulation method is not inverting correcly.
                             This is either that this causal model is not one that can be 
                             handled by this class, or there is an error in the implementation of
                             the simulate() or retrieve_noise() methods.
                             ''')

    def predict(self, noise_df=None, exogenous_only=False, parents_only=False, new_links=None):
        '''
        Predict output based on the noise from the original estimate or a specifed noise. The 
        prediction for each variable uses noise from their parents, so noise for the prediction
        variable is set to zero. 

        Return value:
        A dictionary with a key for each node storing the prediction.

        Keywords:
        noise_df:        Specified noise (default uses noise from original estimate
        exogenous_only:  Only use noise from exogenous variables
        parents_only:    Only use noise from nodes that are parents.
        

        '''
        if noise_df is None:
            noise_df = self._noise.rename(columns=self._labels2nodes)
        else:
            noise_df = noise_df.rename(columns=self._labels2nodes)

        if exogenous_only and parents_only:
            raise ValueError('Cannot choose both exogenous_only and parents_only')
        elif exogenous_only:
            for node in self._nodes:
                if node not in self._exogenous:
                    noise_df[node] *= 0.0
            ans = self.simulate(noise_df=noise_df, new_links=new_links)
        elif parents_only:
            for node in self._nodes:
                if node not in self._parents:
                    noise_df[node] *= 0.0
            ans = self.simulate(noise_df=noise_df, new_links=new_links)
        else:
            ans = noise_df
            base_noise = noise_df.copy()
            for node in self._nodes:
                if node not in self._exogenous:
                    base_noise[node] *= 0.0
            for node in self._nodes:
                if node not in self._exogenous:
                    ancestors = self._model._graph.get_ancestors(node)
                    noise_ = base_noise.copy()
                    for ancestor in ancestors:
                        noise_[ancestor] = noise_df[ancestor]
                ans[node] = self.simulate(noise_df=noise_, new_links=new_links)[node]


        return ans
        
#===========================================================
#===========================================================
#===========================================================




            
class CausalData(object):
    '''
    Stores data in a way that enables causal analysis
    (C) Crown Copyright Met Office, UK.
    
    '''
    
    @staticmethod
    def from_search_string(indir, sstring, label, varnames, names, long_names, target,
                           process_var_dict=None, label_long_name=None,
                           using_ensemble=True, use_standardisation_from=None,
                           rescale_var_dict=None, crit=None):
        clist = iris.cube.CubeList()
        if crit is None:
            crit = iris.Constraint()
        if process_var_dict is None:
            process_var_dict = dict()
        if rescale_var_dict is None:
            rescale_var_dict = dict()
        for name, var in zip(names, varnames):
            cube = iris.load_cube(os.path.join(indir, sstring % (var, label)), crit)
            if name in process_var_dict:
                print('Post-processing %s' % var)
                print('Starting with %s' % repr(cube))
                cube = process_var_dict[name](cube) * rescale_var_dict.get(name, 1.0)
                cube.rename(var)
                print(repr(cube))
            else:
                print('Not processing', var, process_var_dict)
            clist.append(cube)

        clist = match_times(clist, names, target)
        clist = match_members(clist)

        ndims = [c.ndim for c in clist]
        if min(ndims) < 2:
            clist = iris.cube.CubeList([make_cube_2d(c) for c in clist])

        return CausalData(clist, names, long_names, target,
                          label=label, label_long_name=label_long_name,
                          using_ensemble=using_ensemble, use_standardisation_from=use_standardisation_from)
            
    def __init__(self, cntl_cubelist, names, long_names, target,
                 label='cntl', label_long_name='Control',
                 using_ensemble=True, use_standardisation_from=None):
        '''
        Initialise with a set of Nvar control cubes with names and long_names for each of the Nvar variables.
        Keyword:
        label:    Label for the experiment that the cubes came from.
        using_ensemble: if cubes 2-d then take ensemble mean. If 1-d, add realization.
        use_standardisation_from: another CausalData object which supplies the standard deviation and mean for standardisation.
        
        '''
        if len(names) != len(cntl_cubelist):
            raise ValueError('names is length %s but should be length %s like cntl_cubelist' % (len(names), len(cntl_cubelist)))
        if len(long_names) != len(cntl_cubelist):
            raise ValueError('long_names is length %s but should be length %s like cntl_cubelist' % (len(long_names), len(cntl_cubelist)))
        if target not in names:
            raise ValueError('%s needs to be one of %s' % (target, names))
        ndims = list(set([c.ndim for c in cntl_cubelist]))
        if len(ndims) > 1:
            raise ValueError('All cubes need to have same number of dimensions')
        else:
            ndim = ndims[0]
        if ndim < 1 or ndim > 2:
            raise ValueError('Cubes must be 1- or 2-d')

# if single realisation e.g. a reanalysis, then add realization dimension to make 2-d cubes
        self._using_ensemble = using_ensemble
        if not self._using_ensemble:
            if ndim == 2:
                cubes = iris.cube.CubeList([c.collapsed('realization', iris.analysis.MEAN) for c in cntl_cubelist])
            else:
                realn_coord = iris.coords.DimCoord([1], long_name='realization')
                cubes = iris.cube.CubeList()
                for c in cntl_cubelist:
                    print(repr(c))
                    if c.coords('realization'):
                        c.remove_coord(c.coord('realization'))
                    c.add_aux_coord(realn_coord)
                    cubes.append(iris.util.new_axis(c, scalar_coord='realization'))
                cntl_cubelist = cubes
        
        for c in cntl_cubelist:
            if len(c.shape) != 2:
                print(cntl_cubelist)
                raise ValueError('Cubes need to be 2-d')
        
        self._nmem = cntl_cubelist[0].shape[0]
        self._rips = ['%.4i' % (rn - 1100000) for rn in cntl_cubelist[0].coord('realization').points]
        self._names = names
        self._target = target
        self._long_names = dict(zip(names, long_names))
        self._predictions = dict()
        self._causal_strengths = dict()
        
        if use_standardisation_from is None:
            self._mean = dict([n, c.data.mean(-1)] for n, c in zip(names, cntl_cubelist))
            self._stdv = dict([n, c.data.std(-1).mean()] for n, c in zip(names, cntl_cubelist))
        else:
            self._mean = use_standardisation_from._mean.copy()
            self._stdv = use_standardisation_from._stdv.copy()

        self._units = dict([n, c.units] for n, c in zip(names, cntl_cubelist))
        self._cubes = dict()
        self._standardised_cubes = dict()
        self._label_long_names = {label:label_long_name}
        self.add_cubes(label, cntl_cubelist)
        
    @property
    def long_names(self):
        return self._long_names
    
    @property
    def label_long_names(self):
        return self._label_long_names
    
    @property
    def names(self):
        return self._names
    
    @property
    def nvar(self):
        return len(self._names)

    @property
    def nlabels(self):
        return len(self._cubes)
    
    @property
    def nmem(self):
        return self._nmem
    
    @property
    def rips(self):
        return self._rips

    def get_time(self, key=None, coord='season_year'):
        if len(self._cubes) == 1:
            key = list(self._cubes.keys())[0]
            return self._cubes[key][0].coord(coord).points
        else:
            if key is None:
                raise ValueError('key= must be set as multiple cubes')
            else:
                return self._cubes[key][0].coord(coord).points
            
        
    def add_cubes(self, label, cubelist):

        if not self._using_ensemble:
            realn_coord = iris.coords.DimCoord([1], long_name='realization')
            cubes = iris.cube.CubeList()
            for c in cubelist:
                if len(c.shape) < 2:
                    if c.coords('realization'):
                        c.remove_coord(c.coord('realization'))
                    c.add_aux_coord(realn_coord)
                    cubes.append(iris.util.new_axis(c, scalar_coord='realization'))
                else:
                    cubes.append(c)
            cubelist = cubes

        if len(set([c.shape for c in cubelist])) != 1:
            print(cubelist)
            raise ValueError('All cubes need to be same shape')
        ok = True
        for c in cubelist:
            if c.shape[0] != self._nmem:
                ok = False
                print(c)
        if not ok:
            raise ValueError('At least one cube has wrong size of 1st dimension')
            
        if not isinstance(cubelist, iris.cube.CubeList):
            cubelist = iris.cube.CubeList(cubelist)
            
        self._cubes.update({label:cubelist})
        std_cubes = iris.cube.CubeList([c.copy(data=((c.data.T - self._mean[n]).T / self._stdv[n])) for n, c in zip(self.names, cubelist)])
        self._standardised_cubes.update({label:std_cubes})
        self._label_long_names[label] = label
        
    def subselect_experiment(self, label, index, new_label, label_long_name=None):
        if new_label in self._cubes:
            raise ValueError('%s already used in set of labels %s' % (new_label, self._cubes.keys()))
        new_cubelist = iris.cube.CubeList([c[:,index] for c in self._cubes[label]])
        self.add_cubes(new_label, new_cubelist)
        if label_long_name is None:
            self._label_long_names[new_label] = label
        else:
            self._label_long_names[new_label] = label_long_name

    def combine_experiments(self, labels_list, new_label, label_long_name=None, offsets=None):
        if offsets is None:
            offsets = [0] * self.nvar
        if new_label in self._cubes:
            raise ValueError('%s already used in set of labels %s' % (new_label, self._cubes.keys()))
        if label_long_name is None:
            label_long_name = '+'.join(labels_list)
        new_cubelist = iris.cube.CubeList()
        all_cubes = [self._cubes[label] for label in labels_list]
        for c in zip(*all_cubes):
            print(iris.cube.CubeList(c))
            for c_, offset in zip(c, offsets):
                new_coord = c_.coord('time').copy(points=c_.coord('time').points - offset)
                c_.replace_coord(new_coord)
                print(c_.coord('time').points[0], c_.coord('time').points[-1])
                print(c_.coord('time').units)
            new_cubelist.append(iris.cube.CubeList(c).concatenate_cube())
        self.add_cubes(new_label, new_cubelist)   
        self._label_long_names[new_label] = label_long_name    
        
    def add_by_search_string(self, indir, sstring, label, varnames, process_var_dict=None, label_long_name=None):
        if process_var_dict is None:
            process_var_dict = dict()

        clist = iris.cube.CubeList()
        for name, var in zip(names, varnames):
            cube = iris.load_cube(os.path.join(indir, sstring % (var, label)))
            if name in process_var_dict:
                print('Post-processing %s' % var)
                cube = process_var_dict[name](cube)
            else:
                print('Not processing', var, process_var_dict)
            clist.append(cube)

        clist = match_times(clist, self._names, self._target)
        clist = match_members(clist)
        self.add_cubes(label, clist)
        if label_long_name is None:
            self._label_long_names[label] = label
        else:
            self._label_long_names[label] = label_long_name
        
    def _plot_data(self, xx, data):
        p1 = plt.plot(xx, data.T)
      
    def raw_cube(self, label, *var):
        if len(var) == 0:
            return self._cubes[label]
        else:
            return self._cubes[label][self._names.index(var[0])]
        
    def standardised_cube(self, label, *var):
        if len(var) == 0:
            return self._standardised_cubes[label]
        else:
            return self._standardised_cubes[label][self._names.index(var[0])]
        
    def raw_data(self, label, *var):
        return self.raw_cube(label, *var).data
    
    def standardised_data(self, label, *var):
        if len(var) > 0:
            return self.standardised_cube(label, *var).data
        else:
            cubes = self.standardised_cube(label)
            return numpy.array([c.data for c in cubes])

    def standardised_dataframe(self, label, member, *var):
        if len(var) == 0:
            df = pd.DataFrame(self.standardised_data(label)[:,member,:].T, columns=self.names)
        else:
            df = pd.DataFrame(self.standardised_data(label, *var)[:,member,:].T, columns=var)
        return df
        
    def plot_raw_data(self, label, var, tcoord='season_year'):
        xx = self._cubes[label][0].coord(tcoord).points
        p1 = self._plot_data(xx, self.raw_data(label, var))
        plt.title('%s\n%s' % (self._long_names[var], label))
        if 'year' in tcoord:
            plt.xlabel('Year')
        plt.ylabel(self._units[var])
        
    def plot_standardised_data(self, label, var, tcoord='season_year', nmem=None):
        xx = self._cubes[label][0].coord(tcoord).points
        if nmem is None:
            p1 = self._plot_data(xx, self.standardised_data(label, var))
        else:
            p1 = self._plot_data(xx, self.standardised_data(label, var)[nmem])
        plt.title('%s\n%s' % (self._long_names[var], label))
        if 'year' in tcoord:
            plt.xlabel('Year')
        plt.ylabel('Standardised units [1]')
        plt.ylim(-5, 5)
        
    def compare_members(self, label, var, tcoord='season_year', nrow=5, ncol=4, standardised=True, **kwargs):
        xx = self._cubes[label][0].coord(tcoord).points
        if standardised:
            data = self.standardised_data(label, var)
            units = 'Standardised units [1]'
        else:
            data = self.raw_data(label, var)
            units = self._units[var]
            
        plt.suptitle('%s\n%s' % (self._long_names[var], label))
        
        for ii in range(self.nmem):
            plt.subplot(nrow, ncol, ii+1)
            plt.fill_between(xx, data.min(0), y2=data.max(0), color='k', alpha=0.3)
            plt.plot(xx, data.mean(0), linestyle=':', color='k')
            plt.plot(xx, data[ii], **kwargs)
            plt.title(self.rips[ii])
            if 'year' in tcoord and ii % ncol == 0:
                plt.xlabel('Year')
            if (ii+1) >= (nrow - 1) * ncol + 1:
                plt.ylabel(units)
            plt.ylim(-5, 5)

        
    def scatter(self, expt_label, xvar, yvar, member, index=slice(None, None, None), detrend=False, label=None, **kwargs):
        xx = self.standardised_data(expt_label, xvar)[member, index]
        yy = self.standardised_data(expt_label, yvar)[member, index]
        if detrend:
            xx = detrend2d(xx)
            yy = detrend2d(yy)
        if label is None:
            label = self._label_long_names[expt_label]
        pc = plt.scatter(xx, yy, label=label, **kwargs)
        slope, intercept, r, p, se = stats.linregress(xx, yy)
        xl = numpy.array([xx.min(), xx.max()])
        yl = intercept + slope * xl
        plt.plot(xl, yl, linestyle='--')
        sslope, sintercept = stats.siegelslopes(yy, x=xx)
        plt.plot(xl, sintercept + sslope * xl, linestyle=':')
        tslope, tintercept, tlo, tup = stats.theilslopes(yy, x=xx)
        plt.plot(xl, tintercept + tslope * xl, linestyle='-.')
        plt.ylabel(self.long_names[yvar])
        plt.xlabel(self.long_names[xvar])
        return pc

    def plot_convex_hull(self, expt_label, xvar, yvar, member, index=slice(None, None, None), label=None, **kwargs):
        xx = self.standardised_data(expt_label, xvar)[member, index]
        yy = self.standardised_data(expt_label, yvar)[member, index]
        if label is None:
            label = self._label_long_names[expt_label]
        pc = plot_convex_hull(xx, yy, label=label, **kwargs)
        slope, intercept, r, p, se = stats.linregress(xx, yy)
        xl = numpy.array([xx.min(), xx.max()])
        yl = intercept + slope * xl
        plt.plot(xl, yl, linestyle='--', color=kwargs.get('color', None))

        plt.ylabel(self.long_names[yvar])
        plt.xlabel(self.long_names[xvar])
        return pc
    
    def cdf2d(self, expt_label, xvar, yvar, member, index=slice(None, None, None), **kwargs):
        xx = self.standardised_data(expt_label, xvar)[member, index]
        yy = self.standardised_data(expt_label, yvar)[member, index]
        pc = cdf2d(xx, yy, **kwargs)
        plt.ylabel(self.long_names[yvar])
        plt.xlabel(self.long_names[xvar])
        return pc
    
    def estimate_causal_strength(self, link_name, xvar, yvar, condition=None, window=None,
                                 obs_tvalue=None, obs_yvalue=None, tcoord='season_year', detrend=False):
        causal_dict = dict()
        for label in self._cubes.keys():
            xx = self.standardised_data(label, xvar)
            yy = self.standardised_data(label, yvar)
            if condition is None:
                zz = None
            else:
                zz = numpy.array([self.standardised_data(label, cc).T for cc in condition])
                zz = numpy.transpose(zz)
            nrun, ntscen = xx.shape
            if window is None:
                nwindow = ntscen
            else:
                nwindow = window
            slopes = numpy.zeros((nrun, ntscen - nwindow + 1))
            times = numpy.zeros(ntscen - nwindow + 1)
            for i in range(ntscen - nwindow + 1):
#                 pdb.set_trace()
                slopes[:,i] = slope2d(xx[:,i:i+nwindow], yy[:,i:i+nwindow], detrend=detrend, condition=zz[:,i:i+nwindow,:])
                times[i] = self.raw_cube(label, xvar).coord(tcoord).points[i:i+nwindow].mean()
                causal_dict[label] = (times, slopes)

        result = dict(name=link_name, xvar=xvar, yvar=yvar, condition=condition,
                      window=window, obs_tvalue=obs_tvalue, obs_yvalue=obs_yvalue,
                      tcoord=times, causal_strength=causal_dict)
        self._causal_strengths[link_name] = result
        return result
        

    def plot_causal_strength(self, link_name, label, imem, span=False, tstart=None, **kwargs):
        plot_causal_strength(causal_strength, label, imem, span=span, tstart=tstart, **kwargs)
    
    def predict(self, cause_to_use, expt_to_use, expt_to_predict,
                rescale=True, anomaly=True, use_ensemble_mean_strength=False):
        '''
        Inputs:
        cause_to_use:    causal_strength dictionary. Stores causal_strength and x- and y-variables
                           and is produced first by call to self.estimate_causal_strengths.
        expt_to_use:     which label to use in cause_to_use['causal_strength']
        expt_to_predict: label of standardised data to predict
        
        e.g. prediction_result = self.predict(causal_strengths, 'cntl', 'scen')
        
        '''
        cs = cause_to_use['causal_strength']
        times, strengths = cs[expt_to_use]
        if strengths.shape[-1] != 1:
            print('Strengths shape is %s' % strengths.shape)
            raise ValueError('Only strengths estimated from the full time window, not a running window can be used. Second dimenison needs to of size 1.')

        if use_ensemble_mean_strength:
            strengths[:] = strengths.ravel().mean()
        
        xvar = cause_to_use['xvar']
        yvar = cause_to_use['yvar']
        link = cause_to_use['name']
        cube = self.standardised_cube(expt_to_predict, xvar)
        if rescale:
            if anomaly:
                pred = cube.copy(data=(cube.data.T * strengths.ravel() * self._stdv[yvar]).T)
            else:
                pred = cube.copy(data=(cube.data.T * strengths.ravel() * self._stdv[yvar] + self._mean[yvar]).T)
        else:
            pred = cube.copy(data=(cube.data.T * strengths.ravel()).T)
        if use_ensemble_mean_strength:
            link += ' using ensmean'
        self._predictions[(link, xvar, yvar, expt_to_use, expt_to_predict)] = pred
        return pred
    
    def plot_prediction(self, expt_to_use, expt_to_predict, causal_name, nrow=5, ncol=4, tcoord='season_year', colors=('DodgerBlue', 'orange')):
        use_ensemble_mean = ' using ensmean' in causal_name
        causal_name = causal_name.replace(' using ensmean', '')
        cause_to_use = self._causal_strengths[causal_name]
        xvar = cause_to_use['xvar']
        yvar = cause_to_use['yvar']
        link = cause_to_use['name']
        if use_ensemble_mean:
            link += ' using ensmean'
        key = (link, xvar, yvar, expt_to_use, expt_to_predict)
        pred = self._predictions[key]
        cube = self.standardised_cube(expt_to_predict, yvar)
        for i in range(self._nmem):
            plt.subplot(nrow, ncol, i+1)
            times = cube.coord(tcoord).points
            plt.plot(times, cube.data[i,:], color=colors[0])
            plt.plot(times, pred.data[i,:], color=colors[1])
            plt.title(self.rips[i])
            
    
    def plot_residuals(self, expt_to_use, expt_to_predict, causal_name, tcoord='season_year', **kwargs):
        use_ensemble_mean = ' using ensmean' in causal_name
        causal_name = causal_name.replace(' using ensmean', '')
        cause_to_use = self._causal_strengths[causal_name]
        xvar = cause_to_use['xvar']
        yvar = cause_to_use['yvar']
        link = cause_to_use['name']
        if use_ensemble_mean:
            link += ' using ensmean'
        key = (link, xvar, yvar, expt_to_use, expt_to_predict)
        pred = self._predictions[key]
        cube = self.standardised_cube(expt_to_predict, yvar)
        residuals = cube.data - pred.data
        for i in range(self._nmem):
            times = cube.coord(tcoord).points
            plt.plot(times, residuals.data[i,:], **kwargs)
            
        kwargs['linewidth'] = kwargs.get('linewidth', 1) * 3
        plt.plot(times, residuals.mean(0), **kwargs)
    
    def plot_predictions_together(self, expt_to_use, expt_to_predict, causal_name, tcoord='season_year', **kwargs):
        use_ensemble_mean = ' using ensmean' in causal_name
        causal_name = causal_name.replace(' using ensmean', '')
        cause_to_use = self._causal_strengths[causal_name]
        xvar = cause_to_use['xvar']
        yvar = cause_to_use['yvar']
        link = cause_to_use['name']
        if use_ensemble_mean:
            link += ' using ensmean'
        key = (link, xvar, yvar, expt_to_use, expt_to_predict)
        pred = self._predictions[key]
        cube = self.standardised_cube(expt_to_predict, yvar)
        residuals = cube.data - pred.data
        for i in range(self._nmem):
            times = cube.coord(tcoord).points
            kwargs['label'] = self.rips[i]
            plt.plot(times, pred.data[i,:], **kwargs)
            plt.text(times[-1] + times.ptp() * 0.1, pred.data[i,-1], self.rips[i])
            
#         kwargs['linewidth'] = kwargs.get('linewidth', 1) * 3
#         plt.plot(times, residuals.mean(0), **kwargs)
    
#     def compare_causal_strengths(self, list_of_causal_links, expt_labels):
#         pass

    def rolling_scatter(self, axes, expt_label, xvar, yvar, member, window=70, step=10, tcoord='season_year',
                        index=slice(None, None, None), detrend=False, label=None, **kwargs):
        xx = self.standardised_data(expt_label, xvar)[member, index]
        yy = self.standardised_data(expt_label, yvar)[member, index]
        if detrend:
            xx = detrend2d(xx)
            yy = detrend2d(yy)
        if label is None:
            label = self._label_long_names[expt_label]

        nt = xx.shape[-1]
        starts = numpy.arange(0, nt - window, step)
        nroll = starts.size
        print('Number of rolling windows: %s' % nroll)

        times = self.raw_cube(expt_label, xvar).coord(tcoord).points

        for i0, i1 in enumerate(starts):
            rolling_index = numpy.arange(i1, i1 + window)
            xx_ = xx[rolling_index]
            yy_ = yy[rolling_index]
            plt.sca(axes.ravel()[i0])
            t0, t1 = times[rolling_index].min(), times[rolling_index].max()

            pc = plt.scatter(xx_, yy_, label=label, **kwargs)
            pc = plt.scatter(xx_[:step], yy_[:step], color='b')
            pc = plt.scatter(xx_[-step:], yy_[-step:], color='r')
            slope, intercept, r, p, se = stats.linregress(xx_, yy_)
            xl = numpy.array([xx_.min(), xx_.max()])
            yl = intercept + slope * xl
            plt.plot(xl, yl, linestyle='--')
            sslope, sintercept = stats.siegelslopes(yy_, x=xx_)
            plt.plot(xl, sintercept + sslope * xl, linestyle=':')
            tslope, tintercept, tlo, tup = stats.theilslopes(yy_, x=xx_)
            plt.plot(xl, tintercept + tslope * xl, linestyle='-.')
            plt.ylabel(self.long_names[yvar])
            plt.xlabel(self.long_names[xvar])
            plt.title('%s-%s, slope = %.3f' % (t0, t1, slope))



class CausalDataWithLinearStructuralCausalModel(CausalData):

    @staticmethod
    def from_search_string(graph, indir, sstring, label, varnames, names, long_names, target,
                           process_var_dict=None, label_long_name=None,
                           using_ensemble=True, use_standardisation_from=None,
                           rescale_var_dict=None, criteria=None,
                           specified_links=None):
        clist = iris.cube.CubeList()
        if process_var_dict is None:
            process_var_dict = dict()
        if rescale_var_dict is None:
            rescale_var_dict = dict()
        for name, var in zip(names, varnames):
            fcube = os.path.join(indir, sstring % (var, label))
            print(fcube)
            try:
                cube = iris.load_cube(fcube)
                cubes = None
            except:
                cubes = iris.load(fcube)
                cube = None

            if cubes is not None:
                if criteria is not None:
                    cubes = cubes.extract(criteria)
                    if len(cubes) == 0:
                        raise ValueError('No cubes matched criteria')
                if not cubes[0].coords('realization'):
                    print('Need to add realization and merge')
                    sources = sorted([c.attributes['source_id'] for c in cubes])
                    if len(sources) != len(set(sources)):
                        raise ValueError('Sources %s are not unique' % sources)
                    cubes_ = iris.cube.CubeList([iris.util.new_axis(c) for c in cubes])
                    for c in cubes_:
                        c.var_name = None
                    for ii, c in enumerate(cubes_):
                        source_ = c.attributes['source_id']
                        isource = sources.index(source_)
                        c.add_dim_coord(iris.coords.DimCoord([isource], long_name='realization'), 0)
                        c.add_aux_coord(iris.coords.AuxCoord([source_], long_name='source_id'), 0)
                    iris.util.equalise_attributes(cubes_)
                    for c in cubes_:
                        c.long_name = cubes_[0].long_name   # fudge for hi-res MIP siconc
                cube = match_tcoord(cubes_).concatenate_cube()
            else:
                if cube is None:
                    cube = cubes.merge_cube()
                    

            if name in process_var_dict:
                print('Post-processing %s' % var)
                print('Starting with %s' % repr(cube))
                cube = process_var_dict[name](cube) * rescale_var_dict.get(name, 1.0)
#                print(cube)
            else:
                print('Not processing', var, process_var_dict)
            cube.rename(var)
            print('here', repr(cube))

            clist.append(cube)
#            print('Before', clist)

#        pdb.set_trace()
        clist_safe = clist[:]
        clist = match_times(clist, names, target)
#            print('After match time', clist)
        clist = match_members(clist)
#            print('After match members', clist)
        is_ensemble = clist[0].ndim > 1 and clist[0].shape[0] > 1
        return CausalDataWithLinearStructuralCausalModel(graph, clist, names, long_names, target,
                                                         specified_links=specified_links,
                                                         label=label, label_long_name=label_long_name,
                                                         using_ensemble=using_ensemble and is_ensemble, 
                                                         use_standardisation_from=use_standardisation_from)

    def __init__(self, graph, cntl_cubelist, names, long_names, target,
                 specified_links=None,
                 label='cntl', label_long_name='Control',
                 using_ensemble=True, use_standardisation_from=None):
        super(CausalDataWithLinearStructuralCausalModel, self).__init__(cntl_cubelist, names, long_names, target,
                                                                        label=label,
                                                                        label_long_name=label_long_name,
                                                                        using_ensemble=using_ensemble,
                                                                    use_standardisation_from=use_standardisation_from)
        self.set_model(graph, specified_links=specified_links)


    def set_model(self, graph, specified_links=None):   #, nodes, mapper=None):
        self._model = LinearStructuralCausalModel(graph, df=None, specified_links=specified_links)
#        self._mapper = mapper

    def estimate_causal_strength(self, window=None, step=1, detrend=False, obs_tvalue=None, obs_yvalue=None,
                                 tcoord='season_year',  new_variables_from_residuals=None, postprocessor=None):

        causal_dict = dict()

        for label in self._cubes.keys():
            df = self.standardised_dataframe(label, 0)
            cube = self._cubes[label][0]

# test if the mapper which maps names to simple letters implies that the dataframe has columns for each of the longer names
# and that df columns need to be remapped to letters.
            orig_nodes = self._model._labels
            tests = [orig in df.columns for orig in orig_nodes]
            print(orig_nodes, tests)
#            if all(tests):
#                print('Applying mapper in reverse')
#                
#                df = df.rename(columns=self._mapper)
            first = True
            failed = list()
            for imem in range(self._nmem):
                df = self.standardised_dataframe(label, imem)
                if new_variables_from_residuals is not None and postprocessor is not None:
                    df = postprocessor(df, cube, new_variables_from_residuals, imem=imem)
#                    df = add_variable_based_on_residuals(df, new_variables_from_residuals)

                if all(tests):
                    df = df.rename(columns=self._model._labels2nodes)
                ntscen, nvar = df.shape
                if window is None:
                    nwindow = ntscen
                else:
                    nwindow = window

                rolling_window = numpy.array([numpy.arange(i, i+nwindow) for i in numpy.arange(0, ntscen, step) if i+nwindow-1 < ntscen])
                nroll = rolling_window.shape[0]

                for i, iroll in enumerate(rolling_window):
                    df_ = df.iloc[iroll]
                    if detrend:
                        df_ = df_detrend(df_)
                    fitted = self._model.fit(df_)
#                    pdb.set_trace()
                    path_coeffts = fitted._links
                    refute, all_failed_tests = fitted.refute_estimate(df=df_, ignore_unused=True)
                    failed.append(all_failed_tests)
                    noise = fitted.retrieve_noise(df_)
                    nlinks = len(path_coeffts)
                    link_order = sorted(path_coeffts.keys())
                    if first:
                        coeffts = numpy.zeros((self._nmem, nroll, nlinks))
                        times = numpy.zeros(nroll)
                        first = False
                        noises = []

                    for ii, lo in enumerate(link_order):
                        coeffts[imem,i,link_order.index(lo)] = path_coeffts[lo]
                    if imem == 0:
                        times[i] = self._cubes[label][0].coord(tcoord).points[iroll].mean()
                    noises.append(noise)
                
            causal_dict[label] = (times, coeffts, failed, noises)

# Need to mirror this from CausalData method
#        result = dict(name=link_name, xvar=xvar, yvar=yvar, condition=condition,
#                      window=window, obs_tvalue=obs_tvalue, obs_yvalue=obs_yvalue,
#                      tcoord=times, causal_strength=causal_dict)
#        self._causal_strengths[link_name] = result

        names = ['%s->%s' % (self._model._nodes2labels[lo[0]], self._model._nodes2labels[lo[1]]) for lo in link_order]
        xvar = [self._model._nodes2labels[lo[0]] for lo in link_order]
        yvar = [self._model._nodes2labels[lo[1]] for lo in link_order]
        result = dict(names=names,
                      xvar=xvar,
                      yvar=yvar,
                      obs_tvalue=None,
                      obs_yvalue=None,
                      window=window,
                      tcoord=times,
                      causal_strength=causal_dict)

#        print('NEED TO REORDER THIS')

#        self._causal_strengths[link_name] = result
        return result
        

    def predict(self, expt_to_predict):   
#                rescale=True, anomaly=True, use_ensemble_mean_strength=False):
        '''
        Inputs:
        cause_to_use:    causal_strength dictionary. Stores causal_strength and x- and y-variables
                           and is produced first by call to self.estimate_causal_strengths.
        expt_to_use:     which label to use in cause_to_use['causal_strength']
        expt_to_predict: label of standardised data to predict
        
        e.g. prediction_result = self.predict(causal_strengths, 'cntl', 'scen')
        
        '''

        for label in self._cubes.keys():
            pred_df = self.standardised_dataframe(expt_to_predict, 0)

# test if the mapper which maps names to simple letters implies that the dataframe has columns for each of the longer names
# and that df columns need to be remapped to letters.
            orig_nodes = self._model._labels
            tests = [orig in df.columns for orig in orig_nodes]
            print(orig_nodes, tests)
#            if all(tests):
#                print('Applying mapper in reverse')
#                
#                df = df.rename(columns=self._mapper)
            first = True
            for imem in range(self._nmem):
                df = self.standardised_dataframe(expt_to_predict, imem)
                if new_variables_from_residuals is not None:
                    pred_df = add_variable_based_on_residuals(pred_df, new_variables_from_residuals)

                noise_df = pred_df.copy()
                for node in self._model._nodes:
                    if node not in self._model._exogenous:
                        noise_df[node] *= 0.0

                pred = self._model.simulate(noise_df)
                self._predictions[(expt_to_predict, imem)] = pred



#        cs = cause_to_use['causal_strength']
#        times, strengths = cs[expt_to_use]
#        if strengths.shape[-1] != 1:
#            print('Strengths shape is %s' % strengths.shape)
#            raise ValueError('Only strengths estimated from the full time window, not a running window can be used. Second dimenison needs to of size 1.')

#        if use_ensemble_mean_strength:
#            strengths[:] = strengths.ravel().mean()
        
#        xvar = cause_to_use['xvar']
#        yvar = cause_to_use['yvar']
#        link = cause_to_use['name']
#        cube = self.standardised_cube(expt_to_predict, xvar)
#        if rescale:
#            if anomaly:
#                pred = cube.copy(data=(cube.data.T * strengths.ravel() * self._stdv[yvar]).T)
#            else:
#                pred = cube.copy(data=(cube.data.T * strengths.ravel() * self._stdv[yvar] + self._mean[yvar]).T)
#        else:
#            pred = cube.copy(data=(cube.data.T * strengths.ravel()).T)
#        if use_ensemble_mean_strength:
#            link += ' using ensmean'
#        self._predictions[(link, xvar, yvar, expt_to_use, expt_to_predict)] = pred
        return pred


