from .elements import PostProcessorElement
from .elements import MCMCPostProcessorElement, MultinestPostProcessorElement, WeightedMCMCPostProcessorElement
from .elements import Loadable
from .outputs import PostprocessPlot
from ..plotting.kde import KDE
from ..runtime import Parameter
from .utils import std_weight, mean_weight
from .density import smooth_density_estimate_1d, smooth_density_estimate_2d
from . import cosmology_theory_plots
import configparser
import numpy as np
import scipy.optimize
from . import lazy_pylab as pylab
import itertools
import os
import warnings


default_latex_file = os.path.join(os.path.split(__file__)[0], "latex.ini")
legend_locations = {
"BEST": 0,
"UR": 1,
"UL": 2,
"LL": 3,
"LR": 4,
"R": 5,
"CL": 6,
"CR": 7,
"LC": 8,
"UP": 8,
"C": 10}

class Plots(PostProcessorElement):
    excluded_columns = []
    def __init__(self, *args, **kwargs):
        super(Plots, self).__init__(*args, **kwargs)
        self.figures = {}
        self.no_latex = self.options.get("no_latex")
        latex_file = self.options.get("more_latex") 
        self._latex = {}
        self.plot_set = self.source.index
        if self.source.cosmosis_standard_output and not self.no_latex:
            self.load_latex(latex_file)
        self.quiet =  False
        self.truth = None
        self.cache = {}
        truth = self.options.get("truth")
        if truth:
            self.truth = {str(p): p.start for p in Parameter.load_parameters(truth)}

    def finalize(self):
        super(Plots, self).finalize()
        legend = self.options.get("legend", "")
        if legend:
            legend_loc = legend_locations[self.options.get("legend_loc", "best").upper()]
            for name, fig in self.figures.items():
                pylab.figure(fig.number)
                handles, labels = pylab.gca().get_legend_handles_labels()
                for h, l in getattr(fig, "cosmosis_extra_labels", []):
                    handles.append(h)
                    labels.append(l)
                pylab.legend(handles, labels, loc=legend_loc)

    def load_latex(self, latex_file):
        latex_names = {}
        latex_names = configparser.ConfigParser(strict=False)
        latex_names.read(default_latex_file)
        if latex_file:
            latex_names.read(latex_file)
        for i,col_name in enumerate(self.source.colnames):
            display_name=col_name
            if '--' in col_name:
                section,name = col_name.lower().split('--')
                try:
                    display_name = latex_names.get(section,name)
                except (configparser.NoOptionError, configparser.NoSectionError):
                    pass

            else:
                if col_name in ["LIKE","like", "likelihood"]:
                    display_name=r"{\cal L}"
                if col_name in ["POST","post", "Posterior"]:
                    display_name=r"{\cal P}"
                else:
                    try:
                        display_name = latex_names.get("misc",col_name)
                    except (configparser.NoOptionError, configparser.NoSectionError):
                        pass
            if display_name != col_name:
                self._latex[col_name]=display_name

    def latex(self, name, dollar=True):
        l = self._latex.get(name)
        if l is None:
            if '--' in name:
                name = name.split('--', 1)[1]
            return name
        if dollar:
            l = "$"+l+"$"
        return l

    def get_truth_value(self, name):
        if self.truth is None:
            return
        return self.truth.get(name)

    def plot_truth_1d(self, name):
        val = self.get_truth_value(name)
        if val is not None:
            pylab.axvline(val, color='k', linestyle=':', label="Truth")

    def plot_truth_2d(self, name1, name2):
        val1 = self.get_truth_value(name1)
        val2 = self.get_truth_value(name2)
        if (val1 is not None) and (val2 is not None):
            pylab.plot(val1, val2, 'kx', label="Truth")

    def filename(self, base, *bases):
        ftype = self.options.get("file_type", "png")
        filename = super(Plots, self).filename(ftype, base, *bases)
        return filename

    def figure(self, *names):
        #we want to be able to plot multiple chains on the same
        #figure at some point.  So when we make figures we should
        #call this function
        name = "_".join(names)
        filename = self.filename(*names)
        fig = self.get_output(name)
        if fig is None:
            fig = pylab.figure()
            self.set_output(name, PostprocessPlot(name,filename,fig, info=names[:]))
        else:
            fig = fig.value
        self.figures[name] = fig
        return fig, filename

    def run(self):
        print("I do not know how to generate plots for this kind of data")
        return []


    def line_color(self):
        # From https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        # removing white
        possible_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', 
            '#f032e6', '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8',
            '#800000', '#aaffc3', '#000075', '#a9a9a9', '#000000'
        ]

        col = possible_colors[self.plot_set%len(possible_colors)]
        return col

    def shade_colors(self):
        col = self.line_color()
        #convert to tuple
        col = pylab.matplotlib.colors.ColorConverter().to_rgb(col)
        
        if self.options.get("alpha", True):
            #v1 use the same color twice but with low alpha
            col = (col[0], col[1], col[2], 0.2)
            light_col = col
        else:
            #use dark and light variants of the same color
            d1 = 100.0/255.
            d2 = 50.0/255.
            col  = clip_rgb((col[0]+d1, col[1]+d1, col[2]+d1))
            light_col  = clip_rgb((col[0]+d2, col[1]+d2, col[2]+d2))
        return col, light_col


    def parameter_pairs(self):
        swap=self.options.get("swap")
        prefix_only = self.options.get("prefix_only")
        prefix_either = self.options.get("prefix_either")
        prefix_exclude = self.options.get("prefix_exclude")
        #only overrides either

        for name1 in self.source.colnames[:]:
            for name2 in self.source.colnames[:]:
                if name1<=name2: continue
                if prefix_only and not (
                    name1.startswith(prefix_only)
                and name2.startswith(prefix_only)
                    ): continue
                elif prefix_either and not (
                    name1.startswith(prefix_either)
                or name2.startswith(prefix_either)
                    ): continue
                if prefix_exclude and any([name1.startswith(p) for p in prefix_exclude]):
                    continue
                if prefix_exclude and any([name2.startswith(p) for p in prefix_exclude]):
                    continue
                if name1.lower() in self.excluded_columns: continue
                if name2.lower() in self.excluded_columns: continue
                if swap:
                    yield name2,name1
                else:
                    yield name1, name2



class GridPlots(Plots):
    excluded_columns=["post","like", "prior"]
    def __init__(self, *args, **kwargs):
        super(GridPlots, self).__init__(*args, **kwargs)
        self.nsample_dimension = self.source.metadata[0]['nsample_dimension']

    @staticmethod
    def find_grid_contours(like, contour1, contour2):
        like_total = like.sum()
        def objective(limit, target):
            w = np.where(like>limit)
            return like[w].sum() - target
        target1 = like_total*contour1
        target2 = like_total*contour2
        try:
            level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
        except RuntimeError:
            level1 = np.nan
        try:
            level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
        except RuntimeError:
            level2 = np.nan
        return level1, level2


class GridPlots1D(GridPlots):

    def run(self):
        return [self.plot_1d(name) for name in self.source.colnames if name not in self.excluded_columns]

    def plot_1d(self, name1):
        filename = self.filename(name1)
        cols1 = self.source.get_col(name1)
        try: like = self.source.get_col("post")
        except: like = self.source.get_col("like")
        vals1 = np.unique(cols1)
        n1 = len(vals1)
        like_sum = np.zeros(n1)

        #normalizing like this is a bit help 
        #numerically
        like = like-like.max()

        #marginalize
        for k,v1 in enumerate(vals1):
            w = np.where(cols1==v1)
            like_sum[k] = np.log(np.exp(like[w]).sum())
        like = like_sum.flatten()
        like -= like.max()


        #linearly interpolate
        n1_interp = n1*10
        vals1_interp = np.linspace(vals1[0], vals1[-1], n1_interp)
        like_interp = np.interp(vals1_interp, vals1, like)
        if np.isfinite(like_interp).any():
            vals1 = vals1_interp
            like = like_interp
            n1 = n1_interp
        else:
            print()
            print("Parameter %s has a very wide range in likelihoods " % name1)
            print("So I couldn't do a smooth likelihood interpolation for plotting")
            print()


        #normalize
        like[np.isnan(like)] = -np.inf
        like -= like.max()


        #Determine the spacing in the different parameters
        dx = vals1[1]-vals1[0]

        #Set up the figure
        fig, filename = self.figure(name1)
        pylab.figure(fig.number)

        #Plot the likelihood
        pylab.plot(vals1, np.exp(like), linewidth=3, label=self.source.label)

        #Find the levels of the 68% and 95% contours
        X, L = self.find_edges(np.exp(like), 0.68, 0.95, vals1)
        #Plot black dotted lines from the y-axis at these contour levels
        for (x, l) in zip(X,L):
            if np.isnan(x[0]): continue
            pylab.plot([x[0],x[0]], [0, l[0]], ':', color='black')
            pylab.plot([x[1],x[1]], [0, l[1]], ':', color='black')

        #Set the x and y limits
        pylab.xlim(cols1.min()-dx/2., cols1.max()+dx/2.)
        pylab.ylim(0,1.05)
        #Add label
        pylab.xlabel(self.latex(name1))
        if not hasattr(fig, 'cosmosis_done_truth'):
            self.plot_truth_1d(name1)
            fig.cosmosis_done_truth = True

        return filename

    @classmethod
    def find_edges(cls, like, contour1, contour2, vals1):
        X = []
        L = []
        level1,level2=cls.find_grid_contours(like, contour1, contour2)
        for level in [level1, level2]:
            if np.isnan(level):
                X.append((np.nan,np.nan))
                L.append((np.nan,np.nan))
                continue
            above = np.where(like>level)[0]
            left = above[0]
            right = above[-1]
            x1 = vals1[left]
            x2 = vals1[right]
            L1 = like[left]
            L2 = like[right]
            X.append((x1,x2))
            L.append((L1,L2))
        return X,L



class GridPlots2D(GridPlots):
    def run(self):
        if self.options.get("no_2d", False):
            print("Not making any 2D plots because you said --no-2d")
            return []
        filenames=[]
        nv = self.source.metadata[0]['n_varied']
        varied_params = self.source.colnames[:nv]
        for name1, name2 in self.parameter_pairs():
            if (name1 not in varied_params) or (name2 not in varied_params):
                continue
            try:
                filename=self.plot_2d(name1, name2)
            except ValueError:
                if self.options.get("fatal_errors", False):
                    raise
                print("Could not make plot {} vs {} - error in contour".format(name1,name2))
                continue
            if filename: filenames.append(filename)
        return filenames

    def get_grid_like(self, name1, name2):
        # Load the columns
        cols1 = self.source.get_col(name1)
        cols2 = self.source.get_col(name2)
        try: like = self.source.get_col("post")
        except: like = self.source.get_col("like")
        vals1 = np.unique(cols1)
        vals2 = np.unique(cols2)

        n1 = int(self.nsample_dimension)
        n2 = n1

        like = like - like.max()

        #Marginalize over all the other parameters by summing
        #them up
        like_sum = np.zeros((n1,n2))
        for k,(v1, v2) in enumerate(itertools.product(vals1, vals2)):
            w = np.where((cols1==v1)&(cols2==v2))
            i,j = np.unravel_index(k, like_sum.shape)
            like_sum[i,j] = np.log(np.exp(like[w]).sum())
        like = like_sum.flatten()

        #Normalize the log-likelihood to peak=0
        like -= like.max()
        like = np.exp(like).reshape((n1,n2)).T
        extent=(vals1[0], vals1[-1], vals2[0], vals2[-1])

        return extent, like

    def plot_2d(self, name1, name2):
        extent, like = self.get_grid_like(name1, name2) 

        #Choose a color mapping
        norm = pylab.matplotlib.colors.Normalize(like.min(), like.max())
        colormap = pylab.cm.Reds
        do_image = self.options.get("image", True)
        do_fill =  self.options.get("fill", True)

        #Create the figure
        fig,filename = self.figure("2D", name1, name2)
        pylab.figure(fig.number)

        #Decide whether to do a smooth or block plot
        smooth = self.options.get("smooth", True)
        if smooth:
            interpolation='bilinear'
        else:
            interpolation='nearest'

        #Make the plot

        if do_image:
            pylab.imshow(like, extent=extent, 
                aspect='auto', cmap=colormap, norm=norm, interpolation=interpolation, origin='lower')
            
            sm = pylab.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm._A = [] #hack from StackOverflow to make this work
            pylab.colorbar(sm, label='Posterior', ax=pylab.gca())

        #Add contours
        level1, level2 = self.find_grid_contours(like, 0.68, 0.95)
        level0 = like.max() + 1

        #three cases
        if do_image:
            colors = None #auto colors from the contourf
            pylab.contour(like, levels = [level2, level1], extent=extent, linewidths=[1,3], colors=colors)
        elif do_fill:
            dark, light = self.shade_colors()
            pylab.contourf(like, levels = [level2, level0], extent=extent, linewidths=[1,3], colors=[light])
            pylab.contourf(like, levels = [level1, level0], extent=extent, linewidths=[1,3], colors=[dark])
        else:
            colors = [self.line_color(), self.line_color()]
            pylab.contour(like, levels = [level2, level1], extent=extent, linewidths=[1,3], colors=colors)
            
        pylab.xlabel(self.latex(name1))
        pylab.ylabel(self.latex(name2))
        
        if not hasattr(fig, 'cosmosis_done_truth'):
            self.plot_truth_2d(name1, name2)
            fig.cosmosis_done_truth = True

        return filename



class SnakePlots2D(GridPlots2D):
    def get_grid_like(self, name1, name2):
        # Load the columns
        cols1 = self.source.get_col(name1)
        cols2 = self.source.get_col(name2)
        try: like = self.source.get_col("post")
        except: like = self.source.get_col("like")
        vals1 = np.unique(cols1)
        vals2 = np.unique(cols2)
        dx1 = np.min(np.diff(vals1))
        dx2 = np.min(np.diff(vals2))
        left1 = vals1.min()
        left2 = vals2.min()
        right1 = vals1.max()
        right2 = vals2.max()
        n1 = int(np.round((right1-left1)/dx1))+1
        n2 = int(np.round((right2-left2)/dx2))+1

        like = like - like.max()

        #Marginalize over all the other parameters by summing
        #them up
        like_sum = np.zeros((n1,n2))
        for k,(v1, v2) in enumerate(itertools.product(vals1, vals2)):
            w = np.where((cols1==v1)&(cols2==v2))
            i = int(np.round((v1-left1)/dx1))
            j = int(np.round((v2-left2)/dx2))
            like_sum[i,j] = np.log(np.exp(like[w]).sum())
        like = like_sum.flatten()

        #Normalize the log-likelihood to peak=0
        like -= like.max()
        like = np.exp(like).reshape((n1,n2)).T
        extent=(left1, right1, left2, right2)

        return extent, like


class MetropolisHastingsPlotsBase(Plots, MCMCPostProcessorElement):
    excluded_columns = ["like","post", "prior"]

def get_param_limits(source, name):
    values = source.extract_ini("VALUES")
    priors = source.extract_ini("PRIORS")
    params = Parameter.load_parameters(values, [priors])
    i = params.index(name)
    param = params[i]
    return param.limits

def select_limits(x, source, name):
    try:
        xmin, xmax = get_param_limits(source, name)
    except ValueError:
        return (x.min(), x.max(), False)
    std = x.std()
    mu = x.mean()
    near_boundary = False
    if abs(xmax - x.max()) < std:
        near_boundary = True
    else:
        xmax = x.max()

    if abs(x.min() - xmin) < std:
        near_boundary = True
    else:
        xmin = x.min()

    return xmin, xmax, near_boundary


def next_entry(l, m):
    return m[(l.index(m) + 1)%len(m)]


class MetropolisHastingsPlots(MetropolisHastingsPlotsBase):
    def run(self):
        return self.run_1d() + self.run_2d()

    def keywords_1d(self):
        return {}

    def smooth_likelihood_1d(self, x, name):
        #Interpolate using KDE
        if self.options.get("fix_edges"):
            return self.smooth_likelihood_with_boundaries_1d(x, name)
        n = self.options.get("n_kde", 100)
        factor = self.options.get("factor_kde", 2.0)
        kde = KDE(x, factor=factor)
        x_axis, like = kde.grid_evaluate(n, (x.min(), x.max()) )
        return x_axis, like

    def smooth_likelihood_with_boundaries_1d(self, x, name):
        # find the limits on this parameter
        factor = self.options.get("factor_kde", 2.0)
        xmin, xmax, fix = select_limits(x, self.source, name)
        xout, like = smooth_density_estimate_1d(x, xmin, xmax,
                            smoothing=factor, fix_boundary=fix)
        cut = (xout >= x.min()) & (xout <= x.max())
        return xout[cut], like[cut]


    def make_1d_plot(self, name, figure=None):
        x = self.reduced_col(name)
        if not self.quiet:
            print(" - 1D plot ", name)
        if figure is None:
            figure, filename = self.figure(name)
        else:
            filename = None
        if x.max()-x.min()==0: return

        x_axis, like = self.smooth_likelihood_1d(x, name)
        like/=like.max()
        self.cache[name] = x_axis, like

        #Choose colors
        color = self.line_color()

        #Make the plot
        pylab.figure(figure.number)
        keywords = self.keywords_1d()
        pylab.plot(x_axis, like, '-', color=color, lw=2, label=self.source.label,  **keywords)
        pylab.xlabel(self.latex(name, dollar=True))
        if not hasattr(figure, 'cosmosis_done_truth'):
            self.plot_truth_1d(name)
            figure.cosmosis_done_truth = True
        return filename

    def run_1d(self):
        filenames = []
        for name in self.source.colnames:
            if name.lower() in self.excluded_columns: continue
            filename = self.make_1d_plot(name)
            if filename: filenames.append(filename)
        return filenames

    def keywords_2d(self):
        return {}

    def _find_contours(self, like, x, y, x_axis, y_axis, contour1, contour2):
        N = len(x)
        hx = 0.5 * (x_axis[1] - x_axis[0])
        hy = 0.5 * (y_axis[1] - y_axis[0])
        xedge = np.concatenate([x_axis - hx, [x_axis[-1] + hx]])
        yedge = np.concatenate([y_axis - hy, [y_axis[-1] + hy]])

        histogram, _, _ = np.histogram2d(x, y, bins=[xedge, yedge])

        def objective(limit, target):
            w = np.where(like>limit)
            count = histogram[w]
            return count.sum() - target
        target1 = N*(1-contour1)
        target2 = N*(1-contour2)
        level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
        level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
        return level1, level2, like.sum()

    def smooth_likelihood_2d(self, x, y, xname, yname):
        if self.options.get("fix_edges"):
            return self.smooth_likelihood_with_boundaries_2d(x, y, xname, yname)

        n = self.options.get("n_kde", 100)
        factor = self.options.get("factor_kde", 2.0)
        kde = KDE([x,y], factor=factor)
        x_range = (x.min(), x.max())
        y_range = (y.min(), y.max())
        (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
        return x_axis, y_axis, like

    def smooth_likelihood_with_boundaries_2d(self, x, y, xname, yname):
        factor = self.options.get("factor_kde", 2.0)
        xmin, xmax, fix_x = select_limits(x, self.source, xname)
        ymin, ymax, fix_y = select_limits(y, self.source, yname)
        xout, yout, like = smooth_density_estimate_2d(x, y, xmin, xmax, ymin, ymax,
                            smoothing=factor, fix_boundary=fix_x or fix_y)
        xcut = np.where((xout >= x.min()) & (xout <= x.max()))[0]
        ycut = np.where((yout >= y.min()) & (yout <= y.max()))[0]
        xcut0 = xcut.min()
        xcut1 = xcut.max() + 1
        ycut0 = ycut.min()
        ycut1 = ycut.max() + 1
        return xout[xcut0:xcut1], yout[ycut0:ycut1], like[xcut0:xcut1, ycut0:ycut1]

    def make_corner_plot(self, figure=None):
        pairs = list(self.parameter_pairs())

        # parameters, in the order we will use
        params = list(dict.fromkeys([p[0] for p in pairs]))
        nparam = len(params)
        fig, filename = self.figure("corner")

        # If we are making a corner plot with two different chains
        # then there may be different parameters in the two of them.
        # In that case just inherit the parameters from the first one.
        if fig.get_axes():
            params = fig._cosmosis_params
            nparam = len(params)
            axes = fig._cosmosis_axes
            new = False
        else:
            # enlarge for this extra big figure
            size = min(4 * nparam, 24)
            fig.set_size_inches(size, size)
            axes = fig.subplots(nparam, nparam, squeeze=False)
            fig._cosmosis_params = params[:]
            fig._cosmosis_axes = axes
            new = True

        for i in range(nparam):
            for j in range(nparam):
                ax = axes[i, j]

                # Switch on minor ticks and have them point inwards like sensible people
                ax.tick_params(direction='in', which='both', bottom=True, left=True)
                ax.tick_params(which='minor', length=5, bottom=True, left=True)
                ax.minorticks_on()

                # Remove upper right above diagonal
                if j > i and new:
                    fig.delaxes(ax)
                    continue

                p1 = params[j]
                p2 = params[i]
                if i == j:
                    # Left column only has a y-label, and others also have no y tick labels
                    # and only the bottom row as an x-label, and others have no x ticks.
                    # Top left is labelled "Posterior"
                    # Only the bottom row has
                    if i == 0:
                        ax.set_ylabel("Posterior")
                    else:
                        ax.yaxis.set_ticklabels([])
                    if j == nparam - 1:
                        ax.set_xlabel(self.latex(p2))
                    else:
                        ax.xaxis.set_ticklabels([])

                    if nparam > 10:
                        ax.xaxis.set_ticklabels([])

                    if p1 not in self.cache:
                        continue

                    # Use the same 1D information as in the 1D plots
                    x, like = self.cache[p1]

                    # Plot and set the limits explicitly.
                    # Trying to use the sharex / sharey feature here is painful
                    # because we want to remove the diagonal, and that seems to make
                    # the tick stuff really fiddly.
                    ax.plot(x, like / like.max(), color=self.line_color())
                    ax.set_xlim(x.min(), x.max())
                    ax.set_ylim(0, 1.1)

                else:
                    # We might have done the swap thing, in which case
                    # check both directions.
                    # If the user has done some funny things with the --only or --either
                    # parameters then things might get funny here, so allow missing panels.
                    try:
                        x, y, like, levels = self.cache[p1, p2]
                    except KeyError:
                        # we have to switch the parameter ordering in this case,
                        # and flip the contours
                        try:
                            x, y, like, levels = self.cache[p2, p1]
                            x, y = y, x
                            like = like.T
                        except KeyError:
                            continue
                    color = self.line_color()

                    # This is the same contour code as in the main 2D plot code
                    ax.contour(x, y, like, levels=levels[:2], colors=color)

                    # As with the 1D plots on the diagonal we have to remove
                    # the labelling from most of the panels.
                    if i == nparam - 1:
                        ax.set_xlabel(self.latex(p1))
                    else:
                        ax.xaxis.set_ticklabels([])
                    if j == 0:
                        ax.set_ylabel(self.latex(p2))
                    else:
                        ax.yaxis.set_ticklabels([])

                    if nparam > 10:
                        ax.xaxis.set_ticklabels([])

                    # Explicitly set the ranges so that they are the
                    # same for all the panels. To ensure that axes are identical to
                    # the 1D version we set them to the same range as that.

                    x1, _ = self.cache[p1]
                    y1, _ = self.cache[p2]

                    ax.set_xlim(x1.min(), x1.max())
                    ax.set_ylim(y1.min(), y1.max())

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.04, wspace=0.08)
        return filename



    def make_2d_plot(self, name1, name2, figure=None):
        #Get the data
        x = self.reduced_col(name1)
        y = self.reduced_col(name2)


        if x.max()-x.min()==0 or y.max()-y.min()==0:
            return

        if not self.quiet:
            print("  (making %s vs %s)" % (name1, name2))


        #Interpolate using KDE
        try:
            x_axis, y_axis, like = self.smooth_likelihood_2d(x, y, name1, name2)
        except np.linalg.LinAlgError:
            print("  -- these two parameters have singular covariance - probably a linear relation")
            print("Not making a 2D plot of them")
            return []

        if figure is None:
            figure, filename = self.figure("2D", name1, name2)
        else:
            filename = None


        #Choose levels at which to plot contours
        contour1=1-0.68
        contour2=1-0.95
        level1, level2, total_mass = self._find_contours(like, x, y, x_axis, y_axis, contour1, contour2)

        level0 = np.inf
        levels = [level2, level1, level0]

        #Make the plot
        pylab.figure(figure.number)
        keywords = self.keywords_2d()
        fill = self.options.get("fill", True)
        imshow = self.options.get("imshow", False)
        plot_points = self.options.get("plot_points", False)

        self.cache[name1, name2] = (x_axis, y_axis, like.T, levels)

        if imshow:
            pylab.imshow(like.T, extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), aspect='auto', origin='lower')
            pylab.colorbar()
        elif fill:
            dark,light = self.shade_colors()
            pylab.contourf(x_axis, y_axis, like.T, [level2,level0], colors=[light], alpha=0.25)
            pylab.contourf(x_axis, y_axis, like.T, [level1,level0], colors=[dark], alpha=0.25)
        else:
            color = self.line_color()
            cs = pylab.contour(x_axis, y_axis, like.T, [level2,level1], colors=color)
            if not hasattr(figure, "cosmosis_extra_labels"):
                figure.cosmosis_extra_labels = []
            figure.cosmosis_extra_labels.append((cs.legend_elements()[0][0], self.source.label))

        if plot_points:
            pylab.plot(x, y, ',')


        #Do the labels
        pylab.xlabel(self.latex(name1))
        pylab.ylabel(self.latex(name2))
        if not hasattr(figure, 'cosmosis_done_truth'):
            self.plot_truth_2d(name1, name2)
            figure.cosmosis_done_truth = True
        
        return filename        


    def run_2d(self):
        if self.options.get("no_2d", False):
            print("Not making any 2D plots because you said --no-2d")
            return []
        filenames = []
        print("(Making 2D plots using KDE; this takes a while but is really cool)")
        for name1,name2 in self.parameter_pairs():
                try:
                    filename = self.make_2d_plot(name1, name2)
                except KeyboardInterrupt:
                    raise
                except: #any other error we just continue
                    import traceback
                    print("Failed to make plot of {} vs {}.  Here is the error context:".format(name1,name2))
                    filename=None
                    print(traceback.format_exc())
                if filename:
                    filenames.append(filename)
        filenames.append(self.make_corner_plot())
        return filenames


class TestPlots(Plots):
    def run(self):
        dirname = self.source.sampler_option("save_dir")
        output_dir = self.options.get("outdir", "png")
        prefix=self.options.get("prefix","")
        ftype=self.options.get("file_type", "png")
        filenames = []
        for cls in cosmology_theory_plots.Plot.registry:
            fig = None
            try:
                #may return None
                figure = self.get_output(cls.filename)
                if figure is None:
                    print("New plot", cls.filename)
                else:
                    print("Old plot", cls.filename)
                p=cls(dirname, output_dir, prefix, ftype, figure=None)
                filename=p.filename
                fig = p.figure
                p.figure=fig
                p.plot()
                if figure is None:
                    self.set_output(cls.filename,
                                     PostprocessPlot(p.filename,p.outfile,fig))
                filenames.append(filename)
            except IOError as err:
                if fig is not None:
                    #Then we got as far as making the figure before
                    #failing.  so remove it
                    pylab.close()
                print(err)
        return filenames


class WeightedPlots(object):
    excluded_columns = ["like","old_like","post", "weight", "log_weight", "old_log_weight", "old_weight", "old_post", "prior"]

    def smooth_likelihood_1d(self, x, name):
        #Interpolate using KDE
        n = self.options.get("n_kde", 100)
        weights = self.weight_col()
        #speed things up by removing zero-weighted samples
        if self.options.get("fix_edges"):
            return self.smooth_likelihood_with_boundaries_1d(x, name, weights)

        x_range = get_plot_range(x, weights)

        factor = self.options.get("factor_kde", 2.0)
        kde = KDE(x, factor=factor, weights=weights)
        x_axis, like = kde.grid_evaluate(n, x_range )
        return x_axis, like

    def smooth_likelihood_with_boundaries_1d(self, x, name, weights):
        # find the limits on this parameter
        factor = self.options.get("factor_kde", 2.0)
        xmin, xmax, fix = select_limits(x, self.source, name)
        xout, like = smooth_density_estimate_1d(x, xmin, xmax, weights=weights,
                            smoothing=factor, fix_boundary=fix)
        xmin0, xmax0 = get_plot_range(x, weights)
        cut = (xout >= xmin0) & (xout <= xmax0)
        return xout[cut], like[cut]

    def smooth_likelihood_2d(self, x, y, xname, yname):
        weights = self.weight_col()

        if self.options.get("fix_edges"):
            return self.smooth_likelihood_with_boundaries_2d(x, y, weights, xname, yname)

        n = self.options.get("n_kde", 100)
        fill = self.options.get("fill", True)
        factor = self.options.get("factor_kde", 2.0)
        kde = KDE([x,y], factor=factor, weights=weights)
        x_range = get_plot_range(x, weights)
        y_range = get_plot_range(y, weights)
        (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
        return x_axis, y_axis, like

    def smooth_likelihood_with_boundaries_2d(self, x, y, weights, xname, yname):
        factor = self.options.get("factor_kde", 2.0)
        xmin, xmax, fix_x = select_limits(x, self.source, xname)
        ymin, ymax, fix_y = select_limits(y, self.source, yname)
        xout, yout, like = smooth_density_estimate_2d(x, y, xmin, xmax, ymin, ymax, weights=weights,
                            smoothing=factor, fix_boundary=fix_x or fix_y)
        xmin0, xmax0 = get_plot_range(x, weights)
        ymin0, ymax0 = get_plot_range(y, weights)
        xcut = np.where((xout >= xmin0) & (xout <= xmax0))[0]
        ycut = np.where((yout >= ymin0) & (yout <= ymax0))[0]
        xcut0 = xcut.min()
        xcut1 = xcut.max() + 1
        ycut0 = ycut.min()
        ycut1 = ycut.max() + 1

        return xout[xcut0:xcut1], yout[ycut0:ycut1], like[xcut0:xcut1, ycut0:ycut1]


    def _find_contours(self, like, x, y, x_axis, y_axis, contour1, contour2):
        N = len(x)
        weights = self.weight_col()
        hx = 0.5 * (x_axis[1] - x_axis[0])
        hy = 0.5 * (y_axis[1] - y_axis[0])
        xedge = np.concatenate([x_axis - hx, [x_axis[-1] + hx]])
        yedge = np.concatenate([y_axis - hy, [y_axis[-1] + hy]])

        histogram, _, _ = np.histogram2d(x, y, bins=[xedge, yedge], weights=weights)
        def objective(limit, target):
            w = np.where(like>=limit)
            count = histogram[w]
            return count.sum() - target
        target1 = histogram.sum()*(1-contour1)
        target2 = histogram.sum()*(1-contour2)

        level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
        level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
        return level1, level2, like.sum()


class MultinestPlots(WeightedPlots, MultinestPostProcessorElement, MetropolisHastingsPlots):
    excluded_columns = ["like","old_like","post", "weight", "log_weight", "old_log_weight", "old_weight", "old_post", "prior"]

class PolychordPlots(MultinestPlots):
    pass

def get_plot_range(x, weights):
    dx = std_weight(x, weights)*4
    mu_x = mean_weight(x, weights)
    return (max(x.min(), mu_x-dx), min(x.max(), mu_x+dx))


class WeightedMetropolisPlots(WeightedPlots, WeightedMCMCPostProcessorElement, MetropolisHastingsPlots):
    excluded_columns = ["like","old_like","post", "weight", "log_weight", "old_log_weight", "old_weight", "old_post", "prior"]




class TracePlots(Plots, MCMCPostProcessorElement):
    excluded_columns = []
    def run(self):
        return [self.plot_1d(name) for name in self.source.colnames if not name in self.excluded_columns]

    def plot_truth_1d(self, name):
        val = self.get_truth_value(name)
        if val is not None:
            pylab.axhline(val, color='k', linestyle=':')

    def plot_1d(self, name):
        if not self.quiet:
            print(" - Trace plot ", name)
        x = self.reduced_col(name)
        fig, filename = self.figure(f"trace_{name}")
        pylab.figure(fig.number)
        color = self.line_color()
        pylab.plot(x, ',', color=color, label=self.source.label)
        pylab.xlabel("Reduced Chain Position")
        pylab.ylabel(self.latex(name))
        if not hasattr(fig, 'cosmosis_done_truth'):
            self.plot_truth_1d(name)
            fig.cosmosis_done_truth = True
        return filename



class ColorScatterPlotBase(Plots):
    scatter_filename='scatter'
    x_column = None
    y_column = None
    color_column = None

    def run(self):
        if (self.x_column is None) or (self.y_column is None) or (self.color_column is None):
            print("Please specify x_column, y_column, color_column, and scatter_filename")
            print("in color scatter plots")
            return []

        #Get the columns you want to plot
        #"reduced" means that we skip the burn-in
        #and apply any thinning.
        #We get these functions because we inherit
        #from plots.MetropolisHastingsPlots
        x = self.reduced_col(self.x_column)
        y = self.reduced_col(self.y_column)
        c = self.reduced_col(self.color_column)

        # Multinest chains do not contain equally 
        # weighted samples (i.e. the chain rows are not
        # drawn from the posterior) because you need more
        # than that to do the evidence calculation.
        # We need to use a method from MultinestPostProcessorElement
        # to get a posterior sample.
        # On the other hand regular MCMC chains are posteriors.
        # So subclasses will need to inherit from either MultinestPostProcessorElement
        # or MCMCPostProcessorElement.
        sample = self.posterior_sample()
        x = x[sample]
        y = y[sample]
        c = c[sample]

        #Use these to create figures looked
        #after by cosmosis.
        #Though you can also use your own filenames,
        #saving, etc, in which case do not use these functions
        figure, filename = self.figure(self.scatter_filename)

        #Do the actual plotting.
        #By default the saving will be handled later.
        pylab.scatter(x, y, c=c, s=5, lw=0, cmap=pylab.cm.bwr)

        pylab.colorbar(label=self.latex(self.color_column))
        pylab.xlabel(self.latex(self.x_column))
        pylab.ylabel(self.latex(self.y_column))

        self.plot_truth_2d(self.x_column, self.y_column)

        #Return a list of files you create.
        return [filename]

class MCMCColorScatterPlot(MCMCPostProcessorElement, ColorScatterPlotBase):
    pass

class WeightedMCMCColorScatterPlot(WeightedMCMCPostProcessorElement, ColorScatterPlotBase):
    pass


class MultinestColorScatterPlot(MultinestPostProcessorElement, ColorScatterPlotBase):
    pass

class PolychordColorScatterPlot(MultinestColorScatterPlot):
    pass



class CovarianceMatrixGaussians(Plots):
    def run(self):
        filenames = []
        Sigma = np.linalg.inv(self.source.data[0]).diagonal()**0.5
        Mu = [float(self.source.metadata[0]['mu_{0}'.format(i)]) for i in range(Sigma.size)]

        for name, mu, sigma in zip(self.source.colnames, Mu, Sigma):
            filename = self.plot_1d(name, mu, sigma)
            filenames.append(filename)
        return filenames

    def plot_1d(self, name, mu, sigma):
        xmin = mu - 4*sigma
        xmax = mu + 4*sigma
        sigma2 = sigma**2
        x = np.linspace(xmin, xmax, 200)
        p = np.exp(-0.5 * (x-mu)**2 / sigma2)# / np.sqrt(2*np.pi*sigma2)
        figure, filename = self.figure(name)
        pylab.figure(figure.number)
        if not hasattr(figure, "cosmosis_done_truth"):
            self.plot_truth_1d(name)
            figure.cosmosis_done_truth = True
        pylab.plot(x, p, label=self.source.label)
        pylab.xlabel(self.latex(name))
        pylab.ylabel("Posterior")
        return filename



class CovarianceMatrixEllipse(Plots):

    def run(self):
        filenames = []
        self.covmat_estimate = np.linalg.inv(self.source.data[0])
        for name1, name2 in self.parameter_pairs():
            i = self.source.colnames.index(name1)
            j = self.source.colnames.index(name2)
            filename = self.plot_2d(name1, i, name2, j)
            filenames.append(filename)
        return filenames

    def plot_2d(self, name1, i, name2, j):
        #Get the central points about which this was estimated
        mu1 = float(self.source.metadata[0]['mu_{0}'.format(i)])
        mu2 = float(self.source.metadata[0]['mu_{0}'.format(j)])
        pos = np.array([mu1,mu2])

        #Cut the covariance estimate down to the two parameters
        #we are using here
        covmat = self.covmat_estimate[:,[i,j]][[i,j],:]

        #for setting widths we would like a std. dev.
        s11 = covmat[0,0]**0.5
        s22 = covmat[1,1]**0.5

        #Open the figure (new or existing) for this pair
        figure, filename = self.figure("2D", name1, name2)
        pylab.figure(figure.number)

        if not hasattr(figure, "cosmosis_done_truth"):
            self.plot_truth_2d(name1, name2)
            figure.cosmosis_done_truth = True

        #Plot the 1 sigma and 2 sigma ellipses
        self.plot_cov_ellipse(covmat, pos, nstd=1, facecolor=None, 
            edgecolor=self.line_color(), linewidth=2, fill=False, label=self.source.label)
        self.plot_cov_ellipse(covmat, pos, nstd=2, facecolor=None, 
            edgecolor=self.line_color(), linewidth=2, fill=False)


        #Parameter ranges - use 3 sigma.
        #We don't want to cut down on the existing range
        #if it's bigger already so we check for that.
        xmin,xmax = pylab.xlim()
        ymin, ymax = pylab.ylim()
        #The default range is (0,1) in both parameters
        #That would confuse the code so we check whether this
        #is the first run.
        if self.plot_set>0:
            xmin = min(xmin,mu1-3*s11)
            xmax = max(xmax,mu1+3*s11)
            ymin = min(ymin,mu2-3*s22)
            ymax = max(ymax,mu2+3*s22)
        else:
            xmin = mu1-3*s11
            xmax = mu1+3*s11
            ymin = mu2-3*s22
            ymax = mu2+3*s22

        #And finally set the ranges, after all this.
        pylab.xlim(xmin,xmax)
        pylab.ylim(ymin, ymax)

        #Axis labels, finally.
        pylab.xlabel(self.latex(name1))
        pylab.ylabel(self.latex(name2))
        return filename



    @staticmethod
    def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
        """
        Based on code from StackOverflow. 
        http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals

        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.

        Returns
        -------
            A matplotlib ellipse artist
        """
        import matplotlib
        from matplotlib.patches import Ellipse

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        pylab.gca().add_patch(ellip)
        return ellip

class StarPlots(Plots):
    excluded_columns=["post","like", "prior"]

    def star_plot(self, i, name, log):
        n = int(self.source.metadata[0]['nsample_dimension'])
        shp = self.source.get_col(name).shape

        x = self.source.get_col(name)[i*n:(i+1)*n]
        y = self.source.get_col("post")[i*n:(i+1)*n]
        if log:
            figure,filename = self.figure(name+"_log")
        else:
            figure,filename = self.figure(name)
            y = np.exp(y-y.max())
        pylab.figure(figure.number)
        pylab.plot(x, y)
        pylab.xlabel(self.latex(name))
        if log:
            pylab.ylabel("Log Posterior")
        else:
            pylab.ylabel("Posterior")
        return filename

    def run(self):
        filenames = []

        i=0
        # We can only make plots of varied parameters in the star sampler,
        # doing derived parameters doesn't really make sense
        nv = self.source.metadata[0]['n_varied']
        for i in range(nv):
            name = self.source.colnames[i]
            # Do both log and non-log variants
            filename = self.star_plot(i,name, True)
            filenames.append(filename)
            filename = self.star_plot(i,name, False)
            filenames.append(filename)
        return filenames




class Tweaks(Loadable):
    filename="default_nonexistent_filename_ignore"
    _all_filenames='all plots'
    def __init__(self):
        self.has_run=False
        self.info=None

    def run(self):
        print("Please fill in the 'run' method of your tweak to modify a plot")

def clip_unit(x):
    if x<0:
        return 0.0
    elif x>1.:
        return 1.0
    return x

def clip_rgb(c):
    return clip_unit(c[0]), clip_unit(c[1]), clip_unit(c[2])

def color_variant(hex_color, brightness_offset=1):
    """ takes a color like #87c95f and produces a lighter or darker variant """
    if len(hex_color) != 7:
        raise Exception("Passed %s into color_variant(), needs to be in #87c95f format." % hex_color)
    rgb_hex = [hex_color[x:x+2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int] # make sure new values are between 0 and 255
    # hex() produces "0x88", we want just "88"
    return "#" + "".join([hex(i)[2:] for i in new_rgb_int])



# For backwards-compatibility with any poor souls using these, we provide replacements for the old
# separate 1D and 2D classes, which were unified to make it easier to do a corner plot.
class WeightedMetropolisPlots1D(WeightedMetropolisPlots):
    def run(self):
        warnings.warn('WeightedMetropolisPlots1D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_1d()
    def smooth_likelihood(self, x, name):
        return self.smooth_likelihood_1d(x, name)

class MetropolisHastingsPlots1D(MetropolisHastingsPlots):
    def run(self):
        warnings.warn('MetropolisPlots1D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_1d()
    def smooth_likelihood(self, x, name):
        return self.smooth_likelihood_1d(x, name)

class PolychordPlots1D(PolychordPlots):
    def run(self):
        warnings.warn('PolychordPlots1D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_1d()
    def smooth_likelihood(self, x, name):
        return self.smooth_likelihood_1d(x, name)

class MultinestPlots1D(MultinestPlots):
    def run(self):
        warnings.warn('MetropolisHastingsPlots1D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_1d()
    def smooth_likelihood(self, x, name):
        return self.smooth_likelihood_2d(x, name)

# 2D versions of the above

class MetropolisHastingsPlots2D(MetropolisHastingsPlots):
    def run(self):
        warnings.warn('MetropolisHastingsPlots2D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_2d()
    def smooth_likelihood(self, x, y, xname, yname):
        return self.smooth_likelihood_2d(x, y, xname, yname)

class WeightedMetropolisPlots2D(WeightedMetropolisPlots):
    def run(self):
        warnings.warn('WeightedMetropolisPlots2D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_2d()
    def smooth_likelihood(self, x, y, xname, yname):
        return self.smooth_likelihood_2d(x, y, xname, yname)

class MultinestPlots2D(MultinestPlots):
    def run(self):
        warnings.warn('MultinestPlots2D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_2d()
    def smooth_likelihood(self, x, y, xname, yname):
        return self.smooth_likelihood_2d(x, y, xname, yname)

class PolychordPlots2D(PolychordPlots):
    def run(self):
        warnings.warn('Polychord2D and the other 1D and 2D plotters are deprecated', DeprecationWarning, stacklevel=2)
        return self.run_2d()
    def smooth_likelihood(self, x, y, xname, yname):
        return self.smooth_likelihood_2d(x, y, xname, yname)


class AprioriPlots1D(Plots):
    def run(self):
        cols = self.source.colnames[:]
        filenames = []
        for col in cols:
            fig, filename = self.figure(col)
            pylab.hist(self.reduced_col(col), bins=30, histtype='step')
            pylab.xlabel(self.latex(col))
            pylab.ylabel('Number of samples')
            filenames.append(filename)
        return filenames

class AprioriPlots2D(Plots):
    def run(self):
        filenames = []
        post = self.source.get_col("post")
        for col1, col2 in self.parameter_pairs():
            fig, filename = self.figure("2D", col1, col2)
            x1 = self.source.get_col(col1)
            x2 = self.source.get_col(col2)
            good = np.isfinite(post)
            bad = ~good
            if np.any(good):
                pylab.plot(x1[good], x2[good], '.')
            if np.any(bad):
                pylab.plot(x1[bad], x2[bad], 'x')
            pylab.xlabel(self.latex(col1))
            pylab.ylabel(self.latex(col2))
            filenames.append(filename)
        return filenames
