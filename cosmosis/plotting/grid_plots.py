"""
Stop!
This file is not ready yet.

I think I need to re-write the plotting code almost from scratch
as it's grown pretty organically.

"""

import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['font.family']='serif'
matplotlib.rcParams['font.size']=18
matplotlib.rcParams['legend.fontsize']=15
matplotlib.rcParams['xtick.major.size'] = 10.0
matplotlib.rcParams['xtick.minor.size'] = 5.0
matplotlib.rcParams['ytick.major.size'] = 10.0
matplotlib.rcParams['ytick.minor.size'] = 5.0
import pylab
import numpy as np
import collections
from .plotter import Plotter
import itertools
import scipy.optimize
from scipy.special import logsumexp
try:
	from cosmosis import output as output_module
except ImportError:
	print("Running without cosmosis: no pretty section names or running on ini files")


class GridPlotter(Plotter):
	@staticmethod
	def _logsumexp(values):
		"""Stable log(sum(exp(values))) for grid marginalization."""
		if np.size(values) == 0:
			return -np.inf
		return logsumexp(values)

	#def __init__(self, *args, **kwargs):
		# super(GridPlotter, self).__init__(*args, **kwargs)
		# convert the loaded chain data sets into grids
		# need the number of varied parameters
		# how do we get that?

	@classmethod
	def from_outputs(cls, options, **kw):
		column_names, chains, metadata, comments, final_metadata = output_module.input_from_options(options)
		chains = np.vstack(chains).T
		chains = dict(list(zip(column_names,chains)))
		chain_data = {"Chains":chains}
		return cls(chain_data, **kw)

	@classmethod
	def from_chain_files(cls, filenames, **kw):
		nameset = [open(filename).readline().strip().strip('#').replace("/","--").split() for filename in filenames]
		dataset = [(np.loadtxt(filename).T) for filename in filenames]
		chain_data = collections.OrderedDict()
		for filename,names,data in zip(filenames,nameset,dataset):
			chain_datum = dict(list(zip(names,data)))
			chain_data[filename] = chain_datum	
		return cls(chain_data, **kw)

	def _plot_1d(self, name1):
		cols1 = list(self.cols_for_name(name1).values())[0]
		like = list(self.cols_for_name("LIKE").values())[0]
		vals1 = np.unique(cols1)
		n1 = len(vals1)

		#check for derived parameters - this is a weak check
		if len(vals1)==len(cols1) and not len(self.all_names)==2:
			print("Not grid-plotting %s as it seems to be a derived parameter" % name1)
			raise ValueError(name1)
		like_sum = np.zeros(n1)

		#marginalize
		for k,v1 in enumerate(vals1):
			w = np.where(cols1==v1)
			like_sum[k] = self._logsumexp(like[w])
		like = like_sum.flatten()

		#linearly interpolate
		n1 *= 10
		vals1_interp = np.linspace(vals1[0], vals1[-1], n1)
		like_interp = np.interp(vals1_interp, vals1, like)
		vals1 = vals1_interp
		like = like_interp

		#normalize
		like -= like.max()

		#Determine the spacing in the different parameters
		dx = vals1[1]-vals1[0]

		ax = pylab.gca()
		pylab.xlim(cols1.min()-dx/2., cols1.max()+dx/2.)
		pylab.ylim(0,1.05)
		pylab.plot(vals1, np.exp(like), linewidth=3)

		#Find the levels of the 68% and 95% contours
		level1, level2 = self._find_grid_contours(like, 0.68, 0.95)
		above = np.where(like>level1)[0]
		left = above[0]
		right = above[-1]
		x1 = vals1[left]
		x2 = vals1[right]
		#Lmin = ax.get_ylim()[0]
		L1 = like[left]
		L2 = like[right]
		pylab.plot([x1,x1], [0, np.exp(L1)], ':', color='black')
		pylab.plot([x2,x2], [0, np.exp(L2)], ':', color='black')


		above = np.where(like>level2)[0]
		left = above[0]
		right = above[-1]
		x1 = vals1[left]
		x2 = vals1[right]
		#Lmin = ax.get_ylim()[0]
		L1 = like[left]
		L2 = like[right]
		pylab.plot([x1,x1], [0, np.exp(L1)], ':', color='gray')
		pylab.plot([x2,x2], [0, np.exp(L2)], ':', color='gray')
		pylab.xlabel("$"+self._display_names[name1]+"$")


	@staticmethod
	def _find_grid_contours(log_like, contour1, contour2):
		like = np.exp(log_like)
		like_total = like.sum()
		def objective(limit, target):
			w = np.where(like>limit)
			return like[w].sum() - target
		target1 = like_total*contour1
		target2 = like_total*contour2
		level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
		level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
		level1 = np.log(level1)
		level2 = np.log(level2)
		return level1, level2


	def _plot_2d(self, name1, name2, log_like=True):
		# Load the columns
		cols1 = list(self.cols_for_name(name1).values())[0]
		cols2 = list(self.cols_for_name(name2).values())[0]
		like = list(self.cols_for_name("LIKE").values())[0]

		#Marginalize over all the other parameters by summing
		#them up
		vals1 = np.unique(cols1)
		vals2 = np.unique(cols2)
		n1 = len(vals1)
		n2 = len(vals2)
		like_sum = np.zeros((n1,n2))
		for k,(v1, v2) in enumerate(itertools.product(vals1, vals2)):
			w = np.where((cols1==v1)&(cols2==v2))
			i,j = np.unravel_index(k, like_sum.shape)
			like_sum[i,j] = self._logsumexp(like[w])
		like = like_sum.flatten()

		#Normalize the log-likelihood to peak=0
		like -= like.max()

		#Determine the spacing in the different parameters
		dx = vals1[1]-vals1[0]
		dy = vals2[1]-vals2[0]

		# Set up the axis ranges, grid, and labels
		ax = pylab.gca()
		pylab.xlim(cols1.min()-dx/2., cols1.max()+dx/2.)
		pylab.ylim(cols2.min()-dy/2., cols2.max()+dy/2.)
		# pylab.grid()
		pylab.xlabel("$"+self._display_names[name1]+"$")
		pylab.ylabel("$"+self._display_names[name2]+"$")

		#Choose a color mapping
		norm = matplotlib.colors.Normalize(np.exp(like.min()), np.exp(like.max()))
		colormap = pylab.cm.Reds


		edges1 = set()
		edges2 = set()
		def toggle_edge(edges, x1,y1,x2,y2):
			e = ((np.around(x1,4),np.around(y1,4)), (np.around(x2,4),np.around(y2,4)))
			if e in edges: edges.remove(e)
			else: edges.add(e)

		#Find the levels of the 68% and 95% contours
		level1, level2 = self._find_grid_contours(like, 0.68, 0.95)


		#Loop through our grid
		for (px, py, L) in zip(cols1, cols2,like):
			#get the color for the point
			c = colormap(norm(np.exp(L)))
			#create and apply the square colour patch
			r = pylab.Rectangle((px-dx/2.,py-dy/2.), dx, dy, color=c)
			ax.add_artist(r)
			if L>level1:
				toggle_edge(edges1, px-dx/2., py-dy/2., px-dx/2., py+dy/2.)
				toggle_edge(edges1, px-dx/2., py-dy/2., px+dx/2., py-dy/2.)
				toggle_edge(edges1, px-dx/2., py+dy/2., px+dx/2., py+dy/2.)
				toggle_edge(edges1, px+dx/2., py-dy/2., px+dx/2., py+dy/2.)
			if L>level2:
				toggle_edge(edges2, px-dx/2., py-dy/2., px-dx/2., py+dy/2.)
				toggle_edge(edges2, px-dx/2., py-dy/2., px+dx/2., py-dy/2.)
				toggle_edge(edges2, px-dx/2., py+dy/2., px+dx/2., py+dy/2.)
				toggle_edge(edges2, px+dx/2., py-dy/2., px+dx/2., py+dy/2.)

		for ((x1,y1), (x2,y2)) in edges1:
			pylab.plot([x1,x2],[y1,y2], '-', linewidth=3, color='black')
		for ((x1,y1), (x2,y2)) in edges2:
			pylab.plot([x1,x2],[y1,y2], '-', linewidth=3, color='gray')

		#create and add a colorbar
		sm = pylab.cm.ScalarMappable(cmap=colormap, norm=norm)
		sm._A = [] #hack from StackOverflow to make this work
		pylab.colorbar(sm, label='Likelihood')
