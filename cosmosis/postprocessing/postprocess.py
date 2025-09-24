from . import plots
from . import statistics
from .postprocess_base import PostProcessor, postprocessor_registry
import numpy as np
import collections
import hashlib

def postprocessor_for_sampler(sampler):
	return postprocessor_registry.get(sampler)

def blinding_value(name, seed):
    #hex number derived from code phrase
    m = hashlib.md5(name).hexdigest()
    #convert to decimal
    s = int(m, 16) + seed
    # last 8 digits
    f = s%100000000
    # turn 8 digit number into value between 0 and 1
    g = f*1e-8
    #get value between -1 and 1
    return g*2-1


def additive_blinding(postprocessors, seed):
	factors = collections.defaultdict(lambda: -np.inf)
	for P in postprocessors:
		for c,col in enumerate(P.colnames):
			col = col.lower()
			#Skip likelihood columns
			if col in ['like','post', 'weight', 'log_weight', 'old_weight', 'old_log_weight']:continue
			#Work out the approximate scale of the parameters.
			f = P.approximate_scale_ceiling(c)
			#We use the max approx scale over all the chains
			factors[col] = max(factors[col], f)

	#print out scale info
	for col,f in list(factors.items()):
		print("Blinding additive value for %s ~ %.1e" % (col, f))

	for P in postprocessors:
		for c,col in enumerate(P.colnames):
			col = col.lower()
			if col in ['like','post', 'weight', 'log_weight', 'old_weight', 'old_log_weight']:continue
			#add +- the ceiling of the typical standard deviation. e.g. if width is ~ 20% add up to 3.0
			b = 3 * factors[col] * blinding_value(col,seed)
			P.additive_blind_column(c,b)

def multiplicative_blinding(postprocessors, seed):
	#print out scale info
	scale = 0.5
	print("Blinding all parameters by -50% to +50%")
	for P in postprocessors:
		for c,col in enumerate(P.colnames):
			col = col.lower()
			#blind by up to 50%
			f = blinding_value(col,seed) * 0.5
			P.multiplicative_blind_column(c,f)




class MetropolisHastingsProcessor(PostProcessor):
	elements=[
		plots.MetropolisHastingsPlots,
		plots.TracePlots,
		statistics.MetropolisHastingsStatistics,
		statistics.MetropolisHastingsCovariance,
		statistics.Citations,
	]
	def reduced_col(self, name, stacked=True):
		cols = self.get_col(name, stacked=False)
		burn = self.options.get("burn", 0)
		thin = self.options.get("thin", 1)

		if 0.0<burn<1.0:
			burn = int(len(cols[0])*burn)
		else:
			burn = int(burn)
		cols = [col[burn::thin] for col in cols]
		if stacked:
			return np.concatenate(cols)
		else:
			return cols

	def posterior_sample(self):
		"""
		A posterior sample of MCMC is just all the samples.

		Return an array of Trues with the same length as the chain

		"""
		n = self.reduced_col(self.colnames[0]).size
		return np.ones(n, dtype=bool)


class PocoProcessor(MetropolisHastingsProcessor):
	sampler = "pocomc"
	def reduced_col(self, name, stacked=True):
		# PocoMC has the burn-in removed already,
		# so there is nothing to remove here.
		return self.get_col(name, stacked=stacked)



class EmceeProcessor(MetropolisHastingsProcessor):
	sampler="emcee"
	def reduced_col(self, name, stacked=True):
		cols = self.get_col(name, stacked=False)
		burn = self.options.get("burn", 0)
		thin = self.options.get("thin", 1)

		if 0.0<burn<1.0:
			burn = int(len(cols[0])*burn)
		else:
			burn = int(burn)

		cols = [col[burn::] for col in cols]

		if thin!=1:
			walkers=self.sampler_option("walkers")
			index = np.arange(len(cols[0]), dtype=np.int64)
			index = index//int(walkers)
			w = (index%thin)==0
			cols = [col[w] for col in cols]

		if stacked:
			return np.concatenate(cols)
		else:
			return cols

class ZeusProcessor(EmceeProcessor):
	sampler="zeus"

class KombineProcessor(MetropolisHastingsProcessor):
	sampler="kombine"
	pass


class PymcProcessor(MetropolisHastingsProcessor):
	sampler="pymc"
	pass

class MetropolisProcessor(MetropolisHastingsProcessor):
	sampler="metropolis"
	elements=[
		plots.MetropolisHastingsPlots,
		statistics.MetropolisHastingsStatistics,
		statistics.MetropolisHastingsCovariance,
		statistics.DunkleyTest,
		statistics.GelmanRubinStatistic,
		statistics.Citations,
		]


class WeightedMetropolisProcessor(MetropolisHastingsProcessor):
	sampler="weighted_metropolis"
	elements=[
		plots.WeightedMetropolisPlots,
		plots.TracePlots,
		statistics.WeightedMetropolisStatistics,
		statistics.WeightedMetropolisHastingsCovariance,
		#statistics.DunkleyTest,
		statistics.Citations,		
		]

	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		if self.has_col("weight"):
			w = MetropolisHastingsProcessor.reduced_col(self, "weight").copy()
			logw = np.log(w)
		elif self.has_col("log_weight"):
			logw = MetropolisHastingsProcessor.reduced_col(self, "log_weight").copy()
		else:
			raise ValueError("No 'weight' or 'log_weight' column found in chain.")


		if self.has_col("old_weight"):
			old_w = MetropolisHastingsProcessor.reduced_col(self, "old_weight").copy()
			old_logw = np.log(old_w)
			logw += old_logw
			print("Including old_weight in weight")
		elif self.has_col("old_log_weight"):
			old_logw = MetropolisHastingsProcessor.reduced_col(self, "old_log_weight").copy()
			logw += old_logw
			print("Including old_log_weight in weight")
		logw-=np.nanmax(logw)
		self._weight_col = np.exp(logw)
		return self._weight_col    

	def posterior_sample(self):
		"""
		Weighted chains are *not* drawn from the posterior distribution
		but we do have the information we need to construct such a sample.

		This function returns a boolean array with True where we should
		use the sample at that index, and False where we should not.

		"""
		w = self.weight_col()
		w = w / w.max()
		u = np.random.uniform(size=w.size)
		return u<w


class ImportanceProcessor(WeightedMetropolisProcessor):
	sampler="importance"
	elements=[
		plots.WeightedMetropolisPlots,
		statistics.WeightedMetropolisStatistics,
		# statistics.DunkleyTest,
		statistics.Citations,		
		]

class GridProcessor(PostProcessor):
	elements=[
		plots.GridPlots1D,
		plots.GridPlots2D,
		statistics.GridStatistics,
		statistics.Citations,
		]
	sampler="grid"



class TestProcessor(PostProcessor):
	elements = [plots.TestPlots]
	sampler="test"
	cosmosis_standard_output=False

	def load(self, ini):
		if isinstance(ini, str) and os.path.isdir(ini):			
			self._sampler_options['save_dir'] = ini
		else:
			for key,val in ini.items("test"):
				self.sampler_options[key]=str(val)



class MultinestProcessor(WeightedMetropolisProcessor):
	elements = [
		plots.MultinestPlots,
		plots.TracePlots,
		statistics.MultinestStatistics,
		statistics.MultinestCovariance,
		statistics.Citations,
		]
	sampler="multinest"
	def reduced_col(self, name, stacked=True):
		#we only use the last n samples from a multinest output
		#file.  And omit zero-weighted samples.
		n = int(self.final_metadata[0]["nsample"])
		col = self.get_col(name)
		w = self.get_col("weight")[-n:]
		return col[-n:][w>0]
		
	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		n = int(self.final_metadata[0]["nsample"])
		w = self.get_col("weight")[-n:]
		w = w[w>0].copy()
		self._weight_col = w
		return self._weight_col

	def posterior_sample(self):
		"""
		Multinest chains are *not* drawn from the posterior distribution
		but we do have the information we need to construct such a sample.

		This function returns a boolean array with True where we should
		use the sample at that index, and False where we should not.

		"""
		w = self.weight_col()
		w = w / w.max()
		u = np.random.uniform(size=w.size)
		return u<w

class PolyChordProcessor(MultinestProcessor):
    elements = [
            plots.PolychordPlots,
            plots.TracePlots,
            statistics.PolychordStatistics,
            statistics.PolychordCovariance,
            statistics.Citations,
            ]
    sampler="polychord"

class DynestyProcessor(MultinestProcessor):
	elements = [
			plots.PolychordPlots,
			plots.TracePlots,
			statistics.PolychordStatistics,
			statistics.PolychordCovariance,
			statistics.Citations,
			]
	sampler="dynesty"

	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		w = self.get_col("log_weight")
		w = np.exp(w - w.max())
		# w = w[w>0]
		self._good_index = w > 0
		self._weight_col = (w / w.sum())[self._good_index]
		return self._weight_col

	def reduced_col(self, name, stacked=True):
		col = self.get_col(name)
		self.weight_col()
		x = col[self._good_index]
		return x

class NautilusProcess(DynestyProcessor):
	sampler = "nautilus"

		


class PMCPostProcessor(WeightedMetropolisProcessor):
	sampler="pmc"
	elements = [
		plots.WeightedMetropolisPlots,
		statistics.WeightedMetropolisStatistics,
		#statistics.WeightedMetropolisHastingsCovariance,
		#statistics.DunkleyTest,
		statistics.Citations,		
	]
	def reduced_col(self, name, stacked=True):
		#we only use the last n samples from a multinest output
		#file.  And omit zero-weighted samples.
		n = int(self.final_metadata[0]["nsample"])
		col = self.get_col(name)
		w = self.get_col("log_weight")[-n:]
		return col[-n:][np.isfinite(w)]
		
	def weight_col(self):
		if hasattr(self, "_weight_col"):
			return self._weight_col
		n = int(self.final_metadata[0]["nsample"])
		w = self.get_col("log_weight")[-n:]
		w = w[np.isfinite(w)].copy()
		self._weight_col = np.exp(w)
		return self._weight_col

	def posterior_sample(self):
		"""
		PMC chains are *not* drawn from the posterior distribution - they have weights.
		We do have the information we need to construct such a sample.

		This function returns a boolean array with True where we should
		use the sample at that index, and False where we should not.

		"""
		w = self.weight_col()
		w = w / w.max()
		u = np.random.uniform(size=w.size)
		return u<w


class SnakeProcessor(PostProcessor):
	sampler='snake'
	elements = [
		plots.GridPlots1D,
		plots.SnakePlots2D,
		statistics.Citations,		
	]

class FisherProcessor(PostProcessor):
	sampler = 'fisher'
	elements = [
		plots.CovarianceMatrixEllipse,
		plots.CovarianceMatrixGaussians,
		statistics.CovarianceMatrix1D,
		statistics.CovarianceMatrixEllipseAreas,
		statistics.FisherFigureOfMerit,
	]


class StarProcessor(PostProcessor):
	sampler='star'
	elements=[
		plots.StarPlots,
	]


class AprioriProcessor(PostProcessor):
	sampler='apriori'
	elements=[
		plots.AprioriPlots1D,
		plots.AprioriPlots2D,
	]
	def reduced_col(self, name, stacked=True):
		cols = self.get_col(name, stacked=False)
		likes = self.get_col("post", stacked=False)
		cols = [col[np.isfinite(like)] for col, like in zip(cols, likes)]

		if stacked:
			return np.concatenate(cols)
		else:
			return cols
	
