from ...runtime import logs
from numpy import pi, dot, exp, einsum
import numpy as np
from scipy.linalg import solve_triangular


class PopulationMonteCarlo(object):
	"""
	A Population Monte Carlo (PMC) sampler,
	which combines expectation-maximization and
	importance sampling
	
	This code follows the notation and methodolgy in
	http://arxiv.org/pdf/0903.0837v1.pdf


	"""
	def __init__(self, posterior, n, start, sigma, pool=None, student=False, nu=2.0):
		"""
		posterior: the posterior function
		n: number of components to use in the mixture
		start: estimated mean of the distribution
		sigma: estimated covariance matrix
		pool (optional): an MPI or multiprocessing worker pool

		"""
		self.posterior = posterior
		mu = np.random.multivariate_normal(start, sigma, size=n)

		if student:
			self.components = [StudentsTComponent(1.0/n, m, sigma, nu) for m in mu]
		else:
			self.components = [GaussianComponent(1.0/n, m, sigma) for m in mu]
		self.pool = pool

	def sample(self, n, update=True, do_kill=True):
		"Draw a sample from the Gaussian mixture and update the mixture"
		self.kill_count = n*1./len(self.components)/50.
		self.kill = [False for c in self.components]
		#draw sample from current mixture
		component_index, x = self.draw(n)

		#calculate likelihoods
		if self.pool is None:
			samples = list(map(self.posterior, x))
		else:
			samples = self.pool.map(self.posterior, x)

		post = np.array([s[0] for s in samples])
		extra = [s[1] for s in samples]
		post[np.isnan(post)] = -np.inf

		#update components
		log_weights = self.update_components(x, post, update, do_kill)

		return x, post, extra, component_index, log_weights

	def draw(self, n):
		"Draw a sample from the Gaussian mixture"
		A = [m.alpha for m in self.components]
		A = np.array(A)
		A/=A.sum()
		#Components to draw from
		N = np.arange(len(self.components))
		C = np.random.choice(N, size=n, replace=True, p=A)
		for i in N:
			count = np.sum(C==i)
			if count<self.kill_count:
				self.kill[i] = True
				logs.debug(f"Component {i} less than kill count ({count} < {self.kill_count})")
		x = np.array([self.components[c].sample() for c in C])
		return C, x

	def update_components(self, x, log_post, update, do_kill):
		"Equations 13-16 of arxiv.org 0903.0837v1"

		#x #n_sample*n_dim
		log_Aphi = np.array([np.log(m.alpha) + m.log_phi(x) for m in self.components]) #n-component * n_sample
		Aphi = np.array([m.alpha*m.phi(x) for m in self.components]) #n-component * n_sample
		post = np.exp(log_post)
		w = post/Aphi.sum(0) #n_sample
		logw = log_post - np.log(Aphi.sum(0))


		if not update:
			return logw

		w_norm = w/w.sum()  #n_sample

		logw_norm = np.log(w_norm)
		entropy =  -(w_norm*logw_norm).sum()
		perplexity = np.exp(entropy) / len(x)
		logs.debug(f"Perplexity = {perplexity}")


		Aphi[np.isnan(Aphi)] = 0.0
		w_norm[np.isnan(w_norm)] = 0.0
		A = [m.alpha for m in self.components]
		#rho_top = einsum('i,ij->ij', A, phi)  #n_component * n_sample
		rho_bottom = Aphi.sum(0) #n_sample
		rho = [rho_t/rho_bottom for rho_t in Aphi]



		for d,(m,rho_d) in enumerate(zip(self.components, rho)):
			try:
				m.update(w_norm, x, rho_d)
			except np.linalg.LinAlgError as error:
				logs.debug(f"Component not fitting the data very well {d} {error}")
				self.kill[d] = True

		if do_kill:
			self.components = [c for c,kill in zip(self.components,self.kill) if not kill]
		logs.debug("%d components remain" % len(self.components))
		return logw



class GaussianComponent(object):
	"""
	A single Gaussian component of the mixture model.

	"""
	def __init__(self,alpha, mu, sigma):
		self.set(alpha, mu, sigma)

	def set(self, alpha, mu, sigma):
		"Set the parameters of this distribution component"
		self.alpha = alpha
		self.mu = mu
		ndim = len(self.mu)
		self.sigma = sigma
		self.sigma_cholesky = np.linalg.cholesky(self.sigma)
		log_det = 2.0 * np.log(np.diag(self.sigma_cholesky)).sum()
		self.logA = -0.5 * (ndim*np.log(2*pi) + log_det)
		# Retain the historical normalization attribute for callers that
		# inspect it; evaluations use logA to avoid unnecessary overflow.
		self.A = exp(self.logA)

	def update(self, w_norm, x, rho_d):
		"Update the parameters according to the samples and rho values"
		alpha = dot(w_norm, rho_d)  #scalar
		if not alpha>0:
			raise np.linalg.LinAlgError("alpha = %f"%alpha)
		mu = einsum('i,ij,i->j',w_norm, x, rho_d) / alpha  #scalar
		delta = x-mu  #n_sample * n_dim
		logs.debug(f"Updating to mu = {mu}")
		sigma = einsum('i,ij,ik,i->jk',w_norm, delta, delta, rho_d) / alpha  #n_dim * n_dim
		self.set(alpha, mu, sigma)

	def phi(self, x):
		"Evaluate the distribution"
		d = (x-self.mu) #n_sample * n_dim
		z = solve_triangular(self.sigma_cholesky, d.T, lower=True).T
		chi2 = einsum('ij,ij->i', z, z)
		#result size n_sample
		return exp(self.logA - 0.5*chi2)

	def log_phi(self, x):
		"Evaluate the log distribution"
		d = (x-self.mu) #n_sample * n_dim
		z = solve_triangular(self.sigma_cholesky, d.T, lower=True).T
		chi2 = einsum('ij,ij->i', z, z)
		return self.logA - 0.5*chi2


	def sample(self):
		"Draw a sample from the distribution"
		return np.random.multivariate_normal(self.mu, self.sigma)




class StudentsTComponent(object):
	"""
	A single Students-t component of the mixture model.

	Implements (unnumbered) equations between A8
	and A9 in http://arxiv.org/pdf/0903.0837v1.pdf

	Not yet working
	"""
	def __init__(self,alpha, mu, sigma, nu):
		self.nu=nu
		self.ndim = len(mu)
		self.set(alpha, mu, sigma)

	def set(self, alpha, mu, sigma):
		"Set the parameters of this distribution component"
		from scipy.special import gamma
		self.alpha = alpha
		self.mu = mu
		p = self.ndim
		nu=self.nu
		self.sigma = sigma
		self.sigma_cholesky = np.linalg.cholesky(self.sigma)
		log_det = 2.0 * np.log(np.diag(self.sigma_cholesky)).sum()
		self.logA = (np.log(gamma((nu+p)/2.)) - np.log(gamma(nu/2.))
			- (p/2.)*np.log(pi*nu) - 0.5*log_det)
		self.A = exp(self.logA)

	def update(self, w_norm, x, rho_d):
		"Update the parameters according to the samples and rho values"
		alpha = dot(w_norm, rho_d)  #scalar
		if not alpha>0:
			raise np.linalg.LinAlgError("alpha = %f"%alpha)
		nu=self.nu
		p=self.ndim
		gamma = (nu+p)/(nu+self.chi2)  #size n_sampl
		mu = einsum('i,ij,i,i->j',w_norm, x, rho_d,gamma)  #scalar
		mu /= einsum('i,i,i',w_norm,rho_d,gamma)
		delta = x-mu  #n_sample * n_dim
		sigma = einsum('i,ij,ik,i,i->jk',w_norm, delta, delta, rho_d,gamma)  #n_dim * n_dim
		sigma/= einsum('i,i',w_norm,rho_d)
		self.set(alpha, mu, sigma)

	def phi(self, x):
		"Evaluate the distribution.  This is called tau in the paper"
		d = (x-self.mu) #n_sample * n_dim
		z = solve_triangular(self.sigma_cholesky, d.T, lower=True).T
		chi2 = einsum('ij,ij->i', z, z) #size n_sample
		#record this chi2 as we will need it later.
		#valid until x,mu,or sigma_d changes
		self.chi2=chi2
		#result size n_sample
		nu=self.nu
		p=self.ndim
		return exp(self.logA) * (1.0+chi2/nu)**(-(nu+p)/2.0)

	def sample(self):
		"Draw a sample from the distribution"
		y = np.random.multivariate_normal(np.zeros_like(self.mu), self.sigma)
		z = np.random.chisquare(self.nu)
		return self.mu + y*(self.nu/z)**0.5



def test_like(p):
	x,y=p
	a = (x+y)/np.sqrt(2)
	b = (x-y)/np.sqrt(2)
	return 0.2*exp(-0.5*(a**2+b**2/25.0)) + 0.8*exp(-0.5*(a**2/25.0+b**2))


def test():
	pmc = PopulationMonteCarlo(test_like, 2, [0.0, 0.0], [1.0, 1.0])
	for i in range(50):
		s = pmc.sample(4000)
	import pylab
	x = s[:,0]
	y = s[:,1]
	# pylab.plot(x[:,0], x[:,1], '.')
	xmin=x.min()
	xmax=x.max()
	ymin=y.min()
	ymax=y.max()
	Y,X=np.mgrid[ymin:ymax:100j,xmin:xmax:100j]
	r = np.array(list(zip(X.flatten(),Y.flatten())))
	P = np.array([test_like((xi,yi) ) for (xi,yi) in r])
	P = P.reshape(X.shape)
	for c in pmc.components:
		pylab.contour(X, Y, c.phi(r).reshape(X.shape), 3)
	pylab.contourf(X,Y,P,50, cmap='Reds')
	pylab.plot(x, y, 'k,')
	pylab.show()


if __name__ == '__main__':
	test()
