#coding: utf-8


u"""Implementation of :class:`Prior` and the concrete probability distribution specializations.

The specializations are :class:`UniformPrior`, :class:`GaussianPrior`,
:class:`TruncatedGaussianPrior`, :class:`ExponentialPrior`,
:class:`TruncatedExponentialPrior`, :class:`TruncatedOneoverxPrior`,
and :class:`DeltaFunctionPrior`.

Applications can get by only knowing about the :class:`Prior` superclass.

"""

from . import config
import numpy as np
import math
from scipy import interpolate
import copy

class Prior(object):

    u"""This class serves as an abstract base for possible prior distributions, and provides a concrete distribution factory method.

    At high level, each concrete prior distribution is a function which
    takes a value and returns a probability of that value occurring;
    actually, the logarithm of the probability density is returned as this
    is more useful in a Bayesian framework where the value of evidence
    takes the form of a log-likelihood.  There are additionally methods
    which

    * Return a value corresponding to a given cumulated probability,
      i.e. the inverse of the prior distribution.

    * Obtain the value of a number of random samples from the
      distribution.

    * Optionally, a method for returning a truncated version of the
      distribution.

    The application calls a single function, :func:`Prior.load_priors`, to
    obtain a concrete set of all the priors the user specifies in the
    configuration files, and then abstractly calls the above methods on
    those priors to get the work done.

    """

    def __init__(self):
        u"""Do nothing."""
        pass



    def sample(self, n):
        u"""Return an array of `n` samples (default one) from the distribution in the derived object.

        The generic case implemented here in the base class is an
        *extremely* expensive operation in almost all cases.  Other cases
        (such as uniform and delta functions) are specially implemented to
        run much quicker.  Moral: think hard about how you are using this
        method.  If you have to deal with truncated distributions, maybe
        this sledge-hammer approach is the only one feasible.

        """
        if n is None:
            n = 1
        Y = np.random.uniform(0., 1.0, n)
        return np.array([self.denormalize_from_prior(y) for y in Y])



    @classmethod
    def parse_prior(cls, value):
        u"""Produce a concrete :class:`Prior` object based on a line of a .ini ifile.

        This is part of the implementation of :func:`load_priors` and
        should be considered private to the class.

        """
        prior_type, parameters = value.split(' ', 1)
        prior_type = prior_type.lower()
        try:
            # Most of the priors require float
            # parameters, but not all of them -
            # the InverseTransform prior needs
            # a string argument (file name).
            parameters_flt = []
            for p in parameters.split():
                try:
                    p = float(p)
                except:
                    pass
                parameters_flt.append(p)
            parameters = parameters_flt

            if prior_type.startswith("uni"):
                return UniformPrior(*parameters)
            elif prior_type.startswith("gau") or \
                    prior_type.startswith("nor"):
                return GaussianPrior(*parameters)
            elif prior_type.startswith("exp"):
                return ExponentialPrior(*parameters)
            elif  prior_type.startswith("one"):
                return TruncatedOneoverxPrior(*parameters)
            elif prior_type.startswith("tab") or \
                    prior_type.startswith("loa"):
                return TabulatedPDF(*parameters)
            else:
                raise ValueError("Unable to parse %s as prior" %
                                 (value,))
        except TypeError:
            raise ValueError("Unable to parse %s as prior" %
                             (value,))



    @classmethod
    def load_priors(cls,prior_files):
        u"""Produce a dictionary of `(section,name)` -> :class:`Prior` objects as per the instructions in `prior_files`.

        The dictionary values are concrete :class:`Prior` types, and this
        method is the applicationʼs sole constructor of :class:`Prior`
        objects.

        Can also pass in an ini file directly
        """
        priors = {}
        for f in prior_files:
            if isinstance(f, config.Inifile):
                ini = f
            else:
                ini = config.Inifile(f) 
            for option, value in ini:
                if option in priors:
                    raise ValueError("Duplicate prior identified")
                priors[option] = cls.parse_prior(value)

        return priors

class TabulatedPDF(Prior):

    u"""Load from a 2-column ASCII table containing values for x, pdf(x).
    
    pdf(x) not interpolated (assumed to be zero) outisde of range of x

    Heavily 'inspired' by DistDeviate from GalSim
    """

    def __init__(self, function_filename=None, lower=None, upper=None):
        u"""Create an object representing the distribution specified in function_filename"""
        

        self.function_filename = function_filename
        # Basic input checking and setups
        if function_filename is None:
            raise TypeError('You must specify a function_filename for TabulatedPDF!')

        # Load the probability distribution function, pdf(x)
        xarray, pdf = np.loadtxt(function_filename, unpack=True)
        
        if lower==None:
            self.lower = xarray.min()
        else:
            self.lower = lower

        if upper==None:
            self.upper = xarray.max()
        else:
            self.upper = upper

        pdf = pdf[(xarray >= self.lower)*(xarray <= self.upper)]
        xarray = xarray[(xarray >= self.lower)*(xarray <= self.upper)]

        # Set up pdf, so cumsum basically does a cumulative trapz integral
        # On Python 3.4, doing pdf[1:] += pdf[:-1] the last value gets messed up.
        # Writing it this way works.  (Maybe slightly slower though, so if we stop
        # supporting python 3.4, consider switching to the += version.)
        pdf_x = copy.copy(pdf)
        pdf[1:] = pdf[1:] + pdf[:-1]
        pdf[1:] *= np.diff(xarray)
        pdf[0] = 0

        # Check that the probability is nonnegative
        if not np.all(pdf >= 0.):
            raise ValueError('Negative probability found in TabulatedPDF.',function)

        # Compute the cumulative distribution function = int(pdf(x),x)
        cdf = np.cumsum(pdf)

        # Quietly renormalize the probability if it wasn't already normalized
        totalprobability = cdf[-1]
        cdf /= totalprobability

        self.inverse_cdf_interp = interpolate.interp1d(cdf, xarray, kind='linear')
        self.cdf_interp = interpolate.interp1d(xarray, cdf, kind='linear')
        self.pdf_interp = interpolate.interp1d(xarray, pdf_x, kind='linear')

    def __call__(self, x):
        u"""Return the logarithm of the probability density."""
        if x<self.lower:
            return -np.inf
        elif x>self.upper:
            return -np.inf
        return np.log(self.pdf_interp(x))

    def sample(self, n):
        u"""Use interpolation of inverse CDF to give us `n` random samples from a the distribution."""
        if n is None:
            n = 1 
        return self.inverse_cdf_interp(np.random.rand(n))

    def denormalize_from_prior(self, y):
        u"""Get the value for which the cumulated probability is `y`."""
        return self.inverse_cdf_interp(y)

    def normalize_to_prior(self, x):
        if x < self.lower or x > self.upper:
            return np.nan
        return self.cdf_interp(x)

    def __str__(self):
        u"""Tersely describe ourself to a human mathematician."""
        return "Tabulated transform from {0} on range [{1}, {2}]".format(self.function_filename, self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Return a new distribution whose range is the intersection of ours with [`lower`, `upper`].

        A :class:`ValueError` will be thrown if the arguments supplied do
        not make sense.

        """
        lower = max(self.lower, lower)
        upper = min(self.upper, upper)
        if lower>upper:
            raise ValueError("One of your priors is inconsistent with the range described in the values file")
        return TabulatedPDF(self.function_filename, lower, upper)


class UniformPrior(Prior):

    u"""Statistical distribution in which all values in a range have equal probability of occurring."""
    
    def __init__(self, a, b):
        u"""Create an object which encapsulates a Uniform distribution over the interval [`a`, `b`]."""
        self.a=a
        self.b=b
        self.norm = -np.log(b-a)
        super(UniformPrior,self).__init__()


    def __call__(self, x):
        u"""Return the logarithm of the probability density, a constant value independent of `x` so long as `x` is in the proper range."""
        if x<self.a or x>self.b:
            return -np.inf
        return self.norm


    def sample(self, n):
        u"""Use NumPy to obtain a random number in the range [`a`, `b`]."""
        return np.random.uniform(self.a, self.b, n)


    def denormalize_from_prior(self, y):
        u"""Interpolate the cumulated probability `y` to the corresponding value in the interval [`a`, `b`]."""
        if y<0.0:
            x = np.nan
        elif y>1.0:
            x = np.nan
        else:
            x = y * (self.b-self.a) + self.a
        return x

    def normalize_to_prior(self, x):
        if x < self.a:
            return np.nan
        elif x > self.b:
            return np.nan
        else:
            return (x - self.a) / (self.b - self.a)


    def __str__(self):
        u"""Tersely describe ourself to a human mathematician."""
        return "U({}, {})".format(self.a,self.b)


    def truncate(self, lower, upper):
        u"""Return a new Uniform distribution whose range is the intersection of ours with [`lower`, `upper`].

        A :class:`ValueError` will be thrown if the arguments supplied do
        not make sense.

        """
        a = max(lower, self.a)
        b = min(upper, self.b)
        if a>b:
            raise ValueError("One of your priors is inconsistent with the range described in the values file")
        return UniformPrior(a, b)



class GaussianPrior(Prior):

    u"""Encapsulation of a Normal distribution function."""

    def __init__(self, mu, sigma):
        u"""Make a Normal distribution object with mean `mu` and deviation `sigma`."""
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.norm=0.5*np.log(2*np.pi*self.sigma2)
        super(GaussianPrior,self).__init__()

    def __call__(self, x):
        u"""Return the logarithm of the probability density at `x`."""
        return -0.5 * (x-self.mu)**2 / self.sigma2 - self.norm

    def sample(self, n):
        u"""Use NumPy to give us `n` random samples from a Normal distribution."""
        if n is None:
            n = 1        
        return np.random.normal(self.mu, self.sigma, n)

    def denormalize_from_prior(self, y):
        u"""Obtain the value such that the cumulated probability of obtaining a lesser value is `y`.

        Note that this is a very expensive function to call.

        """
        x_normal = normal_ppf(y)
        x = x_normal*self.sigma + self.mu
        return x

    def normalize_to_prior(self, x):
        """Return the cumulative density funtion for this value"""
        return normal_cdf((x - self.mu) / self.sigma)

    def __str__(self):
        u"""Tersely describe ourself to a human mathematician."""
        return "N({}, {} ** 2)".format(self.mu, self.sigma)

    def truncate(self, lower, upper):
        u"""Return a :class:`TruncatedGaussianPrior` object, with our mean and variance and the given `lower` and `upper` limits of non-zero probability."""
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)

    
    
class TruncatedGaussianPrior(Prior):

    u"""A Normal distribution, except that probabilities outside of a specified range are truncated to zero."""

    def __init__(self, mu, sigma, lower, upper):
        u"""Get a concrete object representing a distribution with mode at `mu`, ‘shape width’ `sigma`, and limits of non-zero probability `lower` and `upper`."""
        self.lower = lower
        self.upper = upper
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2
        self.a = (lower-mu)/sigma
        self.b = (upper-mu)/sigma
        self.phi_a = normal_cdf(self.a)
        self.phi_b = normal_cdf(self.b)
        self.norm = np.log(self.phi_b - self.phi_a) + 0.5*np.log(2*np.pi*self.sigma2)
        super(TruncatedGaussianPrior,self).__init__()

    def __call__(self, x):
        u"""Get the logarithm of the probability density at the value `x`."""
        if x<self.lower:
            return -np.inf
        elif x>self.upper:
            return -np.inf
        return -0.5 * (x-self.mu)**2 / self.sigma2 - self.norm

    def denormalize_from_prior(self, y):
        u"""Get the value for which the cumulated probability is `y`.

        This is a very expensive function.

        """
        x_normal = truncated_normal_ppf(y, self.a, self.b)
        x = x_normal*self.sigma + self.mu
        return x

    def normalize_to_prior(self, x):
        """Return the cumulative density function for this value, between 0 and 1"""
        x_normal = (x - self.mu) / self.sigma
        return truncated_normal_cdf(x_normal, self.a, self.b)


    def __str__(self):
        u"""Return a terse description of ourself."""
        return "N({}, {} ** 2)   [{} < x < {}]".format(self.mu, self.sigma, self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Produce a new :class:`Prior` object representing a Normal distribution whose range of non-zero probability is the intersection of our own range with [`lower`, `upper`]."""
        lower = max(self.lower, lower)
        upper = min(self.upper, upper)
        return TruncatedGaussianPrior(self.mu, self.sigma, lower, upper)



class ExponentialPrior(Prior):

    u"""The Exponential Distribution, zero probability of any value less than zero."""

    def __init__(self, beta):
        u"""Create object representing distribution with ‘width’ `beta`."""
        self.beta = beta
        self.log_beta = np.log(beta)
        super(ExponentialPrior,self).__init__()
    
    def __call__(self, x):
        u"""Return logarithm of probability density at `x`."""
        if x<0.0:
            return -np.inf
        return -x/self.beta - self.log_beta
    
    def sample(self,n):
        u"""Use NumPy to obtain random sample of `n` values from Exponential Distribution with our width `beta`."""
        if n is None:
            n = 1        
        return np.random.exponential(self.beta,n)

    def denormalize_from_prior(self, y):
        u"""Return value for which cumulated probability of lesser values occurring is `y`."""
        #y = 1 - exp(-x/beta)
        #exp(-x/beta) = 1 - y
        #-x/beta = log(1-y)
        #x = -beta * log(1-y)
        return -self.beta * np.log(1-y)

    def normalize_to_prior(self, x):
        """Return the cumulative distribution function for this value"""
        return 1 - np.exp(- x / self.beta)
        
    def __str__(self):
        u"""Give a terse description of ourself."""
        return "Expon({})".format(self.beta)

    def truncate(self, lower, upper):
        u"""Return a :class:`Prior` object representing the distribution you get when you take the current distribution but set probability to zero everywhere outside the range [`lower`, `upper`], and re-normalize."""
        return TruncatedExponentialPrior(self.beta, lower, upper)



class TruncatedExponentialPrior(Prior):

    u"""Like the Exponential prior, but truncated."""

    def __init__(self, beta, lower, upper):
        u"""Create a distribution with ‘half-life’ `beta`, `lower` bound of non-zero probability, and `upper` bound."""
        self.beta = beta
        self.log_beta = np.log(beta)
        if lower<0:
            lower = 0.0
        self.lower = lower
        self.upper = upper
        self.a = lower/beta
        self.b = upper/beta
        self.phi_a = exponential_cdf(self.a)
        self.phi_b = exponential_cdf(self.b)
        self.norm = np.log(self.phi_b - self.phi_a) + self.log_beta
        super(TruncatedExponentialPrior,self).__init__()

    
    def __call__(self, x):
        u"""Return the logarithm of probability density at `x`."""
        #(1/beta)*exp(-x/beta)
        if x<self.lower:
            return -np.inf
        if x>self.upper:
            return -np.inf
        return -x/self.beta - self.norm
    

    def denormalize_from_prior(self, y):
        u"""Return the value at which the cumulated probability is `y`."""
        x_normal = truncated_exponential_ppf(y, self.a, self.b)
        x = x_normal * self.beta
        return x

    def normalize_to_prior(self, x):
        x_normal = x / self.beta
        return truncated_exponential_cdf(x_normal, self.a, self.b)

    def __str__(self):
        u"""Give a terse description of ourself."""
        return "Expon({})   [{} < x < {}]".format(self.beta, self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Return a :class:`Prior` like ourself but with range of non-zero probability further restricted to `lower` and `upper` bounds."""
        lower = max(lower, self.lower)
        upper = min(upper, self.upper)
        return TruncatedExponentialPrior(self.beta, lower, upper)


class TruncatedOneoverxPrior(Prior):
    u"""The 1/x distribution, which is a uniform distribution in ln(x). As ln(x) diverges in both directions, we only provide the truncated option."""

    def __init__(self, lower, upper):
        u"""Create a distribution with 1/x, `lower` bound of non-zero probability, and `upper` bound."""
        if lower<=0:
            lower = np.nextafter(0, 1)
        self.lower = lower
        self.upper = upper
        self.ln_lower = np.log(lower)
        self.ln_upper = np.log(upper)
        # Normalization: \int_upper^lower 1/x dx = ln(upper) - ln(lower)
        self.norm = self.ln_upper-self.ln_lower
        self.ln_norm = np.log(self.norm)
        super(TruncatedOneoverxPrior,self).__init__()

    def __call__(self, x):
        u"""Return the logarithm of probability density at `x`."""
        if x<self.lower:
            return -np.inf
        if x>self.upper:
            return -np.inf
        return -np.log(x) - self.ln_norm

    def sample(self, n):
        u"""Obtain random sample of `n` values from 1/x distribution within `lower` and `upper` bounds."""
        if n is None:
            n = 1
        return np.exp(np.random.uniform(self.ln_lower, self.ln_upper, n))

    def denormalize_from_prior(self, y):
        u"""Return the value at which the cumulated probability is `y`."""
        # int_a^x dz 1/z 1/N = (ln(x)-ln(a)) / N != y
        # ln(x) = y*N + ln(a)
        return np.exp(y*self.norm + self.ln_lower)

    def normalize_to_prior(self, x):
        if x < self.lower or x > self.upper:
            return np.nan
        return (np.log(x) - self.ln_lower) / self.norm

    def __str__(self):
        u"""Give a terse description of ourself."""
        return "1/x   [{} < x < {}]".format(self.lower, self.upper)

    def truncate(self, lower, upper):
        u"""Return a :class:`Prior` like ourself but with range of non-zero probability further restricted to `lower` and `upper` bounds."""
        lower = max(lower, self.lower)
        upper = min(upper, self.upper)
        return TruncatedOneoverxPrior(lower, upper)



class DeltaFunctionPrior(Prior):

    u"""Probability distribution with non-zero probability at a single value."""

    # In case this is useful later on
    def __init__(self, x0):
        u"""Create object with atom of probability at `x0`."""
        self.x0 = x0
        super(DeltaFunctionPrior,self).__init__()

    def __call__(self, x):
        u"""The log-density is zero when `x` is `x0`, minus infinity otherwise."""
        if x==self.x0:
            return 0.0
        return -np.inf

    def sample(self, n=None):
        u"""Just return `x0` `n` times."""
        if n is None:
            n = 1
        return np.repeat(self.x0, n)

    def __str__(self):
        u"""Terse description of ourself."""
        return "delta({})".format(self.x0)

    def denormalize_from_prior(self, x):
        u"""Just return `x0`; itʼs the only value with any probability."""
        return self.x0

    def normalize_to_prior(self, x):
        """Normalize a value to a point in the prior; returns 0.5 if x == x0 and np.nan otherwise"""
        if x != self.x0:
            return np.nan
        return 0.5
        


# Helper functions
def inverse_function(f, y, xmin, xmax, *args, **kwargs):
    "Find x in [xmin,xmax] such that f(x)==y, in 1D, with bisection"
    import scipy.optimize
    def g(x):
        return f(x, *args, **kwargs) - y
    x = scipy.optimize.bisect(g, xmin, xmax)
    return x

SQRT2 = np.sqrt(2.)
def normal_cdf(x):
#    return 0.5*math.erf(x) + 0.5
    return 0.5*(math.erf(x/SQRT2) + 1)


def normal_ppf(y):
    if y<0:
        return np.nan
    if y>1:
        return np.nan
    return inverse_function(normal_cdf, y, -20.0, 20.0)

def truncated_normal_cdf(x, a, b):
    if x<a:
        return np.nan
    if x>b:
        return np.nan
    phi_a = normal_cdf(a)
    phi_b = normal_cdf(b)
    phi_x = normal_cdf(x)
    return (phi_x - phi_a) / (phi_b - phi_a)

def truncated_normal_ppf(y, a, b):
    if y<0:
        return np.nan
    if y>1:
        return np.nan
    return inverse_function(truncated_normal_cdf, y, a, b, a, b)

def exponential_cdf(x):
    if x<0.0:
        return np.nan
    return 1 - np.exp(-x)

def truncated_exponential_cdf(x, a, b):
    if x<a:
        return np.nan
    if x>b:
        return np.nan
    phi_a = exponential_cdf(a)
    phi_b = exponential_cdf(b)
    phi_x = exponential_cdf(x)
    return (phi_x - phi_a) / (phi_b - phi_a)

def exponential_ppf(y):
    #y = 1 - exp(-x)
    # exp(-x) = 1-y
    # x = -log(1-y)
    return -np.log(1-y)

def truncated_exponential_ppf(y, a, b):
    if y<0:
        return np.nan
    if y>1:
        return np.nan    
    return inverse_function(truncated_exponential_cdf, y, a, b, a, b)