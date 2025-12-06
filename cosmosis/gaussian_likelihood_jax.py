from .jax import tools as jax_tools
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from functools import partial
import jax
import numpy as np
from .datablock import names, SectionOptions
from .runtime import FunctionModule



MISSING = "if_you_see_this_there_was_a_mistake_creating_a_gaussian_likelihood"

# we do not vary with respect to the data x location, just the theory y values
# could think about the theory x values - probably don't need those in most cases?
@partial(jax.jit, static_argnums=(0,))
def generate_theory_points(data_x, theory_x, theory_y):
    "Generate theory predicted data points by interpolation into the theory"
    jax.lax.stop_gradient(data_x)
    jax.lax.stop_gradient(theory_x)
    s = jax_tools.InterpolatedUnivariateSpline(theory_x, theory_y)
    return s(data_x)


@jax.jit
def _do_likelihood(theory_x, theory_y, data_x, data_y, inv_cov, log_det):
    # These should all be constant for us
    jax.lax.stop_gradient(theory_x)
    jax.lax.stop_gradient(data_x)
    jax.lax.stop_gradient(data_y)
    jax.lax.stop_gradient(inv_cov)
    jax.lax.stop_gradient(log_det)
    #get data x by interpolation
    x = jnp.atleast_1d(generate_theory_points(data_x, theory_x, theory_y))
    mu = jnp.atleast_1d(data_y)

    #gaussian likelihood
    d = x-mu
    chi2 = jnp.einsum('i,ij,j', d, inv_cov, d)
    like = -0.5*chi2

    norm = -0.5 * log_det
    like += norm

    return like, chi2, x

_do_likelihood_jac = jax.jacrev(_do_likelihood, argnums=[1])


@register_pytree_node_class
class GaussianLikelihood:
    """
    Gaussian likelihood with a fixed covariance.  
    
    Subclasses must override build_data and build_covariance,
    e.g. to load from file.
    """
    x_section = MISSING
    x_name    = MISSING
    y_section = MISSING
    y_name    = MISSING
    like_name = MISSING

    #Set this to False to load the covariance at 
    #each cosmology instead of once at the start
    constant_covariance = True

    def __init__(self, data_x, data_y, cov, inv_cov, log_det_constant, kind, likelihood_only=False, **kwargs):
        self.data_x = jnp.array(data_x)
        self.data_y = jnp.array(data_y)
        self.cov = jnp.array(cov)
        self.inv_cov = jnp.array(inv_cov)
        self.log_det_constant = log_det_constant
        self.kind = kind
        self.likelihood_only = likelihood_only
        for key in ["x_section", "x_name", "y_section", "y_name", "like_name"]:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    @classmethod
    def build(cls, options):
        data_x, data_y = cls.build_data(options)
        likelihood_only = options.get_bool('likelihood_only', False)

        if cls.constant_covariance:
            cov = cls.build_covariance(options)
            inv_cov = cls.build_inverse_covariance(options)

            if not likelihood_only:
                chol = jnp.linalg.cholesky(cov)


            # We may want to include the normalization of the likelihood
            # via the log |C| term.
            include_norm = options.get_bool("include_norm", False)
            if include_norm:
                # We have no datablock here so we don't want to call any subclass method
                _, log_inv_det = jnp.linalg.slogdet(inv_cov)
                log_det_constant = -log_inv_det
                print("Including -0.5*|C| normalization in {} likelihood where |C| = {}".format(cls.like_name, log_det_constant))
            else:
                log_det_constant = 0.0
        else:
            raise ValueError("You have set constant_covariance=False in a Gaussian likelihood. This does not work for JAX yet.")

        #Interpolation type, when interpolating into theory vectors
        kind = options.get_string("kind", "cubic")

        #Allow over-riding where the inputs come from in 
        #the options section
        overrides = {}
        if options.has_value("x_section"):
            overrides["x_section"] = options['x_section']
        if options.has_value("y_section"):
            overrides["y_section"] = options['y_section']
        if options.has_value("x_name"):
            overrides["x_name"] = options['x_name']
        if options.has_value("y_name"):
            overrides["y_name"] = options['y_name']
        if options.has_value("like_name"):
            overrides["like_name"] = options['like_name']

        likelihood = cls(
            data_x, data_y, cov, inv_cov, log_det_constant, kind,
            likelihood_only=likelihood_only,
            **overrides
        )

        return likelihood

    def build_data(self):
        """
        Override the build_data method to read or generate 
        the observed data vector to be used in the likelihood
        """
        raise RuntimeError("Your Gaussian covariance code needs to "
            "over-ride the build_data method so it knows how to "
            "load the observed data")

    def build_covariance(self):
        """
        Override the build_covariance method to read or generate 
        the observed covariance
        """
        raise RuntimeError("Your Gaussian covariance code needs to "
            "over-ride the build_covariance method so it knows how to "
            "load the data covariance (or set constant_covariance=False and "
            "over-ride the extract_covariance method)")



    def build_inverse_covariance(self):
        """
        Override the build_inverse_covariance method to change
        how the inverse is generated from the covariance.

        When the covariance is generated from a suite of simulations,
        for example, the simple inverse is not the best estimate.

        """
        # inverse of symmetric matrix should remain symmetric
        if jnp.allclose(self.cov, self.cov.T):
            return jnp.linalg.pinv(self.cov, hermitian=True)
        return jnp.linalg.inv(self.cov)


    def cleanup(self):
        """
        You can override the cleanup method if you do something 
        unusual to get your data, like open a database or something.
        It is run just once, at the end of the pipeline.
        """
        pass
    
    def extract_covariance(self, block):
        """
        Override this and set constant_covariance=False
        to enable a cosmology-dependent covariance.

        Load the covariance from the block here.
        """
        raise RuntimeError("You need to implement the method "
            "'extract_covariance' if you set constant_covariance=False "
            "in a gaussian likelihood")

    def extract_inverse_covariance(self, block):
        """
        Override this and set constant_covariance=False
        to enable a cosmology-dependent inverse
        covariance matrix.

        By default the inverse is just directly calculated from
        the covariance, but you might have a different method.
        """
        return jnp.linalg.inv(self.cov)

    def extract_covariance_log_determinant(self, block):
        """
        If you are using a varying covariance we have to account
        for the dependence of the covariance matrix on parameters
        in the likelihood.

        Override this method if you have a faster way to get |C|
        rather than just taking the determinant of directly (e.g.
        if you know it is diagonal or block-diagonal).

        Since we know that we must have the inverse covariance,
        whereas the covariance itself is optional, we use the former
        in the default implementation.
        """
        sign, log_inv_det = jnp.linalg.slogdet(self.inv_cov)
        log_det = -log_inv_det
        return log_det

    def do_likelihood(self, block):
        theory_x, theory_y = self.extract_theory_samples(block)
        inv_cov = self.inv_cov
        log_det = self.log_det_constant

        like, chi2, x = _do_likelihood(theory_x, theory_y, self.data_x, self.data_y, inv_cov, log_det)

        r = _do_likelihood_jac(theory_x, theory_y, self.data_x, self.data_y, inv_cov, log_det)
        # dlike_dx = r[0][0]
        # dlike_dy = r[0][1]
        dlike_dy = r[0][0]

        # block[names.likelihoods + "_derivative", f"{self.like_name}_like_by_{self.x_section}.{self.x_name}"] = np.array(dlike_dx)
        # block[names.likelihoods + "_derivative", f"{self.like_name}_like_by_{self.y_section}.{self.y_name}"] = np.array(dlike_dy)
        # block.put_derivative(names.likelihoods, self.like_name+"_like", self.x_section, self.x_name, dlike_dx)
        block.put_derivative(names.likelihoods, self.like_name+"_like", self.y_section, self.y_name, dlike_dy)


        #Now save the resulting likelihood
        block[names.likelihoods, self.like_name+"_LIKE"] = float(like)

        #It can be useful to save the chi^2 as well as the likelihood,
        #especially when the covariance is non-constant.
        block[names.data_vector, self.like_name+"_CHI2"] = float(chi2)
        block[names.data_vector, self.like_name+"_N"] = self.data_y.size

        if self.likelihood_only:
            return

        # Save various other quantities
        block[names.data_vector, self.like_name+"_LOG_DET"] = float(log_det)
        block[names.data_vector, self.like_name+"_NORM"] = -0.5 * float(log_det)

        #And also the predicted data points - the vector of observables 
        # that in a fisher approch we want the derivatives of.
        #and inverse cov mat which also goes into the fisher matrix.
        block[names.data_vector, self.like_name + "_theory"] = np.array(x)
        block[names.data_vector, self.like_name + "_data"] = np.array(self.data_y)
        block[names.data_vector, self.like_name + "_inverse_covariance"] = np.array(inv_cov)

        return block


    def simulate_data_vector(self, x):
        "Simulate a data vector by adding a realization of the covariance to the mean"
        #generate a vector of normal deviates
        r = jnp.random.randn(x.size)
        return x + jnp.dot(self.chol, r)


    def extract_theory_samples(self, block):
        "Extract relevant theory from block and get theory at data x values"
        theory_x = block[self.x_section, self.x_name]
        theory_y = block[self.y_section, self.y_name]
        return theory_x, theory_y


    @classmethod
    def build_module(cls):

        def setup(options):
            options = SectionOptions(options)
            likelihoodCalculator = cls.build(options)
            return likelihoodCalculator

        def execute(block, config):
            likelihoodCalculator = config
            likelihoodCalculator.do_likelihood(block)
            return 0

        def cleanup(config):
            likelihoodCalculator = config
            likelihoodCalculator.cleanup()

        return setup, execute, cleanup

    @classmethod
    def as_module(cls, name):
        setup, execute, cleanup = cls.build_module()
        return FunctionModule(name, setup, execute, cleanup)



    def tree_flatten(self):
        children = (self.data_x, self.data_y, self.cov, self.inv_cov)
        aux_data = {
            "log_det_constant": self.log_det_constant,
            "likelihood_only": self.likelihood_only,
            "kind": self.kind,
            "x_section": self.x_section,
            "x_name": self.x_name,
            "y_section": self.y_section,
            "y_name": self.y_name,
            "like_name": self.like_name
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data_x, data_y, cov, inv_cov = children
        return cls(
            data_x, data_y, cov, inv_cov, 
            aux_data["log_det_constant"],
            aux_data["kind"],
            likelihood_only=aux_data["likelihood_only"],
            x_section=aux_data["x_section"],
            x_name=aux_data["x_name"],
            y_section=aux_data["y_section"],
            y_name=aux_data["y_name"],
            like_name=aux_data["like_name"]
        )