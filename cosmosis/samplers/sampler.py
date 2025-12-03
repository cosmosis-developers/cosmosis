from ..runtime.attribution import PipelineAttribution
from ..runtime.utils import get_git_revision
from ..runtime import Inifile, logs
from ..output import InMemoryOutput
import datetime
import platform
import getpass
import os
import uuid
import pickle
from .hints import Hints
import numpy as np
import shutil
import numpy as np
import configparser

# Sampler metaclass that registers each of its subclasses

class RegisteredSampler(type):
    registry = {}
    def __new__(meta, name, bases, class_dict):
        if name.endswith("Sampler"):
            meta.registry = {name : cls for name, cls in meta.registry.items() if cls not in bases}
            config_name = name[:-len("Sampler")].lower()
            cls = type.__new__(meta, name, bases, class_dict)
            cls.name = config_name
            meta.registry[config_name] = cls
            return cls
        else:
            raise ValueError("Sampler classes must be named [Name]Sampler")

class Sampler(metaclass=RegisteredSampler):
    needs_output = True
    sampler_outputs = []
    understands_fast_subspaces = False
    parallel_output = False
    is_parallel_sampler = False
    supports_resume = False
    internal_resume = False

    
    def __init__(self, ini, pipeline, output=None):
        if isinstance(ini, Inifile):
            self.ini = ini
        else:
            self.ini = Inifile(ini)

        self.pipeline = pipeline
        # Default to an in-memory output
        if output is None:
            output = InMemoryOutput()
        self.output = output
        self.attribution = PipelineAttribution(self.pipeline.modules)
        self.distribution_hints = Hints()
        self.write_header()

    def write_header(self, output=None):
        if output is None:
            output = self.output
        if output is None:
            return

        for p in self.pipeline.output_names():
            output.add_column(p, float)
        for p,ptype in self.sampler_outputs:
            output.add_column(p, ptype)
        output.metadata("n_varied", len(self.pipeline.varied_params))
        self.attribution.write_output(output)
        for key, value in self.collect_run_metadata().items():
            output.metadata(key, value)
        blinding_header = self.ini.getboolean("output","blinding-header", fallback=False)
        if blinding_header:
            output.blinding_header()

    def collect_run_metadata(self):
        info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'platform': platform.platform(),
            'platform_version': platform.version(),
            'uuid': uuid.uuid4().hex,
        }
        info['cosmosis_git_version'] = get_git_revision("$COSMOSIS_SRC_DIR")
        info['csl_git_version'] = get_git_revision("$COSMOSIS_SRC_DIR/cosmosis-standard-library")
        info['cwd_git_version'] = get_git_revision("$PWD")

        # The host name and username are (potentially) private information
        # so we only save those if privacy=False, which is not the default
        privacy = self.ini.getboolean('output','privacy', fallback=True)
        save_username = not privacy
        if save_username:
            info['hostname'] = platform.node()
            info['username'] = getpass.getuser()
            info['workdir'] = os.getcwd()

        return info

    def read_ini(self, option, option_type, default=configparser._UNSET):
        """
        Read option from the ini file for this sampler
        and also save to the output file if it exists
        """
        if default is None:
            default = configparser._UNSET
        if option_type is float:
            val = self.ini.getfloat(self.name, option, fallback=default)
        elif option_type is int:
            val = self.ini.getint(self.name, option, fallback=default)
        elif option_type is bool:
            val = self.ini.getboolean(self.name, option, fallback=default)
        elif option_type is str:
            val = self.ini.get(self.name, option, fallback=default)
        else:
            raise ValueError("Internal cosmosis sampler error: "
                "tried to read ini file option with unknown type {}".format(option_type))
        if self.output:
            self.output.metadata(option, str(val))
        return val

    def read_ini_choices(self, option, option_type, choices, default=None):
        value = self.read_ini(option, option_type, default=default)
        if value not in choices:
            name = self.__class__.__name__
            raise ValueError("Parameter {} for sampler {} must be one of: {}\n Parameter file said: {}".format(option, name, choices, value))
        return value


    def config(self):
        ''' Set up sampler (could instead use __init__) '''
        pass

    def execute(self):
        ''' Run one (self-determined) iteration of sampler.
            Should be enough to test convergence '''
        raise NotImplementedError

    def write_resume_info(self, info):
        try:
            filename = self.output.name_for_sampler_resume_info()
        except NotImplementedError:
            return

        # in some fast pipelines like demo 5 a keyboard interrupt
        # is likely to happen in the middle of this dump operation
        tmp_filename = filename + '.tmp'

        with open(tmp_filename, 'wb') as f:
            pickle.dump(info, f)

        shutil.move(tmp_filename, filename)

    def read_resume_info(self):
        filename = self.output.name_for_sampler_resume_info()
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def get_sampler(cls, name):
        try:
            return cls.registry[name.lower()]
        except KeyError:
            raise KeyError(f"Unknown sampler {name}")


    def resume(self):
        raise NotImplementedError("The sampler {} does not support resuming".format(self.name))

    def is_converged(self):
        return False
    


    def start_estimate(self, method=None, input_source=None, prefer_random=False, quiet=False):
        """
        Select a starting parameter set for the sampler.

        The method is chosen by looking at the start_method and start_input
        options in the sampler's ini file.

        This can be:
        - the peak from a previous sampler (default if available)
        - the defined starting position in the values file (if start_method not set)
        - a random point in the prior (if start_method is "prior")
        - a random point from a chain file (if start_method "chain-sample")
        - the last point from a chain file (if start_method is "chain-last")
        - a point from a covariance matrix (if start_method is "cov")

        Returns
        -------
        start : np.ndarray
        """
        if method is None:
            method = self.read_ini("start_method", str, "")
        if input_source is None:
            input_source = self.read_ini("start_input", str, "")

        if method.startswith("chain"):
            if not input_source:
                raise ValueError("If you set the start_method to 'chain-best' you should not also set start_input to the name of a chain file")

            with open(input_source) as f:
                maybe_colnames = f.readline().strip('#').split()
                has_post = 'post' in maybe_colnames
                has_like = 'like' in maybe_colnames

            # User can just specify "chain" and we will try to guess.
            # We may want to draw a sample if we are using an ensemble-based
            # sampling method like emcee.
            if method == "chain" and not self.distribution_hints.has_peak():
                if prefer_random:
                    method = "chain-sample"
                elif has_post:
                    method = "chain-maxpost"
                elif has_like:
                    method = "chain-maxlike"
                else:
                    method = "chain-last"

        # Option 1: if the user is chaining samplers then we always start at the peak
        # from the previous sampler, if available.  Otherwise, why was the user chaining samplers?
        if self.distribution_hints.has_peak():
            if not quiet:
                logs.overview("Starting at max-posterior point from previous sampler")
            start = self.distribution_hints.get_peak()
            if prefer_random:
                covmat = self.distribution_hints.get_cov()
                start = sample_ellipsoid(start, covmat)

        # Option 2: start from a random point following the prior distribution
        elif method == "prior":
            print("Starting at a random point in the prior")
            if not quiet:
                logs.overview("Starting at a random point in the prior")
            start = self.pipeline.randomized_start()

        # Option 3: start from the last point in a previous chain.
        elif method == "chain-last":
            if not quiet:
                logs.overview(f"Starting from last point in file {input_source}")
            start = np.genfromtxt(input_source, invalid_raise=False)[-1, :self.pipeline.nvaried]

        # Option 4: start from a random sample of points in a previous chain.  This only
        # works if we want an ensemble of points. Should really check that.
        elif method == "chain-sample":
            if not quiet:
                logs.overview(f"Starting at random sample of points from chain file {input_source}")
            # assume the chain file has a header. check for weight or log_weight columns
            # otherwise assume that the columns match the varied parameters
            data = np.genfromtxt(input_source, invalid_raise=False)
            with open(input_source) as f:
                maybe_colnames = f.readline().strip('#').split()
            if 'weight' in maybe_colnames:
                weight_index = maybe_colnames.index('weight')
                weight = data[:, weight_index]
            elif 'log_weight' in maybe_colnames:
                log_weight_index = maybe_colnames.index('log_weight')
                weight = np.exp(data[:, log_weight_index] - data[:, log_weight_index].max())
            else:
                weight = np.ones(len(data))
            weight /= weight.sum()
            index = np.random.choice(len(data), p=weight)
            start = data[index, :self.pipeline.nvaried]

        # Option 5: start from a random sample of points drawn from the covariance of a
        # previous chain.
        elif method == "cov":
            if not quiet:
                logs.overview(f"Starting at a random sample of points from the covariance of chain {input_source}")
            if not input_source:
                raise ValueError("If you set the start_method to 'cov' you should not also set start_input to the name of a covariance file")
            covmat = np.loadtxt(input_source)[:self.pipeline.nvaried, :self.pipeline.nvaried]
            start = sample_ellipsoid(self.pipeline.start_vector(), covmat)[0]

        # Option 6: start from the best-fitting point in a previous chain
        elif method in ["chain-maxpost", "chain-maxlike"]:
            data = np.genfromtxt(input_source, invalid_raise=False)

            # read the column names again.  A bit wasteful as we may
            # have done it already, but it's only a single line.
            with open(input_source) as f:
                maybe_colnames = f.readline().strip('#').split()

            if method == "chain-maxpost":
                if not quiet:
                    logs.overview(f"Starting at best posterior point from chain file {input_source}")
                col_index = maybe_colnames.index('post')
            else:
                if not quiet:
                    logs.overview(f"Starting at best likelihood point from chain file {input_source}")
                col_index = maybe_colnames.index('like')

            best_row = data[:, col_index].argmax()
            start = data[best_row, :self.pipeline.nvaried]

        # Fallback option 7: just start at the point specified in the parameter file.
        else:
            # default method is just to use a single starting point
            start = self.pipeline.start_vector()

        return start

    def cov_estimate(self):
        covmat_file = self.read_ini("covmat", str, "")
        n = len(self.pipeline.varied_params)

        if self.distribution_hints.has_cov():
            # hints from a previous sampler take
            # precendence
            covmat = self.distribution_hints.get_cov()

        elif covmat_file:
            covmat = np.loadtxt(covmat_file)
            # Fix the size.
            # If there is only one sample parameter then 
            # we assume it is a 1x1 matrix
            # If it's a 1D vector then assume these are
            # standard deviations
            if covmat.ndim == 0:
                covmat = covmat.reshape((1, 1))
            elif covmat.ndim == 1:
                covmat = np.diag(covmat ** 2)

            # Error on incorrect shapes or sizes
            if covmat.shape[0] != covmat.shape[1]:
                raise ValueError("Covariance matrix from {}"
                                 "not square".format(covmat_file))
            if covmat.shape[0] != n:
                raise ValueError("Covariance matrix from {} "
                                 "is the wrong shape ({}x{}) "
                                 "for the {} varied params".format(
                                    covmat_file, covmat.shape[0], n))
        else:
            # Just try a minimal estimate - 5% of prior width as standard deviation
            covmat_scale = self.read_ini("covmat_scale", float, 0.05)
            covmat = np.diag([covmat_scale*p.width() for p in self.pipeline.varied_params])**2

        return covmat



class ParallelSampler(Sampler):
    parallel_output = True
    is_parallel_sampler = True
    supports_smp = True
    def __init__(self, ini, pipeline, output=None, pool=None):
        Sampler.__init__(self, ini, pipeline, output)
        self.pool = pool

    def worker(self):
        ''' Default to a map-style worker '''
        if self.pool:
            self.pool.wait()
        else:
            raise RuntimeError("Worker function called when no parallel pool exists!")

    def is_master(self):
        return self.pool is None or self.pool.is_master()


# These are marked as deprecated in emcee, so I moved them here.
# I think I wrote the first one.  And I've rewritten the second
# to use the first
def sample_ellipsoid(p0, covmat, size=1):
    """
    Produce an ellipsoid of walkers around an initial parameter value,
    according to a covariance matrix.
    :param p0: The initial parameter value.
    :param covmat:
        The covariance matrix.  Must be symmetric-positive definite or
        it will raise the exception numpy.linalg.LinAlgError
    :param size: The number of samples to produce.
    """
    return np.random.multivariate_normal(
        np.atleast_1d(p0), np.atleast_2d(covmat), size=size
    )

def sample_ball(p0, std, size=1):
    """
    Produce a ball of walkers around an initial parameter value.
    :param p0: The initial parameter value.
    :param std: The axis-aligned standard deviation.
    :param size: The number of samples to produce.
    """
    covmat = np.diag(std**2)
    return sample_ellipsoid(p0, covmat, size)
