#coding: utf-8
from .. import ParallelSampler
from ...runtime import logs
import ctypes as ct
import os
import cosmosis
import numpy as np
import sys

loglike_type = ct.CFUNCTYPE(ct.c_double, 
    ct.POINTER(ct.c_double),  #cube
    ct.c_int,   #ndim
    ct.c_int,   #npars
    ct.c_voidp, #context
)


dumper_type = ct.CFUNCTYPE(None, #void
    ct.c_int,  #nsamples
    ct.c_int,  #nlive
    ct.c_int,  #npars
    ct.POINTER(ct.c_double),   #physlive
    ct.POINTER(ct.c_double),   #posterior
    ct.POINTER(ct.c_double),   #paramConstr
    ct.c_double,   #maxLogLike
    ct.c_double,   #logZ
    ct.c_double,   #INSLogZ
    ct.c_double,   #logZerr
    ct.c_voidp,   #Context
)


multinest_args = [
    ct.c_bool,  #nest_IS   nested importance sampling 
    ct.c_bool,  #nest_mmodal   mode separation
    ct.c_bool,  #nest_ceff   constant efficiency mode
    ct.c_int,   #nest_nlive
    ct.c_double, #nest_tol
    ct.c_double, #nest_ef
    ct.c_int, #nest_ndims,
    ct.c_int, #nest_totPar,
    ct.c_int, #nest_nCdims,
    ct.c_int, #maxClst,
    ct.c_int, #nest_updInt,
    ct.c_double, #nest_Ztol,
    ct.c_char_p, #nest_root,
    ct.c_int, #seed,
    ct.POINTER(ct.c_int), #nest_pWrap,
    ct.c_bool, #nest_fb,
    ct.c_bool, #nest_resume,
    ct.c_bool, #nest_outfile,
    ct.c_bool, #initMPI,
    ct.c_double, #nest_logZero,
    ct.c_int, #nest_maxIter,
    loglike_type, #loglike,
    dumper_type, #dumper,
    ct.c_voidp, #context
]


MULTINEST_SECTION='multinest'

def load_multinest_library(mpi_version):
    if mpi_version:
        libname = "libnest3_mpi.so"
    else:
        libname = "libnest3.so"
    dirname = os.path.split(__file__)[0]
    libname = os.path.join(dirname, "multinest_src", libname)
        
    try:
        libnest3 = ct.cdll.LoadLibrary(libname)
    except Exception as error:
        raise RuntimeError("Multinest could not be loaded.\n"
                            "This may mean an MPI compiler was not found to compile it,\n"
                            "or that some other error occurred.  More info below:\n"
                            + str(error)+'\n')

    return libnest3



class MultinestSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("like", float), ("post", float), ("weight", float)]
    supports_smp=False
    internal_resume = True

    def config(self):
        libnest3 = load_multinest_library(self.pool is not None)

        self._run = libnest3.run
        self._run.restype=None
        self._run.argtypes = multinest_args
        self.converged=False

        self.ndim = len(self.pipeline.varied_params)

        # We add one to the output to save the posterior as well as the
        # likelihood.
        self.npar = self.ndim + self.pipeline.number_extra + 2

        #Required options
        self.max_iterations = self.read_ini("max_iterations", int)
        self.live_points    = self.read_ini("live_points", int)

        #Output and feedback options
        self.feedback               = self.read_ini("feedback", bool, True)
        self.resume                 = self.read_ini("resume", bool, False)
        self.multinest_outfile_root = self.read_ini("multinest_outfile_root", str, "")
        self.update_interval        = self.read_ini("update_interval", int, 200)

        if len(self.multinest_outfile_root)>511:
            raise ValueError("Multinest parameter 'multinest_outfile_root'"
                " cannot be longer than 511 characters.")
        #General run options
        self.random_seed = self.read_ini("random_seed", int, -1)
        self.importance  = self.read_ini("ins", bool, True)
        self.efficiency  = self.read_ini("efficiency", float, 1.0)
        self.tolerance   = self.read_ini("tolerance", float, 0.1)
        self.log_zero    = self.read_ini("log_zero", float, -1e6)

        #Multi-modal options
        self.mode_separation    = self.read_ini("mode_separation", bool, False)
        self.const_efficiency   = self.read_ini("constant_efficiency", bool, False)
        self.max_modes          = self.read_ini("max_modes", int, default=100)
        self.cluster_dimensions = self.read_ini("cluster_dimensions", int, default=-1)
        self.mode_ztolerance    = self.read_ini("mode_ztolerance", float, default=0.5)

        #Parameters with wrap-around edges - can help sampling
        #of parameters which are relatively flat in likelihood
        wrapped_params = self.read_ini("wrapped_params", str, default="")
        wrapped_params = wrapped_params.split()
        self.wrapping = [0 for i in range(self.ndim)]
        for p in wrapped_params:
            try:
                P = p.split('--')
            except ValueError:
                raise ValueError("You included {} in wrapped_params mulitnest option but should be format: section--name".format(p))
            if P in self.pipeline.varied_params:
                index = self.pipeline.varied_params.index(P)
                self.wrapping[index] = 1
                logs.overview("MULTINEST: Parameter {} ({}) will be wrapped around the edge of its prior".format(index,p))
            elif P in self.pipeline.parameters:
                logs.warnings("MULTINEST NOTE: You asked for wrapped sampling on {}. That parameter is not varied in this pipeline, so this will have no effect.".format(p))
            else:
                raise ValueError("You asked for an unknown parameter, {} to be wrapped around in the multinest wrapped_params option.".format(p))



        if self.output:
            def dumper(nsample, nlive, nparam, live, post, paramConstr, max_log_like, logz, ins_logz, log_z_err, context):
                logs.overview("Saving %d samples" % nsample)
                self.output_params(nsample, live, post, logz, ins_logz, log_z_err)
            self.wrapped_output_logger = dumper_type(dumper)
        else:
            def dumper(nsample, nlive, nparam, live, post, paramConstr, max_log_like, logz, ins_logz, log_z_err, context):
                return
            self.wrapped_output_logger = dumper_type(dumper)

        def likelihood(cube_p, ndim, nparam, context_p):
            # The -1 is because we store the likelihood separately.
            nextra = nparam-ndim-2
            #pull out values from cube
            cube_vector = np.array([cube_p[i] for i in range(ndim)])
            vector = self.pipeline.denormalize_vector_from_prior(cube_vector)

            try:
                results = self.pipeline.run_results(vector)
            except KeyboardInterrupt:
                raise sys.exit(1)

            for i in range(ndim):
                cube_p[i] = vector[i]

            for i in range(nextra):
                cube_p[ndim+i] = results.extra[i]

            # posterior column
            cube_p[ndim+nextra] = results.prior
            cube_p[ndim+nextra+1] = results.post

            return results.like
        self.wrapped_likelihood = loglike_type(likelihood)

    def worker(self):
        self.sample()

    def execute(self):
        self.log_z = 0.0
        self.log_z_err = 0.0

        self.sample()

        self.output.final("log_z", self.log_z)
        self.output.final("log_z_error", self.log_z_err)

    def sample(self):
        # only master gets dumper function
        cluster_dimensions = self.ndim if self.cluster_dimensions==-1 else self.cluster_dimensions
        periodic_boundaries = (ct.c_int*self.ndim)()
        for i in range(self.ndim):
            periodic_boundaries[i] = self.wrapping[i]
        context=None
        init_mpi=False
        
        self._run(self.importance, self.mode_separation,
                  self.const_efficiency, self.live_points,
                  self.tolerance, self.efficiency, self.ndim,
                  self.npar, cluster_dimensions, self.max_modes,
                  self.update_interval, self.mode_ztolerance,
                  self.multinest_outfile_root.encode('ascii'), self.random_seed,
                  periodic_boundaries, self.feedback, self.resume,
                  self.multinest_outfile_root!="", init_mpi,
                  self.log_zero, self.max_iterations, 
                  self.wrapped_likelihood, self.wrapped_output_logger,
                  context)

        self.converged = True

    def output_params(self, n, live, posterior, log_z, ins_log_z, log_z_err):
        self.output.reset_to_chain_start()
        self.log_z = ins_log_z if self.importance else log_z
        self.log_z_err = log_z_err
        data = np.array([posterior[i] for i in range(n*(self.npar+2))]).reshape((self.npar+2, n))

        self.distribution_hints.set_from_sample(data[:self.ndim].T, data[self.npar - 1], weights=data[self.npar+1])

        # Note the transposition here to get the right shape
        for row in data.T:
            params = row[:self.ndim]
            extra_vals = row[self.ndim:self.npar-2]
            prior = row[self.npar-2]
            post = row[self.npar-1]
            like = row[self.npar]
            importance = row[self.npar+1]
            self.output.parameters(params, extra_vals, prior, like, post, importance)
        self.output.final("nsample", n)
        self.output.flush()

    def is_converged(self):
        return self.converged
