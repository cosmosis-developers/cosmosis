#coding: utf-8
from .. import ParallelSampler
from ...runtime import logs
import ctypes as ct
import os
import numpy as np
import sys
from ...runtime.utils import mkdir
from ...output import TextColumnOutput
import warnings

prior_type = ct.CFUNCTYPE(None, 
    ct.POINTER(ct.c_double),  #hypercube
    ct.POINTER(ct.c_double),  #physical
    ct.c_int,   #ndim
)

loglike_type = ct.CFUNCTYPE(ct.c_double, 
    ct.POINTER(ct.c_double),  #physical
    ct.c_int,   #ndim
    ct.POINTER(ct.c_double),  #derived
    ct.c_int,   #nderived
)


dumper_type = ct.CFUNCTYPE(None, #void
    ct.c_int,  #ndead
    ct.c_int,  #nlive
    ct.c_int,  #npars
    ct.POINTER(ct.c_double),   #live
    ct.POINTER(ct.c_double),   #dead
    ct.POINTER(ct.c_double),   #logweights
    ct.c_double,   #logZ
    ct.c_double,   #logZerr
)


polychord_args = [
    loglike_type,           #loglike,
    prior_type,             #prior,
    dumper_type,            #dumper,
    ct.c_int,               #nlive
    ct.c_int,               #nrepeats
    ct.c_int,               #nprior
    ct.c_bool,              #do_clustering
    ct.c_int,               #feedback
    ct.c_double,            #precision_criterion
    ct.c_double,            #logzero
    ct.c_int,               #max_ndead
    ct.c_double,            #boost_posterior
    ct.c_bool,              #posteriors
    ct.c_bool,              #equals
    ct.c_bool,              #cluster_posteriors
    ct.c_bool,              #write_resume 
    ct.c_bool,              #write_paramnames
    ct.c_bool,              #read_resume
    ct.c_bool,              #write_stats
    ct.c_bool,              #write_live
    ct.c_bool,              #write_dead
    ct.c_bool,              #write_prior
    ct.c_double,            #compression_factor
    ct.c_int,               #nDims
    ct.c_int,               #nDerived 
    ct.c_char_p,            #base_dir
    ct.c_char_p,            #file_root
    ct.c_int,               #nGrade
    ct.POINTER(ct.c_double),#grade_frac
    ct.POINTER(ct.c_int),   #grade_dims
    ct.c_int,               #n_nlives
    ct.POINTER(ct.c_double),#loglikes
    ct.POINTER(ct.c_int),   #nlives
    ct.c_int,               #seed
]


POLYCHORD_SECTION='polychord'


class PolychordSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("like", float), ("post", float), ("weight", float)]
    supports_smp=False
    internal_resume = True
    understands_fast_subspaces = True

    def config(self):
        if self.pool:
            libname = "libchord_mpi.so"
        else:
            libname = "libchord.so"

        dirname = os.path.split(__file__)[0]
        libname = os.path.join(dirname, "polychord_src", libname)
            
        try:
            libchord = ct.cdll.LoadLibrary(libname)
        except Exception as error:
            sys.stderr.write("PolyChord could not be loaded.\n")
            sys.stderr.write("This may mean an MPI compiler was not found to compile it,\n")
            sys.stderr.write("or that some other error occurred.  More info below.\n")
            sys.stderr.write(str(error)+'\n')
            sys.exit(1)

        self._run = libchord.polychord_c_interface
        self._run.restype=None
        self._run.argtypes = polychord_args
        self.converged=False

        self.ndim = len(self.pipeline.varied_params)
        # We save the prior as well as the other derived params
        self.nderived = self.pipeline.number_extra + 1

        #Required options
        self.live_points    = self.read_ini("live_points", int, 100)

        #Output and feedback options
        self.feedback               = self.read_ini("feedback", int, 1)
        self.resume                 = self.read_ini("resume", bool, False)
        self.polychord_outfile_root = self.read_ini("polychord_outfile_root", str, "")
        self.base_dir               = self.read_ini("base_dir", str, ".")
        self.compression_factor     = self.read_ini("compression_factor", float, np.exp(-1))

        if len(self.polychord_outfile_root)>299:
            raise ValueError("Polychord parameter 'polychord_outfile_root'"
                " cannot be longer than 299s characters.")

        if '/' in self.polychord_outfile_root:
            warnings.warn("\n\nWARNING: / found in polychord_outfile_root\n"
                "In polychord it is better to set base_dir to the directory and \n"
                "have polychord_outfile_root not include the directory.\n"
                "Otherwise you may have to mkdir some directories yourself.\n"
                "This may cause problems below.\n"
                )

        #General run options
        self.max_iterations = self.read_ini("max_iterations", int, -1)
        self.num_repeats = self.read_ini("num_repeats", int, 0)
        self.nprior = self.read_ini("nprior", int, -1)
        self.random_seed = self.read_ini("random_seed", int, -1)
        self.tolerance   = self.read_ini("tolerance", float, 0.1)
        self.log_zero    = self.read_ini("log_zero", float, -1e6)
        self.boost_posteriors = self.read_ini("boost_posteriors", float, 0.0)
        self.weighted_posteriors = self.read_ini("weighted_posteriors", bool, True)
        self.equally_weighted_posteriors = self.read_ini("equally_weighted_posteriors", bool, True)       
        self.cluster_posteriors = self.read_ini("cluster_posteriors", bool, True)

        self.fast_fraction    = self.read_ini("fast_fraction", float, 0.5)

        if self.output:
            def dumper(ndead, nlive, npars, live, dead, logweights, log_z, log_z_err):
                logs.overview("Saving %d samples" % ndead)
                self.output_params(ndead, nlive, npars, live, dead, logweights, log_z, log_z_err)
            self.wrapped_output_logger = dumper_type(dumper)
        else:
            def dumper(ndead, nlive, npars, live, dead, logweights, log_z, log_z_err):
                pass
            self.wrapped_output_logger = dumper_type(dumper)

        def prior(cube, theta, ndim):
            cube_vector = np.array([cube[i] for i in range(ndim)])
            if self.pipeline.do_fast_slow:
                cube_vector = self.reorder_slow_fast(cube_vector)
            try:
                theta_vector = self.pipeline.denormalize_vector_from_prior(cube_vector) 
            except ValueError:
                # Polychord sometimes seems to propose outside the prior.
                # Just give terrible parameters when that happens.
                theta_vector = np.repeat(-np.inf, ndim)
            for i in range(ndim):
                theta[i] = theta_vector[i]

        self.wrapped_prior = prior_type(prior)

        def likelihood(theta, ndim, phi, nderived):
            theta_vector = np.array([theta[i] for i in range(ndim)])

            if np.any(~np.isfinite(theta_vector)):
                return -np.inf

            try:
                r = self.pipeline.run_results(theta_vector)

                # Extract the derived parameters
                for i in range(nderived-1):
                    phi[i] = r.extra[i]
                # And we also handle the prior here
                phi[nderived-1] = r.prior

            except KeyboardInterrupt:
                raise sys.exit(1)

            return r.like
        self.wrapped_likelihood = loglike_type(likelihood)

    def reorder_slow_fast(self, x):
        y = np.zeros_like(x)
        ns = self.pipeline.n_slow_params
        y[self.pipeline.slow_param_indices] = x[0:ns]
        y[self.pipeline.fast_param_indices] = x[ns:]
        return y



    def worker(self):
        self.sample()

    def execute(self):
        self.log_z = 0.0
        self.log_z_err = 0.0


        self.sample()

        self.output.final("log_z", self.log_z)
        self.output.final("log_z_error", self.log_z_err)

    def sample(self):

        if self.pipeline.do_fast_slow and (self.pipeline.n_fast_params > 0):
            logs.overview("Using two grades of parameter speed in polychord.")
            n_grade = 2
        elif self.pipeline.do_fast_slow:
            logs.warning("You asked for fast/slow, but there were no fast parameters, so "
                  "I have switched off Polychord's fast/slow mechanism to avoid a hang")
            n_grade = 1
        else:
            logs.overview("Using a single grade of parameter speeds in polychord.")
            n_grade = 1

        grade_dims = (ct.c_int*n_grade)()
        grade_frac = (ct.c_double*n_grade)()


        if n_grade > 1:
            if self.pipeline.n_slow_params == 0:
                raise ValueError("All your parameters have been classified as fast.  This will now work in polychord.  Change your fast/slow settings.")

            grade_dims[0] = self.pipeline.n_slow_params
            grade_dims[1] = self.pipeline.n_fast_params
            grade_frac[0] = 1 - self.fast_fraction
            grade_frac[1] = self.fast_fraction
            logs.overview("Telling Polychord to spend fraction {} if its time in the fast subspace (adjust with fast_fraction option)".format(self.fast_fraction))
        else:
            grade_dims[0] = self.pipeline.nvaried
            grade_frac[0] = 1.0


        n_nlives = 0
        loglikes = (ct.c_double*n_nlives)()
        nlives = (ct.c_int*n_nlives)()

        base_dir = self.base_dir.encode('ascii')
        polychord_outfile_root = self.polychord_outfile_root.encode('ascii')
        output_to_file = len(polychord_outfile_root) > 0

        if output_to_file:
            mkdir(self.base_dir)
            mkdir(os.path.join(self.base_dir, "clusters"))
            
        if self.num_repeats == 0:
            num_repeats = 3 * grade_dims[0]
            logs.overview("Polychord num_repeats = {}  (3 * n_slow_params [{}])".format(num_repeats, grade_dims[0]))
        else:
            num_repeats = self.num_repeats
            logs.overview("Polychord num_repeats = {}  (from parameter file)".format(num_repeats))

        self._run(
                self.wrapped_likelihood,      #loglike,
                self.wrapped_prior,           #prior,
                self.wrapped_output_logger,   #dumper,
                self.live_points,             #nlive
                num_repeats,                  #nrepeats
                self.nprior,                  #nprior
                True,                         #do_clustering
                self.feedback,                #feedback
                self.tolerance,               #precision_criterion
                self.log_zero,                #logzero
                self.max_iterations,          #max_ndead
                self.boost_posteriors,         #boost_posterior
                self.weighted_posteriors,     #posteriors
                self.equally_weighted_posteriors, #equals
                self.cluster_posteriors,      #cluster_posteriors
                True,                         #write_resume  - always
                False,                        #write_paramnames
                self.resume,                  #read_resume
                output_to_file,  #write_stats
                output_to_file,  #write_live
                output_to_file,  #write_dead
                output_to_file,  #write_prior
                self.compression_factor,      #compression_factor
                self.ndim,                    #nDims
                self.nderived,                #nDerived 
                base_dir,           #base_dir
                polychord_outfile_root,  #file_root
                n_grade,                      #nGrade
                grade_frac,                   #grade_frac
                grade_dims,                   #grade_dims
                n_nlives,                     #n_nlives
                loglikes,                     #loglikes
                nlives,                       #nlives
                self.random_seed,             #seed
                )

        self.converged = True

        if self.is_master() and self.boost_posteriors:
            self.make_boosted_posterior_file()

    def make_boosted_posterior_file(self):
        # two cases where there is nothing to boost
        if not self.polychord_outfile_root:
            return
        if not self.boost_posteriors:
            return

        old = self.output

        boosted_filename = os.path.join(self.base_dir, self.polychord_outfile_root + ".txt")

        # This is a horribly awkward hack
        if isinstance(self.output, TextColumnOutput):
            # make sure everything is written out already
            old.flush()
            
            new_output_filename = old.filename_base + "_boosted.txt"
            main_output_info = TextColumnOutput.load_from_options({"filename":old._filename, "delimiter":old.delimiter})

            _, _, metadata, comments, final_metadata = main_output_info
            metadata = metadata[0]
            comments = comments[0]
            final_metadata = final_metadata[0]

            # make a new output object
            new = TextColumnOutput.from_options({"filename":new_output_filename})

            # write the main bits of metadata from the output file
            new.metadata("sampler", "polychord")
            new.comment("NOTE: Boosted posterior file")
            self.write_header(new)
            for md in metadata:
                new.metadata(md, metadata[md])
            for c in comments:
                new.comment(c)
            
            nsample = 0
            # write the parameter rows from the old file
            with open(boosted_filename, "r")  as f:
                for line in f:
                    values = [float(x) for x in line.split()]
                    weight = values[0]
                    like = -0.5 * values[1]
                    params = values[2:]
                    prior = params[-1]
                    post = like + prior
                    new.parameters(params, like, post, weight)
                    nsample += 1

            metadata = old._final_metadata.copy()
            metadata["nsample"] = nsample
            # This hasn't been written yet because the original output file has not been closed.
            # so we have to use the saved one in the old object. The * is because a comment is also included
            for key, value in metadata.items():
                if isinstance(value, tuple):
                    new.final(key, *value)
                else:
                    new.final(key, value)
            new.final("log_z", self.log_z)
            new.final("log_z_error", self.log_z_err)

            # fake these first two, and add the completion marker.
            new.final("evaluations", 0)
            new.final("successes", 0)
            new.final("complete", 1)
            new.close()



    def output_params(self, ndead, nlive, npars, live, dead, logweights, log_z, log_z_err):
        # Polychord repeats output, but with changed weights, so reset to the start
        # of the chain to overwrite them.
        self.output.reset_to_chain_start()
        self.log_z = log_z
        self.log_z_err = log_z_err
        data = np.array([dead[i] for i in range(npars*ndead)]).reshape((ndead, npars))
        logw = np.array([logweights[i] for i in range(ndead)])
        for row, w in zip(data,logw):
            params = row[:self.ndim]
            extra_vals = row[self.ndim:self.ndim+self.nderived-1]
            prior = row[self.ndim+self.nderived-1]
            birth_like = row[self.ndim+self.nderived]
            like = row[self.ndim+self.nderived+1]
            importance = np.exp(w)
            post = like + prior
            self.output.parameters(params, extra_vals, prior, like, post, importance)

        # priors + likes
        posts = data[:, self.ndim+self.nderived-1] + data[:, self.ndim+self.nderived+1]
        self.distribution_hints.set_from_sample(data[:, :self.ndim], posts, log_weights=logw)

        self.output.final("nsample", ndead)
        self.output.flush()

    def is_converged(self):
        return self.converged

# legacy alias
PolyChordSampler = PolychordSampler
