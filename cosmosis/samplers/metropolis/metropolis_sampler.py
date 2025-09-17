from .. import ParallelSampler
from ...runtime import logs
import numpy as np
from . import metropolis
from cosmosis.runtime.analytics import Analytics
import os

#We need a global pipeline
#object for MPI to work properly
pipeline=None


METROPOLIS_INI_SECTION = "metropolis"

def posterior(p):
    results = pipeline.run_results(p)
    # We never use the block, and it was getting serialized
    # unnecessarily when writing checkpoints
    if results is not None:
        results.block = None
    return results


class MetropolisSampler(ParallelSampler):
    parallel_output = True
    sampler_outputs = [("prior", float),("post", float)]
    understands_fast_subspaces = True
    supports_resume = True

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.samples = self.read_ini("samples", int, default=20000)
        random_start = self.read_ini("random_start", bool, default=False)
        covmat_sample_start = self.read_ini("covmat_sample_start", bool, default=False)
        use_cobaya = self.read_ini("cobaya", bool, default=False)
        self.Rconverge = self.read_ini("Rconverge", float, -1.0)
        self.drag = self.read_ini("drag", int, 0)
        self.oversampling = self.read_ini("oversampling", int, 5)
        tuning_frequency = self.read_ini("tuning_frequency", int, -1)
        tuning_grace = self.read_ini("tuning_grace", int, 5000)
        self.tuning_end = self.read_ini("tuning_end", int, 100000)
        self.save_during_tuning = self.read_ini("save_during_tuning", bool, False)
        self.n = self.read_ini("nsteps", int, default=100)
        self.exponential_probability = self.read_ini("exponential_probability", float, default=0.333)
        self.split = None #work out later
        if self.Rconverge==-1.0:
            self.Rconverge=None
        self.interrupted = False
        self.num_samples = 0
        self.ndim = len(self.pipeline.varied_params)
        self.num_samples_post_tuning = 0
        self.last_accept_count = 0
        #Any other options go here

        # if we are not tunning then there is no tuning phase
        if tuning_frequency == -1:
            self.tuning_end = 0

        if (self.drag > 0) and not self.pipeline.do_fast_slow:
            logs.warning("You asked for dragging, but the pipeline does not have fast/slow enabled"
                  ", so no draggng will be done."
                )

        if (self.pipeline.do_fast_slow) and not (self.pipeline.first_fast_module):
            raise ValueError("To use fast/slow splitting with metropolis please "
                             "manually define first_fast_module in the pipeline "
                             "section.")


        #Covariance matrix
        covmat = self.load_covariance_matrix()

        #start values from prior
        start = self.define_parameters(random_start, covmat_sample_start, covmat)
        logs.overview("MCMC starting point:")
        for param, x in zip(self.pipeline.varied_params, start):
            logs.overview(f"    {param}  {x}")


        if use_cobaya:
            logs.overview("Using the Cobaya proposal")

        logs.overview(f"Will tune every {tuning_frequency} samples, from samples "
              f"{tuning_grace} to {self.tuning_end}.")

        self.sampler = metropolis.MCMC(start, posterior, covmat,
            tuning_frequency=tuning_frequency, # Will be multiplied by the oversampling
            tuning_grace=tuning_grace,         # within the sampler if needed
            tuning_end=self.tuning_end,
            exponential_probability=self.exponential_probability,
            use_cobaya=use_cobaya,
            n_drag = self.drag,
        )
        self.analytics = Analytics(self.pipeline.varied_params, self.pool)
        self.fast_slow_done = False

    def worker(self):
        while not self.is_converged():
            self.execute()
            if self.output:
                self.output.flush()


    def resume(self):
        resume_info = self.read_resume_info()
        if resume_info is None:
            return

        self.sampler, self.num_samples, self.num_samples_post_tuning = resume_info

        # Fast slow is already configured on the sampler.
        self.fast_slow_done = True

        # If we started main sampling (as opposed to tuning phase)
        # then we will have some existing chain, but this is not always the case
        try:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)[:, :self.ndim]
            self.analytics.add_traces(data)
        except IndexError:
            data = None

        
        if self.num_samples >= self.samples:
            logs.important("You told me to resume the chain - it has already completed (with {} samples), so sampling will end.".format(len(data)))
            logs.important("Increase the 'samples' parameter to keep going.")
        elif self.is_converged():
            logs.overview("The resumed chain was already converged.  You can change the converged testing parameters to extend it.")
        elif data is None:
            logs.overview("Continuing metropolis from existing chain - you were in the tuning phase, which will continue")
        else:
            logs.overview("Continuing metropolis from existing chain - have {} samples already".format(len(data)))






    def execute(self):
        #Run the MCMC  sampler.
        if self.pipeline.do_fast_slow and not self.fast_slow_done:
            self.fast_slow_done = True
            self.sampler.set_fast_slow(
                self.pipeline.fast_param_indices,
                self.pipeline.slow_param_indices, 
                self.oversampling
            )

        try:
            samples = self.sampler.sample(self.n)
        except KeyboardInterrupt:
            self.interrupted=True
            return
        self.num_samples += self.n
        self.num_samples_post_tuning = self.num_samples - self.tuning_end


        overall_rate = (self.sampler.accepted * 1.0) / self.sampler.iterations
        recent_accepted = self.sampler.accepted - self.last_accept_count
        recent_rate = recent_accepted / self.n
        logs.overview("Overall accepted {} / {} samples ({:.1%})" .format(
            self.sampler.accepted, self.sampler.iterations, overall_rate))
        logs.overview("Last {0} accepted {1} / {0} samples ({2:.1%})\n" .format(
            self.n, recent_accepted, recent_rate))
        self.last_accept_count = self.sampler.accepted

        # Regardless of save settings we never use tuning samples
        # for analytics
        if self.num_samples_post_tuning > 0:
            traces = np.array([r.vector for r in samples[-self.num_samples_post_tuning:]])
            self.analytics.add_traces(traces)

        vectors = np.array([r.vector for r in samples])
        posts = np.array([r.post for r in samples])
        self.distribution_hints.set_from_sample(vectors, posts)
        if (self.num_samples_post_tuning > 0) or self.save_during_tuning:
            for i, result in enumerate(samples):
                self.output.parameters(result.vector, result.extra, result.prior, result.post)

        if self.num_samples_post_tuning <= 0:
            logs.overview("Tuning ends at {} samples\n".format(self.tuning_end))

        self.write_resume_info([self.sampler, self.num_samples, self.num_samples_post_tuning])

    def is_converged(self):
         # user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.num_samples >= self.samples:
            rank = self.pool.rank if self.pool is not None else 0
            logs.overview(f"Rank {rank}: Full number of samples generated; sampling complete")
            return True
        elif (self.num_samples > 0 and
              self.pool is not None and
              self.Rconverge is not None and
              self.num_samples_post_tuning > 0):
            R = self.analytics.gelman_rubin()
            R1 = abs(R - 1)
            return np.all(R1 <= self.Rconverge)
        else:
            return False



    def load_covariance_matrix(self):
        covmat_filename = self.read_ini("covmat", str, "").strip()
        if covmat_filename == "" and self.distribution_hints.has_cov():
                covmat =  self.distribution_hints.get_cov()
                logs.overview("Using covariance from previous sampler")
        elif covmat_filename == "":
            logs.overview("Using default covariance 1% of param widths")
            covmat = np.array([p.width()/100.0 for p in self.pipeline.varied_params])
        elif not os.path.exists(covmat_filename):
            raise ValueError(
            "Covariance matrix %s not found" % covmat_filename)
        else:
            logs.overview("Loading covariance from {}".format(covmat_filename))
            covmat = np.loadtxt(covmat_filename)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat ** 2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                    "but there are %d varied parameters." %
                    (covmat.shape[0], covmat.shape[1], nparams))
        return covmat



    def define_parameters(self, random_start, covmat_sample_start, covmat):
        if covmat_sample_start:
            p = self.pipeline.start_vector()
            chol = np.linalg.cholesky(covmat)

            # Try 100 times to get points within the prior
            for i in range(200):
                r = np.random.normal(size=covmat.shape[0])
                start = p + chol @ r
                prior = self.pipeline.prior(start, total_only=True)
                if np.isfinite(prior):
                    break
            else:
                raise ValueError("You set covmat_sample_start=T so I tried "
                                 "100 times to get a sample inside the prior, "
                                 "but it was always outside."
                    )
            return start
        elif random_start:
            return self.pipeline.randomized_start()
        else:
            return self.pipeline.start_vector()
