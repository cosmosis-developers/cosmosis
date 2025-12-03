from ...runtime import logs
from .. import ParallelSampler, sample_ellipsoid, sample_ball
import numpy as np
import sys


def log_probability_function(p):
    r = emcee_pipeline.run_results(p)
    return r.post, (r.prior, r.extra)


class EmceeSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global emcee_pipeline
        emcee_pipeline = self.pipeline

        if self.is_master():
            import emcee
            self.emcee = emcee

            self.emcee_version = int(self.emcee.__version__[0])


            # Parameters of the emcee sampler
            self.nwalkers = self.read_ini("walkers", int, 2)
            self.samples = self.read_ini("samples", int, 1000)
            self.nsteps = self.read_ini("nsteps", int, 100)
            self.a = self.read_ini("a", float, 2.0)

            assert self.nsteps>0, "You specified nsteps<=0 in the ini file - please set a positive integer"
            assert self.samples>0, "You specified samples<=0 in the ini file - please set a positive integer"

            random_start = self.read_ini("random_start", bool, False)
            start_file = self.read_ini("start_points", str, "")
            covmat_file = self.read_ini("covmat", str, "")
            self.ndim = len(self.pipeline.varied_params)

            #Starting positions and values for the chain
            self.num_samples = 0
            self.prob0 = None
            self.blob0 = None

            if start_file:
                self.p0 = self.load_start(start_file)
                logs.overview(f"Loaded starting position from {start_file}")
            elif self.distribution_hints.has_cov():
                center = self.start_estimate()
                cov = self.distribution_hints.get_cov()
                self.p0 = sample_ellipsoid(center, cov, size=self.nwalkers)
                logs.overview("Generating starting positions from covmat from earlier in pipeline")
            elif covmat_file:
                center = self.start_estimate()
                cov = self.load_covmat(covmat_file)
                logs.overview(f"Generating starting position from covmat in  {covmat_file}")
                iterations_limit = 100000
                n=0
                p0 = []
                for i in range(iterations_limit):
                    p = sample_ellipsoid(center, cov)[0]
                    if np.isfinite(self.pipeline.prior(p)):
                        p0.append(p)
                    if len(p0)==self.nwalkers:
                        break
                else:
                    raise ValueError("The covmat you used could not generate points inside the prior")
                self.p0 = np.array(p0)
            elif random_start:
                self.p0 = [self.pipeline.randomized_start()
                           for i in range(self.nwalkers)]
                logs.overview("Generating random starting positions from within prior")
            else:
                center_norm = self.pipeline.normalize_vector(self.start_estimate())
                sigma_norm=np.repeat(1e-3, center_norm.size)
                p0_norm = sample_ball(center_norm, sigma_norm, size=self.nwalkers)
                p0_norm[p0_norm<=0] = 0.001
                p0_norm[p0_norm>=1] = 0.999
                self.p0 = [self.pipeline.denormalize_vector(p0_norm_i) for p0_norm_i in p0_norm]
                logs.overview("Generating starting positions in small ball around starting point")

            if self.emcee_version < 3:
                kw = {"a": self.a}
            else:
                kw = {"moves": [(emcee.moves.StretchMove(a=self.a), 1.0)]}

            #Finally we can create the sampler
            self.ensemble = self.emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                                       log_probability_function,
                                                       pool=self.pool, **kw)

    def resume(self):
        if self.output.resumed:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)[:, :self.ndim]
            num_samples = len(data) // self.nwalkers
            self.p0 = data[-self.nwalkers:]
            self.num_samples += num_samples
            if self.num_samples >= self.samples:
                logs.error("You told me to resume the chain - it has already completed (with {} samples), so sampling will end.".format(len(data)))
                logs.error("Increase the 'samples' parameter to keep going.")
            else:
                logs.overview("Continuing emcee from existing chain - have {} samples already".format(len(data)))

    def load_start(self, filename):
        #Load the data and cut to the bits we need.
        #This means you can either just use a test file with
        #starting points, or an emcee output file.
        data = np.genfromtxt(filename, invalid_raise=False)[-self.nwalkers:, :self.ndim]
        if data.shape != (self.nwalkers, self.ndim):
            raise RuntimeError("There are not enough lines or columns "
                               "in the starting point file %s" % filename)
        return list(data)


    def load_covmat(self, covmat_file):
        covmat = np.loadtxt(covmat_file)

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


    def output_samples(self, pos, prob, extra_info):
        self.distribution_hints.set_from_sample(pos, prob)
        for params, post, extra in zip(pos,prob,extra_info):
            prior, extra = extra      
            self.output.parameters(params, extra, prior, post)

    def execute(self):
        #Run the emcee sampler.
        if self.num_samples == 0:
            logs.overview("Begun sampling")
        outputs = []
        if self.emcee_version < 3:
            kwargs = dict(lnprob0=self.prob0, blobs0=self.blob0, 
                          iterations=self.nsteps, storechain=False)
        else:
            # In emcee3 we have to enable storing the chain because
            # we want the acceptance fraction.  Also the name of one
            # of the parameters has changed.
            kwargs = dict(log_prob0=self.prob0, blobs0=self.blob0, 
                          iterations=self.nsteps, store=True)

        for (pos, prob, rstate, extra_info) in self.ensemble.sample(self.p0, **kwargs):
            outputs.append((pos.copy(), prob.copy(), np.copy(extra_info)))
    
        for (pos, prob, extra_info) in outputs:
            self.output_samples(pos, prob, extra_info)

        #Set the starting positions for the next chunk of samples
        #to the last ones for this chunk
        self.p0 = pos
        self.prob0 = prob
        self.blob0 = extra_info
        self.num_samples += self.nsteps
        acceptance_fraction = self.ensemble.acceptance_fraction.mean()
        logs.overview("Done {} iterations of emcee. Acceptance fraction {:.3f}".format(
            self.num_samples, acceptance_fraction))
        sys.stdout.flush()
        self.output.final("mean_acceptance_fraction", acceptance_fraction)

    def is_converged(self):
        return self.num_samples >= self.samples
