from .. import ParallelSampler, sample_ellipsoid, sample_ball
from ...runtime import logs
import numpy as np
import sys


def log_probability_function(p):
    r = zeus_pipeline.run_results(p)
    return r.post, (r.prior, r.extra)


class ZeusSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global zeus_pipeline
        zeus_pipeline = self.pipeline

        if self.is_master():
            import emcee
            import zeus
            self.zeus = zeus

            if not hasattr(zeus, "EnsembleSampler"):
                raise ImportError("There are two python packages called Zeus, and you"
                    " have the wrong one.  Uninstall zeus and install zeus-mcmc.")

            # Parameters of the zeus sampler
            self.nwalkers = self.read_ini("walkers", int)
            self.samples = self.read_ini("samples", int)
            self.nsteps = self.read_ini("nsteps", int, 50)

            assert self.nsteps>0, "You specified nsteps<=0 in the ini file - please set a positive integer"
            assert self.samples>0, "You specified samples<=0 in the ini file - please set a positive integer"

            # Parameters controlling initialization
            random_start = self.read_ini("random_start", bool, False)
            start_file = self.read_ini("start_points", str, "")
            covmat_file = self.read_ini("covmat", str, "")
            self.ndim = len(self.pipeline.varied_params)

            # Zeus move options
            move_names = {
                "differential": zeus.moves.DifferentialMove,
                "gaussian": zeus.moves.GaussianMove,
                "global": zeus.moves.GlobalMove,
                "random": zeus.moves.RandomMove,
                "kde": zeus.moves.KDEMove,
            }

            moves = self.read_ini("moves", str, "differential").strip()

            # parse the moves into the list self.moves.
            # Expected format is, e.g. "differential:1.0  global:0.5"
            # but the weights default to 1
            self.moves = []
            moves = moves.split()
            for move in moves:
                # split if weighting present
                if ':' in move:
                    move_name, move_weight = move.split(':')
                else:
                    move_name = move
                    move_weight = 1.0
                move_cls = move_names[move_name]
                self.moves.append((move_cls(), move_weight))

            logs.overview("Running zeus with moves:")
            logs.overview(str(self.moves))

            # Other zeus options
            self.tune = self.read_ini("tune", bool, True)
            self.tolerance = self.read_ini("tolerance", float, 0.05)
            self.maxsteps = self.read_ini("maxsteps", int, 10000)
            self.patience = self.read_ini("patience", int, 5)
            self.maxiter = self.read_ini("maxiter", int, 10000)
            self.verbose = self.read_ini("verbose", bool, False)

            #Starting positions and values for the chain
            self.num_samples = 0
            self.prob0 = None
            self.blob0 = None

            # Generate starting point
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

            #Finally we can create the sampler
            self.started = False
            self.sampler = self.zeus.EnsembleSampler(self.nwalkers, self.ndim,
                                                     log_probability_function,
                                                     tune=self.tune,
                                                     tolerance=self.tolerance,
                                                     maxsteps=self.maxsteps,
                                                     patience=self.patience,
                                                     maxiter=self.maxiter,
                                                     verbose=self.verbose,
                                                     pool=self.pool)

    def resume(self):
        resume_info = self.read_resume_info()

        if resume_info is None:
            return

        moves, weights, tune, mu, mus = resume_info

        changed = False
        old_classes = [m.__class__.__name__ for m in moves]
        if len(self.sampler._moves) != len(moves):
            changed = True
        else:
            for m, c in zip(self.sampler._moves, old_classes):
                if m.__class__.__name__ != c:
                    changed = True

        if changed:
            raise ValueError("You have changed the number or type of "
                "the Move objects since re-running this chain. "
                "You can't directly resume a chain with different moves, "
                "so either: (a) change them back (to {}), "
                "(b) delete the sample_status output file, "
                "(c) switch off runtime.resume.  If you just want to restart the "
                "walkers from old positions, set zeus.start_points")

        logs.overview("Loaded 'Move' objects from resume file:")
        for m, w in zip(moves, weights):
            logs.overview(    "{} mu0={}  weight={}  tuning={}".format(m.__class__.__name__, m.mu0, w, m.tune))
        self.sampler._moves = moves
        self.sampler._weights = weights
        self.sampler.tune = tune
        self.sampler.mu = mu
        self.sampler.mus = mus

        if tune:
            logs.overview("Resumed sampler is still tuning")
        else:
            logs.overview("Resumed sampler has finished tuning")

        # if we have some chain, read it here and use it to get
        # the p0 value and number of samples
        if self.output.resumed:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)[:, :self.ndim]
            num_samples = len(data) // self.nwalkers
            self.p0 = data[-self.nwalkers:]
            self.num_samples += num_samples
            if self.num_samples >= self.samples:
                logs.error("You told me to resume the chain - it has already completed (with {} samples), so sampling will end.".format(len(data)))
                logs.error("Increase the 'samples' parameter to keep going.")
            else:
                logs.overview("Continuing zeus from existing chain - have {} samples already".format(len(data)))

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

        # Convert a single number to a 1x1 array,
        # although it is ambiguous
        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))

        # Convert an array of std devs to a covmat
        elif covmat.ndim == 1:
            covmat = np.diag(covmat ** 2)

        # Check the size is right.
        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                    "but there are %d varied parameters." %
                    (covmat.shape[0], covmat.shape[1], nparams))
        return covmat


    def execute(self):
        #Run the zeus sampler.
        if self.started:
            start = self.sampler.chain.shape[0]
        else:
            logs.error("Begun sampling")
            self.started = True
            start = 0

        # Main execution
        self.sampler.run(self.p0, self.nsteps)
        # Update patience (relevant when self.nsteps<self.patience)
        self.sampler.patience-= self.nsteps
        # record ending point of this iteration
        end = self.sampler.chain.shape[0]

        post = self.sampler.get_log_prob()
        chain = self.sampler.get_chain()
        blobs = self.sampler.get_blobs()

        chain_flat = self.sampler.get_chain(flat=True)
        post_flat = self.sampler.get_log_prob(flat=True)
        self.distribution_hints.set_from_sample(chain_flat, post_flat)

        # Output results per sampler per walker
        for i in range(start, end):
            for j in range(self.nwalkers):
                prior, extra = blobs[i, j]
                self.output.parameters(chain[i, j], extra, prior, post[i, j])

        #Set the starting positions for the next chunk of samples
        #to the last ones for this chunk
        # get_last_sample changed from an attribute to a method recently
        try:
            self.p0 = self.sampler.get_last_sample()
        except TypeError:
            self.p0 = self.sampler.get_last_sample
        self.num_samples += self.nsteps

        taus = self.zeus.AutoCorrTime(chain)
        logs.overview(f"Have {self.num_samples} samples from zeus. Current auto-correlation estimates are:")
        for par, tau in zip(self.pipeline.varied_params, taus):
            logs.overview("   {}:  {:.2f}".format(par, tau))
        sys.stdout.flush()

        if self.sampler.tune:
            logs.overview("Sampler is (still) tuning")
        else:
            logs.overview("Sampler is no longer tuning")

        self.write_resume_info([self.sampler._moves, self.sampler._weights,
                                self.sampler.tune, self.sampler.mu, self.sampler.mus])

    def is_converged(self):
        return self.num_samples >= self.samples

