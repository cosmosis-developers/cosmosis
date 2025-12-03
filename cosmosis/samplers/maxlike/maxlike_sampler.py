from .. import ParallelSampler
from ...runtime import logs
from ...runtime.pipeline import PipelineResults
import numpy as np
import scipy.optimize
import warnings

# The likelihood function we wish to optimize.
# Not that we are minimizing the negative of the likelihood/posterior
def likefn(p_in):
    global sampler
    # Check the normalization
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
        return np.inf
    p = sampler.pipeline.denormalize_vector(p_in)
    r = sampler.pipeline.run_results(p)
    if sampler.max_posterior:
        return -r.post
    else:
        return -r.like

def run_optimizer(start_vector):
    global sampler
    return sampler.run_optimizer(start_vector)

class MaxlikeSampler(ParallelSampler):
    parallel_output = False
    supports_resume = False # TODO: support resuming

    sampler_outputs = [("prior", float), ("like", float), ("post", float)]

    def config(self):
        global sampler
        sampler = self
        self.tolerance = self.read_ini("tolerance", float, 1e-3)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.output_ini = self.read_ini("output_ini", str, "")
        self.output_cov = self.read_ini("output_covmat", str, "")
        self.output_block = self.read_ini("output_block", str, "")
        self.method = self.read_ini("method",str,"Nelder-Mead")
        self.max_posterior = self.read_ini("max_posterior", bool, False)
        self.repeats = self.read_ini("repeats", int, 1)
        self.reiterations = self.read_ini("reiterations", int, 1)
        if self.repeats == 1:
            nsteps = 1
        elif self.pool is not None:
            nsteps = self.pool.size
        else:
            nsteps = 1
        self.nsteps = self.read_ini("nsteps", int, nsteps)
        start_method = self.read_ini("start_method", str, "")

        if self.repeats > 1:
            start_method_is_random = start_method in ["prior", "cov", "chain", "chain-sample"]
            have_previous_peak = self.distribution_hints.has_peak()
            have_previous_cov = self.distribution_hints.has_cov()

            if have_previous_peak and not have_previous_cov:
                raise ValueError("You have set repeats>1 in maxlike, and have chained multiple samplers together. "
                                 "But the previous sampler(s) did not save a covariance matrix. "
                                 "So we have no way to generate multiple starting points. "
                                 )
            elif not start_method_is_random:
                raise ValueError(f"You set the repeats parameter in the max-like sampler to {self.repeats}"
                                 " but to you this you must also set a start method to choose a new starting point"
                                 " each repeat.  You can set start_method to 'prior' to sample from the prior,"
                                 " or set it to 'cov' or 'chain' and also set start_input to a file to load one"
                                 " either a covariance matrix or a chain of samples."
                                 )
        if self.is_master():
            if self.max_posterior:
                logs.overview("------------------------------------------------")
                logs.overview("NOTE: Running optimizer in **max-posterior** mode:")
                logs.overview("NOTE: Will maximize the combined likelihood and prior")
                logs.overview("------------------------------------------------")
            else:
                logs.overview("--------------------------------------------------")
                logs.overview("NOTE: Running optimizer in **max-like** mode:")
                logs.overview("NOTE: not including the prior, just the likelihood.")
                logs.overview("NOTE: Set the parameter max_posterior=T to change this.")
                logs.overview("NOTE: This won't matter unless you set some non-flat")
                logs.overview("NOTE: priors in a separate priors file.")
                logs.overview("--------------------------------------------------")

        self.best_fit_results = []
        self.start_points = None

    def save_final_outputs(self, best_fit_results, final=False):

        # This can happen if the user ctrl'c's the run before any results are saved
        if not best_fit_results:
            return

        # Sort the repeated best-fit estimates by increasing posterior or likelihood
        if self.max_posterior:
            best_fit_results.sort(key=lambda x: x.post)
        else:
            best_fit_results.sort(key=lambda x: x.like)

        # Save all the results to the main chain output file.
        # We will overwrite previous sets of results in the file here
        self.output.reset_to_chain_start()
        for results in best_fit_results:
            self.output.parameters(results.vector, results.extra, results.prior, results.like, results.post)

        # Get the overall best-fit results
        results = best_fit_results[-1]

        # Also if requested, approximate the covariance matrix with the
        # inverse of the Hessian matrix.
        # For a gaussian posterior this is exact.
        if results.covmat is None:
            if self.output_cov:
               warnings.warn("Sorry - the optimization method you chose does not return a covariance (or Hessian) matrix")
        else:
            if self.output_cov:
                np.savetxt(self.output_cov, results.covmat)

        if self.output_ini:
            self.pipeline.create_ini(results.vector, self.output_ini)
        
        if self.output_block:
            results.block.save_to_directory(self.output_block)


        # We only want to update the distribution hints at the very end
        if final:
            # These values are used by subsequent samplers, if you chain
            # some of them together
            self.distribution_hints.set_peak(results.vector, results.post)
            if results.covmat is not None:
                self.distribution_hints.set_cov(results.covmat)


    def run_optimizer(self, inputs):
        rank = self.pool.rank if self.pool is not None else 0
        start_vector, repeat_index = inputs
        start_vector_denorm = self.pipeline.denormalize_vector(start_vector)
        logs.overview(f"[Rank {rank}] Starting from point: {start_vector_denorm}")
        bounds = [(0.0, 1.0) for p in self.pipeline.varied_params]

        if self.method.lower() == "bobyqa":
            # use the specific pybobyqa minimizer
            import pybobyqa

            starting_tolerance = 0.1

            # this works with bounds in the form of a tuple of two arrays
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            kw = {
                "seek_global_minimum": True,
                "bounds": (lower,upper),
                "print_progress": logs.is_enabled_for(logs.NOISY),
                "rhobeg": starting_tolerance,
                "rhoend": self.tolerance,
            }

            if self.tolerance >= starting_tolerance:
                raise ValueError(f"The tolerance for bobyqa must be less than {starting_tolerance}.  You chose {self.tolerance}.")

            for i in range(self.reiterations):
                optimizer_result = pybobyqa.solve(likefn, start_vector, **kw)
                start_vector = optimizer_result.x

                if start_vector is None:
                    # The default PipelineResults object is empty and has likelihod -inf
                    vector = np.repeat(np.nan, self.pipeline.nvaried)
                    opt = PipelineResults(vector, self.pipeline.number_extra)
                    logs.error(f"[Rank {rank}] BOBYQA Optimization failed with error number {optimizer_result.flag}, message {optimizer_result.msg}")
                    opt.covmat = None
                    return opt

            opt_norm = optimizer_result.x


            # bobyqa calls it .hessian but scipy calls it .hess, so copy it here
            # if available
            if optimizer_result.hessian is not None:
                optimizer_result.hess = optimizer_result.hessian
        else:
            # Use scipy mainimizer instead
            for i in range(self.reiterations):
                optimizer_result = scipy.optimize.minimize(likefn, start_vector, method=self.method,
                    jac=False, tol=self.tolerance,
                    options={'maxiter':self.maxiter, 'disp':True})
                start_vector = optimizer_result.x

            opt_norm = optimizer_result.x

        opt = self.pipeline.denormalize_vector(opt_norm)
        
        # Generate a results object containing the best-fit parameters,
        # their likelihood, posterior, prior, and derived parameters
        results = self.pipeline.run_results(opt)

        # Log some info to the screen
        par_text = '   '.join(str(x) for x in opt)
        logs.overview(f"\n\n[Rank {rank}] Optimization run {repeat_index+1} of {self.repeats}")
        if self.max_posterior:
            logs.overview(f"[Rank {rank}] Best fit (by posterior): {par_text}\n%s")
        else:
            logs.overview(f"[Rank {rank}] Best fit (by likelihood):\n{par_text}")
        logs.overview(f"[Rank {rank}] Posterior: {results.post}")
        logs.overview(f"[Rank {rank}] Likelihood: {results.like}\n")

        # Also attach any covariance matrix saved by the sampler to the results
        # object, in an ad-hoc way
        if hasattr(optimizer_result, 'hess_inv'):
            if self.method == "L-BFGS-B":
                results.covmat = self.pipeline.denormalize_matrix(optimizer_result.hess_inv.todense())
            else:
                results.covmat = self.pipeline.denormalize_matrix(optimizer_result.hess_inv)
        elif hasattr(optimizer_result, 'hess'):
            results.covmat = self.pipeline.denormalize_matrix(np.linalg.inv(optimizer_result.hess))
        else:
            results.covmat = None

        return results
    
    def choose_start(self, n):
        if n == 1:
            # we just want a single starting point. 
            # So we can use the basic one from the values file or
            # previous sampler; we don't have to worry about making
            # it randomized.

            # It's also possible that this is just the last of our repeats
            # to be done, but in that case it's also okay to just use the
            # basic starting point, since we won't already have used it
            start_vector_denorm = self.start_estimate()
            start_vector = self.pipeline.normalize_vector(start_vector_denorm)

            # If that is invalid we will raise an error
            if not np.isfinite(likefn(start_vector)):
                raise RuntimeError("Invalid starting point for maxlike")
            
            start_vectors = np.array([start_vector]) # 1 x n array

        else:
            start_vectors = np.zeros((n, self.pipeline.nvaried))
            for i in range(n):
                # If we are taking a random starting point then there is a chance it will randomly
                # be invalid. So we should try a 20 times to get a valid one.
                for j in range(20):
                    start_vector_denorm = self.start_estimate(prefer_random=True, quiet=(i>0 or j>0))
                    start_vector = self.pipeline.normalize_vector(start_vector_denorm)
                    if np.isfinite(likefn(start_vector)):
                        break
                else:
                    raise RuntimeError("Could not find a valid random starting point for maxlike in 20 tries")
                start_vectors[i] = start_vector

        return start_vectors

    def execute(self):
        # Choose a starting point. If we are repeating our optimization we will need
        # multiple starting points. Otherwise we use a fixed one.
        # If we are re-starting then this will actually generate a different set of
        # starting points compared to the first time, but they are random so it doesn't
        # matter
        if self.start_points is None:
            self.start_points = self.choose_start(self.repeats)

        # Figure out how many steps we need to do now.
        ndone = len(self.best_fit_results)
        if ndone + self.nsteps > self.repeats:
            n = self.repeats - ndone
        else:
            n = self.nsteps

        # Starting points to use for this chunk of steps
        starting_points = self.start_points[ndone:ndone + n]

        if self.pool is None:
            # serial mode. Just do everything on this process.
            # this allows us to also do a nice thing where if we
            # get part way through the results and then the user
            # ctrl-c's then we can still output partial results
            collected_results = []
            for i, start_vector in enumerate(starting_points):
                try:
                    results = self.run_optimizer((start_vector, ndone + i))
                    collected_results.append(results)
                except KeyboardInterrupt:
                    # If we get a ctrl-c we should save the current best fit
                    # and then exit.
                    # Otherwise report what we are doing. It should be basically instant
                    # so it shouldn't annoy anyone
                    logs.overview("Keyboard interrupt - saving current best fit, if any finished")
                    self.save_final_outputs(collected_results, final=True)
                    raise
        else:
            inputs = [(start_vector, ndone + i) for i, start_vector in enumerate(starting_points)]
            collected_results = self.pool.map(run_optimizer, inputs)

        # Add this to our list of best-fits
        self.best_fit_results.extend(collected_results)

        # we re-save the final outputs each time, rewinding the file
        # to overwrite the previous ones, so we can re-sort each time.
        self.save_final_outputs(self.best_fit_results, final=self.is_converged()    )

    def is_converged(self):
        return len(self.best_fit_results) >= self.repeats
