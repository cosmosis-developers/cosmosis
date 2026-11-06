from .. import ParallelSampler
from ...runtime import logs
import numpy as np
import sys

MINUIT_INI_SECTION = "minuit"

STRATEGIES = {"fast": 0, "medium": 1, "safe": 2, "0": 0, "1": 1, "2": 2}
ALGORITHMS = ("migrad", "fallback", "combined", "simplex")


class MinuitSampler(ParallelSampler):
    needs_output = True
    needs_parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        try:
            import iminuit
        except ImportError:
            raise ImportError(
                "The minuit sampler requires the iminuit python package. "
                "Install it with 'pip install iminuit' or 'conda install iminuit'."
            )
        self.iminuit = iminuit

        self.converged = False
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.save_dir = self.read_ini("save_dir", str, "")
        self.save_cov = self.read_ini("save_cov", str, "")
        self.output_ini = self.read_ini("output_ini", str, "")
        self.width_estimate = self.read_ini("width_estimate", float, 0.05)
        self.tolerance = self.read_ini("tolerance", float, 50.0)
        strategy = self.read_ini("strategy", str, "medium").lower()
        self.algorithm = self.read_ini("algorithm", str, "migrad").lower()

        self.strategy = STRATEGIES.get(strategy)
        if self.strategy is None:
            raise ValueError(
                f"Minuit 'strategy' parameter must be one of fast, medium, safe (default=medium) not '{strategy}'"
            )
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"Minuit 'algorithm' parameter must be one of migrad, fallback, simplex (default=migrad) not '{self.algorithm}'"
            )

        self.neval = 0
        self.minuit = None
        self.param_names = [str(p) for p in self.pipeline.varied_params]
        self.ndim = len(self.param_names)

    def _objective(self, vector):
        try:
            like, extra = self.pipeline.posterior(np.array(vector))
        except KeyboardInterrupt:
            sys.exit(1)
        return -like

    @staticmethod
    def _print_level():
        """Choose the iminuit print level from the cosmosis logging level"""
        level = logs.get_level()
        if level >= 30:    # quiet
            return 0
        elif level >= 20:  # standard
            return 1
        elif level >= 15:  # noisy
            return 2
        else:              # debug
            return 3

    def _make_minuit(self):
        start = self.start_estimate()
        lower = self.pipeline.min_vector()
        upper = self.pipeline.max_vector()
        m = self.iminuit.Minuit(self._objective, start, name=self.param_names)
        m.errordef = self.iminuit.Minuit.LIKELIHOOD  # 0.5 for a -log(posterior)
        m.limits = list(zip(lower, upper))
        m.errors = self.width_estimate * (upper - lower)
        m.strategy = self.strategy
        m.tol = self.tolerance
        m.print_level = self._print_level()
        return m

    def _minimize(self, m, ncall):
        """Run the chosen algorithm, using at most roughly ncall evaluations."""
        if self.algorithm == "simplex":
            m.simplex(ncall=ncall)
        else:
            m.migrad(ncall=ncall)
            if self.algorithm in ("fallback", "combined") and not m.valid:
                m.simplex(ncall=max(ncall - m.nfcn, 1))
                m.migrad(ncall=max(ncall - m.nfcn, 1))

    def execute(self):
        if self.minuit is None:
            self.minuit = self._make_minuit()
            logs.overview(f"Using Minuit {self.algorithm} algorithm, strategy {self.strategy}")
            logs.overview(f"Tolerance = {self.tolerance}")

        m = self.minuit
        self._minimize(m, self.maxiter - self.neval)
        self.neval = m.nfcn
        logs.overview(str(m.fmin))

        param_vector = np.array(m.values)
        if m.valid:
            logs.overview("SUCCESS: Minuit has converged!")
            results = self.pipeline.run_results(param_vector)
            self.save_results(param_vector, results)
            self.distribution_hints.set_peak(param_vector, results.post)
            cov = m.covariance
            if cov is not None:
                cov = np.array(cov)
                logs.overview("Posterior covariance matrix has been calculated and will be stored for any later sampling steps.")
                self.distribution_hints.set_cov(cov)
                if self.save_cov:
                    logs.overview(f"Saving posterior covariance matrix estimate to file: {self.save_cov}")
                    np.savetxt(self.save_cov, cov)
            else:
                logs.overview("Posterior covariance matrix could not be calculated.")
        elif self.neval >= self.maxiter:
            logs.error("MINUIT has failed to converge properly in the max number of iterations.  Sorry.")
            logs.error("Saving the best fitting parameters of the ones we tried, though beware: these are probably not the best-fit")
            results = self.pipeline.run_results(param_vector)
            self.save_results(param_vector, results)
        else:
            logs.overview("Minuit did not converge this run; trying again")
            logs.overview("until we run out of iterations.")

    def save_results(self, param_vector, results):
        section = None
        self.output.parameters(param_vector, results.extra, results.prior, results.post)

        if self.output_ini:
            self.pipeline.create_ini(param_vector, self.output_ini)

        for name, value in zip(self.param_names, param_vector):
            sec, name = name.split('--')
            if section != sec:
                logs.important("[%s]" % sec)
                section = sec
            logs.important("%s = %g" % (name, value))

        logs.important(f"Likelihood = {results.like}")

        if self.save_dir:
            logs.overview(f"Saving best-fit model cosmology to {self.save_dir}")
            results.block.save_to_directory(self.save_dir, clobber=True)

        # converged is used to tell the driver to stop, whether or not
        # the minimization really succeeded
        self.converged = True

    def worker(self):
        pass

    def is_converged(self):
        return self.converged
