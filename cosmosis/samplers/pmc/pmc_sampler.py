from .. import ParallelSampler
from ...runtime import logs
from . import pmc
import numpy as np

#needs to be global for MPI to behave
#I think this is still true, at least
global pipeline

def posterior(p):
    # PMC code needs posterior not log posterior
    r = pipeline.run_results(p)
    return r.post, (r.prior, r.extra)


class PmcSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("component", int), ("prior", float), ("post",float), ("log_weight",float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline
        self.n_iterations = self.read_ini("iterations", int, default=30)
        self.n_components = self.read_ini("components", int, default=5)
        self.grace_period = self.read_ini("grace_period", int, default=3)
        self.n_samples = self.read_ini("samples_per_iteration", int, 
            default=1000)
        self.final_samples = self.read_ini("final_samples", int, 
            default=5000)

        #Student's t mode
        student = self.read_ini("student", bool, default=False)
        if student:
            print("WARNING")
            print("Student's t mode probably not working yet")
            print("Unless you are testing you should probably set")
            print("student=F")
            nu = self.read_ini("nu", float, default=2.0)
        else:
            nu = None


        #start values from prior
        start = self.pipeline.start_vector()
        covmat = self.load_covariance_matrix()

        #Sampler object itself.
        self.sampler = pmc.PopulationMonteCarlo(posterior, self.n_components, 
            start, covmat, student=student, nu=nu, pool=self.pool)

        self.interrupted = False
        self.iterations = 0
        self.samples = 0

    def execute(self):
        if self.iterations==self.n_iterations:
            n = self.final_samples
            update=False
        else:
            n = self.n_samples
            update=True
        do_kill = self.iterations>=self.grace_period
        #Run the MCMC  sampler.
        try:
            results = self.sampler.sample(n,update, do_kill)
            #returns samples, like, extra, weights
        except KeyboardInterrupt:
            self.interrupted=True
            return
            
        self.iterations += 1
        self.samples += n

        for (vector, post, extra, component, weight) in zip(*results):
            prior, extra = extra
            self.distribution_hints.set_peak(vector, post)
            self.output.parameters(vector, extra, (component, prior, post, weight))

        logs.overview("Done %d iterations, %d samples" % (self.iterations, self.samples))


    def is_converged(self):
         # user has pressed Ctrl-C
        if self.interrupted:
            return True
        if self.iterations >= self.n_iterations+1:
            logs.overview("Full number of samples generated; sampling complete")
            self.output.final("nsample", self.final_samples)

            return True
        return False



    def load_covariance_matrix(self):
        covmat_filename = self.read_ini("covmat", str, "").strip()

        #No covariance specified - just use 1% of the range of the parameter.
        # This is the std. dev. here and is squared below.
        if covmat_filename == "":
            covmat = np.array([p.width()/100.0 for p in self.pipeline.varied_params])
        elif not os.path.exists(covmat_filename):
            raise ValueError(
            "Covariance matrix %s not found" % covmat_filename)
        else:
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

# legacy alias
PMCSampler = PmcSampler
