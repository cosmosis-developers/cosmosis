from .. import ParallelSampler
import numpy as np



neval = 0
nfail = 0
def minus_log_posterior(p_in):
    global neval, nfail
    p_in, index = p_in
    if maxlike_sampler.pool:
        rank = maxlike_sampler.pool.rank
    else:
        rank = 0
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1.0)):
        nfail += 1
        pstr = '   '.join(str(x) for x in p_in)
        return np.inf
    neval +=1 
    p = maxlike_sampler.pipeline.denormalize_vector(p_in)
    post, extra = maxlike_sampler.pipeline.posterior(p)
    pstr = '   '.join(str(x) for x in p)
    return -post



def posterior_and_gradient(p_in):
    pstr = '   '.join(str(x) for x in p_in)
    print("Calculating gradient about (normalized) point {}".format(pstr))
    points = [(p_in,0)]
    n = len(p_in)
    for i in range(n):
        p = p_in.copy()
        p[i] += maxlike_sampler.epsilon
        points.append((p, i+1))

    if maxlike_sampler.pool:
        results = maxlike_sampler.pool.map(minus_log_posterior, points)
    else:
        results = list(map(minus_log_posterior, points))

    post=results[0]
    grad=np.array([(results[i+1]-post)/maxlike_sampler.epsilon  for i in range(n)])
    print("Norm of gradient: {:.2e}".format(np.linalg.norm(grad)))
    return post, grad




class PmaxlikeSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("like", float), ("post", float)]

    def config(self):
        self.method = self.read_ini("method", str, "CG")
        self.tolerance = self.read_ini("tolerance", float, 1e-3)
        self.maxiter = self.read_ini("maxiter", int, 1000)
        self.output_ini = self.read_ini("output_ini", str, "")
        self.output_cov = self.read_ini("output_covmat", str, "")
        self.epsilon = self.read_ini("gradient_epsilon",float,1e-9)
        self.gradient_tolerance = self.read_ini("gradient_tolerance",float,1e-5)
        self.converged = False
        global maxlike_sampler
        maxlike_sampler = self

    def execute(self):
        import scipy.optimize


        #starting position in the normalized space
        start_vector = self.pipeline.normalize_vector(self.pipeline.start_vector())
        bounds = [(0.0, 1.0) for p in self.pipeline.varied_params]


        result = scipy.optimize.minimize(posterior_and_gradient, start_vector, method=self.method,
          jac=True, tol=self.tolerance,  #bounds=bounds, 
          options={'maxiter':self.maxiter, 'disp':True, 'eps':self.epsilon, 'gtol':self.gradient_tolerance})

        opt_norm = result.x
        opt = self.pipeline.denormalize_vector(opt_norm)
        
        results = self.pipeline.run_results(opt)

        #Some output - first log the parameters to the screen.
        print("Best fit:\n%s"%'   '.join(str(x) for x in opt))
        print("Best likelihood: %f\n", results.like)

        #Next save them to the proper table file
        self.output.parameters(opt, results.extra, results.prior, results.like, results.post)

        #If requested, create a new ini file for the
        #best fit.
        if self.output_ini:
          self.pipeline.create_ini(opt, self.output_ini)

        # If we are coupling to later samplers they can use the peak we have found.
        self.distribution_hints.set_peak(opt, results.post)

        #Also if requested, approximate the covariance matrix with the 
        #inverse of the Hessian matrix.
        #For a gaussian likelihood this is exact.
        covmat = None
        if hasattr(result, 'hess_inv'):
            covmat = self.pipeline.denormalize_matrix(result.hess_inv)
        elif hasattr(result, 'hess'):
            covmat = self.pipeline.denormalize_matrix(np.linalg.inv(result.hess_inv))

        if covmat is None:
            if self.output_cov:
               print("Sorry - the optimization method you chose does not return a covariance (or Hessian) matrix")
        else:
            if self.output_cov:
                np.savetxt(self.output_cov, covmat)
            self.distribution_hints.set_cov(covmat)

        self.converged = True

    def is_converged(self):
        return self.converged

# legacy alias
PMaxlikeSampler = PmaxlikeSampler
