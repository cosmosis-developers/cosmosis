from .. import ParallelSampler
import numpy as np
from ...runtime import prior
import scipy.linalg
import types

def abc_model(p):
    #add check that omega_m >0
   for i,pname in enumerate(abc_pipeline.varied_params):
    if pname.name == 'omega_m' and p[i] <0:
        return np.inf

    data = abc_pipeline.run_parameters(p)
    if data is None:
        return None
    #collect the models for each data set we want to use
    model = []
    for like_name in abc_pipeline.likelihood_names:
        #Look in the standard place for the model
        try:
            model.append(data['data_vector', like_name + '_simulation'])
        #Raise an error in the even of failure - this is a systematic problem
        except BlockError:
            raise ValueError("The module in the ABC pipeline should save a model (i.e. a simulation) with the same dimensions as the data, called NAME_simulation.")
    model = np.concatenate(model)
    
    return model


class AbcSampler(ParallelSampler):
    parallel_output = False #True for 
    sampler_outputs = [("weight", float),("distance",float)]

    def config(self):
        try:
            import abcpmc
        except ImportError:
            raise ValueError("To use ABC PMC you need to install it with pip install abcpmc")
        
        global abc_pipeline
        abc_pipeline = self.pipeline

        self.threshold = self.read_ini("threshold",str, 'LinearEps')
        self.metric_kw = self.read_ini("metric",str, 'chi2') #mean, chi2 or other
        if self.metric_kw =='other':
            self.distance_func = self.read_ini("distance_func",str, None) #only for other metric, 
            self.metric = self.distance_func[1:-1]
        self.epimax = self.read_ini('epimax', float,5.0)
        self.epimin = self.read_ini('epimin',float, 1.0)
        self.part_prop = self.read_ini("particle_prop",str,'weighted_cov')
        self.set_prior = self.read_ini("set_prior",str,'uniform')
        self.param_cov = self.read_ini("param_cov_file",str,'None')
        self.knn = self.read_ini("num_nn",int, 10)
        self.npart = self.read_ini("npart",int,100)
        self.niter = self.read_ini("niter",int,2)
        self.ngauss = self.read_ini("ngauss",int,4)
        self.run_multigauss = self.read_ini("run_multigauss",bool,False)
        self.diag_cov = self.read_ini("diag_cov",bool,False)
        self.ndim = len(self.pipeline.varied_params)

        #options for decreasing threshold
        if self.threshold == 'ConstEps':
            self.eps = abcpmc.ConstEps(self.niter, self.epimax)
        elif self.threshold == 'ExpEps':
            self.eps = abcpmc.ExponentialEps(self.niter, self.epimax,self.epimin)
        else:
            self.eps = abcpmc.LinearEps(self.niter, self.epimax, self.epimin)

        print("\nRunning ABC PMC")
        print("with %d particles, %s prior, %s threshold, %d iterations over (%f,%f), %s kernal \n" % (self.npart,self.set_prior,self.threshold,self.niter,self.epimax,self.epimin,self.part_prop))


        #Initial positions for all of the parameters
        self.p0 = np.array([param.start for param in self.pipeline.varied_params])

        #Data file is read for use in dist() for each step
        #parameter covariance used in the prior   
        self.data, self.cov, self.invcov = self.load_data()
        
        #At the moment the same prior (with variable hyperparameters) is
        # used for all parameters  - would be nice to change this to be more flexible 
        self.pmin = np.zeros(self.ndim)
        self.pmax = np.zeros(self.ndim)
        for i,pi in enumerate(self.pipeline.varied_params):
            self.pmin[i] = pi.limits[0]
            self.pmax[i] = pi.limits[1]


        if self.set_prior.lower() == 'uniform':
            self.prior = abcpmc.TophatPrior(self.pmin,self.pmax)
        elif self.set_prior.lower() == 'gaussian':
            sigma2 = np.loadtxt(self.param_cov)
            if len(np.atleast_2d(sigma2)[0][:]) != self.ndim:
                n_columns = len(np.atleast_2d(sigma2)[0][:])
                raise ValueError("Cov matrix for Gaussian prior has %d columns for %d params" % (n_columns, self.ndim))
            else:
                self.prior = abcpmc.GaussianPrior(self.p0, np.atleast_2d(sigma2)) 
        else:
            raise ValueError("Please set the ABC option 'set_prior' to either 'uniform' or 'gaussian'. At the moment only 'uniform' works in the general case.")
        #create sampler
        self.sampler = abcpmc.Sampler(N=self.npart, Y=self.data, postfn=abc_model, dist=self.dist)

        #set particle proposal kernal
        abcpmc.Sampler.particle_proposal_kwargs = {}
        if self.part_prop == 'KNN':
            abcpmc.Sampler.particle_proposal_kwargs = {'k':self.knn}
            self.sampler.particle_proposal_cls = abcpmc.KNNParticleProposal
        elif self.part_prop == 'OLCM':
            self.sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal

        self.converged = False

    def output_samples(self,out):
        for o in out:
            self.output.parameters(o)

    def execute(self):
        # called only by master. Other processors will be in self.pool
        # generate outputs somehow by running log_probability_function
        outputs = []
        for output in self.sampler.sample(self.prior, self.eps):
            outputs.append(np.c_[output.thetas,output.ws,output.dists])
        for out in outputs:
            self.output_samples(out)

        self.converged = True


    def is_converged(self):
        return  self.converged

    def dist(self,x, y):
        if np.isinf(x).any():
            return np.inf
        if self.metric_kw == "chi2":
            d  = x - y
            return np.dot(d,np.dot(self.invcov,d))
        elif self.metric_kw == "mean":
            return np.sum(np.abs(np.mean(x, axis=0) - np.mean(y, axis=0)))
        elif self.metric_kw == "other":
            exec(self.metric)
            return dist_result

    def load_data(self):
        #Load the data by running the pipeline once
        print("Doing and initial ABC run of the pipeline, just to get the data vector from all the likelihood modules.")
        print("This is a slight waste of time (sorry) but the most general way of doing things.")

        block = self.pipeline.run_parameters(self.p0)
        if block is None:
            raise ValueError("I had an error running the pipeline at the central starting values in the values file")

        data = []
        covs = []
        invcovs = []
        for like_name in self.pipeline.likelihood_names:
            data.append(block["data_vector", like_name + "_data"])
            covs.append(block["data_vector", like_name + "_covariance"])
            invcovs.append(block["data_vector", like_name + "_inverse_covariance"])

        data = np.concatenate(data)
        covs = scipy.linalg.block_diag(*covs)
        invcovs = scipy.linalg.block_diag(*invcovs)
        
        if self.diag_cov:
            diag = np.diag(covs)
            covs = np.diag(diag)
            inv = 1./diag
            invcovs = np.diag(inv)

        return  data, covs,invcovs

# legacy alias
ABCSampler = AbcSampler
