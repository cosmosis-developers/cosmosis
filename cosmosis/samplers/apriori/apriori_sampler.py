import numpy as np
from .. import ParallelSampler
from ...runtime import logs

def task(p):
    i,p = p
    r = sampler.pipeline.run_results(p)
    #If requested, save the data to file
    if sampler.save_name:
        if r.block is None:
            logs.error("Failed to run parameters: {} so not saving".format(p))
        else:
            filename = "{}_{}".format(sampler.save_name, i)
            r.block.save_to_file(filename, clobber=True)

    return (r.prior, r.post, r.extra)




class AprioriSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global sampler
        sampler = self

        self.converged = False
        self.save_name = self.read_ini("save", str, "")
        self.nsample = self.read_ini("nsample", int, 1)
        self.n = 0

    def resume(self):
        if self.output.resumed:
            data = np.genfromtxt(self.output._filename, invalid_raise=False)
            self.n = len(data)
            if self.n >= self.nsample:
                logs.error(f"You told me to resume the apriori sampler - it has already completed (with {self.n} samples), so sampling will end.")
                logs.error("Increase the 'nsample' parameter to keep going.")
            else:
                logs.overview(f"Continuing apriori sampling - have {self.n} samples already")


    def execute(self):
        n = self.n
        nparam = len(self.pipeline.varied_params)

        if self.pool:
            chunk_size = self.pool.size

        def sample_from_prior():
            x = np.random.uniform(0.0,1.0,size=nparam)
            p = self.pipeline.denormalize_vector_from_prior(x)
            return p

        while n < self.nsample:
            if self.pool:
                jobs = [(n+i,sample_from_prior()) for i in range(chunk_size)]
                results = self.pool.map(task, jobs)
                n += chunk_size
            else:
                jobs = [(n,sample_from_prior())]
                results = list(map(task, jobs))
                n += 1

            #Save the results of the sampling
            #We now need to abuse the output code a little.
            for sample, result  in zip(jobs, results):
                i, sample=sample
                #Optionally save all the results calculated by each
                #pipeline run to files
                (prior, post, extra) = result
                #always save the usual text output
                self.output.parameters(sample, extra, prior, post)

                # this will test to see if the new point is the best so far.
                # It's not a good idea to try to estimate the covariance from
                # an apriori sample, so we don't do that here.
                self.distribution_hints.set_peak(sample, post)

        #We only ever run this once, though that could 
        #change if we decide to split up the runs
        self.converged = True

    def is_converged(self):
        return self.converged
