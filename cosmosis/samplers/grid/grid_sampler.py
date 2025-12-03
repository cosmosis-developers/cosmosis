import itertools
import numpy as np
from ... runtime import logs
from .. import ParallelSampler


def task(p):
    i,p = p
    results = grid_sampler.pipeline.run_results(p)
    #If requested, save the data to file
    if grid_sampler.save_name and results.block is not None:
        results.block.save_to_file(grid_sampler.save_name+"_%d"%i, clobber=True)
    return (results.post, results.prior, results.extra)

LARGE_JOB_SIZE = 1000000



class GridSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]
    understands_fast_subspaces = True

    def config(self):
        global grid_sampler
        grid_sampler = self

        self.converged = False
        self.nsample = self.read_ini("nsample_dimension", int, 1)
        self.save_name = self.read_ini("save", str, "")
        self.nstep = self.read_ini("nstep", int, -1)
        self.allow_large = self.read_ini("allow_large", bool, False)
        self.sample_points = None
        self.ndone = 0

    def setup_sampling(self):
        #Number of jobs to do at once.
        #Can be taken from the ini file.
        #Otherwise it is set to -1 by default
        if self.nstep==-1:
            #if in parallel mode do a chunk of 4*the number of tasks to do at once
            #chosen arbitrarily.
            if self.pool:
                self.nstep = 4*(self.pool.size-1)
            #if not parallel then just do a single slice through one dimension each chunk
            else:
                self.nstep = self.nsample


        #Also Generate the complete collection of parameter sets to run over.
        #This doesn't actually keep them all in memory, it is just the conceptual
        #outer product
        total_samples = self.nsample**len(self.pipeline.varied_params)
        logs.overview(f"Total number of grid samples: {total_samples}")

        if total_samples>LARGE_JOB_SIZE:
            logs.error("That is a very large number of samples.")
            if self.allow_large:
                logs.error("But you set allow_large=T so I will continue")
            else:
                logs.error("This is suspicously large so I am going to stop")
                logs.error("If you really want to do this set allow_large=T in the")
                logs.error("[grid] section of the ini file.")
                raise ValueError("Suspicously large number of grid points %d ( = n_samp ^ n_dim = %d ^ %d); set allow_large=T in [grid] section to permit this."%(total_samples,self.nsample,len(self.pipeline.varied_params)))
        
        # If our pipeline allows it we arrange it so that the
        # fast parameters change fastest in the sequence.
        # This is still not optimal for the multiprocessing case
        if self.pipeline.do_fast_slow:
            param_order = self.pipeline.slow_params + self.pipeline.fast_params
        else:
            param_order = self.pipeline.varied_params

        # This little bit of python and numpy wizardry generates
        # an iterator that generates the sequence of grid points
        # which is an outer product of the linearly spaced sample
        # points in each dimension.
        self.sample_points = itertools.product(*[np.linspace(*param.limits,
                                                       num=self.nsample)
                                            for param in param_order])


    def execute(self):
        #First run only:
        if self.sample_points is None:
            self.setup_sampling()

        #Chunk of tasks to do this run through, of size nstep.
        #This advances the self.sample_points forward so it knows
        #that these samples have been done
        samples = list(itertools.islice(self.sample_points, self.nstep))

        #If there are no samples left then we are done.
        if not samples:
            self.converged=True
            return

        #Each job has an index number in case we are saving
        #the output results from each one
        sample_index = np.arange(len(samples)) + self.ndone
        jobs = list(zip(sample_index, samples))

        #Actually compute the likelihood results
        if self.pool:
            results = self.pool.map(task, jobs)
        else:
            results = list(map(task, jobs))

        #Update the count
        self.ndone += len(results)

        #Save the results of the sampling
        for sample, result  in zip(samples, results):
            #Optionally save all the results calculated by each
            #pipeline run to files
            (prob, prior, extra) = result
            #always save the usual text output
            self.output.parameters(sample, extra, prior, prob)
            self.distribution_hints.set_peak(sample, prob)

    def is_converged(self):
        return self.converged
