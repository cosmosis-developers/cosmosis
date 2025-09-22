import itertools
import numpy as np
from scipy.stats import qmc
from ...runtime import logs
from .. import ParallelSampler


def task(p):
    i, p, name, training = p
    results = lh_sampler.pipeline.run_results(p)
    #If requested, save the data to file
    if lh_sampler.save_name and results.block is not None:
        if training:
            results.block.to_pickle(f"{lh_sampler.save_name}{name}_{i}")
        else:
            results.block.save_to_file(f"{lh_sampler.save_name}{name}_{i}", clobber=True)
    return (results.post, results.prior, results.extra)


class LatinHypercubeSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]
    understands_fast_subspaces = True

    def config(self):
        global lh_sampler
        lh_sampler = self

        self.converged = False
        self.nsample = self.read_ini("nsample", int, 1)
        self.save_name = self.read_ini("save", str, "")
        self.nstep = self.read_ini("nstep", int, -1)

        if self.pipeline.training:
            self.nsample = self.pipeline.nsample
            self.nsample_test = self.pipeline.nsample_test
            self.save_name = self.pipeline.save_name
        else:
            self.nsample_test = 0

        self.sample_points = None
        self.ndone = 0
        self.ndone_train = 0
        self.ndone_test = 0

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
                self.nstep = self.nsample + self.nsample_test
        
        # If our pipeline allows it we arrange it so that the
        # fast parameters change fastest in the sequence.
        # This is still not optimal for the multiprocessing case
        if self.pipeline.do_fast_slow:
            param_order = self.pipeline.slow_params + self.pipeline.fast_params
        else:
            param_order = self.pipeline.varied_params
        
        # Create a Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=len(param_order))

        # Generate all the Latin Hypercube samples at once
        lhs = sampler.random(n=self.nsample)

        # Extract the lower and upper bounds for each parameter
        l_bounds = [param.limits[0] for param in param_order]
        u_bounds = [param.limits[1] for param in param_order]

        # Scale the samples to the desired parameter limits
        lhs_scaled = qmc.scale(lhs, l_bounds, u_bounds)
        self.sample_points = iter(lhs_scaled)

        # We create a second LHS if we use the latin hypercube to create samples for
        # ML and / or CosmoPower training
        if self.pipeline.training:
            if self.nsample_test == 0:
                raise ValueError("nsample_test needs to be greater than 0!")
            # Create a Latin Hypercube sampler
            sampler_test = qmc.LatinHypercube(d=len(param_order))

            # Generate all the Latin Hypercube samples at once
            lhs_test = sampler_test.random(n=self.nsample_test)

            # Scale the samples to the desired parameter limits
            # We use the same bounds!
            lhs_scaled_test = qmc.scale(lhs_test, l_bounds, u_bounds)
            self.sample_points_test = iter(lhs_scaled_test)


    def execute(self):
        #First run only:
        if self.sample_points is None:
            self.setup_sampling()

        #Chunk of tasks to do this run through, of size nstep.
        #This advances the self.sample_points forward so it knows
        #that these samples have been done
        samples = list(itertools.islice(self.sample_points, self.nstep))
        if self.pipeline.training:
            samples_test = list(itertools.islice(self.sample_points_test, self.nstep))
        else:
            samples_test = []
        samples_tot = samples + samples_test

        #If there are no samples left then we are done.
        if not samples_tot:
            self.converged=True
            return
        
        #Each job has an index number in case we are saving
        #the output results from each one
        sample_index = np.arange(len(samples)) + self.ndone_train
        sample_name = ["" for _ in sample_index]
        training_flag = [self.pipeline.training for _ in sample_index]
        jobs = list(zip(sample_index, samples, sample_name, training_flag))

        #Actually compute the likelihood results
        if self.pool:
            results = self.pool.map(task, jobs)
        else:
            results = list(map(task, jobs))

        if self.pipeline.training:
            sample_test_index = np.arange(len(samples_test)) + self.ndone_test
            sample_name_test = ["_test" for i in sample_test_index]
            training_flag_test = [self.pipeline.training for i in sample_index]
            jobs_test = list(zip(sample_test_index, samples_test, sample_name_test, training_flag_test))
            if self.pool:
                results_test = self.pool.map(task, jobs_test)
            else:
                results_test = list(map(task, jobs_test))
            self.ndone_test += len(results_test)

        #Update the count
        self.ndone_train += len(results)
        self.ndone = self.ndone_train + self.ndone_test

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
