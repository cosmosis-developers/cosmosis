import itertools
import numpy as np
from ...runtime import logs
from .. import ParallelSampler


def task(p):
    i,p = p
    results = star_sampler.pipeline.run_results(p)
    #If requested, save the data to file
    if star_sampler.save_name and results.block is not None:
        results.block.save_to_file(star_sampler.save_name+"_%d"%i, clobber=True)
    return (results.post, results.prior, results.extra)

LARGE_JOB_SIZE = 1000000



class StarSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float),("post", float)]
    understands_fast_subspaces = True

    def config(self):
        global star_sampler
        star_sampler = self

        self.converged = False
        self.nsample = self.read_ini("nsample_dimension", int, 3)
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
                self.nstep = 4*self.pool.size
            #if not parallel then just do a single slice through one dimension each chunk
            else:
                self.nstep = self.nsample

        if self.output:
            for name,value in zip(self.pipeline.varied_params, self.pipeline.start_vector()):
                self.output.metadata("fid_{0}".format(name), value)


        #Also Generate the complete collection of parameter sets to run over.
        #This doesn't actually keep them all in memory, it is just the conceptual
        #outer product
        total_samples = self.nsample*len(self.pipeline.varied_params)

        logs.overview(f"Total number of star samples: {total_samples}")

        if total_samples>LARGE_JOB_SIZE:
            logs.overview("That is a very large number of samples.")
            if self.allow_large:
                logs.overview("But you set allow_large=T so I will continue")
            else:
                logs.overview("This is suspicously large so I am going to stop")
                logs.overview("If you really want to do this set allow_large=T in the")
                logs.overview("[star] section of the ini file.")
                raise ValueError("Suspicously large number of star points %d ( = n_samp * n_dim = %d * %d); set allow_large=T in [star] section to permit this."%(total_samples,self.nsample,len(self.pipeline.varied_params)))

        

        sample_points = []
        start = self.pipeline.start_vector()
        for i,param in enumerate(self.pipeline.varied_params):
            for p in np.linspace(*param.limits, num=self.nsample):
                v = start.copy()
                v[i] = p
                sample_points.append(v)
        self.sample_points = iter(sample_points)




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
            (post, prior, extra) = result
            #always save the usual text output
            self.output.parameters(sample, extra, prior, post)
            self.distribution_hints.set_peak(sample, post)

    def is_converged(self):
        return self.converged
