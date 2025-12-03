from ...runtime import logs
from .. import ParallelSampler
import numpy as np
import sys


def log_probability_function(p):
    results = pipeline.run_results(p)
    # also save the prior because we want the posterior which dynesty does not report
    blob = np.concatenate([[results.prior], results.extra])
    return results.like, blob

def prior_transform(p):
    return pipeline.denormalize_vector_from_prior(p)

class DynestySampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [('log_weight', float), ("prior", float), ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            self.mode = self.read_ini_choices("mode", str, ["static", "dynamic"], "static")
            self.nlive = self.read_ini("nlive", int, 500)
            self.bound = self.read_ini_choices("bound", str, ["none", "single", "multi", "balls", "cube"], "multi")
            self.sample = self.read_ini_choices("sample", str, ["unif", "rwalk", "slice", "rslice", "hslice", "auto"], "auto")
            self.update_interval = self.read_ini("update_interval", float, 0.6)
            self.min_ncall = self.read_ini("min_ncall",int, 2*self.nlive)
            self.min_eff = self.read_ini("min_eff",float, 10.0)
            default_queue_size = 1 if self.pool is None else self.pool.size
            self.queue_size = self.read_ini("queue_size", int, default_queue_size)
            self.parallel_prior = self.read_ini("parallel_prior", bool, True)
            self.max_call = self.read_ini("max_call", int, sys.maxsize)
            self.dlogz = self.read_ini("dlogz", float, 0.01)
            print_progress_default = logs.is_enabled_for(logs.logging.INFO)
            self.print_progress = self.read_ini("print_progress", bool, print_progress_default)

        self.converged = False



    def execute(self):
        from dynesty import NestedSampler, DynamicNestedSampler

        ndim = self.pipeline.nvaried


        if self.mode == "static":
            sampler = NestedSampler(
                log_probability_function,
                prior_transform, 
                ndim, 
                nlive = self.nlive,
                bound = self.bound,
                sample = self.sample,
                update_interval = self.update_interval,
                first_update = {'min_ncall':self.min_ncall, 'min_eff':self.min_eff},
                queue_size = self.queue_size,
                pool = self.pool,
                blob=True
                )

            sampler.run_nested(dlogz=self.dlogz)

        else:
            sampler = DynamicNestedSampler(
                log_probability_function,
                prior_transform, 
                ndim, 
                bound = self.bound,
                sample = self.sample,
                # update_interval = self.update_interval,
                queue_size = self.queue_size,
                pool = self.pool,
                blob=True
                )
            sampler.run_nested(dlogz_init=self.dlogz)

        results = sampler.results
        results.summary()

        posts = results['blob'][:, 0] + results['logl']
        self.distribution_hints.set_from_sample(results['samples'], posts, log_weights=results['logwt'])

        for sample, logwt, logl, derived in zip(results['samples'],results['logwt'], results['logl'], results['blob']):
            prior = derived[0]
            post = prior + logl
            self.output.parameters(sample, derived[1:], logwt, prior, post)

        self.output.final("efficiency", results['eff'])
        self.output.final("nsample", len(results['samples']))
        self.output.final("log_z", results['logz'][-1])
        self.output.final("log_z_error", results['logzerr'][-1])
        self.converged = True



    def is_converged(self):
        return self.converged
