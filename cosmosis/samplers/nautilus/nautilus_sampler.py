from .. import ParallelSampler
from ...runtime import logs
import numpy as np
import os


def log_probability_function(p):
    r = pipeline.run_results(p)
    out = [r.like, r.prior]

    # Flatten any vector outputs here
    for e in r.extra:
        if np.isscalar(e):
            out.append(e)
        else:
            out.extend(e)
    out = tuple(out)
    return out


def prior_transform(p):
    return pipeline.denormalize_vector_from_prior(p)


class NautilusSampler(ParallelSampler):
    parallel_output = False
    internal_resume = True
    sampler_outputs = [('log_weight', float), ('prior', float),
                       ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            self.n_live = self.read_ini("n_live", int, 2000)
            self.n_update = self.read_ini("n_update", int, self.n_live)
            self.enlarge_per_dim = self.read_ini("enlarge_per_dim", float, 1.1)
            self.n_points_min = self.read_ini("n_points_min", int,
                                              self.pipeline.nvaried + 50)
            self.split_threshold = self.read_ini("split_threshold", float,
                                                 100.0)
            self.n_networks = self.read_ini("n_networks", int, 4)
            self.n_batch = self.read_ini("n_batch", int, 100)
            self.seed = self.read_ini("seed", int, -1)
            if self.seed < 0:
                self.seed = None
            self.resume_ = self.read_ini("resume", bool, False)
            self.f_live = self.read_ini("f_live", float, 0.01)
            self.n_shell = self.read_ini("n_shell", int, self.n_batch)
            self.n_eff = self.read_ini("n_eff", float, 10000.0)
            self.n_like_max = self.read_ini("n_like_max", int, -1)
            if self.n_like_max < 0:
                self.n_like_max = np.inf
            self.discard_exploration = self.read_ini(
                "discard_exploration", bool, False)
            self.verbose = self.read_ini("verbose", bool, False)

        self.converged = False

    def execute(self):
        from nautilus import Sampler

        n_dim = self.pipeline.nvaried

        try:
            resume_filepath = self.output.name_for_sampler_resume_info()
        except NotImplementedError:
            resume_filepath = None

        if resume_filepath is not None:
            resume_filepath = resume_filepath + ".hdf5"
            if self.resume_ and os.path.exists(resume_filepath):
                if self.is_master():
                    logs.overview(f"Resuming Nautilus from file {resume_filepath}")

        sampler = Sampler(
            prior_transform,
            log_probability_function,
            n_dim,
            n_live=self.n_live,
            n_update=self.n_update,
            enlarge_per_dim=self.enlarge_per_dim,
            n_points_min=self.n_points_min,
            split_threshold=self.split_threshold,
            n_networks=self.n_networks,
            n_batch=self.n_batch,
            seed=self.seed,
            filepath=resume_filepath,
            resume=self.resume_,
            pool=self.pool,
            blobs_dtype=float
        )

        sampler.run(f_live=self.f_live,
                    n_shell=self.n_shell,
                    n_eff=self.n_eff,
                    n_like_max=self.n_like_max,
                    discard_exploration=self.discard_exploration,
                    verbose=self.verbose)

        results = sampler.posterior(return_blobs=True)
        if isinstance(results[3][0], float):
            priors = results[3]
        else:
            priors = np.array([r[0] for r in results[3]])

        posts = results[2] + priors
        self.distribution_hints.set_from_sample(results[0], posts, log_weights=results[1])

        for sample, logwt, logl, blob in zip(*results):
            if isinstance(blob, float):
                prior = blob
                self.output.parameters(sample, logwt, prior, logl + prior)
            else:
                prior = blob[0]
                extra = list(blob)[1:]
                self.output.parameters(sample, extra, logwt, prior, logl + prior)

        self.output.final("efficiency", sampler.n_eff / sampler.n_like)
        self.output.final("neff", sampler.n_eff)
        self.output.final("nsample", len(sampler.posterior()[0]))
        self.output.final("log_z", sampler.log_z)
        self.output.final("log_z_error", 1.0 / np.sqrt(sampler.n_eff))
        self.converged = True

    def is_converged(self):
        return self.converged
