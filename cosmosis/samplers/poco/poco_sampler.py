from .. import ParallelSampler
from ...runtime import logs
import numpy as np
import os
import glob

def log_likelihood(p):
    r = pipeline.run_results(p)
    return r.like, r.extra


class Prior:
    def __init__(self):
        self.dim = len(pipeline.varied_params)
        self.bounds = np.array([p.limits for p in pipeline.varied_params])

    def logpdf(self, x):
        return np.array([pipeline.prior(p) for p in x])
    
    def rvs(self, size=None):
        return np.array([pipeline.randomized_start() for i in range(size)])


class PocoMCSampler(ParallelSampler):
    parallel_output = False
    supports_resume = True
    sampler_outputs = [('log_weight', float), ('prior', float),
                       ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline
        import pocomc
        self.pocomc = pocomc

        if self.is_master():
            # Parameters of the sampler
            self.n_effective = self.read_ini("n_effective", int, 512)
            self.n_active = self.read_ini("n_active", int, 256)
            self.flow = self.read_ini("flow", str, "nsf6")
            self.precondition = self.read_ini("precondition", bool, True)
            self.dynamic = self.read_ini("dynamic", bool, True)
            self.n_steps = self.read_ini("n_steps", int, len(pipeline.varied_params))
            self.n_max_steps = self.read_ini("n_max_steps", int, 10*len(pipeline.varied_params))
            seed = self.read_ini("seed", int, 0)
            if seed == 0:
                seed = None

            # Parameters of the run
            self.n_total = self.read_ini("n_total", int, 4096)
            self.n_evidence = self.read_ini("n_evidence", int, 4096)
            self.progress = self.read_ini("progress", bool, True)
            self.save_every = self.read_ini("save_every", int, 10)
            if self.save_every == -1:
                self.save_every = None

            # We only checkpoint if the user is saving output to a
            # file, since otherwise we are probably in a script
            # and so checkopinting would probably not make sense
            try:
                self.output_dir = self.output.name_for_sampler_resume_info()
            except NotImplementedError:
                self.output_dir = None

            if self.output_dir is not None:
                if os.path.isfile(self.output_dir):
                    # Another sampler has been run before this one
                    # and created a checkpoint file. We can delete it
                    os.remove(self.output_dir)

                os.makedirs(self.output_dir, exist_ok=True)

            # Finally we can create the sampler
            self.sampler = self.pocomc.Sampler(
                prior=Prior(),
                likelihood=log_likelihood,
                random_state=seed,
                n_effective=self.n_effective,
                n_active=self.n_active,
                flow=self.flow,
                precondition=self.precondition,
                dynamic=self.dynamic,
                n_steps=self.n_steps,
                n_max_steps=self.n_max_steps,
                output_dir=self.output_dir,
                pool=self.pool,
                blobs_dtype=float,
            )

            self.converged = False
            self.resume_state_path = None
            self.told_to_resume = False

    def resume(self):
        self.told_to_resume = True
        # Here we just work out what file path
        # to use for the resume option. The PocoMC
        # library handles the actual resuming.
        # If this function is never called because
        # the user did not set resume=T then
        # self.resume_state will stay as None

        # If we are not saving to a file
        # then we also don't do any checkpointing
        if self.output_dir is None:
            return

        resume_files = glob.glob(os.path.join(self.output_dir, "pmc_*.state"))
        if len(resume_files) == 0:
            return
        # Parse the file names to find the highest
        # index number
        indices = []
        for f in resume_files:
            if "final" in f:
                index = np.inf
            else:
                index = int(f.split("_")[-1].split(".")[0])
            indices.append(index)
        highest_index = np.argmax(indices)
        self.resume_state_path = resume_files[highest_index]
        print("Will resume from state file ", self.resume_state_path)


    def execute(self):
        if not self.told_to_resume and self.output_dir is not None:
            # in this case the state files in the output_dir
            # are stale and should all be removed.
            for f in glob.glob(os.path.join(self.output_dir, "pmc_*.state")):
                os.remove(f)

        self.sampler.run(
            n_total=self.n_total,
            n_evidence=self.n_evidence,
            progress=self.progress,
            resume_state_path=self.resume_state_path,
            save_every=self.save_every,
        )

        logz, logz_err = self.sampler.evidence()

        results = self.sampler.posterior(return_blobs=True, return_logw=True)
        self.distribution_hints.set_from_sample(results[0], results[2]+results[3], log_weights=results[1])

        for sample, logw, logl, logp, blob in zip(*results):
            extra = np.atleast_1d(blob)
            prior = logp
            logpost = logl + prior
            self.distribution_hints.set_peak(sample, logpost)
            self.output.parameters(sample, extra, logw, prior, logpost)

        self.output.final("log_z", logz)
        self.output.final("log_z_error", logz_err)
        self.converged = True

    def is_converged(self):
        return self.converged
