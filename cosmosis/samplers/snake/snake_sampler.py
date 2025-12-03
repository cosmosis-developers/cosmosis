from .. import ParallelSampler
from ...runtime import logs
import numpy as np
from .snake import Snake

def posterior(p_in):
    #Check the normalization
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
        return -np.inf, ([np.nan for i in range(len(snake_pipeline.extra_saves))], -np.inf)
    p = snake_pipeline.denormalize_vector(p_in)
    results = snake_pipeline.run_results(p)
    return results.post, (results.extra, results.prior)


class SnakeSampler(ParallelSampler):
    sampler_outputs = [("prior", float), ("post", float)]
    parallel_output = False

    def config(self):
        global snake_pipeline
        snake_pipeline=self.pipeline
        if self.is_master():
            threshold = self.read_ini("threshold", float, 4.0)
            self.grid_size = 1.0/self.read_ini("nsample_dimension", int, 10)
            self.maxiter = self.read_ini("maxiter", int, 100000)

            
            origin = self.pipeline.normalize_vector(self.pipeline.start_vector())
            spacing = np.repeat(self.grid_size, len(self.pipeline.varied_params))
            self.snake = Snake(posterior, origin, spacing, threshold, pool=self.pool)

    def execute(self):
        X, P, E = self.snake.iterate()
        for (x,post,(extra, prior)) in zip(X,P,E):
            try:
                x = self.pipeline.denormalize_vector(x)
                self.output.parameters(x, extra, prior, post)
                self.distribution_hints.set_peak(x, post)
            except ValueError:
                logs.noisy("The snake is trying to escape its bounds!")


    def is_converged(self):
        if self.snake.converged():
            logs.overview("Snake has converged!")
            logs.overview("Best post = %f    Best surface point = %f" %(self.snake.best_like_ever, self.snake.best_fit_like))
            return True
        if self.snake.iterations > self.maxiter:
            logs.warning("Run out of iterations.")
            logs.warning("Done %d, max allowed %d" % (self.snake.iterations, self.maxiter))
            return True
        return False
