from .. import ParallelSampler
from ...runtime import logs
import numpy as np


def task(p):
    #pipeline here refers to a global variable
    #created later.  This is the only way we know
    #to get MPI task sharing to work
    results = pipeline.run_results(p)
    return (results.prior, results.post, results.extra)


class GridmaxSampler(ParallelSampler):
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            self.nsteps = self.read_ini("nsteps", int, 24)
            self.tolerance = self.read_ini("tolerance", float, 0.1)
            self.max_iterations = self.read_ini("max_iterations", int, 10000)
            self.output_ini = self.read_ini("output_ini", str, "")
            self.ndim = len(self.pipeline.varied_params)
            self.p = self.pipeline.normalize_vector(self.pipeline.start_vector())
            self.dimension = 0
            self.bounds = [(0,1) for i in range(self.ndim)]
            self.previous_maxlike = -np.inf
            self.maxlike = -np.inf
            self.iterations = 0


    def execute(self):
        #Do a slice maximization through the current
        #dimension, with the current bounds, in parallel
        #Start by setting all the points to the current one

        normed_points = [self.p.copy() for i in range(self.nsteps)]

        #Figure out which points to sample at in the current 
        #dimension
        d = self.dimension
        bounds = self.bounds[d]
        samples = np.linspace(bounds[0], bounds[1], self.nsteps)

        logs.overview("Minimizing in %s"%self.pipeline.varied_params[d].name)

        #Fill in the sample points for the current
        #dimension in the samples
        for (p,s) in zip(normed_points,samples):
            p[d] = s

        #Denormalize the points to get the physical parameters
        points = [self.pipeline.denormalize_vector(p) for p in normed_points]

        #And send them off to the workers
        if self.pool:
            results = self.pool.map(task, points)
        else:
            results = list(map(task, points))

        #Log the results for posterity
        for p, (pr, po, e) in zip(points, results):
            self.output.parameters(p, e, pr, po)
            self.distribution_hints.set_peak(p, po)

        #And now update our information.
        #We need to find the two points either side
        #of the maximum likelihood and set them as our new bound,
        #and set our new starting point to be the max value
        posteriors = np.array([r[1] for r in results])
        best = posteriors.argmax()
        #first, the special cases one of the boundary 
        #points is the best (this might be cause for concern, of course)
        if best==0:
            #If we are up against the lower boundary then
            #use the edge as the lower bound, the first point 
            #as the upper, and the half-way point as the new start
            low = 0.0
            high = normed_points[1][d]
            start = high/2.0
        elif best==self.nsteps-1:
            #if on the upper edge, do the mirror image
            low = normed_points[-1][d]
            high = 1.0
            start = (normed_points[-1][d] + 1.0) / 2.0
        else:
            #but the usual case is to just bracket the
            #best-fit point
            low = normed_points[best-1][d]
            high = normed_points[best+1][d]
            start = normed_points[best][d]

        #update the bounds and start with whatever we found
        self.p[d] = start
        self.bounds[d] = (low, high)
    
        #If we have been around all the parameters 
        #then update the best-fits
        if d==0:
            self.previous_maxlike = self.maxlike
            self.maxlike = results[best][0]

        logs.overview(f"New best fit L = {posteriors.max()} at {self.pipeline.varied_params[d].name} = {points[best][d]}")

        

        #and go on to the next dimension
        self.dimension = (d+1)%self.ndim

        if self.is_converged() and self.output_ini:
            self.pipeline.create_ini(points[best], self.output_ini)
        self.iterations += 1



    def is_converged(self):
        return (self.dimension==1 and (0<self.maxlike-self.previous_maxlike<self.tolerance)) or self.iterations > self.max_iterations

# legay alias
GridMaxSampler = GridmaxSampler
