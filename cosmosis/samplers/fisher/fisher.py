import numpy as np
import pdb

'''
NOTE: Need to recompute the spectra between each iteration (EUGH).

Steps:
    - For a given parameter, choose 4 even variations withing 1 sigma contour (i.e. this will probably be an input)
    - Calculate Fisher as dClobs/dparm
        - Calculate derivative with 5-point stencil method http://en.wikipedia.org/wiki/Five-point_stencil
    - Find the new 1 sigma contour according to this Fisher
    - Iterate until contours converge to some tolerance -- 0.5% as default

Inputs:
    - function to calculate vector of Clobs
Outpus:
    - Converged Fisher matrix

Write to parallelise
will be passed a pool with map function can pool.map(params) will give list of results
make list of params, run all at once, will parallelise nicely
be aware that some of the workers in pool might fail, return e.g. None
can have all paramers normalised to [0,1]

compute_vector(p) runs an entire pipeline on a set of parameters (i.e. needs to be done five times here) and returns vector, covmat where vector is Cl_*** and covmat is the necessary **INVERSE** covmat

can do pool.map(compute_vector, [p1,p2 etc]) where p1 is an array of the parameter values

len(start_vector) will give you the number of parameters we are varying, which is taken from the cosmosis values.ini file

'''

class FisherParameterError(Exception):
    def __init__(self, parameter_index):
        message = "Fisher Matrix likelihood function returned None for parameter: {}".format(parameter_index)
        super(Exception,self).__init__(self, message)
        self.parameter_index = parameter_index

class FisherBase:
    def __init__(self, compute_vector, start_vector, pool=None):
        self.compute_vector = compute_vector
        self.start_vector = start_vector
        self.pool = pool
        self.nparams = start_vector.shape[0]

    def compute_fisher_matrix(self):
        derivatives, inv_cov = self.compute_derivatives()

        if not np.allclose(inv_cov, inv_cov.T):
            print("WARNING: The inverse covariance matrix produced by your pipeline")
            print("         is not symmetric. This probably indicates a mistake somewhere.")
            print("         If you are only using cosmosis-standard-library likelihoods please ")
            print("         open an issue about this on the cosmosis site.")
        fisher_matrix = np.einsum("il,lk,jk->ij", derivatives, inv_cov, derivatives)
        return fisher_matrix


    def compute_derivatives(self):
        derivatives = []
        points = self.generate_sample_points()
        print("Calculating derivatives using {} total models".format(len(points)))
        if self.pool is None:
            results = list(map(self.compute_vector, points))
        else:
            results = self.pool.map(self.compute_vector, points)

        # Bit of a waste of time to compute the inv cov separately,
        # but it's a quick fix to a memory error if we compute the cov
        # for every single value
        _, inv_cov = self.compute_vector(points[0], cov=True)

        derivatives = self.extract_derivatives(results)

        return derivatives, inv_cov

class Fisher(FisherBase):
    def __init__(self, compute_vector, start_vector, step_size, pool=None):
        super().__init__(compute_vector, start_vector, pool)
        self.iterations = 0
        self.step_size = step_size

    def generate_sample_points(self):
        points = []

        #To improve parallelization we first gather all the data points
        #we use in all the dimensions
        for p in range(self.nparams):
            points +=  self.five_points_stencil_points(p)

        return points

    def extract_derivatives(self, results):
        derivatives = []
        #Now get out the results that correspond to each dimension
        for p in range(self.nparams):
            results_p = results[4*p:4*(p+1)]
            derivative = self.five_point_stencil_deriv(results_p, p)
            derivatives.append(derivative)
        return np.array(derivatives)

    def five_points_stencil_points(self, param_index):
        delta = np.zeros(self.nparams)
        delta[param_index] = 1.0
        points = [self.start_vector + x*delta for x in 
            [2*self.step_size, 
             1*self.step_size, 
            -1*self.step_size, 
            -2*self.step_size]
        ]
        return points        

    def five_point_stencil_deriv(self, obs, param_index):
        for r in obs:
            if r is None:
                raise FisherParameterError(param_index)
        deriv = (-obs[0] + 8*obs[1] - 8*obs[2] + obs[3])/(12*self.step_size)
        return deriv



class NumDiffToolsFisher(Fisher):
    def compute_derivatives(self):
        import numdifftools as nd
        def wrapper(param_vector):
            return self.compute_vector(param_vector, cov=False)
        jacobian_calculator = nd.Jacobian(wrapper, step=self.step_size)
        derivatives = jacobian_calculator(self.start_vector)
        _, inv_cov = self.compute_vector(self.start_vector, cov=True)
        return derivatives.T, inv_cov
    

class SmoothingFisher(FisherBase):
    def __init__(self, compute_vector, start_vector,
                step_size_min, step_size_max, step_count, pool=None):
        super().__init__(compute_vector, start_vector, pool)
        # import the "derivative" library here, since otherwise it will
        # not be imported until right at the end, after lots of time
        # has been spent computing the sample points. 
        import derivative
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max
        if step_count % 2 == 0:
            self.half_step_count = step_count // 2
        else:
            self.half_step_count = (step_count - 1) // 2
    
    def generate_sample_points(self):
        points = []
        self.x_values = []

        # Recall that this whole file assumes all the bounds are already
        # normalized, so we can just use the same bounds for all parameters
        delta = np.logspace(np.log10(self.step_size_min), np.log10(self.step_size_max), self.half_step_count)

        #To improve parallelization we first gather all the data points
        #we use in all the dimensions
        for p in range(self.nparams):
            x0 = self.start_vector[p]
            x = np.concatenate([(x0 - delta)[::-1], [x0], x0 + delta])
            self.x_values.append(x)
            for xi in x:
                point = np.copy(self.start_vector)
                point[p] = xi
                points.append(point)

        return points

    def extract_derivatives(self, results):
        import derivative
        chunk = 2*self.half_step_count + 1
        derivatives = []

        # Consider exposing some options for these
        calculator = derivative.SavitzkyGolay(left=.5, right=.5, order=2)

        #Now get out the results that correspond to each dimension
        for p in range(self.nparams):
            x = self.x_values[p]
            y = np.array(results[chunk * p : chunk * (p + 1)])
            d = calculator.d(y, x, axis=0)
            derivatives.append(d[self.half_step_count])
        return np.array(derivatives)



def test():
    def theory_prediction(x, cov=False):
        #same number of data points as parameters here
        x = np.concatenate([x,x])
        theory = x ** 3 + 2*x + 2
        inv_cov = np.diag(np.ones_like(x)**-1)
        if cov:
            return theory, inv_cov
        else:
            return theory

    best_fit_params = np.array([0.1, 1.0, 2.0, 4.0,])
    fisher_calculator = Fisher(theory_prediction, best_fit_params, 0.01)
    F = fisher_calculator.compute_fisher_matrix()
    print(F)

    fisher_calculator = SmoothingFisher(theory_prediction, best_fit_params, 1e-5, 0.01, 20)
    F = fisher_calculator.compute_fisher_matrix()
    print(F)

    fisher_calculator = NumDiffToolsFisher(theory_prediction, best_fit_params, 0.01)
    F = fisher_calculator.compute_fisher_matrix()
    print(F)


if __name__ == '__main__':
    test()