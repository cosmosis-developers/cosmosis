#coding: utf-8


u"""Definition of the :class:`Parameter` class."""
import random
from . import config
from . import prior as priors # to avoid breaking other stuff below
import numpy as np



class Parameter(object):

    u"""Distribution meta-data for a :class:`Pipeline` parameter.
    
    While pipeline modules deal with a dictionary of scalar parameters,
    the pipeline infrastructure (i.e., the :class:`Pipeline` class) must
    track meta-data for each parameter: the allowable range of values the
    parameter is allowed to take, and the (prior) distribution from which
    the values should be sampled.  These data, *but not the parameter
    values themselves*, are stored in this class and kept by the
    :class:`Pipeline` class in an array which parallels the
    :class:`DataBlock` or :class:`Config` which holds the values.

    """

    def __init__(self, section, name, start, limits=None, prior=None):
        u"""Store meta-data for parameter at `(section, name)`.

        If `prior` is not supplied, a uniform distribution is assumed.

        If `limits` are not supplied, the parameter will be considered
        fixed at the `start` value.

        For the available priors, see the :module:Priors module.

        """
        if not limits:
            self.limits = (start, start)
        else:
            self.limits = limits
            if (limits[1]<limits[0]):
                raise ValueError("Your ini file specified that "
                    "parameter %s in section %s had upper limit "
                    "< lower limit"% (name, section))

        self.section = section
        self.name = name
        self.start = start

        if prior is None:
            if limits is None:
                # Parameter has no setting in the priors file and is fixed
                prior = priors.DeltaFunctionPrior(start)
            else:
                # Parameter has no setting in the priors file and is
                # variable
                prior = priors.UniformPrior(limits[0], limits[1])
        else:
            if limits is None:
                # Parameter does have setting in the priors file but is
                # fixed - just fix value
                prior = priors.DeltaFunctionPrior(start)
            else:
                # Parameter does have setting in the priors file and is
                # variable - truncate prior to limits
                prior = prior.truncate(limits[0], limits[1])

        self.prior = prior

        # TODO: check consistency of prior with limits



    def __eq__(self, other):
        u"""Return `True` if `other` stands for the same data block entry as us.

        The same entry means that the `(section, name)` pairs are the
        same.  Note that this is NOT a test of equality of the
        `Parameter`sʼ values!

        """
        if isinstance(other, (list, tuple)):
            try:
                section, name = other
            except ValueError:
                return False
            return (self.section == section and
                    self.name == name)
        elif isinstance(other, Parameter):
            return (self.section == other.section and
                    self.name == other.name)
        elif isinstance(other, str):
            return other==self.__str__()
        else:
            raise NotImplementedError("Tried to do parameter==something where something was not a thing I understand.")



    def __str__(self):
        u"""Return our ID "section--name" as stringified version."""
        return self.section + "--" + self.name



    def __repr__(self):
        u"""Return __str__."""
        return self.__str__()

    def fix(self, value=None):
        """
        Switch this parameter to being fixed, optionally to
        a new specified value.
        """
        if value is not None:
            self.start = value

        self.limits = self.start, self.start

    def is_fixed(self):
        u"""Test whether this parameter is fixed or varied.

        Returns `True` if parameter is fixed to a single value by
        degenerate definition of the limits, and false if there is room
        for it to vary.

        """
        return self.limits[0] == self.limits[1]



    def is_varied(self):
        u"""The opposite of :func:`is_fixed`."""
        return not self.is_fixed()



    def in_range(self, p):
        u"""Check that the value `p` is in this parameterʼs allowed range."""
        return self.limits[0] <= p <= self.limits[1]



    def width(self):
        u"""Return the difference between the upper and lower limits."""
        return self.limits[1] - self.limits[0]



    def random_point(self):
        u"""Return a random number taken from the 'prior' distribution."""
        return self.prior.sample(1)[0]



    def normalize(self, p):
        u"""Return the relative position of `p` between the allowable limits: a value fron 0.0 to 1.0.

        But the return value will go beyond the unit interval if `p` is
        actually outside the limits.

        """
        if self.is_fixed():
            return 0.0
        else:
            return (p - self.limits[0]) / (self.limits[1] - self.limits[0])

    def normalize_to_prior(self, x):
        """
        Return the cumulative distribution function value of this value in the parameter's
        prior. That is, return the probability that p < x. Outside the prior support this returns
        nan.
        """
        if self.is_fixed():
            if math.isclose(x, self.start):
                return 0.5
            else:
                return np.nan
        else:
            return self.prior.normalize_to_prior(x)



    def denormalize(self, p, raise_exception=True):
        u"""Return the value at the relative position `p` between the lower and upper limits.

        If `p` is outside the range [0.0, 1.0], then if `raise_exception`
        is `True` a ValueError will be raised, otherwise a value extrapolated
        outside the range of the limits will be returned.

        """
        if 0.0 <= p <= 1.0:
            return p*(self.limits[1]-self.limits[0]) + self.limits[0]
        elif not raise_exception:
            return p*(self.limits[1]-self.limits[0]) + self.limits[0]
        else:
            raise ValueError("parameter value {} for {} not normalized".format(p,self))



    def denormalize_from_prior(self, p):
        u"""Take `p` as a probability, and find the value which has that (cumulated) probability in the prior distribution."""
        if 0.0 <= p <= 1.0:
            return self.prior.denormalize_from_prior(p)
        else:
            raise ValueError("parameter value {} for {} not normalized".format(p,self))



    def evaluate_prior(self, p):
        u"""Get the probability of `p` coming from the prior distribution."""
        if p < self.limits[0] or p > self.limits[1]:
            return -np.inf
        elif self.prior:
            return self.prior(p)
        else:
            return 0.0



    @staticmethod
    def load_parameters(value_file, priors_files=None, override=None):
        u"""Return array of :class:`Parameters` as directed by the input files.

        Every key in the `value_file` will produce an entry in the
        returned array, with the inferred starting value, and lower and
        upper limits (space-separated list).

        Where a key is also found in a given file in `priors_files` then a
        :class:`Prior` object will be constructed and attached to the
        parameter.

        If `override` contains a corresponding `(section, name)` key, then
        it will provide the start and limit values for the parameter,
        regardless of the file contents.

        """
        if isinstance(value_file, config.Inifile):
            values_ini = value_file
        else:
            values_ini = config.Inifile(value_file)

        if priors_files:
            priors_data = priors.Prior.load_priors(priors_files)
        else:
            priors_data = {}

        parameters = []
        for (section, name), value in values_ini:
            #override if available
            if (override is not None) and (section, name) in override:
                value = override[(section,name)]

            # parse value line
            start, limits = Parameter.parse_parameter(value)

            # check for prior
            pri = priors_data.get((section, name), None)


            parameters.append(Parameter(section, name,
                                        start, limits, pri))

        # Allow parameters not originally specified in the parameter
        # file to be added on the command line
        if override is not None:
            for (section, name) in override:
                if (section, name) not in parameters:
                    value = override[(section,name)]
                    start, limits = Parameter.parse_parameter(value)
                    pri = priors_data.get((section, name), None)
                    parameters.append(Parameter(section, name,
                                            start, limits, pri))

        return parameters



    @staticmethod
    def parse_parameter(line):
        u"""Interpret a `line` of one to three numbers as the start and range of a parameter.

        In all cases, the return will be a scalar start value and 2-tuple
        of limits.

        With one number, the start value will be taken and `None` returned
        as the limits tuple, indicating that the parameter is to be kept
        constant, equivalently a delta-function distribution.

        With two numbers, they will be taken as the limits and the
        starting value will be half way between the two.

        With three numbers, they will be taken directly as the start
        value, lower limit, and upper limit.

        """
        try:
            values = [float(p) for p in line.split()]
            if len(values) == 1:
                if values[0]==int(values[0]):
                    try:
                        v = int(line)
                        return v, None
                    except ValueError:
                        return values[0], None
                return values[0], None
            elif len(values) == 2:
                return 0.5*(values[0]+values[1]), tuple(values)
            elif len(values) == 3:
                return values[1], (values[0], values[2])
            else:
                raise ValueError("Was expecting 1-3 values for "
                                 "parameter value %s" % (line,))
        except ValueError as error:
            raise ValueError("Unable to parse numeric value for "
                             "parameter value %s, error %s" %
                             (line, error))


def register_new_parameter(options,
                           section,
                           name,
                           min_value,
                           start_value,
                           max_value,
                           prior_name="",
                           prior_args=None):
    """
    Add a parameter to the pipeline from a module setup function.

    We allow modules to define and add parameters to the pipeline,
    either fixed or sampled.  This function can be called to do this.

    It will only affect the pipeline from which setup was called.

    options is the datablock passed to setup(options).
    section is the string setting the datablock section for the new parameter
    name is the name of the new parameter
    min_value is the lower limit of the parameter's range
    max_value is the upper limit of the parameter's range
    prior_name (optional) is a string - see prior.py
    prior_args (optional) is any additional arguments to create the prior
    """
    from .pipeline import LikelihoodPipeline

    n = len(LikelihoodPipeline.pipeline_being_set_up)

    if n == 0:
        print("Tried to register_new_parameter from a module not in a pipeline: skipping.")
        return
    elif n != 1:
        raise RuntimeError("Multiple pipelines are currently being set up, cannot register new parameter")

    if len(LikelihoodPipeline.module_being_set_up) != n:
        raise RuntimeError("Internal error: not clear what module is being set up.")

    # Now we have established the numbers are right we
    # register the parameter
    pipeline = LikelihoodPipeline.pipeline_being_set_up[0]
    caller = LikelihoodPipeline.module_being_set_up[0]


    pipeline._register_new_parameter(caller,
                           section,
                           name,
                           start_value,
                           min_value,
                           max_value,
                           prior_name,
                           prior_args)
