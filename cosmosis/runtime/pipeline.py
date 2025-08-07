#coding: utf-8

u"""Definition of :class:`Pipeline` and the specialization :class:`LikelihoodPipeline`."""


import os
import sys
import numpy as np
import time
import collections
import warnings
import traceback
import io
from . import config
from . import parameter
from . import prior
from . import module
from . import logs
from ..datablock.cosmosis_py import block, section_names
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass


class PipelineResults(object):
    def __init__(self, vector, number_extra):
        self.vector = vector
        self.prior = -np.inf
        self.extra = [np.nan for i in range(number_extra)]
        self.block = None
        self.post = -np.inf
        self.like = -np.inf

    def set_like(self, L):
        self.like = L
        self.post = self.prior + self.like
    
    def log(self, i):
        v = ", ".join(str(x) for x in self.vector)
        logs.info(f"Posterior = {self.post} for parameter set #{i}: [{v}]")




PIPELINE_INI_SECTION = "pipeline"
NO_LIKELIHOOD_NAMES = "no_likelihood_names_sentinel"
TRAINING_INI_SECITON = "training"

class MissingLikelihoodError(Exception):

    u"""Class to throw if there are not enough data for a likelihood method.

    This class is specifically designed to be thrown during a
    `pipeline.likelihood()` call when there are insufficient data so
    that the test sampler can provide a detailed report to the user
    about the problem (recall that the typical lifetime of a cosmosis
    run involves running the test sampler first to weed out such
    problems, before a full run takes place).

    """

    def __init__(self, message, data):
        u"""Data to provide to the test sampler in case of likelihood data problems.

        The `message` is for the user generally; for use of `data` see
        the `test_sampler.execute()` method, currently at
        `samplers/test/test_sampler.py:29`.
        
        """
        super(MissingLikelihoodError, self).__init__(message)
        self.pipeline_data = data


class LimitedSizeDict(collections.OrderedDict):
    "From http://stackoverflow.com/questions/2437617/limiting-the-size-of-a-python-dictionary"
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        collections.OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class SlowSubspaceCache(object):
    """
    This tool analyzes pipelines to determine which of their parameters
    are fast and which are slow, and then caches the results of new sets
    of slow parameters so that if only fast parameters have changed the 
    pipeline can be much faster.
    """
    def __init__(self, first_fast_module=None, cache_size_limit=3,):
        self.current_hash = np.nan
        self.analyzed = False
        self.cache = LimitedSizeDict(size_limit=cache_size_limit)
        self.first_fast_module = first_fast_module

    def clear_cache(self):
        self.cache.clear()
        self.current_hash = np.nan

    def hash_slow_parameters(self, block):
        """This is not a general block hash! 
        It just looks at the slow parameters.
        """
        pairs = block.keys()
        pairs.sort()
        values = []
        for section,name in pairs:
            if (section,name) not in self.slow_params:
                continue
            value = block[section, name]
            if not np.isscalar(value):
                #non-scalar in the block - this must
                #reflect some feature not coded yet
                warnings.warn("Vector parameter(s) in the input means that fast-slow split will not work - this can be fixed - please open an issue if it affects you.  Or set fast_slow=F")
                return np.nan # 
            values.append(value)
        values = tuple(values)
        return hash(values)

    def start_pipeline(self, initial_block):
        # We may be in the process of analyzing the pipeline
        # the first time.
        if not self.analyzed:
            return 0
        #
        new_hash = self.hash_slow_parameters(initial_block)
        cached = self.cache.get(new_hash)
        if cached is None:
            self.current_hash = new_hash
            first_module = 0
        else:            
            # Now we need to use the old cached results in the new block.
            # We put everything from the old block into the new block,
            # EXCEPT for the fast parameters themselves.
            for section,name in cached.keys():
                if (section,name) not in self.fast_params:
                    initial_block[section,name] = cached[section,name]
            first_module = self.split_index
        return first_module

    def next_module_results(self, module_index, block):
        if not self.analyzed:
            return

        if module_index != self.split_index-1:
            return

        self.cache[self.current_hash] = block.clone()

    def analyze_pipeline(self, pipeline, all_params=False, grid=False):
        """
        Analyze the pipeline to determine how best to split it into two parts,
        a slow beginning and a faster end, so that we can vary parameters

        """

        #Run the sampler twice first, once to make sure
        #everything is initialized and once to do timing.
        #Use the pipeline starting parameter since that is
        #likely more typical than any random starting position
        start = pipeline.start_vector()

        if pipeline.has_run:
            print("Pipeline has been run once already so no further initialization steps")
        else:
            print("")
            print("Analyzing pipeline to determine fast and slow parameters (because fast_slow=T in [pipeline] section)")
            print("")
            print("Running pipeline once to make sure everything is initialized before timing.")
            print("")
            pipeline.posterior(start)
        print("")
        print("Running the pipeline again to determine timings and fast-slow split")
        print("")
        #Run with timing but make sure to re-set back to the original setting
        #of timing after it finishes
        #This will also print out the timing, which is handy.
        original_timing = pipeline.timing
        pipeline.timing = True
        #Also get the datablock since it contains a log
        #of all the parameter accesses
        _, _, block = pipeline.posterior(start, return_data=True)
        pipeline.timing = original_timing
        timings = pipeline.timings

        if timings is None:
            raise ValueError("Pipeline did not complete, so cannot do fast/slow")

        #Now we have the datablock, which has the log in it, and the timing.
        #The only information that can be of relevance is the fraction
        #of the time pipeline before a given module, and the list of 
        #parameters accessed before that module.
        #We are going to need to go through the log and figure out the latter of
        #these
        if all_params:
            params = pipeline.parameters
        else:
            params = pipeline.varied_params
        first_use = block.get_first_parameter_use(params)
        first_use_count = [len(f) for f in first_use.values()]
        if sum(first_use_count)!=len(params):
            used = sum(first_use.values(), [])
            unused = [p for p in params if p not in used]
            raise ValueError("Tried to do fast-slow split but not all varied parameters ever used in the pipeline: (unused: {})".format(unused))
        print("\n")
        print("Parameters first used in each module:")
        for f, n in zip(first_use.items(), first_use_count):
            name, params = f
            print("{} - {} parameters:".format(name, n))
            for p in params:
                print("     {}--{}".format(*p))

        # Now we have a count of the number of parameters and amount of 
        # time used before each module in the pipeline
        # So we can divide up the parameters into fast and slow.
        self.split_index = self._choose_fast_slow_split(first_use_count, timings, grid)

        self.full_time = sum(timings)
        self.slow_time = sum(timings[:self.split_index])
        self.fast_time = sum(timings[self.split_index:])

        print("Time for full pipeline:  {:.2f}s".format(self.full_time))
        print("Time for slow pipeline:  {:.2f}s".format(self.slow_time))
        print("Time for fast pipeline:  {:.2f}s".format(self.fast_time))
        time_save_percent = 100-100*self.fast_time/self.full_time
        print("Time saving: {:.2f}%".format(time_save_percent))

        self.worth_splitting = time_save_percent > 10.

        if (not self.worth_splitting) and (self.first_fast_module is not None):
            print("")
            print("No significant time saving (<10%) from a fast-slow split.")
            print("But you told me specifically which module to use, so I will")
            print("split the pipeline anyway, though it may be slower.")
            print("You may wish to reconsider your choices, but I do what I'm told.")
            print("")
            self.worth_splitting = True
        elif not self.worth_splitting:
            print("")
            print("No significant time saving (<10%) from a fast-slow split.")
            print("Not splitting pipeline into fast and slow parts.")
            print("")
            self.split_index = len(timings)

        self.slow_modules = self.split_index
        self.fast_modules = len(pipeline.modules) - self.slow_modules
        self.slow_params = sum(list(first_use.values())[:self.split_index], [])
        self.fast_params = sum(list(first_use.values())[self.split_index:], [])

        if self.worth_splitting:
            print("")
            print("Based on this we have decided: ")
            print("   Slow modules ({}):".format(self.slow_modules))
            for module in pipeline.modules[:self.split_index]:
                print("        %s"% module.name)
            print("   Fast modules ({}):".format(self.fast_modules))
            for module in pipeline.modules[self.split_index:]:
                print("        {}".format(module.name))
            print("   Slow parameters ({}):".format(len(self.slow_params)))
            for param in self.slow_params:
                print("        {}--{}".format(*param))
            print("   Fast parameters ({}):".format(len(self.fast_params)))
            for param in self.fast_params:
                print("        {}--{}".format(*param))
            print("")
            print("")
        self.analyzed = True

    def _choose_fast_slow_split(self, first_use_count, timings, grid_mode):
        #some kind of algorithm to loop through the modules
        #and work out the time saving if we did the fast/slow from there.
        #at the moment just return the last module where there are any 
        #parameters
        #n = len(slow_count)
        if self.first_fast_module is not None:
            print("You manually told me in the parameter file to use module {} as the first in the fast block".format(self.first_fast_module))
            return self.first_fast_module


        n_step = len(timings)
        n_total = sum(first_use_count)
        T_total = sum(timings)

        if grid_mode:
            print("Analyzing the fast/slow for grid sampler - different rule used for time saving")

        T = np.zeros(n_step)
        for i in range(n_step):
            T_slow = sum(timings[:i])
            n_slow = sum(first_use_count[:i])
            T_fast = T_total - T_slow
            n_fast = n_total - n_slow
            if n_fast==0:
                T[i] = T_total
            elif grid_mode:
                T[i] = (T_slow/T_total)*10**(-n_fast) + (T_fast/T_total)*(1-10**(-n_fast))
            else:
                T[i] = (T_fast*n_fast)/(T_total*n_total) + (T_slow*n_slow)/(n_total*T_total)

        index = T.argmin()
        Tmin = T.min()
        index = (np.where(T==Tmin)[0]).max()
        print("Will use module {} as the first in the fast block".format(index))

        return index


class Pipeline(object):

    u"""A container of :class:`DataBlock`-processing :class:`Module`ʼs.

    Cosmosis consists of a large number of computational modules, which
    the user can arrange into pipelines in the configuration files
    before submitting a run.  This class encapsulates the basic concept
    of a pipeline, and provides methods for setting one up based on
    options and/or initial parameter vectors, and for running the
    pipeline to obtain a refinement of a :class:`DataBlock` (the
    prototypical use-case is Bayesian updating of prior likelihoods,
    though this is handled by the more specialized
    :class:`LikelihoodPipeline` defined below).

    Sometimes it will be possible to optimize execution of a pipeline by
    providing pre-emptive data which can be taken as the result of running
    the first few modules.  In this case a run will ‘bypass’ those modules
    and start the run at a `shortcut_module`, using the `shortcut_data` as
    the initial data set.

    """

    def __init__(self, arg=None, load=True, modules=None, training=False):

        u"""Pipeline constructor.

        The (poorly named) `arg` needs to be some reference to a parameter 
        :class:`Inifile` which includes all the information to form this
        pipeline: it can be a list of filenames (.ini files and such) to
        read for the parameters, or a :class:`Inifile` object directly.

        If `load` is `True` then all the modules in the pipelineʼs
        configuration will be loaded into memory and initialized.

        """
        self.training = training

        if arg is None:
            arg = list()

        if isinstance(arg, config.Inifile):
            self.options = arg
        else:
            self.options = config.Inifile(arg)

        #This will be set later
        self.root_directory = self.options.get("runtime", "root", fallback=os.getcwd())
        self.run_count = 0
        self.run_count_ok = 0

        self.debug = self.options.getboolean(PIPELINE_INI_SECTION, "debug", fallback=False)
        self.timing = self.options.getboolean(PIPELINE_INI_SECTION, "timing", fallback=False)
        shortcut = self.options.get(PIPELINE_INI_SECTION, "shortcut", fallback="")
        if shortcut=="":
            shortcut=None

        self.do_fast_slow = self.options.getboolean(PIPELINE_INI_SECTION, "fast_slow", fallback=False)
        if self.do_fast_slow and shortcut:
            sys.stderr.write("Warning: you have the fast_slow and shortcut options both set, and we can only do one of those at once (we will do shortcut)\n")
            self.do_fast_slow = False
        self.slow_subspace_cache = None #until set in method
        self.first_fast_module = self.options.get(PIPELINE_INI_SECTION, "first_fast_module", fallback="")

        # initialize modules
        self.modules = []
        self.has_run = False
        if modules is not None:
            self.modules = modules

        elif load and PIPELINE_INI_SECTION in self.options.sections():
            module_list = self.options.get(PIPELINE_INI_SECTION,
                                           "modules", fallback="").split()
            if self.training:
                self.train_on_module = self.options.get(TRAINING_INI_SECITON, "train_on_module", fallback="camb")
                if self.train_on_module != "camb":
                    raise ValueError("Currently we only support training the CosmoPower on CAMB!")
                try:
                    index = module_list.index(self.train_on_module)
                    module_list = module_list[:index + 1].copy()
                except ValueError as e:
                    raise Exception(f"In order to train the CosmoPower on {self.train_on_module}, {self.train_on_module} module needs to be in the specified modules.") from e
            self.modules = [
                module.Module.from_options(module_name,self.options,self.root_directory)
                for module_name in module_list
            ]
        else:
            self.modules = []

        self.shortcut_module=0
        self.shortcut_data=None
        self.timings = None

        if self.modules:
            if shortcut is not None:
                try:
                    index = module_list.index(shortcut)
                except ValueError:
                    raise ValueError("You tried to set a shortcut in "
                        "the pipeline but I do not know module %s"%shortcut)
                if index == 0:
                    print("You set a shortcut in the pipeline but it was the first module.")
                    print("It will make no difference.")
                else:
                    print("Shortcut mode activated: the first pipeline run will proceed as normal.")
                    print("Subsequent runs will use module {}, {}, as their first module".format(index,shortcut))
                    print("and use the cached results from the first run for everything before that.")
                    print("except the input parameter values. Think about this to check it's what you want.")
                self.shortcut_module = index
        # Add a log file where failures are sent
        # We only set this is a method instead of an init variable
        # because we may want to activate it on pipelines created
        # by the user and sent to run_cosmosis
        self.failure_log_file = None

    def set_failure_log_file(self, failure_log_file):
        """
        Set the file where the pipeline will log failures.

        Parameters
        ----------
        failure_log_file : file-like object
            The file where the pipeline will log failures.
            This can be a regular file or an MPILogFile object
        """
        self.failure_log_file = failure_log_file

    def find_module_file(self, path):
        u"""Find a module file, which is assumed to be either absolute or relative to COSMOSIS_SRC_DIR"""
        return os.path.join(self.root_directory, path)



    def setup(self):
        u"""Run all of our modulesʼ `setup()` routines."""
        
        if self.timing:
            timings = [time.time()]

        for module in self.modules:
            # identify parameters needed for module setup
            relevant_sections = self.options.sections()

            #We let the user specify additional global sections that are
            #visible to all modules
            global_sections = self.options.get("runtime", "global", fallback=" ")
            for global_section in global_sections.split():
                relevant_sections.append(global_section)

            config_block = config_to_block(relevant_sections, self.options)
            config_block[PIPELINE_INI_SECTION, 'current_module'] = module.name
            # we need to store an ID here to allow a hack to add new parameters
            # from the modules themselves.  Unfortunately cosmosis stores ints, whereas
            # python IDs are long.  So we split the ID into two parts here

            LikelihoodPipeline.pipeline_being_set_up.append(self)
            LikelihoodPipeline.module_being_set_up.append(module.name)
            try:
                module.setup(config_block)
            finally:
                # This should only go wrong if someone is doing something
                # ridiculous, so I won't catch any error in it.
                LikelihoodPipeline.pipeline_being_set_up.remove(self)
                LikelihoodPipeline.module_being_set_up.remove(module.name)

            if self.timing:
                timings.append(time.time())


        logs.overview("Setup all pipeline modules\n")

        if self.timing:
            timings.append(time.time())
            sys.stdout.write("Module timing:\n")
            for name, t2, t1 in zip(self.modules, timings[1:], timings[:-1]):
                sys.stdout.write("%s %f\n" % (name, t2-t1))


    def setup_fast_subspaces(self, all_params=False, grid=False):
        if self.do_fast_slow:
            print("Doing fast/slow parameter splitting")
            if self.first_fast_module:
                for i,module in enumerate(self.modules):
                    if module.name==self.first_fast_module:
                        first_fast_index=i
                        break
                else:
                    raise ValueError("You set first_fast_module={} but never ran that module".format(self.first_fast_module))
            else:
                first_fast_index = None

            self.slow_subspace_cache = SlowSubspaceCache(first_fast_module=first_fast_index)
            self.slow_subspace_cache.analyze_pipeline(self, all_params=all_params, grid=grid)

            if not self.slow_subspace_cache.worth_splitting:
                self.slow_subspace_cache = None
                self.do_fast_slow = False
                return


            self.fast_time = self.slow_subspace_cache.fast_time
            self.slow_time = self.slow_subspace_cache.slow_time
            # This looks a bit weird but makes sure that self.fast_params
            # and self.slow_params contain objects of type Parameter
            # not just the (section,name) tuples.
            if all_params:
                params = self.parameters
            else:
                params = self.varied_params
            self.fast_params = [p for p in params 
                if p in self.slow_subspace_cache.fast_params]
            self.slow_params = [p for p in params 
                if p in self.slow_subspace_cache.slow_params]
            self.fast_param_indices = [params.index(p)
                for p in self.fast_params]
            self.slow_param_indices = [params.index(p)
                for p in self.slow_params]
            self.n_fast_params = len(self.fast_params)
            self.n_slow_params = len(self.slow_params)
        else:
            self.slow_subspace_cache = None


    def cleanup(self):
        u"""Call every `module`ʼs `cleanup` method."""
        for module in self.modules:
            module.cleanup()



    def make_graph(self, data, filename):
        u"""Put a description of a graphical model in the graphviz format
        of a completed datablock `data' that was run on the pipeline, 
        in `filename`, illustrating the data flow of this pipeline.

        Graphviz tools can then be used to generate an actual image.

        The :class:`DataBlock` `data`ʼs attached log is used to determine
        the state of each module, and colourization is used in the graphic
        to indicate this.

        """
        try:
            import pygraphviz as pgv
        except ImportError:
            print("Cannot generate a graphical pipeline; please install the python package pygraphviz (e.g. with pip install pygraphviz)")
            return
        P = pgv.AGraph(directed=True)
        # P = pydot.Cluster(label="Pipeline", color='black',  style='dashed')
        # G.add_subgraph(P)
        def norm_name(name):
            return name #.replace("_", " ").title()
        known_sections = set()
        P.add_node("Sampler", color='Pink', style='filled', group='pipeline',shape='octagon', fontname='Courier')
        for module in self.modules:
            # module_node = pydot.Node(module.name, color='Yellow', style='filled')
            P.add_node(norm_name(module.name), color='lightskyblue', style='filled', group='pipeline', shape='box')
            known_sections.add(norm_name(module.name))
        P.add_edge("Sampler", norm_name(self.modules[0].name), color='lightskyblue', style='bold', arrowhead='none')
        for i in range(len(self.modules)-1):
            P.add_edge(norm_name(self.modules[i].name),norm_name(self.modules[i+1].name), color='lightskyblue', style='bold', arrowhead='none')
        # D = pydot.Cluster(label="Data", color='red', style='dashed')
        # G.add_subgraph(D)
        # #find
        log = [data.get_log_entry(i) for i in range(data.get_log_count())]
        for entry in log:
            if entry!="MODULE-START":
                section = entry[1]
                if section not in known_sections:
                    if section=="Results":
                        pass
                    else:
                        P.add_node(norm_name(section), color='yellow', style='filled', fontname='Courier', shape='box')
                    known_sections.add(section)
        module="Sampler"
        known_edges = set()
        for entry in log:
            if entry[0]=="MODULE-START":
                module=entry[1]
            elif entry[0]=="WRITE-OK" or entry[0]=="REPLACE-OK":
                section=entry[1]
                if (module,section,'write') not in known_edges:
                    P.add_edge(norm_name(module), norm_name(section), color='green')
                    known_edges.add((module,section,'write'))
            elif entry[0]=="READ-OK":
                section=entry[1]
                if (section,module,'read') not in known_edges:
                    P.add_edge((norm_name(section),norm_name(module)), color='grey50')
                    known_edges.add((section,module,'read'))

        P.write(filename)



    def run(self, data_package):
        u"""Run every module, in sequence, on DataBlock `data_package`.

        Apart from that the function goes to a lot of effort to provide
        run-time diagnostic information to the user.

        However, if any module returns anything but a zero status, the
        whole pipeline will cease to run and a :class:`ValueError` will be
        raised.

        Bear in mind the note on short-cuts in the class description
        above: if `shortcut_data` and `shortcut_module` are defined, then
        the pipeline will start at `shortcut_module` with `shortcut_data`
        as the initial parameter vector.

        """
        self.run_count += 1
        modules = self.modules
        if self.timing:
            self.timings = None

        timings = []
        if self.shortcut_module:
            if self.shortcut_data is None:
                first_module = 0
            else:
                first_module = self.shortcut_module
                existing_keys = set(data_package.keys())
                for (sec, name) in self.shortcut_data.keys():
                    if (sec,name) not in existing_keys:
                        data_package[sec, name] = self.shortcut_data[sec,name]
        elif self.slow_subspace_cache:
            first_module = self.slow_subspace_cache.start_pipeline(data_package)
            if first_module != 0 and (self.debug or self.timing):
                logs.debug(f"Quickstarting pipeline from module {first_module} (fast/slow)")
        else:
            first_module = 0

        if self.timing:
            start_time = time.time()

        for module_number, module in enumerate(modules):
            if module_number<first_module:
                continue
            logs.noisy(f"Running module {module}")
            data_package.log_access("MODULE-START", module.name, "")
            if self.timing:
                t1 = time.time()

            status = module.execute(data_package)

            if status is None:
                raise ValueError(("A module you ran, '{}', did not return a proper status value.\n"+
                    "It should return an integer, 0 if everything worked.\n"+
                    "Sorry to be picky but this kind of thing is important.").format(module))

            logs.noisy("Done %.20s status = %d \n" % (module,status))

            if self.timing:
                t2 = time.time()
                timings.append(t2-t1)
                sys.stdout.write("%s took: %.3f seconds\n"% (module,t2-t1))

            if status:
                if logs.is_enabled_for(logs.logging.DEBUG):
                    data_package.print_log()
                    logs.noisy("Because you set debug verbosity I printed a log of "
                                   "all access to data printed above. "
                                   "Look for the word 'FAIL' \n"
                                   "Though the error message could also be "
                                   "somewhere above that.\n")

                logs.warning(f"Error running pipeline ({status}). Returning zero likelihood. Error may be above.")
                if not logs.is_enabled_for(logs.logging.DEBUG):
                    logs.warning("Set log level to 'debug' for more info.")
                return None

            # If we are using a fast/slow split (and we are not already running on a cached subset)
            # Then see if it wants to cache these results
            if self.slow_subspace_cache and first_module==0:
                self.slow_subspace_cache.next_module_results(module_number, data_package)

            # Alternatively we will do the shortcut thing
            elif self.shortcut_module and (not self.has_run) and module_number==self.shortcut_module-1:
                print("Saving shortcut data")
                self.shortcut_data = data_package.clone()

        if self.timing:
            end_time = time.time()
            sys.stdout.write("Total pipeline time: {:.3} seconds\n".format(end_time-start_time))
            self.timings = timings

        logs.noisy("Pipeline ran okay.")
        self.run_count_ok += 1

        data_package.log_access("MODULE-START", "Results", "")
        # return something
        self.has_run = True
        return True

    def clear_cache(self):
        self.slow_subspace_cache.clear_cache()




class LikelihoodPipeline(Pipeline):

    u"""Very specialized pipeline designed specifically for the prototypical case of Bayes-computed posterior distributions.

    The point of a statistical updating pipeline is that the parameters in
    the datablocks passed down the pipe, as well as having currently
    estimated values, also have allowable ranges and possibly other
    constraints which the user may want to tinker before each run.  Thus
    there is a specialized layout of initialization files in the file
    system, and there is a modified expectation on the modules to perform
    simulation, compute the Bayesian evidence, hence log-likelihood.  The
    pipeline itself will aggregate the results and summarize the net
    effect of all the likelihood estimations, and thence compute the
    Bayesian posterior.

    Because of the necessity of working with distributions of values for
    each parameter, rather than just a scalar, the extra information is
    stored in a shadow array—another dictionary with the same keys but a
    complementary set of values to the original ones—of
    :class:`parameter`s to the :class:`datablock` which the base pipeline
    modifies (actually only a subset of them known as the `varied_params`:
    an array which references the interesting parameters in the full set).
    this shadow array (`parameters`) is often referred to simply as `p`,
    and the two arrays frequently need to be ‘zipped’ together and then
    ‘unzipped’ after computations have completed.

    """
    # These class-level variables are used by the
    # register_new_parameter function.
    pipeline_being_set_up = []
    module_being_set_up = []

    def __init__(self, arg=None, id="", override=None, modules=None, load=True, values=None, priors=None, only=None, training=False):
        u"""Construct a :class:`LikelihoodPipeline`.

        The arguments `arg` and `load` are used in the base-class
        initialization (see above).  The `id` is given to our `id_code`
        (which doesnʼt seem to have a purpose), and `override` is a
        dictionary of `(section, name)->value` which will override any
        settings for those parametersʼ values in the initialization files.
        
        """
        super().__init__(arg=arg, load=load, modules=modules, training=training)

        if id:
            self.id_code = "[%s] " % str(id)
        else:
            self.id_code = ""
        self.n_iterations = 0


        if values is None:
            self.values_file = self.options.get(PIPELINE_INI_SECTION, "values", fallback="")
            self.values_filename = self.values_file
        else:
            self.values_file = values
            self.values_filename = None

        if priors is None:
            priors_files = self.options.get(PIPELINE_INI_SECTION,
                                            "priors", fallback="").split()
            self.priors_files = priors_files
        else:
            if isinstance(priors, config.Inifile):
                priors = [priors]
            self.priors_files = priors
        self.parameters = parameter.Parameter.load_parameters(self.values_file,
                                                              self.priors_files,
                                                              override,
                                                              )     
        # We set up the modules first, so that if they want to e.g.
        # add parameters then they can.
        self.setup()

        if only:
            for p in only:
                if p not in self.parameters:
                    raise ValueError("You used --only but the parameter "
                        "you named ({}) is not in the pipeline".format(p))
            for p in self.parameters:
                if p not in only:
                    p.fix()

        self.reset_fixed_varied_parameters()

        self.print_priors()

        #We want to save some parameter results from the run for further output
        extra_saves = self.options.get(PIPELINE_INI_SECTION,
                                       "extra_output", fallback="")

        self.extra_saves = []
        self.number_extra = 0
        for extra_save in extra_saves.split():
            section, name = extra_save.upper().split('/')
            self.extra_saves.append((section, name))
            # To load an array-type extra_output, we should know beforehand its size
            # So, it reads from the name especification
            # e.g. data_vector/2pt_theory#457, in which case #457 specifies the size
            self.number_extra += int(name.split('#')[-1]) if '#' in name else 1


        #pull out all the section names and likelihood names for later

        likelihood_names = self.options.get(PIPELINE_INI_SECTION,
                                            "likelihoods",
                                            fallback=NO_LIKELIHOOD_NAMES)
        if likelihood_names==NO_LIKELIHOOD_NAMES:
            self.likelihood_names = NO_LIKELIHOOD_NAMES
        else:
            self.likelihood_names = likelihood_names.split()

        if self.training:
            self.setup_training()
            
    
    def setup_training(self):
        print("")
        print("Running pipeline once to determine which free parametes are being used in order to correctly setup the CosmoPower")
        print("-----------------------------------------------------------------------------------------------------------------")

        self.nsample = self.options.getint(TRAINING_INI_SECITON, "nsample")
        self.nsample_test = self.options.getint(TRAINING_INI_SECITON, "ntest")
        self.save_name = f'{self.options.get(TRAINING_INI_SECITON, "save_dir")}/cosmopower_inputs'

        # This seems like a hack...
        zmin = self.options.getfloat(TRAINING_INI_SECITON, "zmin")
        zmax = self.options.getfloat(TRAINING_INI_SECITON, "zmax")
        z_param = parameter.Parameter('redshift_as_parameter', 'z', zmin+0.01, (zmin, zmax), None)
        print("")
        print("Created new parameter {} used for CosmoPower training".format(z_param))
        print("    with start:", z_param.start)
        print("    with limits:", z_param.limits)
        print("    with prior:", z_param.prior)
        self.parameters.append(z_param)
        self.reset_fixed_varied_parameters()
            
        test_results = self.run_results([p.start for p in self.varied_params])
        used_params_per_module = test_results.block.get_all_parameter_use(self.varied_params)
        used_params = used_params_per_module[self.train_on_module]
        for p in self.varied_params:    
             if (p.section, p.name) not in used_params:
                p.fix()
        self.reset_fixed_varied_parameters()
        # We reset the run stats so the rest functions as intended.
        self.run_count = 0
        self.run_count_ok = 0
        self.has_run = False
            

    @classmethod
    def from_chain_file(cls, filename, **kwargs):
        from ..utils import extract_inis_from_chain_header, read_chain_header
        # Pull out all of the comment bits in the header of the chain
        # that start with a #
        header = read_chain_header(filename)

        # Parse he header to pull out the three chunks of INI files
        # that we save there
        param_lines = extract_inis_from_chain_header(header, "params")
        value_lines = extract_inis_from_chain_header(header, "values")
        prior_lines = extract_inis_from_chain_header(header, "priors")

        # convert all these into Inifile objects
        params = config.Inifile.from_lines(param_lines)
        values = config.Inifile.from_lines(value_lines)
        priors = config.Inifile.from_lines(prior_lines)

        # Build the pipeline from these
        return cls(arg=params, values=values, priors=priors, **kwargs)


    def print_priors(self):
        u"""Pretty-print a table of priors for human inspection."""
        print("")
        print("Parameter Priors")
        print("----------------")
        if self.parameters:
            n = max([len(p.section)+len(p.name)+2 for p in self.parameters])
        else:
            n=1
        for param in self.parameters:
            s = "{}--{}".format(param.section,param.name)
            print("{0:{1}}  ~ {2}" .format(s, n, param.prior))
        print("")

    def _register_new_parameter(self,
                                module_name,
                                section,
                                name,
                                start,
                                min_value,
                                max_value,
                                prior_name,
                                prior_args):
        # This is designed to be called by the register_new_parameter
        # function below, from modules themselves, to support the case
        # where modules create their own parameter
        if max_value == min_value:
            limits = None
        else:
            limits = (min_value, max_value)
        if prior_name == "":
            prior_obj = None
        else:
            if prior_args is None:
                prior_args = []
            prior_text = prior_name + " " + " ".join([str(x) for x in prior_args])
            prior_obj = prior.Prior.parse_prior(prior_text)
        param = parameter.Parameter(section, name, start, limits, prior_obj)
        print("Pipeline module {} created new parameter {}".format(module_name, param))
        print("    with start:", param.start)
        print("    with limits:", param.limits)
        print("    with prior:", param.prior)
        self.parameters.append(param)

    def reset_fixed_varied_parameters(self):
        u"""Identify the sub-set of parameters which are fixed, and those which are to be varied."""
        self.varied_params = [param for param in self.parameters
                              if param.is_varied()]
        self.fixed_params = [param for param in self.parameters
                             if param.is_fixed()]
        self.nvaried = len(self.varied_params)
        self.nfixed = len(self.fixed_params)



    def parameter_index(self, section, name):
        u"""Return the sequence number of the parameter `name` in `section`.

        If the parameter is not found then :class:`ValueError` will be raised.

        """
        i = self.parameters.index((section, name))
        if i==-1:
            raise ValueError("Could not find index of parameter %s in section %s"%(name, section))
        return i



    def set_varied(self, section, name, lower, upper):
        u"""Indicate that the parameter (`section`, `name`) is to be varied between the `lower` and `upper` bounds."""
        i = self.parameter_index(section, name)
        self.parameters[i].limits = (lower,upper)
        self.reset_fixed_varied_parameters()



    def set_fixed(self, section, name, value):
        u"""Indicate that the parameter (`section`, `name`) must be held fixed at `value`."""
        i = self.parameter_index(section, name)
        self.parameters[i].limits = (value, value)
        self.reset_fixed_varied_parameters()



    def output_names(self):
        u"""Return a list of strings, each the name of a non-fixed parameter."""
        param_names = [str(p) for p in self.varied_params]
        extra_names = []
        for section,name in self.extra_saves:
            if ('#' in name):
                n,l = name.split('#')
                for i in range(int(l)):
                    extra_names.append('{}--{}_{}'.format(section,n,i))
            else:
                extra_names.append('%s--%s'%(section,name))
        return param_names + extra_names



    def randomized_start(self):
        u"""Give each varied parameter an independent random value distributed according
        to the parameter prior.

        The return is a `NumPy` :class:`array` of the random values.

        """
        
        # should have different randomization strategies
        # (uniform, gaussian) possibly depending on prior?
        
        return np.array([p.random_point() for p in self.varied_params])



    def is_out_of_range(self, p):
        u"""Determine if any parameter is not in its allowed range."""
        return any([not param.in_range(x) for
                    param, x in zip(self.varied_params, p)])



    def denormalize_vector(self, p, raise_exception=True):
        u"""Convert an array of normalized parameter values, one for each varied parameter,
        in the range [0.0,1.0] into their original values using only the lower and upper limits of the parameter.
    
        Use denormalize_vector_from_prior to convert according to the prior instead.

        """
        return np.array([param.denormalize(x, raise_exception) for param, x
                         in zip(self.varied_params, p)])



    def denormalize_vector_from_prior(self, p):
        r"""Convert an array of normalized parameter values, one for each varied parameter,
        in the range [0.0,1.0] into their original values according to the prior for each parameter.

        i.e. 
        v -> x  such that \int_{-inf}^{x} p(x') dx' = v

        """
        return np.array([param.denormalize_from_prior(x) for param, x
                         in zip(self.varied_params, p)])



    def normalize_vector(self, p):
        u"""Convert an array of parameter values, one for each varied parameter,
         into a normalized form all in the range [0.0,1.0] using only the lower and upper limits for each parameter."""
        return np.array([param.normalize(x) for param, x
                         in zip(self.varied_params, p)])



    def normalize_matrix(self, c):
        u"""Roughly, return a correlation matrix corresponding to the 
        covariance matrix `c`, of `varied_params` values.

        Except that the elements of `c` are not probabilities but
        dimensional values, and the ‘normalization’ is relative to the
        range of values the ‘covariance’s can take given the lower and
        upper limits on the variates.

        """
        c = c.copy()
        n = c.shape[0]
        assert n==c.shape[1], "Cannot normalize a non-square matrix"
        for i in range(n):
            pi = self.varied_params[i]
            ri = pi.limits[1] - pi.limits[0]
            for j in range(n):
                pj = self.varied_params[j]
                rj = pj.limits[1] - pj.limits[0]
                c[i,j] /= (ri*rj)
        return c



    def denormalize_matrix(self, c, inverse=False):
        u"""Perform the inverse operation to the normalize_matrix function above.

        Note that if `inverse` is `True` the action is *exactly* the same
        as the function above, i.e. it *normalizes* the matrix.

        """
        c = c.copy()
        n = c.shape[0]
        assert n==c.shape[1], "Cannot normalize a non-square matrix"
        for i in range(n):
            pi = self.varied_params[i]
            ri = pi.limits[1] - pi.limits[0]
            for j in range(n):
                pj = self.varied_params[j]
                rj = pj.limits[1] - pj.limits[0]
                if inverse:
                    c[i,j] /= (ri*rj)
                else:
                    c[i,j] *= (ri*rj)
        return c



    def start_vector(self, all_params=False, as_array=True):
        u"""Return a vector of starting values for parameters.

        If `all_params` is specified as `True` then the return will include all
        our `parameters`, otherwise only the varying ones are included.

        If `as_array` is specified as `False` then a Python list is
        returned, otherwise, the default, a NumPy array is returned.

        """
        if all_params:
            p = [param.start for param in self.parameters]
        else:            
            p =[param.start for param in self.varied_params]
        if as_array:
            p = np.array(p)
        return p



    def min_vector(self, all_params=False):
        u"""Return a NumPy array of lower limits for the parameters in the pipeline.

        If `all_params` is specified as `True` then the return will 
        include all parameters, including fixed ones. Otherwise it will just be the 
        varying parameters.

        """
        if all_params:
            return np.array([param.limits[0] for
                 param in self.parameters])
        else:
            return np.array([param.limits[0] for
                         param in self.varied_params])



    def max_vector(self, all_params=False):
        u"""Return a NumPy array of upper limits for the parameters in the pipeline.

        If `all_params` is specified as `True` then the return will 
        include all parameters, including fixed ones. Otherwise it will just be the 
        varying parameters.

        """
        if all_params:
            return np.array([param.limits[1] for
                 param in self.parameters])
        else:
            return np.array([param.limits[1] for
                         param in self.varied_params])

    def build_starting_block(self, p, check_ranges=False, all_params=False):
        u"""Assemble :class:`DataBlock` data based on parameter values in `p`, and return it.


        If `check_ranges` is indicated, the function will return `None` if
        **any** of our parameters are out of their indicated range.

        If `all_params` is indicated, then the `p` run data will be
        assumed to match all the pipeline parameter, including fixed ones.
        Otherwise (the default) it should match the list ‘varied_params’, and all
        of our ‘fixed’ parameters are added to the run-set.

        """
        if check_ranges:
            if self.is_out_of_range(p):
                return None

        data = block.DataBlock()

        if all_params:
            for param, x in zip(self.parameters, p):
                data[param.section, param.name] = x
        else:
            # add varied parameters
            for param, x in zip(self.varied_params, p):
                data[param.section, param.name] = x

            # add fixed parameters
            for param in self.fixed_params:
                data[param.section, param.name] = param.start

        return data


    def run_parameters(self, p, check_ranges=False, all_params=False):
        u"""Assemble :class:`DataBlock` data based on parameter values in `p`, and run the pipeline on those data.

        If `check_ranges` is indicated, the function will return `None` if
        **any** of our parameters are out of their indicated range.

        If `all_params` is indicated, then the `p` run data will be
        assumed to match all the pipeline parameter, including fixed ones.
        Otherwise (the default) it should match the list ‘varied_params’, and all
        of our ‘fixed’ parameters are added to the run-set.

        """
        data = self.build_starting_block(p, check_ranges=check_ranges, all_params=all_params)

        if self.run(data):
            # First run.  If we have not set the likelihood names in the parameter
            # file then get them from the block
            if self.likelihood_names == NO_LIKELIHOOD_NAMES:
                self._set_likelihood_names_from_block(data)
                
            return data
        else:
            sys.stderr.write("Pipeline failed on these parameters: {}\n".format(p))
            if self.failure_log_file is not None:
                msg = " ".join([str(x) for x in p])
                self.failure_log_file.write(msg + "\n")
            return None



    def create_ini(self, p, filename):
        u"""Dump the parameters `p` as a new ini file at `filename`"""
        output = collections.defaultdict(list)
        for param, x in zip(self.varied_params, p):
            output[param.section].append("%s  =  %r    %r    %r\n" % (
                param.name, param.limits[0], x, param.limits[1]))
        for param in self.fixed_params:
            output[param.section].append("%s  =  %r\n" % (param.name, param.start))
        ini = open(filename, 'w')
        for section, params in sorted(output.items()):
            ini.write("[%s]\n"%section)
            for line in params:
                ini.write(line)
            ini.write("\n")
        ini.close()



    def prior(self, p, all_params=False, total_only=True):
        u"""Compute the probability of all values in `p` based on their prior distributions.

        The array `p` should match the length of of all of our `parameters` if
        `all_params` is `True`, and our `varied_params` otherwise.

        If `total_only` is `True` (the default), then the scalar sum of
        all the prior probabilities is returned.  Otherwise a list
        of pairs is returned, with each element a stringified version of
        the parameter name, and the prior probability:
        [(name1, prior1), (name2,prior2), ...]

        """
        if all_params:
            params = self.parameters
        else:
            params = self.varied_params
        priors = [(str(param),param.evaluate_prior(x)) for param,x in zip(params,p)]
        if total_only:
            return sum(pr[1] for pr in priors)
        else:
            return priors

    def run_results(self, p, all_params=False):
        u"""Run the pipeline on the given parameters and get a results object.

        The argument `p` is a vector or list of the input parameters.
        if (as in the default) `all_params` is False, then it should be the
        same length and order as `self.varied_params`.

        Otherwise, if all_params is False, then it should match the length and order
        of `self.parameters`.

        The method returns a `PipelineResults` object with the following attributes:

        * results.post, the posterior (float);

        * results.extra, the reqiured additional output parameters (numpy array)

        * results.prior, the total prior (float)

        * results.block, the updated data block

        If there is a problem anywhere in the computations which does
        *not* cause a run-time exception to be raised—including the case
        where a parameter goes outside of its alloted range—, then
        `-numpy.inf` will be returned as the final posterior (i.e., zero
        probability of this set of parameter values being correct).

        """
        r = PipelineResults(p, self.number_extra)

        priors = self.prior(p, all_params=all_params, total_only=False)
        r.prior = sum(pr[1] for pr in priors)

        if np.isnan(r.prior):
            r.prior = -np.inf

        if not np.isfinite(r.prior):
            logs.info("Proposed outside bounds: prior -infinity")
            return r
        try:
            like, r.extra, r.block = self.likelihood(p, return_data=True, all_params=all_params)
            r.set_like(like)

            if r.block is not None:
                for name,pr in priors:
                    r.block["priors", name] = pr

        except Exception:
            logs.error(f"Exception running likelihood at parameters: {p}."
                            "You should fix this; for now, using zero likelihood.")
            logs.error(traceback.format_exc())
            if self.debug:
                sys.stderr.write("\nBecause you have debug=T I will let this kill the chain.\n")
                sys.stderr.write("The input parameters were:{}\n".format(repr(p)))
                raise

        if np.isnan(r.post):
            r.post = -np.inf

        if np.isnan(r.like):
            r.like = -np.inf
        
        r.log(self.run_count)

        return r




    def posterior(self, p, return_data=False, all_params=False):
        u"""Use the above methods to obtain prior and updated log-likelihoods, sum together to get Bayesian posterior.

        The argument `p` is a vector or list of the input parameters.
        if (as in the default) `all_params` is False, then it should be the
        same length and order as `self.varied_params`.

        Otherwise, if all_params is False, then it should match the length and order
        of `self.parameters`.
        The method returns two or three values depending on `return_data`:

        * The posterior;

        * a vector (NumPy array) of updated parameter values as specified
          in `self.extra_saves`;

        * if `return_data` was specified, the updated data block.

        If there is a problem anywhere in the computations which does
        *not* cause a run-time exception to be raised—including the case
        where a parameter goes outside of its alloted range—, then
        `-numpy.inf` will be returned as the final posterior (i.e., zero
        probability of this set of parameter values being correct).

        """

        r = self.run_results(p, all_params=all_params)
        if return_data:
            return r.post, r.extra, r.block
        else:
            return r.post, r.extra



    def _set_likelihood_names_from_block(self, data):
        likelihood_names = []
        for _,key in data.keys(section_names.likelihoods):
            if key.endswith("_like"):
                name = key[:-5]
                likelihood_names.append(name)
        self.likelihood_names = likelihood_names

        # Tell the user what we found.
        logs.noisy("Using likelihooods from first run:")
        for name in self.likelihood_names:
            logs.noisy(f" - {name}")
        if not self.likelihood_names:
            logs.noisy(" - (None found)")

    def _extract_likelihoods(self, data):
        "Extract the likelihoods from the block"

        section_name = section_names.likelihoods

        # loop through named likelihoods and sum their values
        likelihoods = []
        for likelihood_name in self.likelihood_names:
            try:
                L = data.get_double(section_name,likelihood_name+"_like")
                likelihoods.append(L)
                logs.noisy(f"Likelihood {likelihood_name} = {L}")
            # Complain if not found
            except block.BlockError:
                raise MissingLikelihoodError(likelihood_name, data)

        # Total likelihood
        like = sum(likelihoods)

        if np.isnan(like):
            like = -np.inf

        return like

        
    def likelihood(self, p, return_data=False, all_params=False):
        u"""Run the simulation pipeline, computing any log-likelihoods in the pipeline 
        given the given input parameter values, and return the sum of these.

        The parameter vector `p` must match the length of `self.varied_params`,
        unless `all_params` is specified as `True` in which case it must match
        `self.parameters', i.e. must correspond to the complete parameter set.

        If `return_data` are requested, then the updated data block will
        be returned as the third return item.

        The return will consist of two or three items, depending on
        `return_data`:

          * A scalar holding the sum of all computed log-likelihoods of
            the updated parameter value vector;

          * a vector (NumPy array) of updated parameter values as
            specified in `self.extra_saves`;

          * if `return_data` was specified, the updated data block.

        If anything goes wrong in any of the computation which does *not*
        result in a run-time error being raised (which would include the
        case of a parameter going outside of its stipulated limits), then
        the returned log-likelihood will be `-np.inf`.

        """

        data = self.run_parameters(p, all_params=all_params)
        if data is None:
            if return_data:
                return -np.inf, [np.nan for i in range(self.number_extra)], data
            else:
                return -np.inf, [np.nan for i in range(self.number_extra)]

        like = self._extract_likelihoods(data)

        extra_saves = []
        for option in self.extra_saves:
            try:
                #JAZ - should this be just .get(*option) ?
                # If the # symbol is present, we have an array-type extra_output and it is
                # read as numpy.ndarray. We append all of its elements to the extra_saves.
                # If the total array size does not match the sum of the sizes indicated in the
                # name #, an exception will be raised.
                if('#' in option[1]):
                    value = data.get(option[0], option[1].split('#')[0])
                else:
                    value = data.get(*option)

            except block.BlockError:
                value = np.nan

            if (type(value)==np.ndarray):
                for e in value:
                    extra_saves.append(e)
            else:
                extra_saves.append(value)
                # ---------------------------- otavio end ---------------------------

        self.n_iterations += 1
        if return_data:
            return like, extra_saves, data
        else:
            return like, extra_saves

    @classmethod
    def from_likelihood_function(cls, log_likelihood_function, param_ranges, priors=None, debug=False, derived=None):
        """
        Make a pipeline from a simple likelihood function.

        Parameters
        ----------
        log_likelihood_function : function
            A function that takes a list of parameters and returns either a single
            number (the log-likelihood) or a tuple of two things, the first being
            the log-likelihood and the second being a dictionary of extra derived
            parameters.
        param_ranges : list of tuples
            A list of tuples of the form (min, starting_point, max) for each parameter.
        priors : list of tuples, optional
            A dictionary if priors i the form name:prior (see documentation for prior format).
            If not specified then uniform priors are used.
        debug : bool, optional
            If True then exceptions in the likelihood function will be raised.
            If False then they will be ignored and the likelihood will be set to -inf.
        derived : list of strings, optional
            A list of names of derived parameters to save in the output.

        Returns
        -------
        pipeline : LikelihoodPipeline
            A pipeline object that can be run.
        """
        nparam = len(param_ranges)

        def setup(options):
            return {}

        def execute(block, config):
            parameters = np.array([block["params", f"p{i}"]  for i in range(nparam)])
            try:
                p = log_likelihood_function(parameters)
            except:
                if debug:
                    raise
                else:
                    return 1

            if isinstance(p, tuple):
                like = p[0]
                extra = p[1]

                if not isinstance(extra, dict):
                    raise ValueError("The extra output from the likelihood function must be a dictionary")

                for key, value in extra.items():
                    block['derived', key] = value
            else:
                like = p

            block['likelihoods', 'a_like'] = like

            return 0

        mod = module.FunctionModule("log_likelihood_function", setup, execute)

        parameters = {
            f"p{i}": f"{param_ranges[i][0]} {param_ranges[i][1]} {param_ranges[i][2]}"
            for i in range(nparam)
        }

        debug_str = "T" if debug else "F"
        extra_saves = " ".join([f"derived/{d}" for d in derived]) if derived is not None else ""

        config = {
            "pipeline": {
                "likelihoods": "a",
                "debug": debug_str,
                "extra_output": extra_saves,
            }
        }

        values = {
            "params": parameters
        }

        if priors is not None:
            priors = [{
                "params": priors
            }]

        pipeline = cls(config, values = values, modules=[mod], priors=priors)
        return pipeline



def config_to_block(relevant_sections, options):
    u"""Compose :class:`DataBlock` of parameters only in `relevant_sections` of complete set of `options`."""
    config_block = block.DataBlock()

    for (section, name), value in options:
        if section in relevant_sections:
            # add back a default section?
            val = options.gettyped(section, name)
            if val is not None:
                config_block.put(section, name, val)
    # For book-keeping stuff later we also need to record
    # which keys were in the default section.  If
    # we want to keep track of unused parameters in
    # Module.access_check_report then we need to ensure
    # we don't include parameters that are only in the section
    # because they are in every section.  So we keep track of
    # such parameters by including them here
    for (key, val) in options.items("DEFAULT"):
        config_block.put("_cosmosis_default_section", key, val)
    return config_block
