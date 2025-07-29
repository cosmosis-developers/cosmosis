#!/usr/bin/env python
import sys
import configparser
import argparse
import os
import pdb
import cProfile
import contextlib
from .runtime.config import Inifile, CosmosisConfigurationError
from .runtime.pipeline import LikelihoodPipeline
from .runtime import mpi_pool
from .runtime import logs
from .runtime import process_pool
from .runtime.utils import ParseExtraParameters, stdout_redirected, import_by_path, under_over_line, underline, overline
from .samplers.sampler import Sampler, ParallelSampler, Hints
from . import output as output_module
from .runtime.handler import activate_segfault_handling
from .version import __version__


RUNTIME_INI_SECTION = "runtime"


def demo_1_special (inifile):
    if "demo1.ini" in inifile:
        print("""
Congratulations: you have just run cosmosis demo one!

You can make plots of the outputs of this using this command:
    cosmosis-postprocess demos/demo1.ini -o outputs/demo1

Then you can try out the other demos...
... and read the information about plotting their output and what they are doing online.
Please get in touch with any problems, ideally by filing an Issue. Thanks!
""")


def demo_10_special(inifile):
    if   "demo10.ini" in inifile   and   not os.getenv ("HALOFIT", ""):
        print()
        print("Welcome to demo10!")
        print()
        print("**PLEASE NOTE**:")
        print()
        print("There are two flavours of this demo, selected through an ")
        print("environment variable called `HALOFIT'; this variable is not ")
        print("currently set, so we are giving it the value `takahashi'.")

        os.environ ["HALOFIT"] = "takahashi"



def demo_20a_special(inifile):
    if  "demo20a.ini" in inifile:
        print ()
        print ("You have completed demo20a, now run demo20b and compare")
        print ("results with demo5!")


    if   "demo20b.ini" in inifile   and   not os.path.isfile ("./demo20a.txt"):
        print(under_over_line("YOU MUST RUN demo20a BEFORE YOU CAN RUN demo20b.", chr='*'))

def sampler_main_loop(sampler, output, pool, is_root):
    # Run the sampler until convergence
    # which really means "finished" here - 
    # a sampler can "converge" just by reaching the 
    # limit of the number of samples it is allowed.
    if is_root:
        while not sampler.is_converged():
            sampler.execute()
            #Flush any output. This is to stop
            #a problem in some MPI cases where loads
            #of output is built up before being written.
            if output:
                output.flush()
        # If we are in parallel tell the other processors to end the 
        # loop and prepare for the next sampler
        if pool and sampler.is_parallel_sampler:
            pool.close()
    else:
        if sampler.is_parallel_sampler:
            sampler.worker()



def write_header_output(output, params, values, pipeline, values_override=None):
    # If there is an output file, save the ini information to
    # it as well.  We do it here since it's nicer to have it
    # after the sampler options that get written in sampler.config.
    # Create a buffer to store the output:
    output.comment("START_OF_PARAMS_INI")
    comment_wrapper = output.comment_file_wrapper()
    params.write(comment_wrapper)
    output.comment("END_OF_PARAMS_INI")
    # Do the same with the values file.
    # Unfortunately that means reading it in again;
    # if we ever refactor this bit we could eliminate that.
    if isinstance(values, Inifile):
        values_ini = values
    elif values is None:
        values_ini=Inifile(pipeline.values_filename, override=values_override)
    else:
        values_ini=values
    output.comment("START_OF_VALUES_INI")
    values_ini.write(comment_wrapper)
    output.comment("END_OF_VALUES_INI")

    # And the same with the priors
    output.comment("START_OF_PRIORS_INI")
    for priors_file in pipeline.priors_files:
        if isinstance(priors_file, Inifile):
            prior_ini = priors_file
        else:
            prior_ini=Inifile(priors_file)
        prior_ini.write(comment_wrapper)
    output.comment("END_OF_PRIORS_INI")

def setup_output(sampler_class, sampler_number, ini, pool, sample_methods, sample_method, resume, output):

    output_original = output

    needs_output = sampler_class.needs_output and \
       (pool is None or pool.is_master() or sampler_class.parallel_output)

    if not needs_output:
        return None

    if output_original is None:
        #create the output files and methods.
        try:
            output_options = dict(ini.items('output'))
        except configparser.NoSectionError:
            raise ValueError("ERROR:\nFor the sampler (%s) you chose in the [runtime] section of the ini file I also need an [output] section describing how to save results\n\n"%sample_method)
        #Additionally we tell the output here if
        #we are parallel or not.
        if (pool is not None) and (sampler_class.parallel_output):
            output_options['rank'] = pool.rank
            output_options['parallel'] = pool.size

        number_samplers = len(sample_methods)

        #Give different output filenames to the different sampling steps
        #Only change if this is not the last sampling step - the final
        #one retains the name in the output file.
        # Change, e.g. demo17.txt to demo17.fisher.txt
        if ("filename" in output_options) and (sampler_number<number_samplers-1):

            sampler_count = sample_methods.count(sample_method)
            filename = output_options['filename']
            filename, ext = os.path.splitext(filename)
            if sampler_count == 1:
                filename += '.' + sampler_class.name
                filename += ext
            else:
                sampler_repeat_index = 0
                for i in range(sampler_number):
                    if sample_methods[i] == sample_method:
                        sampler_repeat_index += 1
                filename += '.' + sampler_class.name + '.' + str(sampler_repeat_index)
                filename += ext
            output_options['filename'] = filename


        if ("filename" in output_options):
            print("* Saving output -> {}".format(output_options['filename']))

        #Generate the output from a factory
        output = output_module.output_from_options(output_options, resume)
    elif isinstance(output_original, output_module.OutputBase):
        output = output_original
    elif isinstance(output_original, str):
        if output_original == "astropy":
            output = output_module.AstropyOutput()
        elif output_original == "none":
            output = output_module.NullOutput()
        elif output_original == "in_memory":
            output = output_module.InMemoryOutput()
        else:
            raise ValueError(f"Unknown output option {output_original}")
    else:
        raise ValueError(f"Unknown output type {type(output_original)}")
        

    output.metadata("sampler", sample_method)

    return output


def run_cosmosis(ini, pool=None, pipeline=None, values=None, priors=None, override=None,
                 profile_mem=0, profile_cpu="", variables=None, only=None, output=None,
                 train_cosmopower=False, train_cosmopower_force=False):
    """
    Execute cosmosis.

    Parameters
    ----------
    ini: str, cosmosis.Inifile, or None
        The parameter file from which to build the cosmosis run. If set to a string the
        file is read from disc. If set to None, the other parameters must contain
        all the required CosmoSIS parameters.
    
    pool: None, cosmosis.MPIPool, or cosmosis.process_pool.Pool
        A pool object to enable multi-process parallel execution. If left as the default
        None then the code is run with a single process (though modules may still run
        using OpenMP parallelism).
    
    pipeline: None or cosmosis.LikelihoodPipeline
        If set, ignore the pipeline definition in the ini file and use this pipeline
        instead.
    
    values: None or dict[str, str]->str
        If set, ignore the numerical parameter values in the ini file and use these 
        instead.
    
    priors: None or dict[str, str]->str
        If set, ignore the prior values in the ini file and use these 
        instead. 

    override: None or dict[str, str]->str
        If set, override parameter values in the ini file from the dictionary.
    
    profile_mem: int
        If changed from the default zero value, print a memory profile every 
        profile_mem seconds.

    profile_cpu: str
        If changed from the default empty string, print CPU profile information
        and also save to the named file. If running in parallel, save to 
        {profile_cpu}.{rank}.
    
    variables: None or dict[str, str]->str
        If set, override variable values in the ini file from the dictionary.

    only: None or str
        If set, fix all the variable values except the one supplied.

    output: None or cosmosis.Output
        If set, use this output object to save the results. If not set, create
        an output object from the ini file.

    train_cosmopower: bool
        If set, it will run a specific set of samplers and modules up to and including CAMB to 
        train the CosmoPower emulator, which can then be used as a drop-in replacement to CAMB.
        Will refuse to re-train if emulator exists.

    train_cosmopower_force: bool
        Force re-train even if emulator exists.
    """
    no_subprocesses = os.environ.get("COSMOSIS_NO_SUBPROCESS", "") not in ["", "0"]

    smp = isinstance(pool, process_pool.Pool)

    # Load configuration.
    is_root = (pool is None) or pool.is_master()
    ini_is_str = isinstance(ini, str)
    ini_original = ini
    output_original = output
    ini = Inifile(ini, override=override, print_include_messages=is_root)

    pre_script = ini.get(RUNTIME_INI_SECTION, "pre_script", fallback="")
    post_script = ini.get(RUNTIME_INI_SECTION, "post_script", fallback="")

    verbosity = ini.get(RUNTIME_INI_SECTION, "verbosity", fallback="")
    
    if not verbosity:
        if ini.has_option("pipeline", "quiet"):
            quiet = ini.getboolean("pipeline", "quiet", fallback=False)
            verbosity = "quiet" if quiet else "standard"
        else:
            verbosity = ini.get(RUNTIME_INI_SECTION, "output", fallback="standard")
    logs.set_verbosity(verbosity)

    if ini.has_option("pipeline", "quiet") and is_root:
        logs.warning("Deprecated: The [pipeline] quiet option is deprecated.  Set [runtime] verbosity instead.")

    if is_root and pre_script:
        if no_subprocesses:
            print("Warning: subprocesses not allowed on this system as")
            print("COSMOSIS_NO_SUBPROCESS variable was set.")
            print("Ignoring pre-script.")
        else:
            status = os.WEXITSTATUS(os.system(pre_script))
            if status:
                raise RuntimeError("The pre-run script {} retuned non-zero status {}".format(
                    pre_script, status))

    if is_root and profile_mem:
        from cosmosis.runtime.memmon import MemoryMonitor
        # This launches a memory monitor that prints out (from a new thread)
        # the memory usage every profile_mem seconds
        mem = MemoryMonitor.start_in_thread(interval=profile_mem)

    if profile_cpu:
        profile = cProfile.Profile()
        profile.enable()

    training = False
    if train_cosmopower:
        root_dir_name = ini.get("training", "save_dir")
        save_name = f'{root_dir_name}/cosmopower_emulator_fixed_params.pkl'
        if train_cosmopower:
            if os.path.exists(save_name) and not train_cosmopower_force:
                if is_root:
                    print("Tried to train the CosmoPower emulator, but emulator already exists. If you want to really re-train the emulator, add option --force. Quitting.")
                raise SystemExit
            else:
                training = True
                if is_root:
                    print("Training the CosmoPower emulator!" if not train_cosmopower_force else "Re-training the CosmoPower emulator!")

    # Create pipeline.
    if pipeline is None:
        cleanup_pipeline = True
        pool_stdout = ini.getboolean(RUNTIME_INI_SECTION, "pool_stdout", fallback=False)
        if is_root:
            if ini_is_str:
                print(underline(f"Setting up pipeline from parameter file {ini_original}"))
            else:
                print(underline(f"Setting up pipeline from pre-constructed configuration"))

        if is_root or pool_stdout:
            pipeline = LikelihoodPipeline(ini, override=variables, values=values, only=only, priors=priors, training=training)
        else:
            if pool_stdout:
                pipeline = LikelihoodPipeline(ini, override=variables, values=values, only=only, priors=priors, training=training)
            else:
                # Suppress output on everything except the root process
                with stdout_redirected():
                    pipeline = LikelihoodPipeline(ini, override=variables, values=values, only=only, priors=priors, training=training)

        if pipeline.do_fast_slow:
            pipeline.setup_fast_subspaces()
    else:
        # We should not cleanup a pipeline which we didn't make
        cleanup_pipeline = False

    # This feature lets us import additional samplers at runtime
    sampler_files = ini.get(RUNTIME_INI_SECTION, "import_samplers", fallback="").split()
    for i, sampler_file in enumerate(sampler_files):
        # give the module a new name to avoid name clashes if people
        # just call their thing by the same name
        import_by_path('additional_samplers_{}'.format(i), sampler_file)
    

    # determine the type(s) of sampling we want.
    sample_methods = ini.get(RUNTIME_INI_SECTION, "sampler", fallback="test").split()

    # if we run the training option, we override the selected samplers, as they need to be fixed!
    if training:
        sample_methods = ["latinhypercube", "cosmopower"]

    for sample_method in sample_methods:
        if sample_method not in Sampler.registry:
            raise ValueError("Unknown sampler method %s" % (sample_method,))

    #Get that sampler from the system.
    sampler_classes = [Sampler.registry[sample_method] for sample_method in sample_methods]

    if pool:
        if not any(issubclass(sampler_class,ParallelSampler) for sampler_class in sampler_classes):
            if len(sampler_classes)>1:
                raise ValueError("None of the samplers you chose support parallel execution!")
            else:
                raise ValueError("The sampler you chose does not support parallel execution!")
        for sampler_class in sampler_classes:
            if isinstance(pool, process_pool.Pool) and issubclass(sampler_class,ParallelSampler) and not sampler_class.supports_smp:
                name = sampler_class.__name__[:-len("Sampler")].lower()
                raise ValueError("Sorry, the {} sampler does not support the --smp flag.".format(name))

    number_samplers = len(sampler_classes)

    if ini.has_option("pipeline", "failure_log"):
        filename = ini.get("pipeline", "failure_log")
        failure_log_file = mpi_pool.MPILogFile(filename, pool)
        pipeline.set_failure_log_file(failure_log_file)
    else:
        failure_log_file = None



    #To start with we do not have any estimates of 
    #anything the samplers might give us like centers
    #or covariances. 
    distribution_hints = Hints()

    #Now that we have a sampler we know whether we will need an
    #output file or not.  By default new samplers do need one.
    for sampler_number, (sampler_class, sample_method) in enumerate(
            zip(sampler_classes, sample_methods[:])):
        sampler_name = sampler_class.__name__[:-len("Sampler")].lower()

        # The resume feature lets us restart from an existing file.
        # It's not fully rolled out to all the suitable samplers yet though.
        resume = ini.getboolean(RUNTIME_INI_SECTION, "resume", fallback=False)

        # Polychord, multinest, and nautilus have their own internal
        # mechanism for resuming chains.
        if sampler_class.internal_resume:
            resume2 = ini.getboolean(sampler_name, "resume", fallback=False)
            resume = resume or resume2

            if resume and is_root:
                print(f"Resuming sampling using {sampler_name} internal mechanism, "
                      "so starting a new output chain file.")

            # Tell the sampler to resume directly
            if not ini.has_section(sampler_name):
                ini.add_section(sampler_name)
            ini.set(sampler_name, "resume", str(resume))

            # Switch off the main cosmosis resume mechanism
            resume = False

        # Not all samplers can be resumed.
        if resume and not sampler_class.supports_resume:
            print("NOTE: You set resume=T in the [runtime] section but the sampler {} does not support resuming yet.  I will ignore this option.".format(sampler_name))
            resume=False
        

        if is_root:
            print("****************************************************")
            print("* Running sampler {}/{}: {}".format(sampler_number+1,number_samplers, sampler_name))
            if pool and smp:
                print(f"* Using multiprocessing (SMP) with {pool.size} processes.")
            elif pool:
                print(f"* Using MPI with {pool.size} processes.")
            else:
                print("* Running in serial mode.")

        output = setup_output(sampler_class, sampler_number, ini, pool, sample_methods, sample_method, resume, output_original)

        if is_root:
            print("****************************************************")

        #Initialize our sampler, with the class we got above.
        #It needs an extra pool argument if it is a ParallelSampler.
        #All the parallel samplers can also act serially too.
        if pool and sampler_class.is_parallel_sampler:
            sampler = sampler_class(ini, pipeline, output, pool)
        else:
            sampler = sampler_class(ini, pipeline, output)
         
        #Set up the sampler - for example loading
        #any resources it needs or checking the ini file
        #for additional parameters.
        sampler.distribution_hints.update(distribution_hints)
        sampler.config()

        # Potentially resume
        if resume and sampler_class.needs_output and \
            sampler_class.supports_resume and \
            (is_root or sampler_class.parallel_output):
           sampler.resume()

        if output:
            write_header_output(output, ini, values, pipeline, values_override=variables)

        sys.stdout.flush()
        sys.stderr.flush()


        sampler_main_loop(sampler, output, pool, is_root)
        distribution_hints.update(sampler.distribution_hints)

        # get total number of evaluations on all MPI processes
        if pool is None:
            run_count_total = pipeline.run_count
            run_count_ok_total = pipeline.run_count_ok
        else:
            run_count_total = pool.allreduce(pipeline.run_count)
            run_count_ok_total = pool.allreduce(pipeline.run_count_ok)
        
        if is_root and sampler_name != 'test':
            logs.overview(f"Total posterior evaluations = {run_count_total} across all processes")
            logs.overview(f"Successful posterior evaluations = {run_count_ok_total} across all processes")
            if output:
                output.final("evaluations", run_count_total)
                output.final("successes", run_count_ok_total)
                output.final("complete", "1")


        if output:
            output.close()

    if cleanup_pipeline:
        pipeline.cleanup()

    if failure_log_file is not None:
        failure_log_file.close()

    if profile_cpu:
        profile.disable()
        if (pool is not None) and (not smp):
            profile_name = profile_cpu + f'.{pool.rank}'
        else:
            profile_name = profile_cpu
        profile.dump_stats(profile_name)
        profile.print_stats("cumtime")

    if is_root and profile_mem:
        mem.stop()

    # Extra-special actions we take to help new users playing with the demos
    demo_1_special(str(ini))
    demo_10_special(str(ini))
    demo_20a_special (str(ini))

    # User can specify in the runtime section a post-run script to launch.
    # In general this may be less useful than the pre-run script, because
    # often chains time-out instead of actually completing.
    # But we still offer it
    if post_script and is_root:
        # This decodes the exist status
        if no_subprocesses:
            print("Warning: subprocesses not allowed on this system as")
            print("COSMOSIS_NO_SUBPROCESS variable was set.")
            print("Ignoring post-script.")
        else:
            status = os.WEXITSTATUS(os.system(post_script))
            if status:
                sys.stdout.write("WARNING: The post-run script {} failed with error {}".format(
                    post_script, status))

    if isinstance(output_original, str):
        if output_original == "astropy":
            return 0, output.table
        else:
            return 0, output
    else:
        return 0


def make_graph(inifile, dotfile, params=None, variables=None):
    """
    Make a graphviz "dot" format file, describing the pipeline
    and how data is passed from section to section.

    Requires pygraphviz.

    Parameters
    ----------

    inifile: str
        A path to a pipeline file or an Inifile object

    dotfile: str
        Path to the new graph output file

    params: dict or None
        Dictionary of parameter overrides

    variables: dict or None
        Dictionary of value overrides
    """
    ini = Inifile(inifile, override=params)
    pipeline = LikelihoodPipeline(ini, override=variables)
    data = pipeline.run_parameters(pipeline.start_vector())
    pipeline.make_graph(data, dotfile)


# Make this global because it is useful for testing
parser = argparse.ArgumentParser(description="Run a pipeline with a single set of parameters", add_help=True)
parser.add_argument("inifile", help="Input ini file of parameters")
parser.add_argument("--mpi",action='store_true',help="Run in MPI mode.")
parser.add_argument("--smp",type=int,default=0,help="Run with the given number of processes in shared memory multiprocessing (this is experimental and does not work for multinest).")
parser.add_argument("--pdb",action='store_true',help="Start the python debugger on an uncaught error. Only in serial mode.")
parser.add_argument("--segfaults", "--experimental-fault-handling", action='store_true',help="Activate a mode that gives more info on segfault")
parser.add_argument("--mem", type=int, default=0, help="Print out memory usage every this many seconds from root process")
parser.add_argument("-p", "--params", nargs="*", action=ParseExtraParameters, help="Override parameters in inifile, with format section.name1=value1 section.name2=value2...")
parser.add_argument("-v", "--variables", nargs="*", action=ParseExtraParameters, help="Override variables in values file, with format section.name1=value1 section.name2=value2...")
parser.add_argument("--only", nargs="*", help="Fix all parameters except the ones listed")
parser.add_argument("--graph", type=str, default='', help="Do not run a sampler; instead make a graphviz dot file of the pipeline")
parser.add_argument('--version', action='version', version=__version__, help="Print out a version number")
parser.add_argument('--profile' , help="Save profiling (timing) information to this named file")
parser.add_argument('--train', action='store_true', help="Train the CosmoPower emulator")
parser.add_argument('-f', '--force', action='store_true', help="Force train the CosmoPower emulator (will overwrite existing emulator)")


@contextlib.contextmanager
def run_under_debugger():
    try:
        yield
    except Exception as error:
        print("There was an exception - starting python debugger because you ran with --pdb")
        print(error)
        pdb.post_mortem()




def main():
    try:
        args = parser.parse_args(sys.argv[1:])

        if args.graph:
            make_graph(args.inifile, args.graph, args.params, args.variables)
            return 0

        if args.segfaults:
            activate_segfault_handling()

        # initialize parallel workers
        if args.mpi:
            with mpi_pool.MPIPool() as pool:
                return run_cosmosis(ini=args.inifile, pool=pool, override=args.params, profile_mem=args.mem, profile_cpu=args.profile, variables=args.variables, only=args.only, train_cosmopower=args.train, train_cosmopower_force=args.force)
        elif args.smp:
            with process_pool.Pool(args.smp) as pool:
                return run_cosmosis(ini=args.inifile, pool=pool, override=args.params, profile_mem=args.mem, profile_cpu=args.profile, variables=args.variables, only=args.only, train_cosmopower=args.train, train_cosmopower_force=args.force)
        elif args.pdb:
            with run_under_debugger():
                return run_cosmosis(ini=args.inifile, pool=None, override=args.params, profile_mem=args.mem, profile_cpu=args.profile, variables=args.variables, only=args.only, train_cosmopower=args.train, train_cosmopower_force=args.force)
        else:
            return run_cosmosis(ini=args.inifile, pool=None, override=args.params, profile_mem=args.mem, profile_cpu=args.profile, variables=args.variables, only=args.only, train_cosmopower=args.train, train_cosmopower_force=args.force)

    except CosmosisConfigurationError as e:
        print(e)
        return 1


if __name__=="__main__":
    status = main()
    sys.exit(status)
