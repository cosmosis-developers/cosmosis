from .runtime import Inifile, MPIPool
from .main import run_cosmosis, run_under_debugger
from .runtime.utils import underline
import time
import os
import yaml
import sys
import warnings
import subprocess
import contextlib

class UniqueKeyLoader(yaml.SafeLoader):
    """
    This is a YAML loader that raises an error if there are duplicate keys.

    It is based on the discussion here:
    https://gist.github.com/pypt/94d747fe5180851196eb
    """
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            if ":merge" in key_node.tag:
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)

def load_yaml(stream):
    """
    Load YAML markup from a stream, with a check for duplicate keys.

    Parameters
    ----------
    stream : file
        The open file-like object to load from

    Returns
    -------
    data : object
        The dict or list loaded from the YAML file
    """
    return yaml.load(stream, Loader=UniqueKeyLoader)


def pipeline_after(params, after_module, modules):
    """
    Insert item(s) in the pipeline after the given module.

    Parameters
    ----------
    params : Inifile
        The parameter file to modify
    after_module : str
        The name of the module after which to insert the new modules
    modules : list
        The list of modules to insert

    Returns
    -------
    None
    """
    pipeline = params['pipeline', 'modules']
    pipeline = pipeline.split()
    if isinstance(modules, str):
        modules = [modules]

    i = pipeline.index(after_module)

    for m in modules[::-1]:
        pipeline.insert(i+1, m)
    params.set('pipeline', 'modules', ' '.join(pipeline))

def pipeline_before(params, before_module, modules):
    """
    Insert item(s) in the pipeline before the given module.

    If before_module is "$" then the item is inserted at the end

    Parameters
    ----------
    params : Inifile
        The parameter file to modify
    before_module : str 
        The name of the module before which to insert the new modules
    modules : list
        The list of modules to insert
    """
    if isinstance(modules, str):
        modules = [modules]

    pipeline = params['pipeline', 'modules']
    pipeline = pipeline.split()
    i = pipeline.index(before_module)

    for m in modules[::-1]:
        pipeline.insert(i, m)
    params.set('pipeline', 'modules', ' '.join(pipeline))

def pipeline_replace(params, original_module, modules):
    """
    Replace an item in the pipeline with one or more new items.

    If the original module is not found then this will fail.

    Parameters
    ----------
    params : Inifile
        The parameter file to modify
    original_module : str
        The name of the module to replace
    modules : list
        The list of modules to insert

    Returns
    -------
    None
    """
    pipeline_after(params, original_module, modules)
    pipeline_delete(params, [original_module])

def pipeline_delete(params, modules):
    """
    Delete one or more items in the pipeline.

    Parameters
    ----------
    params : Inifile
        The parameter file to modify
    modules : list
        The list of modules to delete

    Returns
    -------
    None
    """
    if isinstance(modules, str):
        modules = [modules]
    pipeline = params['pipeline', 'modules']
    pipeline = pipeline.split()

    for m in modules:
        pipeline.remove(m)
    params.set('pipeline', 'modules', ' '.join(pipeline))

def pipeline_append(params, modules):
    """
    Append item(s) to the end of the pipeline.

    Parameters
    ----------
    params : Inifile
        The parameter file to modify
    modules : list
        The list of modules to append
    
    Returns
    -------
    None
    """
    pipeline = params['pipeline', 'modules']
    pipeline = pipeline.split()

    for m in modules:
        pipeline.append(m)
    params.set('pipeline', 'modules', ' '.join(pipeline))

def pipeline_prepend(params, modules):
    """
    Prepend item(s) to the start of the pipeline.

    Parameters
    ----------
    params : Inifile
        The parameter file to modify
    modules : list

    Returns
    -------
    """
    pipeline = params['pipeline', 'modules']
    pipeline = pipeline.split()

    for m in modules[::-1]:
        pipeline.insert(0, m)
    params.set('pipeline', 'modules', ' '.join(pipeline))


def apply_update(ini, update, is_params=False):
    """
    Apply an update to a parameters, value, or priors configuration.

    Parameters
    ----------
    ini : Inifile
        The values or priors file to update
    update : str
        The update to apply. This should be of the form:
        section.option=value
        sampler=sampler_name
        del section.option
        del section
    """
    if "=" in update:
        keys, value = update.split("=", 1)
        keys = keys.strip()
        value = value.strip()
        if keys == "sampler":
            if not is_params:
                raise ValueError("You can only set the sampler in the parameters file")
            ini.set("runtime", "sampler", value.strip())
        else:
            section, option = keys.split(".", 1)
            section = section.strip()
            option = option.strip()
            if not ini.has_section(section) and section != "DEFAULT":
                ini.add_section(section)
            ini.set(section, option, value)
    elif update.startswith("del"):
        cmd, keys = update.split(maxsplit=1)
        if cmd not in ["del", "delete"]:
            raise ValueError(f"Unknown command {cmd}")
        if "." in keys:
            section, option = keys.split('.', 1)
            section = section.strip()
            option = option.strip()
            ini.remove_option(section, option)
        else:
            keys = keys.strip().split()
            if len(keys) > 1:
                raise ValueError("Can only delete one section at a time")
            else:
                # delete entire section
                ini.remove_section(keys[0])
    else:
        raise ValueError(f"Unknown update {update}")

def apply_updates(ini, updates, is_params=False):
    """
    Apply a list of updates to a parameters, values, or priors configuration.

    Parameters
    ----------
    ini : Inifile
        The values or priors file to update
    updates : list
        See apply_update for details.
    
    Returns
    -------
    None
    """
    for update in updates:
        try:
            apply_update(ini, update, is_params=is_params)
        except:
            raise ValueError(f"Malformed update: {update}")
        

def apply_pipeline_update(ini, update):
    """
    Apply an update to the list of modules forming the pipeline.
    
    Available actions are:
    - after <module> <new_module> [<new_module> ...]
    - before <module> <new_module> [<new_module> ...]
    - replace <module> <new_module> [<new_module> ...]
    - del <module> [<module> ...]
    - append <new_module> [<new_module> ...]
    - prepend <new_module> [<new_module> ...]

    Parameters
    ----------
    ini : Inifile
        The parameters file to update
    update : list
        The update to apply.  This should be a list of strings
        of the form [action, module, *values]
        values can be a single string or a list of strings,
        which will be joined with spaces.

    """
    action = update[0]
    if action == "after":
        pipeline_after(ini, update[1], update[2:])
    elif action == "before":
        pipeline_before(ini, update[1], update[2:])
    elif action == "replace":
        pipeline_replace(ini, update[1], update[2:])
    elif action == "del" or action == "delete":
        pipeline_delete(ini, update[1:])
    elif action == "append":
        pipeline_append(ini, update[1:])
    else:
        raise ValueError(f"Unknown pipeline action {action}")


def apply_pipeline_updates(params, pipeline_updates):
    """
    Apply a list of pipeline updates.

    Parameters
    ----------
    params : Inifile
        The parameters file to update
    pipeline_updates : list
        The list of updates to apply.  Each update should be a list of strings.
        See apply_pipeline_update for details.

    Returns
    -------
    None
    """
    for update in pipeline_updates:
        apply_pipeline_update(params, update.split())



@contextlib.contextmanager
def temporary_environment(env):
    """
    Temporary set the environment variables in the given dictionary.
    Once the with block is exited the environment is restored to its
    original state.

    Parameters
    ----------
    env : dict
        A dictionary of environment variables to set

    Returns
    -------
    None
    """
    original_environment = os.environ.copy()
    try:
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_environment)

def set_output_dir(params, name, output_dir, output_name):
    """
    Modify a parameters file to set the output directory and file names to be
    based on the run name and output directory.

    Parameters
    ----------
    params : Inifile
        The parameters file to update
    name : str
        The name of the run
    output_dir : str
        The output directory to use
    output_name : str
        The format string to use to generate the output file name, using {name}
        for the run name.
        
    Returns
    -------
    None
    """
    # Always override the output file and various other auxiliary files
    if not params.has_section("output"):
        params.add_section("output")
    if not params.has_section("test"):
        params.add_section("test")

    output_name_base = output_name.format(name=name)
    output_name = output_name_base + ".txt"
    params.set("output", "filename", os.path.join(output_dir, output_name))
    params.set("test", "save_dir", os.path.join(output_dir, output_name_base))

    # Polychord and Multinest have their own output files. They are handled slightly
    # differently. It would be nicer to put all these files in specific directories
    # instead of just using specific base names, and I will do that at the next breaking
    # change, but right now doing that would mean I had to change the multinest sampler
    # to create the directories, which would be backwards incompatible.
    if params.has_section("multinest"):
        params.set("multinest", "multinest_outfile_root", os.path.join(output_dir, f"{output_name_base}.multinest"))

    if params.has_section("polychord"):
        params.set("polychord", "polychord_outfile_root", f"{output_name_base}.polychord")
        params.set("polychord", "base_dir", output_dir)

def build_run(name, run_info, runs, components, output_dir, submission_info, output_name="{name}"):
    """
    Generate a dictionary specifying a CosmoSIS run from a run_info dictioary.

    Parameters
    ----------
    name : str
        The name of the run
    run_info : dict
        A dictionary specifying the run.  This can have the following keys:
        - base : the name of the base parameter file OR
        - parent : the name of the parent run to use as a base (one of these is required)
        - params : a list of updates to apply to the parameters file (optional)
        - values : a list of updates to apply to the values file (optional)
        - priors : a list of updates to apply to the priors file (optional)
        - pipeline : a list of updates to apply to the pipeline (optional)
        - components : a list of components to include (optional)
        - env : a dictionary of environment variables to set (optional)
    runs : dict
        A dictionary of previously built runs
    components : dict
        A dictionary of previously built components
    output_dir : str
        The output directory to use
    submission_info : dict
        A dictionary of submission information
    output_name : str
        The format string to use to generate the output file name, using {name}
        for the run name.

    Returns
    -------
    """
    run = run_info.copy()

    # We want to delay expanding environment variables so that child runs
    # have a chance to override them. So we set no_expand_vars=True on all of these
    if "base" in run_info:
        params = Inifile(run_info["base"], print_include_messages=False, no_expand_vars=True)
    elif "parent" in run_info:
        try:
            parent = runs[run_info["parent"]]
        except KeyError:
            warnings.warn(f"Run {name} specifies parent {run_info['parent']} but there is no run with that name yet")
            return None
        params = Inifile(parent["params"], print_include_messages=False, no_expand_vars=True)
    else:
        warnings.warn(f"Run {name} specifies neither 'parent' nor 'base' so is invalid")
        return None
    
    # Build environment variables
    # These are inherited from the parent run, if there is one,
    # and then updated with any specific to this run, which can overwrite.
    # env vars are only applied right at the end when all runs are collected
    if "parent" in run_info:
        env_vars = parent["env"].copy()
    else:
        env_vars = {}
    env_vars.update(run_info.get("env", {}))
    run["env"] = env_vars

    # Build values file, which is mandatory
    if "parent" in run_info:
        values = Inifile(parent["values"], print_include_messages=False, no_expand_vars=True)
    else:
        values_file = params.get('pipeline', 'values')
        values = Inifile(values_file, print_include_messages=False, no_expand_vars=True)

    # Build optional priors file
    if "parent" in run_info:
        priors = Inifile(parent["priors"], print_include_messages=False, no_expand_vars=True)
    elif "priors" in params.options("pipeline"):
        priors_file = params.get('pipeline', 'priors')
        priors = Inifile(priors_file, print_include_messages=False, no_expand_vars=True)
    else:
        priors = Inifile(None, no_expand_vars=True)

    # Make a list of all the modifications to be applied
    # to the different bits of this pipeline

    param_updates = []
    value_updates = []
    prior_updates = []
    pipeline_updates = []

    # First from any generic components specified
    for component in run_info.get("components", []):
        component_info = components[component]
        param_updates.extend(component_info.get("params", []))
        value_updates.extend(component_info.get("values", []))
        prior_updates.extend(component_info.get("priors", []))
        pipeline_updates.extend(component_info.get("pipeline", []))

    # And then for anything specific to this pipeline
    param_updates.extend(run_info.get("params", []))
    value_updates.extend(run_info.get("values", []))
    prior_updates.extend(run_info.get("priors", []))
    pipeline_updates.extend(run_info.get("pipeline", []))

    # Now apply all the steps
    apply_updates(params, param_updates, is_params=True)
    apply_updates(values, value_updates)
    apply_updates(priors, prior_updates)
    apply_pipeline_updates(params, pipeline_updates)

    output_name = run_info.get("output_name", output_name)
    set_output_dir(params, name, output_dir, output_name)

    # Finally, set the submission information
    submission_info = submission_info.copy()
    submission_info["output_dir"] = output_dir
    submission_info.update(run_info.get("submission", {}))
    run["submission"] = submission_info

    run["params"] = params
    run["values"] = values
    run["priors"] = priors

    return run


def expand_environment_variables(runs):
    """
    For each run, expand any environment variables in the all three parameter files.

    Environment variables are only supported in values, not in keys or sections.

    Parameters
    ----------
    runs : dict
        A dictionary of runs, keyed by name
    """
    for run in runs.values():
        # This sets environment varibles for the duration of the with block
        with temporary_environment(run["env"]):

            # We apply this to all the different ini files
            for ini in [run["params"], run["values"], run["priors"]]:
                for section in ini.sections():
                    # Now we can expand all the actual options
                    for option in ini.options(section):
                        value = ini.get(section, option)
                        new_value = os.path.expandvars(value)
                        if new_value != value:
                            ini.set(section, option, new_value)

def parse_yaml_run_file(run_config):
    """
    Parse a yaml file (or process a previously parsed file)
    containing a list of runs to perform.

    The file should have the following format:

    output_dir: <output directory>

    include: <list of other run files to include>

    components:
        <name of component>:
            params:
            - <list of updates to apply to the parameters file>
            values:
            - <list of updates to apply to the values file>
            priors:
            - <list of updates to apply to the priors file>
            pipeline:
            - <list of updates to apply to the pipeline>
    

    runs:
        - name: <name of the run>
        base: <name of the base parameter file>
        parent: <name of the parent run>
        components:
        - <list of components to include from the list above>
        env:
        <dictionary of environment variables to set>
        params:
        - <list of updates to apply to the parameters file>
        values:
        - <list of updates to apply to the values file>
        priors:
        - <list of updates to apply to the priors file>
        pipeline:
        - <list of updates to apply to the pipeline>

    submission:
        submit: a command to submit runs from batch files (default sbatch)
        cancel: a command to cancel runs from batch files (default scancel)
        template: a template for the batch file (default is a SLURM one suitable for NERSC)
        # remaining variables are passed to the template and can be overridden in specific runs
        time: Wall time for the job, e.g. 00:30:00
        nodes: Number of nodes, e.g. 1
        tasks: Number of tasks in total, e.g. 1
        cores_per_task: Number of cores (threads) per task, e.g. 1
        queue: Queue to submit to, e.g. regular

                
    Parameters
    ----------
    run_config : str or dict
        The name of the file to parse, or a previously loaded configuration

    Returns
    -------
    runs : dict
        A dictionary of runs, keyed by name
    
    components : dict
        A dictionary of components, keyed by name
    """
    if isinstance(run_config, dict):
        info = run_config
    else:
        with open(run_config, 'r') as f:
            info = load_yaml(f)
    
    output_dir = info.get("output_dir", ".")
    output_name = info.get("output_name", "{name}")

    include = info.get("include", [])
    if isinstance(include, str):
        include = [include]

    # Can include another run file, which we deal with
    # recursively.  
    runs = {}
    components = {}
    for include_file in include:
        inc_runs, inc_comps = parse_yaml_run_file(include_file)
        components.update(inc_comps)
        runs.update(inc_runs)

    # But we override the output directory
    # of any imported runs with the one we have here   
    for name, run in runs.items():
        set_output_dir(run["params"], name, output_dir, output_name)
    
    # deal with re-usable components
    components.update(info.get("components", {}))

    submission_info = info.get("submission", {})

    # Build the parameter, value, and prior objects for this run
    for run_dict in info["runs"]:
        name = run_dict["name"]
        runs[name] = build_run(name, run_dict, runs, components, output_dir, submission_info, output_name)

    # Only now do we expand environment variables in the runs.  This gives the child runs
    # a chance to override the environment variables of their parents.
    expand_environment_variables(runs)

    return runs, components


def show_run(run):
    """
    Print a complete description of the run to the screen

    Parameters
    ----------
    run : dict
        The run to print
    
    Returns
    -------
    None
    """
    print(underline(f"Run {run['name']}", '='))
    print(underline("Parameters"))
    run["params"].write(sys.stdout)
    print("")
    print(underline("Values"))
    run["values"].write(sys.stdout)
    print("")
    print(underline("Priors"))
    run["priors"].write(sys.stdout)
    print("")

def perform_test_run(run, use_pdb=False):
    """
    Launch a run under the "test" sampler, which just runs the pipeline
    and does not do any sampling.

    Parameters
    ----------
    run : dict
        The run to perform
    
    Returns
    -------
    status: int
        The exit status of the run
    """
    params = run["params"]
    values = run["values"]
    priors = run["priors"]
    env = run["env"]
    params.set("runtime", "sampler", "test")
    params.set("pipeline", "debug", "T")
    params.set("runtime", "verbosity", "debug")
    params.set("runtime", "resume", "F")

    with temporary_environment(env):
        if use_pdb:
            with run_under_debugger():
                return run_cosmosis(params, values=values, priors=priors)
        else:
            return run_cosmosis(params, values=values, priors=priors)

def chain_status(filename, include_comments=False):
    n = 0
    line = ""
    for line in open(filename):
        if include_comments or not line.startswith("#"):
            n += 1
    # if the last line
    complete = line.startswith("#complete=1")
    last_update_time = os.path.getmtime(filename)
    time_ago_seconds = time.time() - last_update_time
    time_ago_minutes = time_ago_seconds / 60
    return n, complete, time_ago_minutes

def show_run_status(runs, names=None):
    """
    Report the status of the output of one or more runs.

    Parameters
    ----------
    runs : dict
        A dictionary of runs, keyed by name
    names : list
        The names of the runs to report on.  If None then all runs are reported on.
    
    Returns
    -------
    None
    """
    if not names:
        names = runs.keys()

    for name in names:
        run = runs[name]
        sampler = run["params"].get("runtime", "sampler")
        if sampler == "test":
            output_dir = run["params"].get("test", "save_dir")
            if os.path.exists(output_dir):
                print(f"🟢 {name} has been run [test sampler]")
            else:
                print(f"🔴 {name} has not been run [test sampler]")
        else:
            output_file = run["params"].get("output", "filename")
            if os.path.exists(output_file):
                n, complete, last_update = chain_status(output_file)
                if complete:
                    print(f"🟢 {name} output complete with {n} samples, updated {last_update:.1f} minutes ago")
                elif n:
                    print(f"🟡 {name} output exists with {n} samples, updated {last_update:.1f} minutes ago")
                else:
                    print(f"🟠 {name} output exists with 0 samples, updated {last_update:.1f} minutes ago")
            else:
                print(f"🔴 {name} output missing with 0 samples")


def launch_run(run, mpi=False):
    """
    Launch a CosmoSIS run.

    Parameters
    ----------
    run : dict
        The run to launch
    
    Returns
    -------
    status: int
        The exit status of the run
    """
    params = run["params"]
    values = run["values"]
    priors = run["priors"]
    env = run["env"]

    with temporary_environment(env):
        if mpi:
            with MPIPool() as pool:
                return run_cosmosis(params, values=values, priors=priors, pool=pool)
        else:
            return run_cosmosis(params, values=values, priors=priors)


def submit_run(run_file, run):
    """
    Subnmit a CosmoSIS run using a batch system such as SLURM.
    """

    submission_info = run["submission"]
    template = submission_info["template"]
    keys = submission_info.copy()
    name = run["name"]
    keys["job_name"] = name
    output_dir = keys["output_dir"]

    # We use the campaign program to actually run the jobs
    keys["command"]  = f"cosmosis-campaign {run_file} --run {name} --mpi"

    # choose where the jobs stdout / stderr should go
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    keys["log"] = os.path.join(output_dir, "logs", f"{name}.log")

    # fill in the template
    sub_script = template.format(**keys).lstrip()

    # write the submission file to a batch subdir
    os.makedirs(os.path.join(output_dir, "batch"), exist_ok=True)
    sub_file = os.path.join(output_dir, "batch", f"{name}.sub")
    with open(sub_file, "w") as f:
        f.write(sub_script)

    # Actually submit the job using slurm or similar
    submit = submission_info.get("submit", "sbatch")
    subprocess.check_call(f"{submit} {sub_file}", shell=True)
    print(f"Submitted {sub_file}\nJob output in", keys["log"])



import argparse
parser = argparse.ArgumentParser(description="Manage and launch CosmoSIS runs")
parser.add_argument("run_config", help="The yaml file containing the runs to perform")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--list", "-l", action="store_true", help="List all available runs")
group.add_argument("--cat", "-c",  type=str, help="Show a single run")
group.add_argument("--status", "-s", default="_unset", nargs="*", help="Show the status of a single run, or all runs if called with no argument")
group.add_argument("--run", "-r",  help="Run the named run")
group.add_argument("--test", "-t",  help="Test the named run")
group.add_argument("--submit", "-x",  help="Submit the named run to a batch system")
parser.add_argument("--mpi", action="store_true", help="Use MPI to launch the runs")
parser.add_argument("--pdb",  action="store_true", help="When testing a run, enter the debugger in case of failure")



def main(args):
    runs, _ = parse_yaml_run_file(args.run_config)

    if args.mpi and not args.run:
        raise ValueError("MPI can only be used when running a single run")

    status_set = args.status != "_unset"

    # The various command line options, which are mutually exclusive.
    # This is enforced by the parser above.
    if args.list:
        for name in runs:
            print(name)
    elif args.cat:
        show_run(runs[args.cat])
    elif args.test:
        perform_test_run(runs[args.test], args.pdb)
    elif status_set:
        show_run_status(runs, args.status)
    elif args.run:
        launch_run(runs[args.run], mpi=args.mpi)
    elif args.submit:
        submit_run(args.run_config, runs[args.submit])



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

