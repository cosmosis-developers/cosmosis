#coding: utf-8


u"""Definition of `everythingisnan`, :class:`ParseExtraParameters`, :func:`mkdir`, :class:`Timer`, :func:`symmetrized_matrix` and :func:`symmetric_positive_definite_inverse`."""


import numpy as np
import argparse
import os
import errno
from timeit import default_timer
import sys
from contextlib import contextmanager
import tempfile
import subprocess
from functools import wraps
import pathlib
try:
    import dulwich
except:
    dulwich = None


class EverythingIsNan:

    u"""An object which, when iterated over or indexed directly, always returns NumPyʼs `np.nan` as a value."""
    
    def __getitem__(self, param):
        u"""Just return NaN."""
        return np.nan


everythingIsNan = EverythingIsNan()



def datablock_to_astropy(block):
    from .datablock import names
    cosmo = names.cosmological_parameters
    from astropy.cosmology import FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM, w0waCDM, Flatw0waCDM

    # Required parameters that must be in the data block
    # for this to work
    omega_lambda = block[cosmo, 'omega_lambda']
    omega_m = block[cosmo, 'omega_m']
    omega_b = block[cosmo, 'omega_b']
    H0 = block[cosmo, 'h0'] * 100
    m_nu = block[cosmo, 'mnu']

    # Optional parameters with simple default values
    omega_k = block.get_double(cosmo, 'omega_k', 0.0)
    w0 = block.get_double(cosmo, 'w', -1.0)
    wa = block.get_double(cosmo, 'wa', 0.0)

    kw = {
        "H0": H0,
        "Ob0": omega_b,
        "Om0": omega_m,
        "m_nu": m_nu,
    }

    if omega_k == 0:
        if wa == 0:
            if w0 == -1.0:
                model = FlatLambdaCDM
            else:
                model = FlatwCDM
                kw["w0"] = w0
        else:
            model = Flatw0waCDM
            kw["w0"] = w0
            kw["wa"] = wa
    else:
        if wa == 0:
            if w0 == -1.0:
                model = LambdaCDM
                kw["Ode0"] = omega_lambda
            else:
                model = wCDM
                kw["Ode0"] = omega_lambda
                kw["w0"] = w0
        else:
            model = w0waCDM
            kw["Ode0"] = omega_lambda
            kw["w0"] = w0
            kw["wa"] = wa

    return model(**kw)



class ParseExtraParameters(argparse.Action):

    u"""Extended command-line argument parser :class:`Action` which knows how to read arguments of the form ‘section.name=value’."""

    def __call__(self, parser, args, values, option_string=None):
        if getattr(args, self.dest, self.default) is not None:
            parser.error(option_string + " appears several times")
        result = {}
        for arg in values:
            section, param_value = arg.split('.',1)
            param,value = param_value.split('=',1)
            result[(section,param)] = value
        setattr(args, self.dest, result)



def mkdir(path):
    u"""Ensure that all the components in the `path` exist in the file system.

    If there is a file in the file system blocking the creation of a
    directory there, a :class:`ValueError` will be raised.  Any other
    problem will raise an underlying `os` exception.

    """
    #This is much nicer in python 3.
    # Avoid trying to make an empty path
    if not path.strip():
        return
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if os.path.isdir(path):
                #error is that dir already exists; fine - no error
                pass
            else:
                #error is that file with name of dir exists already
                raise ValueError("Tried to create dir %s but file with name exists already"%path)
        elif error.errno == errno.ENOTDIR:
            #some part of the path (not the end) already exists as a file 
            raise ValueError("Tried to create dir %s but some part of the path already exists as a file"%path)
        else:
            #Some other kind of error making directory
            raise


class Timer(object):
    u"""Object to be use with `with` instruction, so that when enclosed code completes a message will appear with the elapsed wall-clock time."""
    def __init__(self, msg):
        u"""Set `msg` as a label which identifies the code block being timed."""
        self.msg = msg

    def __enter__(self):
        u"""Start the timer running."""
        self.start = default_timer()
        return self

    def __exit__(self, *args):
        u"""Print out the elapsed time."""
        interval = default_timer() - self.start
        print("Time taken by step '{}': {}".format(self.msg, interval))



def symmetrized_matrix(U):
    u"""Return a new matrix like `U`, but with upper-triangle elements copied to lower-triangle ones."""
    M = U.copy()
    inds = np.triu_indices_from(M,k=1)
    M[(inds[1], inds[0])] = M[inds]
    return M



def symmetric_positive_definite_inverse(M):
    u"""Compute the inverse of a symmetric positive definite matrix `M`.

    A :class:`ValueError` will be thrown if the computation cannot be
    completed.

    """
    import scipy.linalg
    U,status = scipy.linalg.lapack.dpotrf(M)
    if status != 0:
        raise ValueError("Non-symmetric positive definite matrix")
    M,status = scipy.linalg.lapack.dpotri(U)
    if status != 0:
        raise ValueError("Error in Cholesky factorization")
    M = symmetrized_matrix(M)
    return M

# These parts from:
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    from .runtime import logs
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        log_level = logs.get_level()
        logs.set_level(51)
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            logs.set_level(log_level)








def read_comment_section(filename):
    """Pull out the comments section from the top of a chain file"""
    lines = []
    for line in open(filename):
        if not line.startswith('#'):
            break
        lines.append(line)
    return lines


def extract_section(lines, section):
    """Extract the PARAMS, VALUES, or PRIORS section from a group
    of comment lines from a chain file"""
    start = "## START_OF_{}_INI".format(section).upper()
    end = "## END_OF_{}_INI".format(section).upper()
    in_section = False
    output_lines = []
    for line in lines:
        if line.startswith(start):
            in_section = True
            continue
        elif line.startswith(end):
            break
        elif in_section:
            output_lines.append(line[3:])
    return output_lines

def save_section(lines, section, prefix):
    """Save a group of lines to a file"""
    filename = "{}_{}.ini".format(prefix, section)
    open(filename,'w').writelines(lines)

def extract_params(chain, prefix):
    """ Extract the parameters, values, and priors files from file "chain"
    and save them to prefix_params.ini etc. """
    lines = read_comment_section(chain)

    for section in ['params', 'values', 'priors']:
        section_lines = extract_section(lines, section)
        save_section(section_lines, section, prefix)


def tempdir_safe_docker():
    # TemporaryDirectory is in the tempfile standard library package in Python 3
    return tempfile.TemporaryDirectory()


class PriorFunction(object):
    def __init__(self, chain_filename):
        "Build a prior function from an input chain file"
        import cosmosis.runtime.parameter


        # Read the parameter files to temporary files
        # they will be deleted afterwards
        with tempdir_safe_docker() as tmpdir:
            dirname = tmpdir + os.path.sep
            extract_params(chain_filename, dirname+"tmp")

            # Load in the extracter parameters, including their priors
            self.all_params = cosmosis.runtime.parameter.Parameter.load_parameters(
                dirname+'tmp_values.ini',
                priors_files=[dirname+'tmp_priors.ini'])

        # Pull out the priors from the parameters
        self.all_priors = [p.prior for p in self.all_params]
        self.varied_priors = [p.prior for p in self.all_params if p.is_varied()]

    def _evaluate(self, p_in, priors):
        # Internal method
        # Convert to 2D array
        p = np.atleast_2d(p_in)

        #Check shape of array
        if not p.shape[1]==len(priors):
            raise ValueError("Wrong dimension in evaluate_all")
        # Number of samples
        n = p.shape[0]
        # Output value
        logp = np.zeros(n)
        # Loop through samples and then through parameters
        # in that sample.
        for i,p_i in enumerate(p):
            logp_i = 0.0
            for x, prior in zip(p_i, priors):
                logp_i += prior(x)  # Calling a prior returns log(P(x))
            logp[i] = logp_i
        return logp

    def evaluate_p(self, p_in):
        """
        Evaluate P(x) of the varied parameters.
        p_in is a 1D array if length nparam
        or a 2D array of shape (nsample, nparam)
        """
        return np.exp(self.evaluate_logp(p_in))

    def evaluate_p_all(self, p_in):
        """
        Evaluate log(P(x)) of all the parameters, including fixed ones.
        p_in is a 1D array if length nparam
        or a 2D array of shape (nsample, nparam)
        """
        return np.exp(self.evaluate_logp_all(p_in))

    def evaluate_logp(self, p_in):
        """
        Evaluate log(P(x)) of the varied parameters.
        p_in is a 1D array if length nparam
        or a 2D array of shape (nsample, nparam)
        """
        return self._evaluate(p_in, self.varied_priors)

    def evaluate_logp_all(self, p_in):
        """
        Evaluate log(P(x)) of all the parameters, including fixed ones.
        p_in is a 1D array if length nparam
        or a 2D array of shape (nsample, nparam)
        """
        return self._evaluate(p_in, self.all_priors)

def get_git_revision_dulwich(directory):
    import dulwich.repo
    path = pathlib.Path(directory)
    while not (path / ".git").exists():
        if str(path) == path.root:
            # not a git repo
            return ""
        path = path.parent.absolute()
    repo = dulwich.repo.Repo(path)
    return repo.head().decode('utf-8')


def get_git_revision(directory):
    try:
        # Turn e.g. $COSMOSIS_SRC_DIR into a proper path
        directory = os.path.expandvars(directory)
        if not os.path.isdir(directory):
            return ""

        if dulwich is not None:
            return get_git_revision_dulwich(directory)

        if os.environ.get("COSMOSIS_NO_SUBPROCESS", "") not in ["", "0"]:
            return ""

        # this git command gives the current commit ID of the
        # directory it is run from
        cmd = "git rev-parse HEAD".split()

        # run, capturing stderr and stdout to read the hash from
        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
            cwd=directory)

        # Read stdout.  Discard stderr, which will be None
        rev, _ = p.communicate()

        # If there are any errors then we ignore everything
        if p.returncode:
            return ""
        # There shouldn't be any newlines here, but in case there are in future
        # we replace them with spaces to avoid messing up output file formats
        return rev.decode('utf-8').strip().replace("\n", " ")
    
    # It's usually bad practice to catch all exceptions, but we really don't
    # want to crash if we can't get the metadata.
    except Exception as e:
        print(f"Unable to get metadata on git revision for {directory}: error {e}")
        return ""



def import_by_path(name, path):
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def underline(s, char='-'):
    """
    Return a string with a line of the given character
    underneath it.

    Parameters
    ----------
    s : str
        The string to underline
    char : str, optional
        The character to use for the line
    """
    return s + "\n" + char * len(s)

def overline(s, char='-'):
    """
    Return a string with a line of the given character
    over it.

    Parameters
    ----------
    s : str
        The string to overline
    char : str, optional
        The character to use for the line
    """
    return char * len(s) + "\n" + s

def under_over_line(s, char='-'):
    """
    Return a string with a line of the given character
    over and under it.

    Parameters
    ----------
    s : str
        The string to overline
    char : str, optional
        The character to use for the line
    """
    return underline(overline(s, char), char)


def read_chain_header(filename):
    lines = []
    for line in open(filename):
        if not line.startswith('#'):
            break
        lines.append(line)
    return lines


def extract_inis_from_chain_header(lines, section):
    start = "## START_OF_{}_INI".format(section).upper()
    end = "## END_OF_{}_INI".format(section).upper()
    in_section = False
    output_lines = []
    for line in lines:
        if line.startswith(start):
            in_section = True
            continue
        elif line.startswith(end):
            break
        elif in_section:
            output_lines.append(line[3:])
    return output_lines