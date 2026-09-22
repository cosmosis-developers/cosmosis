"""
Utility functions for ROSE sampler.

This module contains shared utility functions and constants used across
the ROSE sampler implementation.
"""

import os
import errno
import logging
from typing import Optional, Tuple, Union, List, Any

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Save output modes
SAVE_NONE = 0   # Save nothing (not recommended)
SAVE_MODEL = 1  # Save only final trained emulator
SAVE_ALL = 2    # Save all training data, models, and diagnostics

_sampler = None

# Value for the signed log norm transform
# optimised for 3x2pt with C_ell ~1e-11-1e-9
SIGNED_LOG_NORM_TRANSFORM_SCALE = 1e-7  #1e-10

# Sentinel used to detect "no emulator has been loaded yet on this process".
_UNSET_EMU_PATH = object()


def _ensure_emulator(sampler: Any, model_path: Optional[str]) -> None:
    """Make sure this process has the emulator the master is currently using.

    In MPI runs only the master process executes the training loop, so the
    worker processes never see the master's in-memory emulator updates (there
    is no broadcast). Each worker therefore loads the emulator lazily from disk.

    Because the master retrains the emulator every iteration -- and, when
    ``kl_convergence = T``, again during the final-stage holdout retrain loop --
    a worker that loaded an emulator once must *reload* it whenever
    the master moves on to a new model. We track the path of the emulator this
    process last loaded in ``sampler._loaded_emu_path`` and reload whenever the
    requested ``model_path`` differs from it.

    ``model_path`` is the per-iteration model directory (written by
    ``train_emulator``) for from-scratch runs, or ``None`` for a pre-trained
    emulator (in which case ``load_emulator(None)`` uses ``load_emu_filename``).
    """
    if sampler.emu_pipeline is None:
        logger.warning("emu_pipeline is not initialized, setting it up now")
        sampler.compute_fiducial_setup_emu_pipeline()
        sampler.load_emulator(model_path)
        sampler._loaded_emu_path = model_path
        return

    if getattr(sampler, "_loaded_emu_path", _UNSET_EMU_PATH) != model_path:
        logger.info(
            "Emulator on this process is out of date (had %r, need %r); reloading",
            getattr(sampler, "_loaded_emu_path", None), model_path,
        )
        sampler.load_emulator(model_path)
        sampler._loaded_emu_path = model_path


def mkdir(path: str) -> None:
    """Ensure that all the components in the `path` exist in the file system.
    
    Args:
        path: Directory path to create
        
    Raises:
        ValueError: If path conflicts with existing files or directory structure
    """
    # Avoid trying to make an empty path
    if not path.strip():
        return
        
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if not os.path.isdir(path):
                raise ValueError(f"Tried to create directory {path} but file with same name exists")
        elif error.errno == errno.ENOTDIR:
            raise ValueError(f"Tried to create directory {path} but part of path exists as a file")
        else:
            raise ValueError(f"Failed to create directory {path}: {error}")


def task(p: np.ndarray, sampler: Any, return_all: bool = False) -> Optional[Union[Tuple, Any]]:
    """Execute pipeline for given parameters and extract relevant data vectors.
    
    This function runs the full cosmological pipeline for a given parameter vector
    and extracts the theory predictions, data vectors, and likelihood information
    needed for emulator training or MCMC sampling.
    
    The theory/data vectors are grouped per-likelihood so that independent
    emulators can be trained for each likelihood. The grouping key is:
    - the ``base_key`` (likelihood name) when scanning ``data_vector/*_theory``
      entries (e.g. ``desi_bao``, ``lsst``);
    - the ``key`` string when ``sampler.keys`` is set (user-specified outputs).
    
    Args:
        p: Parameter vector to evaluate
        sampler: RoseSampler instance (needed for pipeline and keys)
        return_all: If True, return additional data for emulator training
                   (covariance matrices, error vectors, etc.)
                   
    Returns:
        If return_all=False: (likelihood, data_vectors_theory, prior, posterior)
          where data_vectors_theory is a dict {likelihood_name: np.ndarray}
        If return_all=True: (likelihood, data_vectors_theory, data_vectors,
                           inv_covariance, error_vectors, block)
          where all four vector containers are dicts keyed by likelihood name.
          data_vectors and inv_covariance only contain entries for likelihoods
          whose {base_key}_data / {base_key}_inverse_covariance blocks were found.
        None if pipeline execution failed
    """
    r = sampler.pipeline.run_results(p)
    block = r.block
    if block is None:
        logger.warning(f"(Within Task) Pipeline execution failed for parameters: {p}")
        return None

    data_vectors_theory: dict[str, np.ndarray] = {}
    data_vectors: dict[str, np.ndarray] = {}
    error_vectors: dict[str, np.ndarray] = {}
    data_inv_covariance: dict[str, np.ndarray] = {}

    if sampler.keys:
        # User has specified which keys to emulate: use `key` as the likelihood id
        for sec, key in sampler.keys:
            value = block[sec, key]
            if isinstance(value, (int, float)):
                data_vectors_theory[key] = np.array([value])
            else:
                data_vectors_theory[key] = np.asarray(value)

        if return_all:
            if sampler.error_keys:
                if len(sampler.error_keys) != len(sampler.keys):
                    raise ValueError(
                        f"error_keys length ({len(sampler.error_keys)}) must match "
                        f"keys length ({len(sampler.keys)})"
                    )
                for (_, out_key), (sec, err_key) in zip(sampler.keys, sampler.error_keys):
                    error_vectors[out_key] = np.asarray(block[sec, err_key])
            else:
                for name, d in data_vectors_theory.items():
                    error_vectors[name] = np.ones_like(d)
    else:
        # Group by likelihood (base_key) from the data_vector section
        for sec, key in block.keys(section="data_vector"):
            if not key.endswith("_theory"):
                continue
            base_key = key[:-7]  # Remove '_theory' suffix
            data_vectors_theory[base_key] = np.asarray(r.block[sec, key])

            if return_all:
                try:
                    covmat = block[sec, base_key + "_covariance"]
                    error_vectors[base_key] = np.sqrt(np.diag(covmat))
                    data_inv_covariance[base_key] = block[sec, base_key + "_inverse_covariance"]
                    data_vectors[base_key] = block[sec, base_key + "_data"]
                except KeyError as e:
                    logger.warning(f"Missing covariance data for {base_key}: {e}")
                    error_vectors[base_key] = np.ones_like(data_vectors_theory[base_key])

    if return_all:
        if set(error_vectors.keys()) != set(data_vectors_theory.keys()):
            raise ValueError(
                f"Mismatch between error vectors ({sorted(error_vectors)}) "
                f"and theory vectors ({sorted(data_vectors_theory)})"
            )
        return r.like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, r.block
    else:
        return r.like, data_vectors_theory, r.prior, r.post


#def log_probability_function(u: np.ndarray, sampler: Any, tempering: float) -> Tuple[float, Tuple[float, List[float]]]:
def log_probability_function(u: np.ndarray, tempering: float, model_path: Optional[str] = None):
    """Log probability function using emulated pipeline.
    
    Args:
        u: Parameter vector in unit hypercube [0,1]^ndim
        tempering: Tempering factor to apply to posterior (0 < tempering <= 1)
                  Lower values flatten the likelihood for better exploration
        model_path: Directory of the emulator the master is currently using.
                  Passed so worker processes load/reload the matching emulator
                  (see :func:`_ensure_emulator`). ``None`` for a pre-trained run.

    Returns:
        Tuple of (tempered_posterior, (prior, extra_parameters))
        Returns (-inf, (-inf, [nan, ...])) if parameters are outside prior bounds
    """
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    # Make sure this process has the same (current) emulator as the master.
    _ensure_emulator(_sampler, model_path)
    sampler = _sampler
    try:
        # Transform from unit hypercube to physical parameter space
        p = sampler.pipeline.denormalize_vector_from_prior(u)
    except ValueError:
        # Parameters outside prior bounds
        return (-np.inf, (-np.inf, [np.nan for i in range(sampler.pipeline.number_extra)]))
    # Run emulated pipeline
    r = sampler.emu_pipeline.run_results(p)
    return tempering * r.post, (r.prior, r.extra)
    #return tempering * r.like, (r.prior, r.extra)

def log_probability_function_nautilus(p, model_path=None):
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    # Make sure this process has the same (current) emulator as the master.
    _ensure_emulator(_sampler, model_path)
    sampler = _sampler
    r = sampler.emu_pipeline.run_results(p)
    log_prob, blobs =  r.post, (r.prior, r.extra)
    # Flatten blobs to match cosmosis nautilus format
    if blobs is None:
        return log_prob
    elif isinstance(blobs, (int, float)):
        return log_prob, blobs
    elif isinstance(blobs, tuple):
        # Flatten the tuple to scalars only
        flattened = []
        for item in blobs:
            if np.isscalar(item):
                flattened.append(item)
            else:
                # Flatten arrays
                flattened.extend(np.atleast_1d(item).flatten())
        return log_prob, tuple(flattened)
    else:
        # Convert other types to scalar
        return log_prob, float(blobs)

def nuts_chain_worker(task: dict) -> dict:
    """Picklable MPI worker: run one NUTS chain on this process.

    Uses the global RoseSampler (set in ``RoseSampler.config``) and loads the
    emulator from disk via :func:`_ensure_emulator`, matching the emcee path.
    ``task`` must be a plain dict of numpy / Python scalars (picklable).
    """
    global _sampler
    if _sampler is None:
        raise RuntimeError(
            "Global sampler not set. This should be set in RoseSampler.config()"
        )
    model_path = task.get("model_path")
    _ensure_emulator(_sampler, model_path)
    return _sampler._execute_one_nuts_chain(task)


def prior_transform(p, model_path=None):
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    # Make sure this process has the same (current) emulator as the master.
    _ensure_emulator(_sampler, model_path)
    sampler = _sampler
    return sampler.pipeline.denormalize_vector_from_prior(p)

