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
    
    Args:
        p: Parameter vector to evaluate
        sampler: RoseSampler instance (needed for pipeline and keys)
        return_all: If True, return additional data for emulator training
                   (covariance matrices, error vectors, etc.)
                   
    Returns:
        If return_all=False: (likelihood, data_vectors_theory, prior, posterior)
        If return_all=True: (likelihood, data_vectors_theory, data_vectors, 
                           inv_covariance, error_vectors, block)
        None if pipeline execution failed
    """
    r = sampler.pipeline.run_results(p)
    block = r.block
    if block is None:
        logger.warning(f"(Within Task) Pipeline execution failed for parameters: {p}")
        return None

    data_vectors_theory = []
    data_vectors = []
    error_vectors = []
    data_inv_covariance = []
    
    if sampler.keys:
        # User has specified which keys to emulate
        for sec, key in sampler.keys:
            value = block[sec, key]
            # Ensure value is always an array for consistency
            if isinstance(value, (int, float)):
                data_vectors_theory.append(np.array([value]))
            else:
                data_vectors_theory.append(np.asarray(value))
                
        if return_all:
            if sampler.error_keys:
                for sec, key in sampler.error_keys:
                    error_vectors.append(np.asarray(block[sec, key]))
            else:
                # Default to unit errors if none specified
                for d in data_vectors_theory:
                    error_vectors.append(np.ones_like(d))
    else:
        # Use all theory predictions from data_vector section
        for sec, key in block.keys(section="data_vector"):
            if not key.endswith("_theory"):
                continue
                
            data_vectors_theory.append(r.block[sec, key])
            
            if return_all:
                # Extract corresponding data and covariance
                base_key = key[:-7]  # Remove '_theory' suffix
                try:
                    covmat = block[sec, base_key + "_covariance"]
                    sigma = np.sqrt(np.diag(covmat))
                    error_vectors.append(sigma)
                    data_inv_covariance.append(block[sec, base_key + "_inverse_covariance"])
                    data_vectors.append(block[sec, base_key + "_data"])
                except KeyError as e:
                    logger.warning(f"Missing covariance data for {base_key}: {e}")
                    # Use unit errors as fallback
                    error_vectors.append(np.ones_like(data_vectors_theory[-1]))
       
    if return_all:
        if len(error_vectors) != len(data_vectors_theory):
            raise ValueError(f"Mismatch: {len(error_vectors)} error vectors vs "
                           f"{len(data_vectors_theory)} theory vectors")
        return r.like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, r.block
    else:
        return r.like, data_vectors_theory, r.prior, r.post


#def log_probability_function(u: np.ndarray, sampler: Any, tempering: float) -> Tuple[float, Tuple[float, List[float]]]:
def log_probability_function(u: np.ndarray, tempering: float):
    """Log probability function using emulated pipeline.
    
    Args:
        u: Parameter vector in unit hypercube [0,1]^ndim
        tempering: Tempering factor to apply to posterior (0 < tempering <= 1)
                  Lower values flatten the likelihood for better exploration
                  
    Returns:
        Tuple of (tempered_posterior, (prior, extra_parameters))
        Returns (-inf, (-inf, [nan, ...])) if parameters are outside prior bounds
    """
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    # Check if emu_pipeline is initialized
    if _sampler.emu_pipeline is None:
        #raise RuntimeError("emu_pipeline is not initialized. This may happen if execute() "
        #                    "hasn't been called yet, or if compute_fiducial_setup_emu_pipeline() "
        #                    "failed on this process.")
        logger.warning("emu_pipeline is not initialized, setting it up now")
        _sampler.compute_fiducial_setup_emu_pipeline()
        _sampler.load_emulator()
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

def log_probability_function_nautilus(p):
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    # Check if emu_pipeline is initialized
    if _sampler.emu_pipeline is None:
        #raise RuntimeError("emu_pipeline is not initialized. This may happen if execute() "
        #                    "hasn't been called yet, or if compute_fiducial_setup_emu_pipeline() "
        #                    "failed on this process.")
        logger.warning("emu_pipeline is not initialized, setting it up now")
        _sampler.compute_fiducial_setup_emu_pipeline()
        _sampler.load_emulator()
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

def prior_transform(p):
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    # Check if emu_pipeline is initialized
    if _sampler.emu_pipeline is None:
        logger.warning("emu_pipeline is not initialized, setting it up now")
        _sampler.compute_fiducial_setup_emu_pipeline()
        _sampler.load_emulator()
    sampler = _sampler
    return sampler.pipeline.denormalize_vector_from_prior(p)

