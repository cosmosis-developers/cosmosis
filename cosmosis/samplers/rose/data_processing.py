"""
Data processing and sample generation for ROSE sampler.

This module contains methods for generating training samples, processing results,
and managing training/test datasets.
"""

import logging
from typing import List, Dict, Any
from timeit import default_timer

import numpy as np

from .utils import task, SAVE_ALL

logger = logging.getLogger(__name__)

# Global sampler reference for picklable task wrapper
_sampler = None


def _task_wrapper(p: np.ndarray) -> Any:
    """Wrapper function for task that can be pickled for MPI.
    
    This function uses the global _sampler variable set by RoseSampler.config().
    It's needed because lambda functions cannot be pickled for MPI communication.
    
    Args:
        p: Parameter vector to evaluate
        
    Returns:
        Result from task(p, _sampler)
    """
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    return task(p, _sampler)


class RoseDataProcessingMixin:
    """Mixin class providing data processing methods for RoseSampler."""
    
    def generate_initial_sample(self) -> None:
        """Generate initial training sample using Latin Hypercube sampling.
        
        This method creates the initial training dataset by:
        1. Sampling parameters from priors using Latin Hypercube
        2. Running the full pipeline for each parameter set
        3. Filtering out poor fits based on chi2 cutoff
        4. Storing results for emulator training
        """
        import scipy.stats
        
        logger.info("Generating initial training sample")
        
        # Generate Latin Hypercube sample in unit cube
        hypercube = scipy.stats.qmc.LatinHypercube(self.ndim, seed=self.seed)
        unit_sample = hypercube.random(n=self.initial_size)
        
        # Transform to physical parameter space
        sample = np.array([
            self.pipeline.denormalize_vector_from_prior(p) for p in unit_sample
        ])
        
        logger.info(f"Generated {len(sample)} parameter combinations")
        logger.info(f"Parallelization enabled: {self.pool is not None}")
        
        # Run pipeline for all samples
        start_time = default_timer()
        if self.pool:
            sample_results = self.pool.map(_task_wrapper, sample)
        else:
            sample_results = [task(p, self) for p in sample]
        
        end_time = default_timer()
        logger.info(f"Initial sample evaluation took {end_time - start_time:.1f} seconds")
        
        # Process results and apply chi2 cutoff
        self._process_initial_results(sample_results, sample, unit_sample)
        
        # Initialize test sample storage
        self._initialize_test_storage()

    def _process_initial_results(self, sample_results: List, sample: np.ndarray, 
                                unit_sample: np.ndarray) -> None:
        """Process initial sample results and apply chi2 filtering."""
        # Extract successful results
        valid_results = [s for s in sample_results if s is not None]
        
        if not valid_results:
            raise RuntimeError("No valid pipeline evaluations in initial sample")
        
        # Extract components
        sample_likes = np.array([s[0] for s in valid_results])
        sample_data_vectors = np.array([np.concatenate(s[1]) for s in valid_results])
        sample_priors = np.array([s[2] for s in valid_results])
        sample_posts = np.array([s[3] for s in valid_results])
        
        # Apply chi2 cutoff
        chi2_values = -2 * sample_likes
        cut = chi2_values < self.chi2_cut_off
        
        logger.info(f"Data vector shape: {sample_data_vectors.shape}")
        logger.info(f"Chi2 range: [{chi2_values.min():.1f}, {chi2_values.max():.1f}]")
        
        # Filter arrays
        self.sample_likes = sample_likes[cut]
        self.sample_priors = sample_priors[cut]
        self.sample_posts = sample_posts[cut]
        self.sample_data_vectors = sample_data_vectors[cut]

        
        # Filter corresponding parameter arrays
        valid_indices = np.arange(len(sample_results))[np.array([s is not None for s in sample_results])]
        filtered_indices = valid_indices[cut]
        self.sample = sample[filtered_indices]
        self.unit_sample = unit_sample[filtered_indices]
        
        n_kept = len(self.sample_likes)
        n_total = len(sample_results)
        logger.info(f"Kept {n_kept}/{n_total} samples after chi2 < {self.chi2_cut_off} filter")
        self.points_per_iteration = np.array([n_kept])
        # For the case of rejected points due to additional priors, for instance, w0+wa<0
        if n_kept < 0.8*self.initial_size:
            #raise RuntimeError(f"Only {n_kept} samples passed chi2 filter, but need "
            #                 f"at least {0.8*self.initial_size} (80% of initial size) --> increase the initial size")
            logger.warning(f"Only {n_kept} samples passed chi2 filter, but need "
                           f"at least {0.8*self.initial_size} (80% of initial size) --> increase the initial size")
        
        
        self.initial_size_cut = n_kept

    def _initialize_test_storage(self) -> None:
        """Initialize storage for test samples."""
        self.sample_test = np.array([]).reshape(0, self.ndim)
        self.sample_data_vectors_test = np.array([]).reshape(0, self.sample_data_vectors.shape[1])
        self.unit_sample_test = np.array([]).reshape(0, self.ndim)
        self.sample_likes_test = np.array([])
        self.sample_priors_test = np.array([])
        self.sample_posts_test = np.array([])
        self.points_per_iteration_test = np.array([])

    def generate_updated_sample(self) -> None:
        """Generate additional training samples from MCMC chain.
        
        This method selects high-likelihood samples from the current MCMC chain,
        evaluates them with the exact pipeline, and adds them to the training set.
        """
        logger.info(f"Selecting {self.resample_size} samples from MCMC chain for training")
        
        # Select random samples from chain
        chain_length = len(self.chain)
        random_indices = np.random.choice(
            chain_length, size=self.resample_size, replace=False
        )
        
        unit_sample = self.unit_chain[random_indices]
        sample = self.chain[random_indices]
        
        # Select test samples (20% of resample size)
        test_size = max(1, int(0.2 * self.resample_size))
        test_indices = self._select_non_overlapping_indices(chain_length, random_indices, test_size)
        
        unit_sample_test = self.unit_chain[test_indices]
        sample_test = self.chain[test_indices]
        
        logger.info(f"Running exact pipeline on {len(sample)} training + {len(sample_test)} test samples")
        
        # Evaluate samples with exact pipeline
        start_time = default_timer()
        if self.pool:
            sample_results = self.pool.map(_task_wrapper, sample)
            sample_results_test = self.pool.map(_task_wrapper, sample_test)
        else:
            sample_results = [task(p, self) for p in sample]
            sample_results_test = [task(p, self) for p in sample_test]
        
        end_time = default_timer()
        logger.info(f"Sample evaluation took {end_time - start_time:.1f} seconds")
        
        # Update training and test sets
        self._update_training_set(sample_results, sample, unit_sample)
        self._update_test_set(sample_results_test, sample_test, unit_sample_test)
        
        # Save datasets if requested
        if self.save_outputs == SAVE_ALL and self.iterations == (self.max_iterations - 1):
            self._save_datasets()

    def _select_non_overlapping_indices(self, chain_length: int, 
                                      excluded_indices: np.ndarray, 
                                      n_select: int) -> np.ndarray:
        """Select indices that don't overlap with excluded set."""
        available_indices = np.setdiff1d(np.arange(chain_length), excluded_indices)
        
        if len(available_indices) < n_select:
            logger.warning(f"Only {len(available_indices)} non-overlapping indices available, "
                          f"requested {n_select}")
            n_select = len(available_indices)
        
        return np.random.choice(available_indices, size=n_select, replace=False)

    def _update_training_set(self, sample_results: List, sample: np.ndarray, 
                           unit_sample: np.ndarray) -> None:
        """Update training set with new samples."""
        valid_results = [s for s in sample_results if s is not None]
        
        #if not valid_results:
        #    logger.warning("No valid results in training sample update")
        #    return
        
        # For the case of rejected points due to additional priors, for instance, w0+wa<0
        if len(valid_results) < 0.8*self.resample_size:
            #raise RuntimeError(f"Only {len(valid_results)} samples passed chi2 filter, but need "
            #                 f"at least {0.8*self.resample_size} (80% of resample size) --> increase the resample size")
            logger.warning(f"Only {len(valid_results)} samples passed chi2 filter, but need "
                           f"at least {0.8*self.resample_size} (80% of resample size) --> increase the resample size")
        # Extract new data
        new_data_vectors = np.array([np.concatenate(s[1]) for s in valid_results])
        new_likes = np.array([s[0] for s in valid_results])
        new_priors = np.array([s[2] for s in valid_results])
        new_posts = np.array([s[3] for s in valid_results])
        
        # Filter for valid indices
        valid_mask = np.array([s is not None for s in sample_results])
        valid_sample = sample[valid_mask]
        valid_unit_sample = unit_sample[valid_mask]
        
        # Append to existing arrays
        self.sample = np.vstack([self.sample, valid_sample])
        self.sample_data_vectors = np.vstack([self.sample_data_vectors, new_data_vectors])
        self.unit_sample = np.vstack([self.unit_sample, valid_unit_sample])
        self.sample_likes = np.concatenate([self.sample_likes, new_likes])
        self.sample_priors = np.concatenate([self.sample_priors, new_priors])
        self.sample_posts = np.concatenate([self.sample_posts, new_posts])
        self.points_per_iteration = np.concatenate([self.points_per_iteration, np.array([len(valid_results)])])

    def _update_test_set(self, sample_results_test: List, sample_test: np.ndarray,
                        unit_sample_test: np.ndarray) -> None:
        """Update test set with new samples."""
        valid_results = [s for s in sample_results_test if s is not None]
        
        if not valid_results:
            logger.warning("No valid results in test sample update")
            return
        
        # Extract new test data
        new_data_vectors = np.array([np.concatenate(s[1]) for s in valid_results])
        new_likes = np.array([s[0] for s in valid_results])
        new_priors = np.array([s[2] for s in valid_results])
        new_posts = np.array([s[3] for s in valid_results])
        
        # Filter for valid indices
        valid_mask = np.array([s is not None for s in sample_results_test])
        valid_sample = sample_test[valid_mask]
        valid_unit_sample = unit_sample_test[valid_mask]

        
        
        # Append to test arrays
        self.sample_test = np.vstack([self.sample_test, valid_sample])
        self.sample_data_vectors_test = np.vstack([self.sample_data_vectors_test, new_data_vectors])
        self.unit_sample_test = np.vstack([self.unit_sample_test, valid_unit_sample])
        self.sample_likes_test = np.concatenate([self.sample_likes_test, new_likes])
        self.sample_priors_test = np.concatenate([self.sample_priors_test, new_priors])
        self.sample_posts_test = np.concatenate([self.sample_posts_test, new_posts])
        self.points_per_iteration_test = np.concatenate([self.points_per_iteration_test, np.array([len(valid_results)])])

    def _save_datasets(self) -> None:
        """Save training and test datasets."""
        logger.info("Saving training and test datasets")
        
        # Training set
        training_dict = self._build_dataset_dict(
            self.sample, self.unit_sample, self.sample_data_vectors,
            self.sample_likes, self.sample_priors, self.sample_posts,
            self.points_per_iteration
        )
        np.savez(f'{self.save_outputs_dir}/total_training_set.npz', **training_dict)
        
        # Test set
        test_dict = self._build_dataset_dict(
            self.sample_test, self.unit_sample_test, self.sample_data_vectors_test,
            self.sample_likes_test, self.sample_priors_test, self.sample_posts_test,
            self.points_per_iteration_test
        )
        np.savez(f'{self.save_outputs_dir}/total_testing_set.npz', **test_dict)

    def _build_dataset_dict(self, sample: np.ndarray, unit_sample: np.ndarray,
                           data_vectors: np.ndarray, likes: np.ndarray,
                           priors: np.ndarray, posts: np.ndarray,
                           points_per_iteration: np.ndarray) -> Dict[str, Any]:
        """Build dataset dictionary for saving."""
        param_names = [str(param) for param in self.pipeline.varied_params]
        
        dataset_dict = {}
        
        # Add parameter arrays
        for i, param in enumerate(param_names):
            dataset_dict[param] = sample[:, i]
            dataset_dict[f"{param}--norm"] = unit_sample[:, i]
        
        # Add metadata and results
        dataset_dict.update({
            'fixed_keys': [str(key) for key in self.fixed_keys] if self.fixed_keys else '',
            'fixed_features': self.fixed_vector,
            'output_keys': [str(key) for key in self.keys] if self.keys else 'data_vector',
            'features_size': self.data_vector_sizes,
            'features': data_vectors,
            'chi2': likes,
            'priors': priors,
            'posts': posts,
            'points_per_iteration': points_per_iteration
        })
        
        return dataset_dict


