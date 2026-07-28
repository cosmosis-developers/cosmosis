"""
ROSE: Rapid Online Sampling Emulator for CosmoSIS

This module implements an iterative emulator-based sampler that uses neural networks
to accelerate cosmological parameter estimation. The sampler alternates between:
1. Training neural network emulators on exact pipeline calculations
2. Running MCMC using the fast emulator predictions
3. Improving the emulator with new training data from high-likelihood regions

Authors: CosmoSIS Team
License: BSD 2-Clause
"""

import logging
import os
import time
from typing import Any

from .. import ParallelSampler

from .config import RoseConfigMixin, _parse_per_likelihood
from .data_processing import RoseDataProcessingMixin
import cosmosis.samplers.rose.data_processing as data_processing_module
from .pipeline_setup import RosePipelineSetupMixin
from .emulator_management import RoseEmulatorManagementMixin
from .sampling import RoseSamplingMixin
from .convergence import RoseConvergenceMixin
import cosmosis.samplers.rose.utils as utils_module

# Configure logging
logger = logging.getLogger(__name__)


class RoseSampler(
    ParallelSampler,
    RoseConfigMixin,
    RoseDataProcessingMixin,
    RosePipelineSetupMixin,
    RoseEmulatorManagementMixin,
    RoseSamplingMixin,
    RoseConvergenceMixin
):
    """Emulator-accelerated MCMC sampler for CosmoSIS.
    
    This sampler uses neural network emulators to speed up cosmological parameter
    estimation by 10-1000x. It works by:
    
    1. Generating an initial training set with exact (slow) pipeline calculations
    2. Training a neural network emulator on this data
    3. Running MCMC using the fast emulator for likelihood evaluation
    4. Iteratively improving the emulator with new training data
    5. Final sampling with the best emulator
    
    The sampler supports:
    - Partial pipeline emulation (specify last_emulated_module)
    - Custom data vector components (specify keys)
    - Likelihood tempering for better exploration
    - Comprehensive diagnostics and model saving
    - Reuse of pre-trained emulators
    
    Attributes:
        parallel_output: Whether to use parallel output (always False)
        sampler_outputs: Output columns for chains
    """
    
    parallel_output = False
    sampler_outputs = [("prior", float), ("tempered_post", float), ("post", float)]

    def config(self) -> None:
        """Configure the emulator sampler from ini file parameters.
        
        This method reads all configuration parameters from the ini file and
        sets up the sampler for training and MCMC sampling. It validates
        parameters and sets up the global sampler reference needed by helper functions.
        
        Raises:
            ValueError: If configuration parameters are invalid or inconsistent
        """
        self.converged = False
        
        # Parse emulation target settings
        keys = self.read_ini("keys", str, "")
        fixed_keys = self.read_ini("fixed_keys", str, "")
        error_keys = self.read_ini("error_keys", str, "")
        
        # Convert space-separated strings to lists of (section, key) tuples
        self.keys = [k.split(".") for k in keys.split()] if keys else []
        self.fixed_keys = [k.split(".") for k in fixed_keys.split()] if fixed_keys else []
        self.error_keys = [k.split(".") for k in error_keys.split()] if error_keys else []
        
        # Configure output saving
        self._configure_output_saving()
        self._timing_file = os.path.join(self.save_dir, "rose_timing.txt")
        
        # Configure emulator loading. Accepts either a single directory/base
        # path (applied to all likelihoods) or per-likelihood 'name:path'
        # tokens, e.g.
        #   load_emu_filename = planck:/a/emumodel_5/planck desi_bao:/b/emumodel_5/desi_bao
        self.load_emu_filename = _parse_per_likelihood(
            self.read_ini("load_emu_filename", str, ""), str, default=""
        )
        self.trained_before = self.read_ini("trained_before", bool, False)
        # When loading a pre-trained emulator into a pipeline that varies extra
        # parameters the emulator does not depend on (e.g. combining a
        # CMB-only planck emulator with supernova_params--m), ignore those
        # extra parameters at prediction time instead of raising.
        self.ignore_missing_emu_params = self.read_ini(
            "ignore_missing_emu_params", bool, True
        )
        
        if self.trained_before and not self.load_emu_filename:
            raise ValueError("trained_before=true requires load_emu_filename to be specified")

        # Initialize state
        self.ndim = len(self.pipeline.varied_params)
        self.emu_pipeline = None
        self.iterations = 0
        # Monotonically increasing id of the emulator model directory
        # (emumodel_{version}). Normally version == iterations + 1 (one training
        # per iteration), but the final-step KL-convergence loop retrains
        # additional times; each retrain bumps this so a fresh emumodel_N+1 dir
        # is written instead of overwriting the previous one.
        self._current_emu_version = 0
        
        # Configure training parameters
        self._configure_training_parameters()
        
        # Configure MCMC parameters
        self._configure_mcmc_parameters()
        
        # Configure advanced options
        self._configure_advanced_options()
        
        # Set global sampler reference for picklable task wrapper
        data_processing_module._sampler = self
        utils_module._sampler = self
        
        logger.info(f"RoseSampler configured with {self.ndim} parameters, "
                   f"{self.max_iterations} iterations, initial training size {self.initial_size}")

    def execute(self) -> None:
        """Execute one iteration of the emulator sampler.
        
        This method performs one complete iteration:
        1. Training (if not using pre-trained emulator)
        2. MCMC sampling with current emulator
        3. Output processing and chain storage
        """
        time_training_set_s = 0.0
        time_train_emulator_s = 0.0

        # Handle pre-trained emulator case
        if self.trained_before:
            self.compute_fiducial_setup_emu_pipeline()
            logger.info("Using pre-trained emulator, proceeding to final sampling")
            self.load_emulator()
            # Match the path token workers are given (None for pre-trained) so
            # utils._ensure_emulator does not reload on the master.
            self._loaded_emu_path = self._worker_emu_model_path()
            self.iterations = self.max_iterations - 1
        else:
            # Normal training workflow: time training set generation
            t0 = time.perf_counter()
            if self.iterations == 0:
                # First iteration: setup and initial training
                self.compute_fiducial_setup_emu_pipeline()
                self.generate_initial_sample()
            else:
                # Subsequent iterations: update training set
                self.generate_updated_sample()
            time_training_set_s = time.perf_counter() - t0

            # Train emulator
            logger.info(f"Training emulator (iteration {self.iterations + 1}/{self.max_iterations})")
            t1 = time.perf_counter()
            self.train_emulator()
            time_train_emulator_s = time.perf_counter() - t1
        
        # Set up sampling
        tempering = self._get_current_tempering()
        
        # Check if this is the final iteration and we should use nautilus or NUTS
        is_final_iteration = (self.iterations == self.max_iterations - 1)

        # Run convergence diagnostics on the test points collected from the
        # 1-sigma region of the one-before-last chain, using the freshly trained
        # emulator, just before entering the final sampling stage.
        if is_final_iteration:
            self.run_convergence_tests()
        
        # Sampling methods return MCMC wall time only (excluding chain file I/O
        # in _process_*_results, which can dominate for large chains).
        if is_final_iteration and self.final_sampler == "nuts":
            logger.info("Using NUTS for final iteration")
            time_sampling_s = self._run_nuts_sampling(tempering)
        elif is_final_iteration and self.final_sampler == "nautilus":
            logger.info("Using Nautilus for final iteration")
            time_sampling_s = self._run_nautilus_sampling(tempering)
        else:
            logger.info("Using emcee for sampling")
            time_sampling_s = self._run_emcee_sampling(tempering)

        # Compute a per-iteration KL divergence directly from the MCMC chains
        # (ignoring the tempering used to produce them) and append it to
        # rose_kl.txt, mirroring the per-iteration timing bookkeeping below.
        self._record_iteration_kl()

        self._append_timing_row(
            self.iterations + 1,
            time_training_set_s,
            time_train_emulator_s,
            time_sampling_s,
        )
        
        # Increment iteration counter
        self.iterations += 1

    def _append_timing_row(
        self,
        iteration: int,
        time_training_set_s: float,
        time_train_emulator_s: float,
        time_sampling_s: float,
    ) -> None:
        """Append one timing row to ``rose_timing.txt``."""
        write_header = not os.path.isfile(self._timing_file)
        with open(self._timing_file, "a") as f:
            if write_header:
                f.write(
                    "iteration\ttime_training_set_s\ttime_train_emulator_s\ttime_sampling_s\n"
                )
            f.write(
                f"{iteration}\t{time_training_set_s:.6f}\t"
                f"{time_train_emulator_s:.6f}\t{time_sampling_s:.6f}\n"
            )
        logger.info(
            f"Timing saved to {self._timing_file} "
            f"(iteration={iteration}, training_set={time_training_set_s:.1f}s, "
            f"train_emu={time_train_emulator_s:.1f}s, sampling={time_sampling_s:.1f}s)"
        )

    def is_converged(self) -> bool:
        """Check if sampler has completed all iterations.
        
        Returns:
            True if all iterations are complete
        """
        return self.iterations >= self.max_iterations
