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
from typing import Any

from .. import ParallelSampler

from .config import RoseConfigMixin
from .data_processing import RoseDataProcessingMixin
import cosmosis.samplers.rose.data_processing as data_processing_module
from .pipeline_setup import RosePipelineSetupMixin
from .emulator_management import RoseEmulatorManagementMixin
from .sampling import RoseSamplingMixin
import cosmosis.samplers.rose.utils as utils_module

# Configure logging
logger = logging.getLogger(__name__)


class RoseSampler(
    ParallelSampler,
    RoseConfigMixin,
    RoseDataProcessingMixin,
    RosePipelineSetupMixin,
    RoseEmulatorManagementMixin,
    RoseSamplingMixin
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
        
        # Configure emulator loading
        self.load_emu_filename = self.read_ini("load_emu_filename", str, "")
        self.trained_before = self.read_ini("trained_before", bool, False)
        
        if self.trained_before and not self.load_emu_filename:
            raise ValueError("trained_before=true requires load_emu_filename to be specified")

        # Initialize state
        self.ndim = len(self.pipeline.varied_params)
        self.emu_pipeline = None
        self.iterations = 0    
        
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
        # Handle pre-trained emulator case
        if self.trained_before:
            self.compute_fiducial_setup_emu_pipeline()
            logger.info("Using pre-trained emulator, proceeding to final sampling")
            self.load_emulator()
            self.iterations = self.max_iterations - 1
        else:
            # Normal training workflow
            if self.iterations == 0:
                # First iteration: setup and initial training
                self.compute_fiducial_setup_emu_pipeline()
                self.generate_initial_sample()
            else:
                # Subsequent iterations: update training set
                self.generate_updated_sample()
            
            # Train emulator
            logger.info(f"Training emulator (iteration {self.iterations + 1}/{self.max_iterations})")
            self.train_emulator()
        
        # Set up sampling
        tempering = self._get_current_tempering()
        
        # Check if this is the final iteration and we should use nautilus or NUTS
        is_final_iteration = (self.iterations == self.max_iterations - 1)
        
        if is_final_iteration and self.use_nuts_final:
            logger.info("Using NUTS for final iteration")
            self._run_nuts_sampling(tempering)
        elif is_final_iteration and self.use_nautilus_final:
            logger.info("Using Nautilus for final iteration")
            self._run_nautilus_sampling(tempering)
        else:
            logger.info("Using emcee for sampling")
            self._run_emcee_sampling(tempering)
        
        # Increment iteration counter
        self.iterations += 1

    def is_converged(self) -> bool:
        """Check if sampler has completed all iterations.
        
        Returns:
            True if all iterations are complete
        """
        return self.iterations >= self.max_iterations
