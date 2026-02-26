"""
Configuration methods for ROSE sampler.

This module contains methods for reading and validating configuration
parameters from ini files.
"""

import logging
import numpy as np

from .utils import SAVE_NONE, SAVE_MODEL, SAVE_ALL, mkdir

logger = logging.getLogger(__name__)


class RoseConfigMixin:
    """Mixin class providing configuration methods for RoseSampler."""
    
    def _configure_output_saving(self) -> None:
        """Configure output saving options."""
        save_outputs = self.read_ini("save_outputs", str, "")
        
        if save_outputs:
            self.save_outputs_dir = self.read_ini("save_outputs_dir", str, "")
            if not self.save_outputs_dir:
                raise ValueError("save_outputs_dir must be specified when save_outputs is set")
                
            mkdir(self.save_outputs_dir)
            
            if save_outputs == "model":
                self.save_outputs = SAVE_MODEL
            elif save_outputs == "all":
                self.save_outputs = SAVE_ALL
            else:
                raise ValueError(f"Unknown save_outputs option '{save_outputs}' - "
                               "should be 'model', 'all', or empty")
        else:
            self.save_outputs = SAVE_NONE
            # Set a default directory for model saving (even when not saving outputs)
            # This is needed because train_emulator always needs a model_filename
            import tempfile
            import os
            self.save_outputs_dir = os.path.join(tempfile.gettempdir(), "rose_emulator")
            mkdir(self.save_outputs_dir)
            logger.warning("No outputs will be saved (save_outputs not specified), "
                         f"but models will be saved to temporary directory: {self.save_outputs_dir}")
    
    def _configure_training_parameters(self) -> None:
        """Configure neural network training parameters."""
        self.max_iterations = self.read_ini("iterations", int, 4)
        self.initial_size = self.read_ini("initial_size", int, 9600)
        self.resample_size = self.read_ini("resample_size", int, 4800)
        self.chi2_cut_off = self.read_ini("chi2_cut_off", float)
        self.batch_size = self.read_ini("batch_size", int, 32)
        self.training_iterations = self.read_ini("training_iterations", int, 5)
        
        # Validate training parameters
        if self.max_iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.initial_size < 10:
            raise ValueError("initial_size must be >= 10")
        if self.resample_size < 1:
            raise ValueError("resample_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.training_iterations < 1:
            raise ValueError("training_iterations must be >= 1")
    
    def _configure_mcmc_parameters(self) -> None:
        """Configure MCMC sampling parameters."""
        self.emcee_walkers = self.read_ini("emcee_walkers", int)
        self.emcee_samples = self.read_ini("emcee_samples", int)
        self.emcee_burn = self.read_ini("emcee_burn", float, 0.3)
        self.emcee_thin = self.read_ini("emcee_thin", int, 1)
        
        # Validate MCMC parameters
        if self.emcee_walkers >= self.initial_size:
            raise ValueError("emcee_walkers must be < initial_size for proper initialization")
        if self.emcee_walkers < 2 * self.ndim:
            logger.warning(f"emcee_walkers ({self.emcee_walkers}) < 2*ndim ({2*self.ndim}) "
                          "may lead to poor sampling")
        if self.emcee_samples < 100:
            logger.warning(f"emcee_samples ({self.emcee_samples}) is very small")
        if not (0 <= self.emcee_burn <= 1) and self.emcee_burn >= self.emcee_samples:
            raise ValueError("emcee_burn must be fraction in [0,1] or integer < emcee_samples")
        if self.emcee_thin < 1:
            raise ValueError("emcee_thin must be >= 1")
    
    def _configure_nautilus_parameters(self) -> None:
        """Configure Nautilus sampling parameters for final iteration."""
        self.nautilus_n_live = self.read_ini("nautilus_n_live", int, 2000)
        self.nautilus_n_update = self.read_ini("nautilus_n_update", int, self.nautilus_n_live)
        self.nautilus_enlarge_per_dim = self.read_ini("nautilus_enlarge_per_dim", float, 1.1)
        self.nautilus_n_points_min = self.read_ini("nautilus_n_points_min", int, self.ndim + 50)
        self.nautilus_split_threshold = self.read_ini("nautilus_split_threshold", float, 100.0)
        self.nautilus_n_networks = self.read_ini("nautilus_n_networks", int, 4)
        self.nautilus_n_batch = self.read_ini("nautilus_n_batch", int, 100)
        self.nautilus_f_live = self.read_ini("nautilus_f_live", float, 0.01)
        self.nautilus_n_shell = self.read_ini("nautilus_n_shell", int, self.nautilus_n_batch)
        self.nautilus_n_eff = self.read_ini("nautilus_n_eff", float, 10000.0)
        self.nautilus_n_like_max = self.read_ini("nautilus_n_like_max", int, 10000000000000000000000)
        self.nautilus_discard_exploration = self.read_ini("nautilus_discard_exploration", bool, False)
        
        # Validate nautilus parameters
        if self.nautilus_n_live < 2 * self.ndim:
            logger.warning(f"nautilus_n_live ({self.nautilus_n_live}) < 2*ndim ({2*self.ndim}) "
                          "may lead to poor sampling")
        if self.nautilus_n_points_min < self.ndim:
            raise ValueError("nautilus_n_points_min must be >= ndim")
        if self.nautilus_n_eff < 100:
            logger.warning(f"nautilus_n_eff ({self.nautilus_n_eff}) is very small")
        
        logger.info(f"Nautilus configured for final iteration: n_live={self.nautilus_n_live}, "
                   f"n_eff={self.nautilus_n_eff}")
    
    def _configure_nuts_parameters(self) -> None:
        """Configure NUTS sampling parameters for final iteration."""
        self.nuts_step_size = self.read_ini("nuts_step_size", float, 0.05)
        self.nuts_use_fixed_step_size = self.read_ini("nuts_use_fixed_step_size", bool, False)
        self.nuts_fixed_step_size = self.read_ini("nuts_fixed_step_size", float, 0.3)
        self.nuts_max_tree_depth = self.read_ini("nuts_max_tree_depth", int, 12)
        self.nuts_max_energy_diff = self.read_ini("nuts_max_energy_diff", float, 1000.0)
        self.nuts_unrolled_leapfrog_steps = self.read_ini("nuts_unrolled_leapfrog_steps", int, 1) #do not increase it unless profiling shows benefit
        self.nuts_parallel_iterations = self.read_ini("nuts_parallel_iterations", int, 10)
        self.nuts_num_adaptation_steps = self.read_ini("nuts_num_adaptation_steps", int, 1000)
        self.nuts_num_burnin_steps = self.read_ini("nuts_num_burnin_steps", int, 1000)
        self.nuts_num_results = self.read_ini("nuts_num_results", int, 2000)
        self.nuts_num_chains = self.read_ini("nuts_num_chains", int, 1)
        
        # Validate NUTS parameters
        if self.nuts_step_size <= 0:
            raise ValueError("nuts_step_size must be > 0")
        if self.nuts_max_tree_depth < 1:
            raise ValueError("nuts_max_tree_depth must be >= 1")
        if self.nuts_num_results < 100:
            logger.warning(f"nuts_num_results ({self.nuts_num_results}) is very small")
        if self.nuts_num_chains < 1:
            raise ValueError("nuts_num_chains must be >= 1")
        
        logger.info(f"NUTS configured for final iteration: step_size={self.nuts_step_size}, "
                   f"max_tree_depth={self.nuts_max_tree_depth}, num_results={self.nuts_num_results}")
    
    def _configure_advanced_options(self) -> None:
        """Configure advanced and experimental options."""
        # Pipeline emulation settings
        self.last_emulated_module = self.read_ini("last_emulated_module", str, "")
        
        # Tempering settings
        tempering = self.read_ini("tempering", float, 0.05)
        self.tempering = np.full(self.max_iterations, tempering)
        
        tempering_file = self.read_ini("tempering_file", str, "")
        if tempering_file:
            try:
                custom_tempering = np.genfromtxt(tempering_file)
                if len(custom_tempering) < self.max_iterations:
                    logger.warning(f"Tempering file has {len(custom_tempering)} values but "
                                 f"{self.max_iterations} iterations requested")
                self.tempering = custom_tempering[:self.max_iterations]
            except Exception as e:
                raise ValueError(f"Failed to read tempering file {tempering_file}: {e}")
        
        logger.info(f"Tempering schedule: {self.tempering}")
        
        # Random seed
        self.seed = self.read_ini("seed", int, 0)
        if self.seed == 0:
            self.seed = None  # Use random seed
        
        # Neural network options (currently fixed)
        self.data_trafo = self.read_ini("data_trafo", str, "log_norm")
        self.n_pca = self.read_ini("n_pca", int, 32)
        self.loss_function = self.read_ini("loss_function", str, "standard")
        
        if self.loss_function.startswith("weighted") and self.keys:
            raise ValueError("Weighted loss function can only be used with full data vector "
                           "(empty keys parameter)")
        
        # Nautilus configuration for final iteration
        self.use_nautilus_final = self.read_ini("use_nautilus_final", bool, False)
        if self.use_nautilus_final:
            self._configure_nautilus_parameters()
            # Add log_weight column to output when using nautilus (for all iterations)
            if self.output is not None:
                self.output.add_column("log_weight", float)
        
        # NUTS configuration for final iteration
        self.use_nuts_final = self.read_ini("use_nuts_final", bool, False)
        if self.use_nuts_final:
            self._configure_nuts_parameters()
        
        # Neural network architecture
        self.nn_model = self.read_ini("nn_model", str, 'MLP')  # Could be made configurable in future

