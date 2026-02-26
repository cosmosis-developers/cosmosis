"""
Emulator training and loading for ROSE sampler.

This module contains methods for training neural network emulators and
loading pre-trained emulators.
"""

import os
import logging
from timeit import default_timer
from typing import Any

import numpy as np

from .nn_emulator import NNEmulator

logger = logging.getLogger(__name__)


class RoseEmulatorManagementMixin:
    """Mixin class providing emulator management methods for RoseSampler."""
    
    def train_emulator(self) -> None:
        """Train neural network emulator on current training data.
        
        This method:
        1. Sets up training parameters
        2. Trains the neural network
        3. Updates the emulated pipeline
        """
        n_samp, n_in = self.unit_sample.shape
        n_out = self.sample_data_vectors.shape[1]
        
        logger.info(f"Training emulator: {n_in} parameters -> {n_out} outputs "
                   f"using {n_samp} training points")
        
        # Set up training parameters
        model_filename = f'{self.save_outputs_dir}/emumodel_{self.iterations+1}'
        kwargs = {
            "model_filename": model_filename,
            "n_cycles": self.training_iterations,
            "batch_size": self.batch_size * (self.iterations + 1)
        }
        
        # Create emulator instance
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        logger.info(f"Model parameters: {model_parameters}")
        
        emu = NNEmulator(
            model_parameters, 
            np.arange(n_out), 
            self.nn_model, 
            self.loss_function,
            self.iterations + 1,
            self.data_trafo, 
            self.n_pca, 
            self.data, 
            self.inv_cov
        )
        
        # Prepare training data
        X = {str(param): self.sample[:, i] 
             for i, param in enumerate(self.pipeline.varied_params)}
        
        # Train emulator
        start_time = default_timer()
        emu.train(X, self.sample_data_vectors, **kwargs)
        end_time = default_timer()
        
        logger.info(f"Emulator training took {end_time - start_time:.1f} seconds")
        
        # Store emulator and update pipeline
        self.emulator = emu
        self.emu_module.data.set_emulator(emu)

    def load_emulator(self) -> None:
        """Load pre-trained emulator from file.
        
        This method loads a previously trained emulator, useful for
        parameter studies or continuing interrupted runs.
        
        When load_emu_filename is not set (e.g. on worker processes that did not
        run training), loads from the current-run path: save_outputs_dir/emumodel_{iterations+1}.
        
        Expected file structure:
        - Base model: {load_emu_filename} (e.g., emumodel_5)
        - Info file: {load_emu_filename}.npz (e.g., emumodel_5.npz)
        """
        # Use current-run path when no pre-trained path is specified (e.g. workers loading after main trained)
        load_path = self.load_emu_filename
        if not load_path:
            load_path = f'{self.save_outputs_dir}/emumodel_{self.iterations + 1}'
        info_file = load_path + ".npz"
        
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Emulator info file not found: {info_file}")

        logger.info(f"Loading pre-trained emulator from {load_path}")
        
        # Load info to get the correct output size and parameters
        with np.load(info_file) as data:
            # Get the actual parameters used in training
            if 'parameters' in data:
                model_parameters = list(data['parameters'])
                logger.info(f"Using parameters from trained model: {model_parameters}")
            else:
                # Fallback to pipeline parameters
                model_parameters = [str(param) for param in self.pipeline.varied_params]
                logger.warning("Could not find parameters in info file, using pipeline parameters")
            
            # Get output size from modes
            if 'modes' in data:
                output_size = len(data['modes'])
                logger.info(f"Output size from modes: {output_size}")
            else:
                # Fallback: try to infer from other data
                logger.warning("Could not determine output size from info file, using default")
                output_size = 1000  # Default fallback

        emu = NNEmulator(model_parameters, np.ones(output_size))
        
        # Load trained model - CosmoPowerNN expects the .npz file
        emu.load(load_path)
        
        self.emulator = emu
        self.emu_module.data.set_emulator(emu)
        
        logger.info(f"Pre-trained emulator loaded successfully with output size {output_size}")

