"""
Emulator training and loading for ROSE sampler.

This module contains methods for training neural network emulators and
loading pre-trained emulators.
"""

import os
import logging
from timeit import default_timer
from typing import Any, Optional

import numpy as np

from .nn_emulator import NNEmulator

logger = logging.getLogger(__name__)


def _tagged_row_keys(item: dict[str, Any], n: int) -> list[tuple[tuple[int, int, float], int]]:
    """Row keys for (bin1, bin2, angle) with ordinal disambiguation.

    The same triple can appear more than once per theory block (e.g. different
    probe combinations sharing bin indices). Pair duplicates by occurrence order
    so the k-th duplicate matches between trained and current metadata.
    """
    bin1 = np.asarray(item["bin1"])
    bin2 = np.asarray(item["bin2"])
    angle = np.asarray(item["angle"])
    counts: dict[tuple[int, int, float], int] = {}
    keys: list[tuple[tuple[int, int, float], int]] = []
    for i in range(n):
        triple = (
            int(bin1[i]),
            int(bin2[i]),
            round(float(angle[i]), 12),
        )
        occ = counts.get(triple, 0)
        counts[triple] = occ + 1
        keys.append((triple, occ))
    return keys


class RoseEmulatorManagementMixin:
    """Mixin class providing emulator management methods for RoseSampler."""

    def _save_vector_metadata(self, model_filename: str) -> None:
        """Persist fiducial vector metadata into emulator npz file."""
        if not hasattr(self, "fiducial_vector_metadata"):
            return

        info_file = model_filename + ".npz"
        if not os.path.exists(info_file):
            return

        with np.load(info_file, allow_pickle=True) as data:
            save_dict = {key: data[key] for key in data.files}

        save_dict["rose_vector_metadata"] = np.array(
            self.fiducial_vector_metadata, dtype=object
        )
        np.savez_compressed(info_file, **save_dict)

    def _compute_output_indices_from_metadata(
        self, trained_metadata: list[dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Build map from trained vector indices to current fiducial ordering."""
        current_metadata = getattr(self, "fiducial_vector_metadata", None)
        if not current_metadata:
            return None

        indices = []
        trained_offset = 0
        current_offset = 0

        for train_item, current_item in zip(trained_metadata, current_metadata):
            train_size = int(train_item.get("size", 0))
            current_size = int(current_item.get("size", 0))

            train_has_tags = all(k in train_item for k in ("angle", "bin1", "bin2"))
            current_has_tags = all(k in current_item for k in ("angle", "bin1", "bin2"))

            if train_has_tags and current_has_tags:
                train_keys: dict[tuple[tuple[int, int, float], int], int] = {}
                for i, key in enumerate(_tagged_row_keys(train_item, train_size)):
                    train_keys[key] = trained_offset + i
                    #print('train_keys: ', key, train_keys[key])

                for i, key in enumerate(_tagged_row_keys(current_item, current_size)):
                    if key not in train_keys:
                        return None
                    #print('current_keys: ', key, train_keys[key])
                    indices.append(train_keys[key])
            else:
                if train_size != current_size:
                    return None
                indices.extend(range(trained_offset, trained_offset + train_size))

            trained_offset += train_size
            current_offset += current_size
        #print('indices: ', indices)
        return np.asarray(indices, dtype=int)
    
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
        self._save_vector_metadata(model_filename)
        end_time = default_timer()
        
        logger.info(f"Emulator training took {end_time - start_time:.1f} seconds")
        
        # Store emulator and update pipeline
        self.emulator = emu
        self.emu_module.data.set_emulator(emu)

    def load_emulator(self, path: Optional[str] = None) -> None:
        """Load pre-trained emulator from file.
        
        This method loads a previously trained emulator, useful for
        parameter studies or continuing interrupted runs.
        
        When load_emu_filename is not set (e.g. on worker processes that did not
        run training), path can be provided by the main process (e.g. via a
        closure passed to the pool) so workers load the same iteration. If path
        is not provided, uses save_outputs_dir/emumodel_{iterations+1} (stale on
        workers).
        
        Args:
            path: Optional path to the model (directory + basename, no .npz).
                  When set, this overrides load_emu_filename and iterations.
        
        Expected file structure:
        - Base model: {load_emu_filename} (e.g., emumodel_5)
        - Info file: {load_emu_filename}.npz (e.g., emumodel_5.npz)
        """
        if path is not None:
            load_path = path
        elif self.load_emu_filename:
            load_path = self.load_emu_filename
        else:
            load_path = f'{self.save_outputs_dir}/emumodel_{self.iterations + 1}'
        info_file = load_path + ".npz"
        
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"Emulator info file not found: {info_file}")

        logger.info(f"Loading pre-trained emulator from {load_path}")
        
        # Load info to get the correct output size and parameters
        data_trafo_init = self.data_trafo
        with np.load(info_file, allow_pickle=True) as data:
            if "data_transformation" in data:
                dt = data["data_transformation"]
                if hasattr(dt, "item"):
                    dt = dt.item()
                if isinstance(dt, dict) and "data_trafo" in dt:
                    data_trafo_init = str(dt["data_trafo"])
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
                # Fallback: throw error since output size cannot be determined
                raise ValueError("Could not determine output size from info file. Please ensure that the 'modes' entry is present in the emulator info file (.npz), or retrain the emulator.")
            trained_metadata = None
            if "rose_vector_metadata" in data:
                trained_metadata = list(data["rose_vector_metadata"])

        # Match training transform so __init__ logging and state agree before load()
        emu = NNEmulator(
            model_parameters,
            np.ones(output_size),
            data_trafo=data_trafo_init,
        )
        
        # Load trained model - CosmoPowerNN expects the .npz file
        emu.load(load_path)
        
        self.emulator = emu
        self.emulator_output_indices = None
        expected_size = int(self.fiducial_data_vector.size)
        if output_size != expected_size:
            if trained_metadata is not None:
                output_indices = self._compute_output_indices_from_metadata(trained_metadata)
                if output_indices is not None and output_indices.size == expected_size:
                    self.emulator_output_indices = output_indices
                    logger.info(
                        "Applying output index remapping for loaded emulator "
                        f"(trained size={output_size}, current size={expected_size})"
                    )
                else:
                    raise ValueError(
                        "Loaded emulator output size does not match current data vector size, "
                        "and automatic remapping from metadata failed. "
                        "Please retrain emulator with the current scale cuts."
                    )
            else:
                raise ValueError(
                    "Loaded emulator output size does not match current data vector size "
                    f"(trained={output_size}, current={expected_size}) and no metadata is available "
                    "for remapping. Please retrain emulator once with metadata support."
                )
        self.emu_module.data.set_emulator(emu)
        self.emu_module.data.output_indices = self.emulator_output_indices
        
        logger.info(f"Pre-trained emulator loaded successfully with output size {output_size}")

