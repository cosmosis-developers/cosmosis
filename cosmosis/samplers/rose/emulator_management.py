"""
Emulator training and loading for ROSE sampler.

This module contains methods for training neural network emulators and
loading pre-trained emulators. One independent emulator is trained per
likelihood so they can later be configured/tuned independently.
"""

import os
import logging
from timeit import default_timer
from typing import Any, Dict, Optional

import numpy as np

from .nn_emulator import NNEmulator
from .utils import mkdir

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

    def _save_vector_metadata(self, model_filename: str, likelihood_name: str) -> None:
        """Persist per-likelihood fiducial vector metadata into emulator npz file."""
        metadata_all = getattr(self, "fiducial_vector_metadata", None)
        if not metadata_all:
            return
        item = metadata_all.get(likelihood_name)
        if item is None:
            return

        info_file = model_filename + ".npz"
        if not os.path.exists(info_file):
            return

        with np.load(info_file, allow_pickle=True) as data:
            save_dict = {key: data[key] for key in data.files}

        save_dict["rose_vector_metadata"] = np.array([item], dtype=object)
        save_dict["rose_likelihood_name"] = np.array(likelihood_name)
        np.savez_compressed(info_file, **save_dict)

    def _compute_output_indices_from_metadata(
        self, trained_item: dict[str, Any], likelihood_name: str
    ) -> Optional[np.ndarray]:
        """Build map from trained vector indices to current fiducial ordering for
        a single likelihood.
        """
        metadata_all = getattr(self, "fiducial_vector_metadata", None)
        if not metadata_all:
            return None
        current_item = metadata_all.get(likelihood_name)
        if current_item is None:
            return None

        train_size = int(trained_item.get("size", 0))
        current_size = int(current_item.get("size", 0))

        train_has_tags = all(k in trained_item for k in ("angle", "bin1", "bin2"))
        current_has_tags = all(k in current_item for k in ("angle", "bin1", "bin2"))

        if train_has_tags and current_has_tags:
            train_keys: dict[tuple[tuple[int, int, float], int], int] = {}
            for i, key in enumerate(_tagged_row_keys(trained_item, train_size)):
                train_keys[key] = i

            indices = []
            for key in _tagged_row_keys(current_item, current_size):
                if key not in train_keys:
                    return None
                indices.append(train_keys[key])
            return np.asarray(indices, dtype=int)

        if train_size != current_size:
            return None
        return np.arange(current_size, dtype=int)
    
    def train_emulator(self) -> None:
        """Train one independent emulator per likelihood on current training data.

        Each likelihood gets its own :class:`NNEmulator` trained on its own
        slice of the sample (``self.sample_data_vectors[name]``) using the
        matching reference data vector and inverse covariance. The resulting
        dict of emulators is handed to the :class:`EmulatorModule` so that the
        downstream pipeline can look each one up by name.
        """
        n_samp, n_in = self.unit_sample.shape
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        logger.info(f"Model parameters: {model_parameters}")

        iter_dir = os.path.join(self.save_outputs_dir, f"emumodel_{self.iterations + 1}")
        mkdir(iter_dir)

        X = {str(param): self.sample[:, i]
             for i, param in enumerate(self.pipeline.varied_params)}

        trained_emulators: Dict[str, NNEmulator] = {}
        start_time = default_timer()

        for name in self.likelihood_names:
            y = self.sample_data_vectors[name]
            n_out = y.shape[1]
            logger.info(
                f"Training emulator for '{name}': {n_in} params -> {n_out} outputs "
                f"using {n_samp} training points"
            )

            model_filename = os.path.join(iter_dir, name)
            kwargs = {
                "model_filename": model_filename,
                "n_cycles": self._resolve_training_setting("training_iterations", name),
                "batch_size": self._resolve_training_setting("batch_size", name)
                               * (self.iterations + 1),
            }

            emu = NNEmulator(
                model_parameters,
                np.arange(n_out),
                self._resolve_training_setting("nn_model", name),
                self._resolve_training_setting("loss_function", name),
                self.iterations + 1,
                self._resolve_training_setting("data_trafo", name),
                self._resolve_training_setting("n_pca", name),
                self.data.get(name),
                self.inv_cov.get(name),
            )
            emu.train(X, y, **kwargs)
            self._save_vector_metadata(model_filename, name)
            trained_emulators[name] = emu

        end_time = default_timer()
        logger.info(
            f"Trained {len(trained_emulators)} likelihood emulator(s) in "
            f"{end_time - start_time:.1f} seconds"
        )

        self.emulator = trained_emulators
        self.emu_module.data.set_emulator(trained_emulators)

    def _resolve_training_setting(self, attr: str, likelihood_name: str) -> Any:
        """Resolve a training setting with optional per-likelihood override.

        The sampler stores each setting (e.g. ``data_trafo``, ``loss_function``,
        ``n_pca``) as either a single value (applied to all likelihoods) or a
        dict mapping likelihood name to value. Missing entries fall back to
        the default stored under the ``__default__`` key or, if absent, to
        whichever single value was parsed.
        """
        value = getattr(self, attr)
        if isinstance(value, dict):
            if likelihood_name in value:
                return value[likelihood_name]
            if "__default__" in value:
                return value["__default__"]
            raise KeyError(
                f"No value for setting '{attr}' for likelihood '{likelihood_name}'"
            )
        return value

    def load_emulator(self, path: Optional[str] = None) -> None:
        """Load per-likelihood pre-trained emulators from disk.

        Expected directory layout (as written by :meth:`train_emulator`):
            {load_dir}/{likelihood_name}.npz

        Args:
            path: Optional directory containing per-likelihood model files.
                  When set this overrides ``load_emu_filename`` and iterations.
        """
        if path is not None:
            load_dir = path
        elif self.load_emu_filename:
            load_dir = self.load_emu_filename
        else:
            load_dir = os.path.join(
                self.save_outputs_dir, f"emumodel_{self.iterations + 1}"
            )

        if not os.path.isdir(load_dir):
            raise FileNotFoundError(
                f"Emulator directory not found: {load_dir}. Expected one .npz "
                f"per likelihood."
            )

        logger.info(f"Loading pre-trained emulators from {load_dir}")

        loaded_emulators: Dict[str, NNEmulator] = {}
        output_indices: Dict[str, Optional[np.ndarray]] = {}

        for name in self.likelihood_names:
            base_path = os.path.join(load_dir, name)
            info_file = base_path + ".npz"
            if not os.path.exists(info_file):
                raise FileNotFoundError(
                    f"Emulator info file for likelihood '{name}' not found: {info_file}"
                )

            default_trafo = self._resolve_training_setting("data_trafo", name)
            with np.load(info_file, allow_pickle=True) as data:
                data_trafo_init = default_trafo
                if "data_transformation" in data:
                    dt = data["data_transformation"]
                    if hasattr(dt, "item"):
                        dt = dt.item()
                    if isinstance(dt, dict) and "data_trafo" in dt:
                        data_trafo_init = str(dt["data_trafo"])
                if "parameters" in data:
                    model_parameters = list(data["parameters"])
                else:
                    model_parameters = [str(p) for p in self.pipeline.varied_params]
                    logger.warning(
                        f"[{name}] Could not find parameters in info file, "
                        "using pipeline parameters"
                    )
                if "modes" in data:
                    output_size = len(data["modes"])
                else:
                    raise ValueError(
                        f"[{name}] Could not determine output size from info file. "
                        "Please ensure 'modes' is present, or retrain the emulator."
                    )
                trained_item = None
                if "rose_vector_metadata" in data:
                    md = list(data["rose_vector_metadata"])
                    if md:
                        trained_item = md[0]

            emu = NNEmulator(
                model_parameters,
                np.ones(output_size),
                data_trafo=data_trafo_init,
            )
            emu.load(base_path)
            loaded_emulators[name] = emu

            expected_size = int(self.data_vector_sizes[name])
            if output_size != expected_size:
                if trained_item is None:
                    raise ValueError(
                        f"[{name}] Loaded emulator output size ({output_size}) does not "
                        f"match current data vector size ({expected_size}) and no metadata "
                        "is available for remapping. Please retrain with metadata support."
                    )
                mapped = self._compute_output_indices_from_metadata(trained_item, name)
                if mapped is None or mapped.size != expected_size:
                    raise ValueError(
                        f"[{name}] Loaded emulator output size does not match current data "
                        "vector size and automatic remapping from metadata failed. "
                        "Please retrain with the current scale cuts."
                    )
                output_indices[name] = mapped
                logger.info(
                    f"[{name}] Applying output index remapping for loaded emulator "
                    f"(trained size={output_size}, current size={expected_size})"
                )
            else:
                output_indices[name] = None

            logger.info(
                f"[{name}] Pre-trained emulator loaded successfully "
                f"(output size {output_size})"
            )

        self.emulator = loaded_emulators
        self.emulator_output_indices = output_indices
        self.emu_module.data.set_emulator(loaded_emulators)
        self.emu_module.data.output_indices = output_indices

