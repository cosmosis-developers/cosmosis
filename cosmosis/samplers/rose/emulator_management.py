"""
Emulator training and loading for ROSE sampler.

This module contains methods for training neural network emulators and
loading pre-trained emulators. One independent emulator is trained per
likelihood so they can later be configured/tuned independently.
"""

import os
import logging
from timeit import default_timer
from typing import Any, Dict, List, Optional

import numpy as np


from .utils import mkdir
from .amplitude_prefactor import parse_amplitude_prefactor
from .vector_blocks import (
    slice_vector_metadata,
    spectrum_mode_indices,
    spectrum_model_parameters,
)

logger = logging.getLogger(__name__)


def _tagged_row_keys(item: dict[str, Any], n: int) -> list[tuple[tuple, int]]:
    """Row keys for (name?, bin1, bin2, angle) with ordinal disambiguation.

    The same (bin1, bin2, angle) triple can appear more than once per theory
    block (different probes). When spectrum ``name`` is available it is included
    in the key; otherwise occurrence order disambiguates duplicates.
    """
    bin1 = np.asarray(item["bin1"])
    bin2 = np.asarray(item["bin2"])
    angle = np.asarray(item["angle"])
    names = item.get("name")
    if names is not None:
        names = np.asarray(names).astype(str)
        if names.size != n:
            names = None
    counts: dict[tuple, int] = {}
    keys: list[tuple[tuple, int]] = []
    for i in range(n):
        if names is not None:
            triple: tuple = (
                str(names[i]),
                int(bin1[i]),
                int(bin2[i]),
                round(float(angle[i]), 12),
            )
        else:
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
            # Prefer spectrum-name keys when *both* sides have them; otherwise
            # fall back to (bin1, bin2, angle) so older emulators still remap.
            train_use = dict(trained_item)
            current_use = dict(current_item)
            if "name" not in trained_item or "name" not in current_item:
                train_use.pop("name", None)
                current_use.pop("name", None)

            train_keys: dict[tuple, int] = {}
            for i, key in enumerate(_tagged_row_keys(train_use, train_size)):
                train_keys[key] = i

            indices = []
            for key in _tagged_row_keys(current_use, current_size):
                if key not in train_keys:
                    return None
                indices.append(train_keys[key])
            return np.asarray(indices, dtype=int)

        if train_size != current_size:
            return None
        return np.arange(current_size, dtype=int)
    
    def train_emulator(self, model_version: Optional[int] = None) -> None:
        """Train one independent emulator per likelihood on current training data.

        Each likelihood gets its own :class:`NNEmulator` trained on its own
        slice of the sample (``self.sample_data_vectors[name]``) using the
        matching reference data vector and inverse covariance. The resulting
        dict of emulators is handed to the :class:`EmulatorModule` so that the
        downstream pipeline can look each one up by name.

        Args:
            model_version: Integer id for the on-disk model directory
                ``emumodel_{model_version}``. Defaults to ``self.iterations + 1``
                (the normal one-training-per-iteration numbering). The final-step
                KL-convergence loop passes successive ids (N+1, N+2, ...) so each
                retrain writes a *new* directory instead of overwriting the last
                one, and the freshly trained model is both kept in memory and
                (being on disk) picked up by the MPI worker processes.
        """
        from .nn_emulator import NNEmulator
        from .spectrum_emulator import CompositeSpectrumEmulator

        if model_version is None:
            model_version = self.iterations + 1
        self._current_emu_version = model_version

        n_samp, n_in = self.unit_sample.shape
        model_parameters = [str(param) for param in self.pipeline.varied_params]
        logger.info(f"Model parameters: {model_parameters}")

        pruned = getattr(self, "_pruned_training", None)
        if pruned is not None:
            n_pruned = len(pruned["sample_likes"])
            logger.info(
                f"emumodel_{model_version}: training on {n_samp} points "
                f"({n_pruned} outliers pruned from the training set and "
                f"stashed under pruned/* in total_training_set.npz)"
            )
        else:
            logger.info(
                f"emumodel_{model_version}: training on {n_samp} points"
            )

        iter_dir = os.path.join(self.save_dir, f"emumodel_{model_version}")
        mkdir(iter_dir)

        X = {str(param): self.sample[:, i]
             for i, param in enumerate(self.pipeline.varied_params)}

        trained_emulators: Dict[str, Any] = {}
        start_time = default_timer()

        for name in self.likelihood_names:
            y = self.sample_data_vectors[name]
            n_out = y.shape[1]
            split = parse_amplitude_prefactor(
                self._resolve_training_setting("spectrum_emulators", name)
            )
            if split:
                trained_emulators[name] = self._train_spectrum_emulators(
                    name=name,
                    X=X,
                    y=y,
                    model_parameters=model_parameters,
                    n_samp=n_samp,
                    n_in=n_in,
                    iter_dir=iter_dir,
                )
            else:
                logger.info(
                    f"Training emulator for '{name}': {n_in} params -> {n_out} outputs "
                    f"using {n_samp} training points"
                )
                model_filename = os.path.join(iter_dir, name)
                kwargs = self._training_kwargs(name, model_filename, n_samp)
                emu = self._build_nn_emulator(
                    name=name,
                    model_parameters=model_parameters,
                    n_out=n_out,
                    datavector=self.data.get(name),
                    inv_cov=self.inv_cov.get(name),
                    metadata=(getattr(self, "fiducial_vector_metadata", None) or {}).get(
                        name
                    ),
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
        # Record which on-disk model this (master) process now holds in memory,
        # so the worker-synchronization logic in utils._ensure_emulator does not
        # needlessly reload it. iter_dir is exactly the path that worker
        # processes are told to (re)load from via _worker_emu_model_path().
        self._loaded_emu_path = iter_dir

    def _training_kwargs(
        self, name: str, model_filename: str, n_samp: int
    ) -> Dict[str, Any]:
        """Build CosmoPowerNN / NNEmulator.train kwargs for one likelihood."""
        n_cycles_per_training = self._resolve_training_setting(
            "n_cycles_per_training", name
        )
        n_hidden = list(self._resolve_training_setting("n_hidden", name))
        batch_sizes = list(self._resolve_training_setting("batch_sizes", name))
        if len(batch_sizes) == 1:
            batch_sizes = batch_sizes * n_cycles_per_training
        elif len(batch_sizes) != n_cycles_per_training:
            raise ValueError(
                f"batch_sizes for likelihood '{name}' has length "
                f"{len(batch_sizes)} but n_cycles_per_training is "
                f"{n_cycles_per_training} (use a single value to broadcast)"
            )
        ref_n = getattr(self, "_batch_sizes_ref_n", None)
        if ref_n is None or ref_n <= 0:
            self._batch_sizes_ref_n = n_samp
            ref_n = n_samp
        scale = np.sqrt(n_samp / ref_n)
        val_split = float(self._resolve_training_setting("validation_split", name))
        if scale != 1.0:
            batch_sizes = [max(1, int((1. - val_split) * n_samp / 5)) for b in batch_sizes]
            logger.info(
                f"Scaled batch_sizes for '{name}' by {scale:.3f} "
                f"(n_train={n_samp}, ref={ref_n}): {batch_sizes}"
            )
        kwargs: Dict[str, Any] = {
            "model_filename": model_filename,
            "n_cycles_per_training": n_cycles_per_training,
            "batch_sizes": batch_sizes,
            "test_split": self._resolve_training_setting("validation_split", name),
            "n_hidden": n_hidden,
        }
        for attr, kw in (
            ("learning_rates", "learning_rates"),
            ("gradient_accumulation_steps", "gradient_accumulation_steps"),
            ("patience_values", "patience_values"),
            ("max_epochs", "max_epochs"),
        ):
            values = self._resolve_optional_training_setting(attr, name)
            if values is None:
                continue
            values = list(values)
            if len(values) != n_cycles_per_training:
                raise ValueError(
                    f"{attr} for likelihood '{name}' has length "
                    f"{len(values)} but n_cycles_per_training is "
                    f"{n_cycles_per_training}"
                )
            kwargs[kw] = values
        return kwargs

    def _build_nn_emulator(
        self,
        *,
        name: str,
        model_parameters: List[str],
        n_out: int,
        datavector: Optional[np.ndarray],
        inv_cov: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]],
        spectra: Optional[Any] = None,
    ):
        """Construct and configure an :class:`NNEmulator` for training."""
        from .nn_emulator import NNEmulator

        if spectra is None:
            spectra = self._resolve_training_setting(
                "amplitude_prefactor_spectra", name
            )
        emu = NNEmulator(
            model_parameters,
            np.arange(n_out),
            self._resolve_training_setting("nn_model", name),
            self._resolve_training_setting("loss_function", name),
            self.iterations + 1,
            self._resolve_training_setting("data_transformation", name),
            self._resolve_training_setting("n_pca", name),
            datavector,
            inv_cov,
            amplitude_prefactor=self._resolve_training_setting(
                "amplitude_prefactor", name
            ),
        )
        emu.configure_amplitude_prefactor(metadata, spectra=spectra)
        if str(
            self._resolve_training_setting("data_transformation", name)
        ) == "PCA_per_bin":
            emu.configure_bin_pair_pca(metadata, spectra=spectra)
        return emu

    def _train_spectrum_emulators(
        self,
        *,
        name: str,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        model_parameters: List[str],
        n_samp: int,
        n_in: int,
        iter_dir: str,
    ):
        """Train one NNEmulator per spectrum and wrap as a composite."""
        from .spectrum_emulator import CompositeSpectrumEmulator

        metadata_all = getattr(self, "fiducial_vector_metadata", None) or {}
        metadata = metadata_all.get(name)
        if not metadata:
            raise ValueError(
                f"spectrum_emulators=T for '{name}' requires fiducial vector "
                "metadata (name/bin1/bin2 or angle+bin1+bin2)."
            )
        spectra = self._resolve_training_setting("amplitude_prefactor_spectra", name)
        mode_indices = spectrum_mode_indices(metadata, spectra=spectra)
        spectrum_order = list(mode_indices.keys())
        if not spectrum_order:
            raise ValueError(
                f"spectrum_emulators=T for '{name}' found no spectrum slices "
                "in vector metadata."
            )
        n_out = int(y.shape[1])
        logger.info(
            f"Training spectrum emulators for '{name}': {n_out} modes -> "
            f"{spectrum_order} ({n_samp} training points)"
        )

        data_full = self.data.get(name)
        inv_full = self.inv_cov.get(name)
        emulators = {}
        for spectrum, idx in mode_indices.items():
            params_s = spectrum_model_parameters(spectrum, model_parameters)
            if not params_s:
                raise ValueError(
                    f"[{name}/{spectrum}] spectrum_model_parameters removed all "
                    "varied parameters; check exclusion lists."
                )
            X_s = {p: X[p] for p in params_s}
            y_s = y[:, idx]
            meta_s = slice_vector_metadata(metadata, idx, spectrum=spectrum)
            data_s = None
            inv_s = None
            if data_full is not None:
                data_s = np.asarray(data_full).ravel()[idx]
            if inv_full is not None:
                inv_arr = np.atleast_2d(np.asarray(inv_full, dtype=float))
                inv_s = inv_arr[np.ix_(idx, idx)]
            dropped = [p for p in model_parameters if p not in params_s]
            logger.info(
                f"  [{name}/{spectrum}] {len(params_s)}/{len(model_parameters)} "
                f"params -> {y_s.shape[1]} modes"
                + (f" (dropped {dropped})" if dropped else "")
            )
            model_filename = os.path.join(iter_dir, f"{name}__{spectrum}")
            kwargs = self._training_kwargs(name, model_filename, n_samp)
            # Pass only this spectrum name so amp/PCA name inference matches the
            # sliced vector (full data_sets list would expect 3 blocks).
            emu = self._build_nn_emulator(
                name=name,
                model_parameters=params_s,
                n_out=int(y_s.shape[1]),
                datavector=data_s,
                inv_cov=inv_s,
                metadata=meta_s,
                spectra=spectrum,
            )
            emu.ignore_extra_params = True
            emu.train(X_s, y_s, **kwargs)
            emulators[spectrum] = emu

        composite = CompositeSpectrumEmulator(
            emulators=emulators,
            mode_indices=mode_indices,
            n_modes=n_out,
            spectrum_order=spectrum_order,
            likelihood_name=name,
        )
        composite.trained = True
        model_filename = os.path.join(iter_dir, name)
        composite.save_to(model_filename)
        # Persist sliced metadata onto each spectrum file (after save_to, which
        # rewrites the npz) and full metadata onto the composite manifest.
        for spectrum, idx in mode_indices.items():
            meta_s = slice_vector_metadata(metadata, idx, spectrum=spectrum)
            info_file = os.path.join(iter_dir, f"{name}__{spectrum}") + ".npz"
            if os.path.exists(info_file):
                with np.load(info_file, allow_pickle=True) as data:
                    save_dict = {key: data[key] for key in data.files}
                save_dict["rose_vector_metadata"] = np.array([meta_s], dtype=object)
                save_dict["rose_likelihood_name"] = np.array(name)
                save_dict["rose_spectrum_name"] = np.array(spectrum)
                np.savez_compressed(info_file, **save_dict)
        self._save_vector_metadata(model_filename, name)
        return composite

    def _worker_emu_model_path(self) -> Optional[str]:
        """Path token identifying the emulator the master is currently using.

        This is handed to worker processes (through the emcee/nautilus log-prob
        and prior-transform callables) so they load/reload the *same* emulator
        the master holds -- see :func:`utils._ensure_emulator`. For a run that
        trains from scratch this is the per-iteration model directory written by
        :meth:`train_emulator`; for a pre-trained run there is no per-iteration
        directory, so we return ``None`` and workers fall back to
        ``load_emulator(None)`` (i.e. ``load_emu_filename``).
        """
        if getattr(self, "trained_before", False):
            return None
        version = getattr(self, "_current_emu_version", 0) or (self.iterations + 1)
        return os.path.join(self.save_dir, f"emumodel_{version}")

    def _resolve_training_setting(self, attr: str, likelihood_name: str) -> Any:
        """Resolve a training setting with optional per-likelihood override.

        The sampler stores each setting (e.g. ``data_transformation``, ``loss_function``,
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

    def _resolve_optional_training_setting(
        self, attr: str, likelihood_name: str
    ) -> Any:
        """Like :meth:`_resolve_training_setting`, but missing entries mean unset.

        Used for optional schedule overrides (``learning_rates``,
        ``patience_values``, ...). A partial per-likelihood dict only overrides
        the listed likelihoods; others fall back to the built-in auto schedule
        (``None``) instead of raising.
        """
        value = getattr(self, attr)
        if value is None:
            return None
        if isinstance(value, dict):
            if likelihood_name in value:
                return value[likelihood_name]
            if "__default__" in value:
                return value["__default__"]
            return None
        return value

    def _resolve_emu_base_path(
        self, name: str, load_setting: Any, load_dir: Optional[str]
    ) -> str:
        """Resolve the emulator base path (without ``.npz``) for a likelihood.

        ``load_setting`` is either a single directory/base path (in which case
        ``load_dir`` is used and the file is ``{load_dir}/{name}.npz``) or a
        per-likelihood dict mapping likelihood name to a path. A per-likelihood
        value may be given as a directory (``{dir}/{name}.npz``), a base path
        (``{base}.npz``), or the full ``.npz`` file itself.
        """
        if not isinstance(load_setting, dict):
            return os.path.join(load_dir, name)

        if name in load_setting:
            val = load_setting[name]
        elif "__default__" in load_setting:
            val = os.path.join(load_setting["__default__"], name)
        else:
            raise FileNotFoundError(
                f"No emulator path specified for likelihood '{name}' in "
                "load_emu_filename. Provide a 'name:path' token per likelihood "
                "or a 'default:dir' fallback."
            )

        if val.endswith(".npz"):
            return val[:-4]
        if os.path.isdir(val):
            return os.path.join(val, name)
        return val

    def load_emulator(self, path: Optional[str] = None) -> None:
        """Load per-likelihood pre-trained emulators from disk.

        Expected directory layout (as written by :meth:`train_emulator`):
            {load_dir}/{likelihood_name}.npz

        ``load_emu_filename`` may instead specify per-likelihood paths using
        ``name:path`` tokens, allowing each likelihood's emulator to come from
        a different location.

        Args:
            path: Optional directory containing per-likelihood model files.
                  When set this overrides ``load_emu_filename`` and iterations.
        """
        from .nn_emulator import NNEmulator
        from .spectrum_emulator import CompositeSpectrumEmulator

        load_setting = self.load_emu_filename
        per_likelihood = path is None and isinstance(load_setting, dict)

        load_dir: Optional[str]
        if path is not None:
            load_dir = path
        elif per_likelihood:
            load_dir = None
        elif load_setting:
            load_dir = load_setting
        else:
            version = getattr(self, "_current_emu_version", 0) or (self.iterations + 1)
            load_dir = os.path.join(
                self.save_dir, f"emumodel_{version}"
            )

        if not per_likelihood:
            if not os.path.isdir(load_dir):
                raise FileNotFoundError(
                    f"Emulator directory not found: {load_dir}. Expected one .npz "
                    f"per likelihood."
                )
            logger.info(f"Loading pre-trained emulators from {load_dir}")
        else:
            logger.info("Loading pre-trained emulators from per-likelihood paths")

        loaded_emulators: Dict[str, Any] = {}
        output_indices: Dict[str, Optional[np.ndarray]] = {}

        for name in self.likelihood_names:
            base_path = self._resolve_emu_base_path(name, load_setting, load_dir)
            info_file = base_path + ".npz"
            if not os.path.exists(info_file):
                raise FileNotFoundError(
                    f"Emulator info file for likelihood '{name}' not found: {info_file}"
                )

            default_trafo = self._resolve_training_setting("data_transformation", name)
            if CompositeSpectrumEmulator.is_manifest(info_file):
                emu = CompositeSpectrumEmulator.load(
                    base_path, default_data_transformation=str(default_trafo)
                )
                with np.load(info_file, allow_pickle=True) as data:
                    output_size = int(np.asarray(data["n_modes"]).item())
                    trained_item = None
                    if "rose_vector_metadata" in data.files:
                        md = list(data["rose_vector_metadata"])
                        if md:
                            trained_item = md[0]
                for spectrum, sub in emu.emulators.items():
                    self._restore_amplitude_prefactor_on_load(name, sub, None)
                    sub.ignore_extra_params = True
                logger.info(
                    f"[{name}] Loaded composite spectrum emulators: "
                    f"{list(emu.spectrum_order)} (n_modes={output_size})"
                )
            else:
                with np.load(info_file, allow_pickle=True) as data:
                    data_transformation_init = default_trafo
                    if "data_transformation" in data:
                        dt = data["data_transformation"]
                        if hasattr(dt, "item"):
                            dt = dt.item()
                        if isinstance(dt, dict) and "data_transformation" in dt:
                            data_transformation_init = str(dt["data_transformation"])
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
                    data_transformation=data_transformation_init,
                )
                emu.load(base_path)
                self._restore_amplitude_prefactor_on_load(name, emu, trained_item)

                ignore_extra = getattr(self, "ignore_missing_emu_params", True)
                emu.ignore_extra_params = ignore_extra
                if ignore_extra:
                    trained_params = set(model_parameters)
                    current_params = [str(p) for p in self.pipeline.varied_params]
                    extra = [p for p in current_params if p not in trained_params]
                    missing = [p for p in trained_params if p not in current_params]
                    if extra:
                        logger.info(
                            f"[{name}] Ignoring pipeline parameter(s) not seen during "
                            f"training: {extra}"
                        )
                    if missing:
                        logger.warning(
                            f"[{name}] Emulator expects parameter(s) not varied by the "
                            f"current pipeline: {missing}. Predictions may be invalid."
                        )

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

    def _restore_amplitude_prefactor_on_load(
        self,
        name: str,
        emu,
        trained_item: Optional[Dict[str, Any]],
    ) -> None:
        """Rebuild / validate amplitude_prefactor after loading an NNEmulator."""
        ini_amp = parse_amplitude_prefactor(
            self._resolve_training_setting("amplitude_prefactor", name)
        )
        if emu.amplitude_prefactor_enabled and emu.amplitude_prefactor is None:
            md = trained_item or (
                getattr(self, "fiducial_vector_metadata", {}) or {}
            ).get(name)
            if md is None and getattr(emu, "vector_metadata", None) is not None:
                md = emu.vector_metadata
            if md is None:
                raise ValueError(
                    f"[{name}] Loaded emulator has amplitude_prefactor_enabled=T "
                    "but no amplitude_prefactor_state and no rose_vector_metadata "
                    "to rebuild it. Retrain with amplitude_prefactor=T."
                )
            emu.amplitude_prefactor_enabled = True
            emu.configure_amplitude_prefactor(
                md,
                spectra=self._resolve_training_setting(
                    "amplitude_prefactor_spectra", name
                ),
            )
        if ini_amp and emu.amplitude_prefactor is None:
            raise ValueError(
                f"[{name}] amplitude_prefactor=T in the ini, but the loaded "
                "emulator was trained without amplitude_prefactor. The prefactor "
                "cannot be applied after the fact (weights learned the unscaled "
                "vector). Load an emulator trained with amplitude_prefactor=T "
                "(e.g. rose_lssty1_nonlin_v4/emumodel_5), or retrain."
            )
        if (not ini_amp) and emu.amplitude_prefactor is not None:
            logger.warning(
                "[%s] amplitude_prefactor=F in the ini, but the loaded emulator "
                "was trained with amplitude_prefactor=T. Keeping the saved "
                "prefactor (required for correct predictions). The ini flag "
                "only affects training, not trained_before loads.",
                name,
            )
        if emu.amplitude_prefactor is not None:
            logger.info(
                "[%s] Loaded emulator uses amplitude_prefactor (%s): "
                "WL=%d XC=%d GC=%d",
                name,
                emu.amplitude_prefactor.amp_family,
                int(emu.amplitude_prefactor.wl_mask.sum()),
                int(emu.amplitude_prefactor.xc_mask.sum()),
                int(emu.amplitude_prefactor.gc_mask.sum()),
            )

