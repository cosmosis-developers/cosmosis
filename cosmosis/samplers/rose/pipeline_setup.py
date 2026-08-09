"""
Pipeline setup and configuration for ROSE sampler.

This module contains methods for setting up the emulated pipeline and
configuring the integration between the emulator and CosmoSIS pipeline.
"""

import logging
import types
import copy
from typing import Any

import numpy as np

from ...runtime import LikelihoodPipeline
from .emulator_module import EmulatorModule
from .utils import task
import cosmosis.samplers.rose.utils as utils_module


logger = logging.getLogger(__name__)


class RosePipelineSetupMixin:
    """Mixin class providing pipeline setup methods for RoseSampler."""

    def _extract_vector_metadata(self, block: Any) -> dict[str, dict[str, Any]]:
        """Extract per-likelihood metadata from data_vector block entries.

        The metadata is keyed by likelihood name (``base_key``), aligned with
        the dicts returned by ``utils.task`` when ``sampler.keys`` is not set.
        """
        metadata: dict[str, dict[str, Any]] = {}
        for sec, key in block.keys(section="data_vector"):
            if not key.endswith("_theory"):
                continue
            base_key = key[:-7]
            angle_key = base_key + "_angle"
            bin1_key = base_key + "_bin1"
            bin2_key = base_key + "_bin2"
            name_key = base_key + "_name"

            item = {
                "base_key": base_key,
                "size": int(np.asarray(block[sec, key]).size),
            }

            if (
                block.has_value(sec, angle_key)
                and block.has_value(sec, bin1_key)
                and block.has_value(sec, bin2_key)
            ):
                item["angle"] = np.asarray(block[sec, angle_key]).astype(float)
                item["bin1"] = np.asarray(block[sec, bin1_key]).astype(int)
                item["bin2"] = np.asarray(block[sec, bin2_key]).astype(int)

            # Per-element spectrum name (shear_cl / galaxy_shear_cl / galaxy_cl).
            # Needed for block-wise amplitude_prefactor; absent or placeholder on
            # CosmoSIS <= 3.19.
            if block.has_value(sec, name_key):
                raw_name = block[sec, name_key]
                name_arr = np.asarray(raw_name)
                if name_arr.dtype.kind in ("U", "S", "O") and name_arr.size == item["size"]:
                    item["name"] = name_arr.astype(str)
                elif name_arr.size == item["size"]:
                    item["name"] = name_arr
                else:
                    logger.warning(
                        "data_vector/%s has unexpected shape %s (expected %d); "
                        "not storing spectrum names.",
                        name_key, getattr(name_arr, "shape", None), item["size"],
                    )

            metadata[base_key] = item

        return metadata
    
    def inject_emulator_into_likemodule(self, module: Any) -> None:
        """Inject emulator into likelihood module by monkey-patching.
        
        This method modifies the likelihood module to use emulator predictions
        instead of extracting theory points from the pipeline. When multiple
        independent emulators are trained (one per likelihood), the patched
        ``extract_theory_points`` looks up the per-likelihood slice stored in
        the block by :class:`EmulatorModule`.
        
        Args:
            module: Likelihood module to modify
        """
        original_setup = module.setup_function

        def setup_wrapper(config):
            instance = original_setup(config)
            def emulated_extract_theory_points(self, block):
                like_name = getattr(self, "like_name", None)
                if like_name and block.has_value("data_vector", f"{like_name}_theory_emulated"):
                    return block["data_vector", f"{like_name}_theory_emulated"]
                return block["data_vector", "theory_emulated"]
            instance.extract_theory_points = types.MethodType(emulated_extract_theory_points, instance)
            return instance
        module.setup_function = setup_wrapper

    def compute_fiducial_setup_emu_pipeline(self) -> None:
        """Compute fiducial data vector and set up emulated pipeline.
        
        This method:
        1. Computes fiducial model at parameter center
        2. Determines pipeline structure for emulation
        3. Sets up emulated pipeline with emulator module
        4. Configures fixed inputs and outputs
        """
        logger.info("Computing fiducial data vector and setting up emulated pipeline")
        
        # Get fiducial parameter vector
        p = self.pipeline.start_vector()
        p_unit = self.pipeline.normalize_vector(p)
        
        # Run full pipeline to get fiducial results.
        # Each returned container is a dict keyed by likelihood name so that we
        # can train one emulator per likelihood downstream.
        _, data_vectors, self.data, self.inv_cov, errors, block = task(p, self, return_all=True)

        # Canonical ordering for downstream concatenation (matches task() insertion order).
        self.likelihood_names = list(data_vectors.keys())
        self.data_vector_sizes = {name: int(len(v)) for name, v in data_vectors.items()}
        self.fiducial_data_vector = {name: np.asarray(v) for name, v in data_vectors.items()}
        self.fiducial_errors = {name: np.asarray(v) for name, v in errors.items()}
        self.fiducial_vector_metadata = self._extract_vector_metadata(block)
        self.emulator_output_indices = {name: None for name in self.likelihood_names}

        total_size = sum(self.data_vector_sizes.values())
        logger.info(
            f"Fiducial data vectors per likelihood: "
            f"{[(n, self.data_vector_sizes[n]) for n in self.likelihood_names]} "
            f"(total size={total_size})"
        )
        
        # Determine emulation structure
        self._setup_emulation_structure(block)
        
        # Create emulated pipeline
        self._create_emulated_pipeline()
        
        logger.info("Emulated pipeline setup complete")

    def _setup_emulation_structure(self, block: Any) -> None:
        """Set up the structure for emulation based on pipeline modules."""
        # Get module information
        module_names = [m.name for m in self.pipeline.modules]
        logger.info(f"Pipeline modules: {module_names}")
        
        # Find emulation cutoff point
        if self.last_emulated_module:
            try:
                emu_index = module_names.index(self.last_emulated_module)
            except ValueError:
                raise ValueError(f"Module '{self.last_emulated_module}' not found in pipeline")
        else:
            emu_index = len(module_names)  # Emulate entire pipeline
        
        logger.info(f"Emulation cutoff at module index: {emu_index}")
        
        # Get modules to include in emulated pipeline
        emu_modules = self.pipeline.modules[emu_index + 1:]
        
        # Set up fixed inputs
        fixed_inputs = {(sec, key): block[sec, key] for (sec, key) in self.fixed_keys}
        logger.info(f"Fixed inputs: {list(fixed_inputs.keys())}")
        
        # Build fixed vector for emulator
        self.fixed_vector = self._build_fixed_vector(block)
        
        # Get fiducial chi2 for diagnostics
        self._extract_fiducial_chi2(block)
        
        # Create emulator module
        emu_module = EmulatorModule.as_module("emulator")
        self.emu_module = emu_module
        
        # Configure emulated modules
        if emu_modules:
            # Partial pipeline emulation: prepend emulator module and leave the
            # remaining pipeline modules (including likelihoods) untouched.
            emu_modules.insert(0, emu_module)
        else:
            # Full pipeline emulation: the emulator must supply the theory
            # vector that each likelihood module would normally extract from
            # the block. We copy + patch the trailing N modules, one per
            # likelihood, so each consumes its own per-likelihood emulated
            # slice written by :class:`EmulatorModule`.
            n_like = max(1, len(self.likelihood_names))
            trailing = self.pipeline.modules[-n_like:]
            patched = []
            for m in trailing:
                m_copy = copy.copy(m)
                self.inject_emulator_into_likemodule(m_copy)
                patched.append(m_copy)
            emu_modules = [emu_module, *patched]

        # Add prior modules if they exist
        if self.prior_module:
            try:
                prior_index = module_names.index(self.prior_module)
                emu_modules.insert(0, self.pipeline.modules[prior_index])
            except ValueError:
                raise ValueError(f"Module '{self.prior_module}' not found in pipeline")
        else:
            logger.warning("No prior module found")
        logger.info(f"Emulated modules: {[m.name for m in emu_modules]}")
        self.emu_modules = emu_modules
        self.fixed_inputs = fixed_inputs

    def _build_fixed_vector(self, block: Any) -> np.ndarray:
        """Build vector of fixed parameters for emulator."""
        fixed_vector = []
        
        for (sec, key) in self.fixed_keys:
            value = block[sec, key]
            if isinstance(value, (int, float)):
                fixed_vector.append(value)
            else:
                fixed_vector.extend(np.asarray(value).flatten())
        
        return np.array(fixed_vector) if fixed_vector else np.array([])

    def _extract_fiducial_chi2(self, block: Any) -> None:
        """Extract fiducial chi2 for diagnostics."""
        chi2_fid = None
        for sec, key in block.keys(section="data_vector"):
            if key.endswith("_chi2"):
                chi2_fid = block[sec, key]
                break
        
        if chi2_fid is not None:
            logger.info(f"Fiducial chi2: {chi2_fid:.2f}, cutoff: {self.chi2_cut_off}")
        else:
            logger.warning("No chi2 value found in fiducial evaluation")

    def _create_emulated_pipeline(self) -> None:
        """Create the emulated pipeline object."""
        logger.info("Creating emulated pipeline")
        
        self.emu_pipeline = LikelihoodPipeline(
            self.pipeline.options,
            modules=self.emu_modules,
            values=self.pipeline.values_file
        )
        
        # Configure emulator module. We pass the canonical ordered list of
        # likelihood names and the per-likelihood sizes so the module can
        # iterate emulators in a stable order and write per-likelihood slices
        # (`data_vector/{name}_theory_emulated`) plus a concatenated fallback
        # (`data_vector/theory_emulated`).
        self.emu_module.data.set_emulator_info({
            "fixed_inputs": self.fixed_inputs,
            "pipeline": self.pipeline,
            "outputs": self.keys,
            "sizes": self.data_vector_sizes,
            "likelihood_names": self.likelihood_names,
            "nn_model": self.nn_model,
            "output_indices": self.emulator_output_indices,
        })
        utils_module._sampler = self

    
