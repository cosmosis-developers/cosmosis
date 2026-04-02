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

    def _extract_vector_metadata(self, block: Any) -> list[dict[str, Any]]:
        """Extract per-vector metadata from data_vector block entries.

        The metadata is aligned with the ordering used in utils.task when
        sampler.keys is not set, i.e. scanning all ``*_theory`` entries.
        """
        metadata = []
        for sec, key in block.keys(section="data_vector"):
            if not key.endswith("_theory"):
                continue
            base_key = key[:-7]
            angle_key = base_key + "_angle"
            bin1_key = base_key + "_bin1"
            bin2_key = base_key + "_bin2"

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

            metadata.append(item)

        return metadata
    
    def inject_emulator_into_likemodule(self, module: Any) -> None:
        """Inject emulator into likelihood module by monkey-patching.
        
        This method modifies the likelihood module to use emulator predictions
        instead of extracting theory points from the pipeline.
        
        Args:
            module: Likelihood module to modify
        """
        original_setup = module.setup_function

        def setup_wrapper(config):
            # Step 1: Call the original setup to get the instance
            instance = original_setup(config)
            # Step 2: (Monkey) Patch the method
            def emulated_extract_theory_points(self, block):
                data_vector = block["data_vector", "theory_emulated"]
                return data_vector
            instance.extract_theory_points = types.MethodType(emulated_extract_theory_points, instance)
            # Step 3: Return the patched instance
            return instance
        # Replace the setup_function with our wrapper
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
        
        # Run full pipeline to get fiducial results
        _, data_vectors, self.data, self.inv_cov, errors, block = task(p, self, return_all=True)
        
        # Process data vectors
        self.data_vector_sizes = [len(x) for x in data_vectors]
        self.fiducial_data_vector = np.concatenate(data_vectors)
        self.fiducial_errors = np.concatenate(errors)
        self.fiducial_vector_metadata = self._extract_vector_metadata(block)
        self.emulator_output_indices = None
        
        logger.info(f"Fiducial data vector shape: {self.fiducial_data_vector.shape}")
        
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
            # Partial pipeline emulation
            emu_modules.insert(0, emu_module)
        else:
            # Full pipeline emulation - modify likelihood module
            like_module = copy.copy(self.pipeline.modules[-1])
            self.inject_emulator_into_likemodule(like_module)
            emu_modules = [emu_module, like_module]

        # Add prior modules if they exist
        logger.info(f"Emulated modules: {[m.name for m in emu_modules]}")
        if self.prior_module:
            try:
                prior_index = module_names.index(self.prior_module)
                emu_modules.insert(0, self.pipeline.modules[prior_index])
            except ValueError:
                raise ValueError(f"Module '{self.prior_module}' not found in pipeline")
            logger.info(f"Emulated modules: {[m.name for m in emu_modules]}")
        else:
            logger.warning("No prior module found")
        
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
        
        # Configure emulator module
        self.emu_module.data.set_emulator_info({
            "fixed_inputs": self.fixed_inputs,
            "pipeline": self.pipeline,
            "outputs": self.keys,
            "sizes": self.data_vector_sizes,
            "nn_model": self.nn_model,
            "output_indices": self.emulator_output_indices,
        })
        utils_module._sampler = self

    
