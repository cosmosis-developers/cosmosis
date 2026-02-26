"""
Emulator module for CosmoSIS pipeline integration.

This module provides a CosmoSIS module class that replaces pipeline calculations
with emulator predictions.
"""

from typing import Dict, Any

from ...runtime import ClassModule


class EmulatorModule(ClassModule):
    """CosmoSIS module that replaces pipeline calculations with emulator predictions.
    
    This module intercepts the pipeline execution and replaces slow calculations
    with fast neural network predictions. It handles parameter transformation,
    emulator prediction, and output formatting.
    """
    
    def __init__(self, options: Dict[str, Any]) -> None:
        pass

    def set_emulator_info(self, info: Dict[str, Any]) -> None:
        """Set emulator configuration information.
        
        Args:
            info: Dictionary containing:
                - pipeline: Original CosmoSIS pipeline object
                - fixed_inputs: Parameters that don't vary during emulation
                - outputs: List of (section, key) tuples for emulator outputs
                - sizes: Sizes of each output vector
                - nn_model: Neural network model type
        """
        self.pipeline = info["pipeline"]
        self.fixed_inputs = info["fixed_inputs"]
        self.inputs = [(p.section, p.name) for p in self.pipeline.varied_params]
        self.outputs = info["outputs"]
        self.sizes = info["sizes"]
        self.nn_model = info["nn_model"]

    def set_emulator(self, emu: Any) -> None:
        """Set the trained emulator object.
        
        Args:
            emu: Trained emulator object with predict() method
        """
        self.emulator = emu

    def execute(self, block: Any) -> int:
        """Execute emulator prediction and populate data block.
        
        Args:
            block: CosmoSIS data block to populate with predictions
            
        Returns:
            0 on success (CosmoSIS convention)
            
        Raises:
            RuntimeError: If emulator is not set or prediction fails
        """
        if self.emulator is None:
            raise RuntimeError("Emulator not set - call set_emulator() first")

        # Prepare input dictionary for emulator
        # Use '--' separator to avoid conflicts with parameter names
        p_dict = {f"{sec}--{key}": block[sec, key] for (sec, key) in self.inputs}
        
        # Get emulator prediction
        prediction = self.emulator.predict(p_dict)[0]
        
        # Populate outputs
        if self.outputs==[]:  # Empty outputs means full data vector
            block["data_vector", "theory_emulated"] = prediction
        else:
            # Split prediction into specified output components
            start_idx = 0
            for (sec, key), size in zip(self.outputs, self.sizes):
                block[sec, key] = prediction[start_idx:start_idx + size]
                start_idx += size

        # Set fixed inputs that don't vary during emulation
        for (sec, key), val in self.fixed_inputs.items():
            block[sec, key] = val
            
        return 0

