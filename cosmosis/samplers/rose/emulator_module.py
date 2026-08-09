"""
Emulator module for CosmoSIS pipeline integration.

This module provides a CosmoSIS module class that replaces pipeline calculations
with emulator predictions. One independent emulator is applied per likelihood;
outputs are written both as per-likelihood block entries and as a concatenated
vector for consumers that still expect the flat layout.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ...runtime import ClassModule


class EmulatorModule(ClassModule):
    """CosmoSIS module that replaces pipeline calculations with emulator predictions.

    Args:
        options: CosmoSIS-style options (unused).
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        self.emulators: Dict[str, Any] = {}
        self.output_indices: Dict[str, Optional[np.ndarray]] = {}

    def set_emulator_info(self, info: Dict[str, Any]) -> None:
        """Set emulator configuration information.

        Expected keys:
            - pipeline: original CosmoSIS pipeline object
            - fixed_inputs: dict of (section, key) -> fixed value
            - outputs: list of (section, key) tuples (user-specified) or [] for
              full-data-vector mode
            - sizes: dict {likelihood_name: size}
            - likelihood_names: ordered list of likelihood names (canonical order)
            - nn_model: neural network model type string
            - output_indices: optional dict {name: indices or None}
        """
        self.pipeline = info["pipeline"]
        self.fixed_inputs = info["fixed_inputs"]
        self.inputs = [(p.section, p.name) for p in self.pipeline.varied_params]
        self.outputs: List = info.get("outputs", []) or []
        self.sizes: Dict[str, int] = dict(info["sizes"])
        self.likelihood_names: List[str] = list(
            info.get("likelihood_names", list(self.sizes.keys()))
        )
        self.nn_model = info["nn_model"]
        self.output_indices = info.get("output_indices") or {
            n: None for n in self.likelihood_names
        }

    def set_emulator(self, emu: Any) -> None:
        """Set the trained emulator(s).

        Accepts either a dict ``{likelihood_name: emulator}`` or a single
        emulator (backwards-compatible path used only when exactly one
        likelihood is registered).
        """
        if isinstance(emu, dict):
            self.emulators = emu
        else:
            if len(self.likelihood_names) != 1:
                raise ValueError(
                    "Single emulator provided but multiple likelihoods configured; "
                    "pass a dict {likelihood_name: emulator} instead."
                )
            self.emulators = {self.likelihood_names[0]: emu}

    def execute(self, block: Any) -> int:
        """Execute per-likelihood emulator predictions and populate data block."""
        if not self.emulators:
            raise RuntimeError("Emulators not set - call set_emulator() first")

        p_dict = {f"{sec}--{key}": block[sec, key] for (sec, key) in self.inputs}

        predictions: Dict[str, np.ndarray] = {}
        for name in self.likelihood_names:
            emu = self.emulators.get(name)
            if emu is None:
                raise RuntimeError(f"No emulator registered for likelihood '{name}'")

            pred = emu.predict(p_dict)[0]
            indices = self.output_indices.get(name) if self.output_indices else None
            if indices is not None:
                pred = pred[indices]
            predictions[name] = pred

        if not self.outputs:
            # Full-data-vector mode: write per-likelihood slices so the patched
            # likelihood modules can pick the right one, and also write the
            # concatenated vector for any legacy consumers.
            concatenated = np.concatenate(
                [predictions[n] for n in self.likelihood_names]
            )
            block["data_vector", "theory_emulated"] = concatenated
            for name, pred in predictions.items():
                block["data_vector", f"{name}_theory_emulated"] = pred
        else:
            # User specified explicit output keys. We assume there is one key
            # per likelihood in the same canonical order.
            if len(self.outputs) != len(self.likelihood_names):
                raise ValueError(
                    f"Number of output keys ({len(self.outputs)}) does not match "
                    f"number of likelihoods ({len(self.likelihood_names)})"
                )
            for (sec, key), name in zip(self.outputs, self.likelihood_names):
                block[sec, key] = predictions[name]

        for (sec, key), val in self.fixed_inputs.items():
            block[sec, key] = val

        return 0
