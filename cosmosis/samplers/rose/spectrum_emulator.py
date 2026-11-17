"""
Composite emulator that concatenates per-spectrum NNEmulator predictions.

Used when ``spectrum_emulators=T``: shear_cl / galaxy_shear_cl / galaxy_cl
(or xi equivalents) are trained as separate networks with probe-specific
parameter subsets, then scattered back into the full likelihood data vector
so the 2pt module still sees ``{like}_theory_emulated``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .nn_emulator import NNEmulator

logger = logging.getLogger(__name__)

# Marker stored in the likelihood-level manifest npz.
COMPOSITE_MANIFEST_KEY = "rose_spectrum_emulators"


class CompositeSpectrumEmulator:
    """Reconstruct a full theory vector from per-spectrum emulators."""

    def __init__(
        self,
        emulators: Dict[str, NNEmulator],
        mode_indices: Dict[str, np.ndarray],
        n_modes: int,
        spectrum_order: Sequence[str],
        likelihood_name: str = "",
    ):
        if not emulators:
            raise ValueError("CompositeSpectrumEmulator needs at least one sub-emulator")
        self.emulators = dict(emulators)
        self.mode_indices = {
            s: np.asarray(idx, dtype=int).ravel() for s, idx in mode_indices.items()
        }
        self.n_modes = int(n_modes)
        self.spectrum_order = [str(s) for s in spectrum_order]
        self.likelihood_name = str(likelihood_name)
        self.ignore_extra_params = False
        missing = [s for s in self.spectrum_order if s not in self.emulators]
        if missing:
            raise ValueError(
                f"CompositeSpectrumEmulator missing sub-emulators for: {missing}"
            )
        covered = np.zeros(self.n_modes, dtype=bool)
        for s in self.spectrum_order:
            idx = self.mode_indices[s]
            if idx.size == 0:
                raise ValueError(f"Spectrum '{s}' has empty mode index list")
            if np.any(idx < 0) or np.any(idx >= self.n_modes):
                raise ValueError(f"Spectrum '{s}' mode indices out of range")
            if np.any(covered[idx]):
                raise ValueError(f"Overlapping mode indices for spectrum '{s}'")
            covered[idx] = True
        if not np.all(covered):
            raise ValueError(
                f"CompositeSpectrumEmulator mode indices cover "
                f"{int(covered.sum())}/{self.n_modes} modes"
            )

    @property
    def trained(self) -> bool:
        return all(bool(getattr(e, "trained", False)) for e in self.emulators.values())

    @trained.setter
    def trained(self, value: bool) -> None:
        for emu in self.emulators.values():
            emu.trained = bool(value)

    def predict(self, X: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
        """Predict full data vector; returns shape ``(batch, n_modes)``."""
        if not self.trained:
            raise RuntimeError("CompositeSpectrumEmulator must be trained before predict")

        # Infer batch size from the first available array-valued parameter.
        batch = 1
        for val in X.values():
            arr = np.atleast_1d(val)
            if arr.size > 1:
                batch = int(arr.shape[0]) if arr.ndim >= 1 else int(arr.size)
                break

        out = np.zeros((batch, self.n_modes), dtype=float)
        for spectrum in self.spectrum_order:
            emu = self.emulators[spectrum]
            emu.ignore_extra_params = True
            pred = np.asarray(emu.predict(X), dtype=float)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            idx = self.mode_indices[spectrum]
            if pred.shape[1] != idx.size:
                raise ValueError(
                    f"[{self.likelihood_name}/{spectrum}] predicted "
                    f"{pred.shape[1]} modes but index map has {idx.size}"
                )
            if pred.shape[0] == 1 and batch > 1:
                pred = np.broadcast_to(pred, (batch, pred.shape[1]))
            out[:, idx] = pred
        return out

    def save_to(self, filename: str) -> None:
        """Save each spectrum emulator and a likelihood-level manifest.

        Layout::

            {filename}__{spectrum}.npz   # per-spectrum NNEmulator
            {filename}.npz               # composite manifest
        """
        if not self.trained:
            raise RuntimeError("CompositeSpectrumEmulator must be trained before saving")
        for spectrum in self.spectrum_order:
            path = f"{filename}__{spectrum}"
            self.emulators[spectrum].save_to(path)
            logger.info(
                "Saved spectrum emulator '%s/%s' to %s",
                self.likelihood_name,
                spectrum,
                path,
            )
        self._save_manifest(filename)

    def _save_manifest(self, filename: str) -> None:
        save_dict = {
            COMPOSITE_MANIFEST_KEY: True,
            "spectrum_order": np.array(self.spectrum_order, dtype=object),
            "n_modes": np.asarray(self.n_modes),
            "likelihood_name": np.array(self.likelihood_name),
            "mode_indices": np.array(
                {s: self.mode_indices[s] for s in self.spectrum_order},
                dtype=object,
            ),
            # Satisfy load paths that peek at modes/parameters on the like key.
            "modes": np.arange(self.n_modes),
            "parameters": np.array([], dtype=object),
        }
        np.savez_compressed(filename + ".npz", **save_dict)
        logger.info("Saved composite spectrum manifest to %s.npz", filename)

    @classmethod
    def load(
        cls,
        filename: str,
        *,
        default_data_transformation: str = "log_norm",
    ) -> "CompositeSpectrumEmulator":
        """Load a composite emulator from ``{filename}.npz`` + spectrum files."""
        manifest_path = filename + ".npz"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Composite manifest not found: {manifest_path}")
        with np.load(manifest_path, allow_pickle=True) as data:
            if COMPOSITE_MANIFEST_KEY not in data.files:
                raise ValueError(f"{manifest_path} is not a spectrum-emulator manifest")
            spectrum_order = [str(s) for s in np.asarray(data["spectrum_order"]).tolist()]
            n_modes = int(np.asarray(data["n_modes"]).item())
            likelihood_name = str(np.asarray(data["likelihood_name"]).item())
            raw_idx = data["mode_indices"].item()
            mode_indices = {
                str(s): np.asarray(raw_idx[s], dtype=int).ravel() for s in spectrum_order
            }

        emulators: Dict[str, NNEmulator] = {}
        for spectrum in spectrum_order:
            base = f"{filename}__{spectrum}"
            info = base + ".npz"
            if not os.path.exists(info):
                raise FileNotFoundError(
                    f"Spectrum emulator file not found: {info}"
                )
            with np.load(info, allow_pickle=True) as data:
                data_transformation_init = default_data_transformation
                if "data_transformation" in data.files:
                    dt = data["data_transformation"]
                    if hasattr(dt, "item"):
                        dt = dt.item()
                    if isinstance(dt, dict) and "data_transformation" in dt:
                        data_transformation_init = str(dt["data_transformation"])
                if "parameters" in data.files:
                    model_parameters = list(data["parameters"])
                else:
                    raise ValueError(
                        f"[{likelihood_name}/{spectrum}] missing parameters in {info}"
                    )
                if "modes" in data.files:
                    output_size = len(data["modes"])
                else:
                    raise ValueError(
                        f"[{likelihood_name}/{spectrum}] missing modes in {info}"
                    )
            emu = NNEmulator(
                model_parameters,
                np.ones(output_size),
                data_transformation=data_transformation_init,
            )
            emu.load(base)
            emu.ignore_extra_params = True
            emulators[spectrum] = emu

        return cls(
            emulators=emulators,
            mode_indices=mode_indices,
            n_modes=n_modes,
            spectrum_order=spectrum_order,
            likelihood_name=likelihood_name,
        )

    @staticmethod
    def is_manifest(path: str) -> bool:
        """True if ``path`` (with or without .npz) is a composite manifest."""
        if not path.endswith(".npz"):
            path = path + ".npz"
        if not os.path.exists(path):
            return False
        with np.load(path, allow_pickle=True) as data:
            return COMPOSITE_MANIFEST_KEY in data.files
