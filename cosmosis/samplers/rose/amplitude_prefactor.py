"""
Block-wise amplitude prefactors for 3x2pt theory vectors.

Applied *before* ``data_transformation`` (train) and *after* its inverse
(predict), so the neural network learns residual shape rather than the leading
S8 / sigma8 / As amplitude of each probe.

WL (shear_cl):
    / S8^2  or  / (sigma8^2 * Omega_m/0.3)  or  / (As_1e9 * Omega_m/0.3)
XC (galaxy_shear_cl), lens bin i:
    / (b_i * S8^2)  or  / (b_i * As_1e9 * Omega_m/0.3)
GC (galaxy_cl), bins i,j:
    / (b_i * b_j * sigma8^2)  or  / (b_i * b_j * As_1e9)

where As_1e9 = As / 1e-9 keeps the As-proxy O(1), comparable to sigma8^2.

S8 and sigma8 are used when present among the varied parameters; otherwise the
As_1e9 proxy is used. If only S8 is varied, sigma8 is recovered as
S8 / sqrt(Omega_m/0.3) for the GC block.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Spectrum names written by cosmosis-standard-library likelihood/2pt (FITS / data_sets).
WL_SPECTRA = frozenset({"shear_cl", "shear_xi_plus", "shear_xi_minus"})
XC_SPECTRA = frozenset({"galaxy_shear_cl", "galaxy_shear_xi"})
GC_SPECTRA = frozenset({"galaxy_cl", "galaxy_xi"})

OMEGA_M_KEYS = ("cosmological_parameters--omega_m",)
SIGMA8_KEYS = (
    "cosmological_parameters--sigma_8",
    "cosmological_parameters--sigma8",
)
S8_KEYS = (
    "cosmological_parameters--s8",
    "cosmological_parameters--s_8",
)
LOG_AS_KEYS = (
    "cosmological_parameters--log1e10as",
    "cosmological_parameters--logas",
)

DEFAULT_3X2PT_SPECTRA = ("shear_cl", "galaxy_shear_cl", "galaxy_cl")

_AMP_FLOOR = 1e-30


def infer_spectrum_names_from_tags(
    bin1: np.ndarray,
    bin2: np.ndarray,
    angle: np.ndarray,
    spectra: Sequence[str],
) -> np.ndarray:
    """Infer per-mode spectrum names when ``data_vector/<like>_name`` is missing.

    CosmoSIS ≤3.19 cannot store the string name array. Each 2pt spectrum is a
    contiguous block of unique ``(bin1, bin2, angle)`` triples; a repeated
    triple marks the start of the next spectrum (same bins/ℓ reused by another
    probe). ``spectra`` must list probe names in ``data_sets`` order.
    """
    bin1 = np.asarray(bin1)
    bin2 = np.asarray(bin2)
    angle = np.asarray(angle)
    n = len(bin1)
    if not (len(bin2) == n and len(angle) == n):
        raise ValueError("bin1, bin2, angle must have the same length")
    if n == 0:
        return np.array([], dtype=str)

    seen: set = set()
    cuts = [0]
    for i in range(n):
        triple = (int(bin1[i]), int(bin2[i]), round(float(angle[i]), 12))
        if triple in seen:
            cuts.append(i)
            seen = {triple}
        else:
            seen.add(triple)
    cuts.append(n)
    n_blocks = len(cuts) - 1
    if n_blocks != len(spectra):
        raise ValueError(
            f"Inferred {n_blocks} contiguous 2pt blocks from (bin1,bin2,angle) "
            f"repeats (sizes {list(np.diff(cuts))}), but amplitude_prefactor_spectra "
            f"has {len(spectra)} name(s): {list(spectra)}. Set "
            "amplitude_prefactor_spectra to match data_sets order, or upgrade "
            "CosmoSIS so data_vector/<like>_name is stored."
        )
    names = np.empty(n, dtype=object)
    for spec, a, b in zip(spectra, cuts[:-1], cuts[1:]):
        names[a:b] = str(spec)
    logger.info(
        "Inferred spectrum names from (bin1,bin2,angle) repeats: %s",
        list(zip(spectra, np.diff(cuts).tolist())),
    )
    return names.astype(str)


def _parse_spectra_list(raw: Any) -> List[str]:
    if raw is None:
        return list(DEFAULT_3X2PT_SPECTRA)
    if isinstance(raw, (list, tuple)):
        vals = [str(x).strip() for x in raw if str(x).strip()]
        return vals or list(DEFAULT_3X2PT_SPECTRA)
    s = str(raw).strip()
    if not s:
        return list(DEFAULT_3X2PT_SPECTRA)
    s = s.replace(",", " ")
    vals = [t for t in s.split() if t]
    return vals or list(DEFAULT_3X2PT_SPECTRA)


def parse_amplitude_prefactor(raw: Any) -> bool:
    """Parse ini / per-likelihood amplitude_prefactor value to bool."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("", "none", "f", "false", "0", "off", "no"):
        return False
    if s in ("t", "true", "1", "on", "yes", "3x2pt", "s8_3x2pt"):
        return True
    raise ValueError(
        f"Unknown amplitude_prefactor value '{raw}'. "
        "Use F/none or T/3x2pt."
    )


def _first_present(keys: Sequence[str], available: Sequence[str]) -> Optional[str]:
    avail = set(available)
    for k in keys:
        if k in avail:
            return k
    return None


def _as_1d(value: Any, n_samples: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if n_samples is not None and arr.size == 1 and n_samples > 1:
        arr = np.full(n_samples, float(arr.ravel()[0]))
    return arr.reshape(-1)


def _as_1e9_from_log1e10as(log1e10as: np.ndarray) -> np.ndarray:
    """Convert CosmoSIS/CAMB log1e10As = ln(1e10 As) to As/1e-9.

    Returns an O(1) amplitude proxy (As ~ 2e-9 → As_1e9 ~ 2), comparable to
    sigma8^2, rather than raw As ~ 1e-9.
    """
    As = np.exp(np.asarray(log1e10as, dtype=float)) * 1e-10
    return As / 1e-9


class BlockAmplitudePrefactor:
    """Per-mode amplitude factors for a flat 3x2pt theory vector."""

    def __init__(
        self,
        n_modes: int,
        spectrum: np.ndarray,
        bin1: np.ndarray,
        bin2: np.ndarray,
        model_parameters: Sequence[str],
    ):
        self.n_modes = int(n_modes)
        self.spectrum = np.asarray(spectrum)
        self.bin1 = np.asarray(bin1, dtype=int)
        self.bin2 = np.asarray(bin2, dtype=int)
        self.model_parameters = list(model_parameters)

        if self.spectrum.shape[0] != self.n_modes:
            raise ValueError(
                f"spectrum length {self.spectrum.shape[0]} != n_modes {self.n_modes}"
            )
        if self.bin1.shape[0] != self.n_modes or self.bin2.shape[0] != self.n_modes:
            raise ValueError("bin1/bin2 length must match n_modes")

        # Classify each mode
        names = np.asarray([str(x) for x in self.spectrum])
        self.wl_mask = np.array([n in WL_SPECTRA for n in names], dtype=bool)
        self.xc_mask = np.array([n in XC_SPECTRA for n in names], dtype=bool)
        self.gc_mask = np.array([n in GC_SPECTRA for n in names], dtype=bool)
        known = self.wl_mask | self.xc_mask | self.gc_mask
        if not np.any(known):
            raise ValueError(
                "amplitude_prefactor=T but no shear_cl / galaxy_shear_cl / galaxy_cl "
                "(or xi equivalents) entries found in vector metadata 'name'. "
                "Check that the 2pt likelihood writes data_vector/<like>_name."
            )
        n_unknown = int(np.sum(~known))
        if n_unknown:
            logger.warning(
                "amplitude_prefactor: %d / %d modes have unrecognized spectrum "
                "names and will not be rescaled: %s",
                n_unknown,
                self.n_modes,
                sorted(set(names[~known].tolist())),
            )

        self.omega_m_key = _first_present(OMEGA_M_KEYS, self.model_parameters)
        self.sigma8_key = _first_present(SIGMA8_KEYS, self.model_parameters)
        self.s8_key = _first_present(S8_KEYS, self.model_parameters)
        self.log_as_key = _first_present(LOG_AS_KEYS, self.model_parameters)

        if self.s8_key or self.sigma8_key:
            self.amp_family = "s8"
            if self.omega_m_key is None and self.s8_key and not self.sigma8_key:
                raise ValueError(
                    "amplitude_prefactor needs cosmological_parameters--omega_m "
                    "to recover sigma8 from S8 for the GC block."
                )
            if self.omega_m_key is None and self.sigma8_key and not self.s8_key:
                raise ValueError(
                    "amplitude_prefactor needs cosmological_parameters--omega_m "
                    "to build S8^2 = sigma8^2 * Omega_m/0.3 for WL/XC."
                )
        elif self.log_as_key:
            self.amp_family = "as"
            if self.omega_m_key is None and (np.any(self.wl_mask) or np.any(self.xc_mask)):
                raise ValueError(
                    "amplitude_prefactor As-proxy mode needs "
                    "cosmological_parameters--omega_m for WL/XC blocks."
                )
        else:
            raise ValueError(
                "amplitude_prefactor requires varied S8, sigma8, or log1e10As "
                f"among model parameters; got {self.model_parameters}"
            )

        # Bias keys used by XC/GC (1-indexed bin numbers from metadata)
        needed_bins = set()
        if np.any(self.xc_mask):
            needed_bins.update(self.bin1[self.xc_mask].tolist())
        if np.any(self.gc_mask):
            needed_bins.update(self.bin1[self.gc_mask].tolist())
            needed_bins.update(self.bin2[self.gc_mask].tolist())
        self.bias_keys: Dict[int, str] = {}
        for b in sorted(needed_bins):
            key = f"bin_bias--b{int(b)}"
            if key not in self.model_parameters:
                raise ValueError(
                    f"amplitude_prefactor needs varied parameter '{key}' "
                    f"for bin {b} (XC/GC blocks)."
                )
            self.bias_keys[int(b)] = key

        logger.info(
            "amplitude_prefactor enabled (%s): WL=%d XC=%d GC=%d modes; "
            "omega_m=%s sigma8=%s S8=%s As=%s",
            self.amp_family,
            int(self.wl_mask.sum()),
            int(self.xc_mask.sum()),
            int(self.gc_mask.sum()),
            self.omega_m_key,
            self.sigma8_key,
            self.s8_key,
            self.log_as_key,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_metadata(
        cls,
        metadata: Dict[str, Any],
        model_parameters: Sequence[str],
        spectra: Optional[Sequence[str]] = None,
    ) -> "BlockAmplitudePrefactor":
        n_modes = int(metadata.get("size", 0))
        if "bin1" not in metadata or "bin2" not in metadata:
            raise ValueError(
                "amplitude_prefactor=T requires bin1/bin2 in vector metadata."
            )
        name = metadata.get("name")
        name_arr = None
        if name is not None:
            name_arr = np.asarray(name)
            if name_arr.dtype.kind in ("U", "S", "O"):
                # Guard against the cosmosis<=3.19 placeholder string.
                if name_arr.ndim == 0 or (
                    name_arr.size == 1
                    and "not saved" in str(name_arr.ravel()[0]).lower()
                ):
                    name_arr = None
                elif name_arr.size != (n_modes or name_arr.size):
                    name_arr = None
        if name_arr is None:
            if "angle" not in metadata:
                raise ValueError(
                    "amplitude_prefactor=T requires either data_vector/<like>_name "
                    "or angle+bin1+bin2 metadata to infer WL/XC/GC blocks "
                    "(CosmoSIS ≤3.19 cannot store the name array)."
                )
            spectra_list = _parse_spectra_list(spectra)
            name_arr = infer_spectrum_names_from_tags(
                metadata["bin1"], metadata["bin2"], metadata["angle"], spectra_list
            )
        return cls(
            n_modes=n_modes or len(name_arr),
            spectrum=name_arr,
            bin1=metadata["bin1"],
            bin2=metadata["bin2"],
            model_parameters=model_parameters,
        )

    def to_state(self) -> Dict[str, Any]:
        """Serializable state for emulator npz save/load."""
        return {
            "n_modes": self.n_modes,
            "spectrum": self.spectrum,
            "bin1": self.bin1,
            "bin2": self.bin2,
            "model_parameters": list(self.model_parameters),
            "amp_family": self.amp_family,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BlockAmplitudePrefactor":
        return cls(
            n_modes=int(state["n_modes"]),
            spectrum=state["spectrum"],
            bin1=state["bin1"],
            bin2=state["bin2"],
            model_parameters=list(state["model_parameters"]),
        )

    # ------------------------------------------------------------------
    # Amplitude evaluation
    # ------------------------------------------------------------------
    def _cosmo_pieces(self, X: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (s8_sq, sigma8_sq, as_om) with shape (n_samples,)."""
        # Infer n_samples from any present array
        n_samples = None
        for key in self.model_parameters:
            if key in X:
                v = np.asarray(X[key])
                if v.ndim > 0 and v.size > 1:
                    n_samples = int(v.size)
                    break
        if n_samples is None:
            n_samples = 1

        if self.amp_family == "s8":
            if self.s8_key is not None:
                s8 = _as_1d(X[self.s8_key], n_samples)
                s8_sq = s8 ** 2
                if self.sigma8_key is not None:
                    sigma8_sq = _as_1d(X[self.sigma8_key], n_samples) ** 2
                else:
                    om = _as_1d(X[self.omega_m_key], n_samples)
                    sigma8_sq = s8_sq / np.maximum(om / 0.3, _AMP_FLOOR)
            else:
                sigma8 = _as_1d(X[self.sigma8_key], n_samples)
                sigma8_sq = sigma8 ** 2
                om = _as_1d(X[self.omega_m_key], n_samples)
                s8_sq = sigma8_sq * (om / 0.3)
            as_om = np.ones(n_samples)  # unused
        else:
            log_as = _as_1d(X[self.log_as_key], n_samples)
            As_1e9 = _as_1e9_from_log1e10as(log_as)
            om = _as_1d(X[self.omega_m_key], n_samples) if self.omega_m_key else np.ones(n_samples)
            as_om = As_1e9 * (om / 0.3)
            s8_sq = as_om  # WL/XC use As_1e9*Om/0.3
            sigma8_sq = As_1e9  # GC uses As_1e9
        return s8_sq, sigma8_sq, as_om

    def factors(self, X: Dict[str, Any]) -> np.ndarray:
        """Amplitude A(θ) with shape (n_samples, n_modes)."""
        s8_sq, sigma8_sq, _ = self._cosmo_pieces(X)
        n_samples = int(s8_sq.shape[0])
        A = np.ones((n_samples, self.n_modes), dtype=float)

        if np.any(self.wl_mask):
            A[:, self.wl_mask] = s8_sq[:, None]

        if np.any(self.xc_mask):
            for b, key in self.bias_keys.items():
                sel = self.xc_mask & (self.bin1 == b)
                if not np.any(sel):
                    continue
                bias = _as_1d(X[key], n_samples)
                A[:, sel] = (bias * s8_sq)[:, None]

        if np.any(self.gc_mask):
            for i, key_i in self.bias_keys.items():
                for j, key_j in self.bias_keys.items():
                    sel = self.gc_mask & (self.bin1 == i) & (self.bin2 == j)
                    if not np.any(sel):
                        continue
                    bi = _as_1d(X[key_i], n_samples)
                    bj = _as_1d(X[key_j], n_samples)
                    A[:, sel] = (bi * bj * sigma8_sq)[:, None]

        return np.maximum(A, _AMP_FLOOR)

    def divide(self, X: Dict[str, Any], y: np.ndarray) -> np.ndarray:
        """Forward: y' = y / A."""
        y = np.asarray(y, dtype=float)
        A = self.factors(X)
        if y.ndim == 1:
            return y / A[0]
        if A.shape[0] == 1 and y.shape[0] > 1:
            A = np.broadcast_to(A, y.shape)
        return y / A

    def multiply(self, X: Dict[str, Any], y: np.ndarray) -> np.ndarray:
        """Inverse: y = y' * A."""
        y = np.asarray(y, dtype=float)
        A = self.factors(X)
        if y.ndim == 1:
            return y * A[0]
        if A.shape[0] == 1 and y.shape[0] > 1:
            A = np.broadcast_to(A, y.shape)
        return y * A

    def factors_tf(self, physical_params_tf, param_index: Dict[str, int], dtype):
        """TensorFlow amplitude vector of shape (n_modes,) for a 1D param vector."""
        import tensorflow as tf

        def _get(key: str):
            return physical_params_tf[param_index[key]]

        if self.amp_family == "s8":
            if self.s8_key is not None:
                s8 = _get(self.s8_key)
                s8_sq = s8 * s8
                if self.sigma8_key is not None:
                    s8p = _get(self.sigma8_key)
                    sigma8_sq = s8p * s8p
                else:
                    om = _get(self.omega_m_key)
                    sigma8_sq = s8_sq / tf.maximum(om / 0.3, _AMP_FLOOR)
            else:
                s8p = _get(self.sigma8_key)
                sigma8_sq = s8p * s8p
                om = _get(self.omega_m_key)
                s8_sq = sigma8_sq * (om / 0.3)
        else:
            log_as = _get(self.log_as_key)
            As_1e9 = (tf.exp(log_as) * 1e-10) / 1e-9
            om = _get(self.omega_m_key) if self.omega_m_key else 1.0
            s8_sq = As_1e9 * (om / 0.3)
            sigma8_sq = As_1e9

        ones = tf.ones([self.n_modes], dtype=dtype)
        A = ones

        if np.any(self.wl_mask):
            mask = tf.constant(self.wl_mask, dtype=dtype)
            A = A * (1.0 - mask) + mask * s8_sq

        if np.any(self.xc_mask):
            for b, key in self.bias_keys.items():
                sel = self.xc_mask & (self.bin1 == b)
                if not np.any(sel):
                    continue
                mask = tf.constant(sel, dtype=dtype)
                amp = _get(key) * s8_sq
                A = A * (1.0 - mask) + mask * amp

        if np.any(self.gc_mask):
            for i, key_i in self.bias_keys.items():
                for j, key_j in self.bias_keys.items():
                    sel = self.gc_mask & (self.bin1 == i) & (self.bin2 == j)
                    if not np.any(sel):
                        continue
                    mask = tf.constant(sel, dtype=dtype)
                    amp = _get(key_i) * _get(key_j) * sigma8_sq
                    A = A * (1.0 - mask) + mask * amp

        return tf.maximum(A, _AMP_FLOOR)
