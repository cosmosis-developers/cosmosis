"""
Bin-pair / probe block indexing for flat 3x2pt theory vectors.

Uses the same ``name`` / ``bin1`` / ``bin2`` / ``angle`` metadata that
:mod:`amplitude_prefactor` already consumes. Contiguous runs of equal
``(name, bin1, bin2)`` define one redshift-bin pair (ℓ or θ varies within).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .amplitude_prefactor import (
    DEFAULT_3X2PT_SPECTRA,
    GC_SPECTRA,
    WL_SPECTRA,
    XC_SPECTRA,
    _parse_spectra_list,
    infer_spectrum_names_from_tags,
)

# CosmoSIS sections dropped from spectrum-emulator model_parameters.
# XC (galaxy_shear_cl) keeps all varied parameters.
_SPECTRUM_DROP_SECTIONS = {
    **{s: frozenset({"bin_bias", "bias_lens", "photoz_lens_errors", "lens_photoz_errors", "mag_alpha_lens"}) for s in WL_SPECTRA},
    **{
        s: frozenset(
            {
                "shear_calibration_parameters",
                "photoz_source_errors", "wl_photoz_errors",
                "intrinsic_alignment_parameters",
            }
        )
        for s in GC_SPECTRA
    },
    **{s: frozenset() for s in XC_SPECTRA},
}


def resolve_spectrum_names(
    metadata: Dict[str, Any],
    spectra: Optional[Any] = None,
) -> np.ndarray:
    """Return per-mode spectrum names from metadata, inferring if needed."""
    n_modes = int(metadata.get("size", 0))
    if "bin1" not in metadata or "bin2" not in metadata:
        raise ValueError(
            "Bin-pair indexing requires bin1/bin2 in vector metadata."
        )
    name = metadata.get("name")
    name_arr = None
    if name is not None:
        name_arr = np.asarray(name)
        if name_arr.dtype.kind in ("U", "S", "O"):
            if name_arr.ndim == 0 or (
                name_arr.size == 1
                and "not saved" in str(name_arr.ravel()[0]).lower()
            ):
                name_arr = None
            elif n_modes and name_arr.size not in (1, n_modes):
                name_arr = None
            elif name_arr.size == 1 and n_modes > 1:
                name_arr = None
    if name_arr is None:
        if "angle" not in metadata:
            raise ValueError(
                "Bin-pair indexing requires either data_vector/<like>_name "
                "or angle+bin1+bin2 metadata to infer probe names "
                "(CosmoSIS ≤3.19 cannot store the name array)."
            )
        spectra_list = _parse_spectra_list(spectra)
        if not spectra_list:
            spectra_list = list(DEFAULT_3X2PT_SPECTRA)
        name_arr = infer_spectrum_names_from_tags(
            metadata["bin1"], metadata["bin2"], metadata["angle"], spectra_list
        )
    names = np.asarray([str(x) for x in np.asarray(name_arr).ravel()])
    if n_modes and names.size != n_modes:
        raise ValueError(
            f"Spectrum name length {names.size} != metadata size {n_modes}"
        )
    return names


def iter_bin_pairs(
    metadata: Dict[str, Any],
    spectra: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """List contiguous redshift-bin pairs in a flat theory vector.

    Each entry is::

        {
            "name": str,          # e.g. shear_cl
            "bin1": int,
            "bin2": int,
            "indices": ndarray,   # int indices into the flat vector
        }

    Empty pairs (no surviving modes after scale cuts) are omitted.
    """
    names = resolve_spectrum_names(metadata, spectra=spectra)
    bin1 = np.asarray(metadata["bin1"], dtype=int).ravel()
    bin2 = np.asarray(metadata["bin2"], dtype=int).ravel()
    n = names.size
    if bin1.size != n or bin2.size != n:
        raise ValueError("name/bin1/bin2 lengths must match")
    if n == 0:
        return []

    blocks: List[Dict[str, Any]] = []
    start = 0
    for i in range(1, n + 1):
        boundary = i == n or (
            names[i] != names[start]
            or int(bin1[i]) != int(bin1[start])
            or int(bin2[i]) != int(bin2[start])
        )
        if boundary:
            blocks.append(
                {
                    "name": str(names[start]),
                    "bin1": int(bin1[start]),
                    "bin2": int(bin2[start]),
                    "indices": np.arange(start, i, dtype=int),
                }
            )
            start = i
    return blocks


def spectrum_mode_indices(
    metadata: Dict[str, Any],
    spectra: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Map spectrum name → flat-vector indices for that probe.

    Spectra appear in ``spectra`` / ``DEFAULT_3X2PT_SPECTRA`` order when listed
    there; any additional names found in metadata are appended.
    """
    names = resolve_spectrum_names(metadata, spectra=spectra)
    spectra_list = _parse_spectra_list(spectra)
    if not spectra_list:
        spectra_list = list(DEFAULT_3X2PT_SPECTRA)
    present = {str(s) for s in np.unique(names)}
    ordered: List[str] = [s for s in spectra_list if s in present]
    for s in sorted(present):
        if s not in ordered:
            ordered.append(s)
    return {s: np.where(names == s)[0].astype(int) for s in ordered}


def spectrum_model_parameters(
    spectrum: str,
    varied_params: Sequence[str],
) -> List[str]:
    """Filter varied ``section--name`` params for a spectrum emulator.

    Drops CosmoSIS sections that do not affect that probe (see
    ``_SPECTRUM_DROP_SECTIONS``). Unknown spectrum names keep all parameters.
    """
    drop = _SPECTRUM_DROP_SECTIONS.get(str(spectrum), frozenset())
    if not drop:
        return [str(p) for p in varied_params]
    kept: List[str] = []
    for p in varied_params:
        key = str(p)
        section = key.split("--", 1)[0]
        if section not in drop:
            kept.append(key)
    return kept


def slice_vector_metadata(
    metadata: Dict[str, Any],
    indices: np.ndarray,
    spectrum: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a metadata dict for a contiguous or scattered mode subset.

    If ``spectrum`` is given, set per-mode ``name`` to that probe so amp/PCA
    configuration does not re-infer against the full ``data_sets`` list on a
    single-spectrum slice (CosmoSIS ≤3.19 / missing name arrays).
    """
    idx = np.asarray(indices, dtype=int).ravel()
    out: Dict[str, Any] = {"size": int(idx.size)}
    for key in ("name", "bin1", "bin2", "angle"):
        if key not in metadata:
            continue
        arr = np.asarray(metadata[key])
        if arr.ndim == 0 or arr.size == 1:
            # Scalar / placeholder — only keep if it is a real per-vector tag;
            # otherwise wait for the forced spectrum name below.
            if key == "name" and (
                arr.size == 1
                and "not saved" in str(arr.ravel()[0]).lower()
            ):
                continue
            out[key] = arr
        else:
            out[key] = arr.ravel()[idx]
    if spectrum is not None:
        out["name"] = np.full(idx.size, str(spectrum), dtype=object)
    return out
