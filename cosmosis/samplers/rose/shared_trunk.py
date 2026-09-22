"""Probe partition and parameter routing for ``nn_model = SharedTrunkMLP``.

The cosmological integrand behind a 3x2pt data vector is common to all three
probes; only the redshift-kernel weighting and the nuisance response differ.
``SharedTrunkMLP`` encodes that: one trunk over the cosmological parameters is
shared by every probe, and each probe then gets its own head fed with the trunk
latent plus only the nuisance parameters that enter that probe.

The head parameter subsets are exactly the ones ``spectrum_emulators = T`` gives
its independent networks (:func:`vector_blocks.spectrum_model_parameters`), so
the two options differ only in whether the cosmology is learned once or three
times.

Output layout: each head predicts its probe's modes in ascending flat-vector
order, the head outputs are concatenated, and ``gather_index`` puts them back
into likelihood order.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .vector_blocks import spectrum_mode_indices, spectrum_model_parameters

logger = logging.getLogger(__name__)

# CosmoSIS section whose parameters feed the shared trunk by default.
COSMOLOGY_SECTION = "cosmological_parameters"


def default_trunk_parameters(model_parameters: Sequence[str]) -> List[str]:
    """Parameters routed to the shared trunk when ``trunk_params`` is unset."""
    return [
        str(p)
        for p in model_parameters
        if str(p).split("--", 1)[0] == COSMOLOGY_SECTION
    ]


def split_trunk_head_parameters(
    model_parameters: Sequence[str],
    probes: Sequence[str],
    trunk_params: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Route varied parameters to the shared trunk and the per-probe heads.

    Trunk membership defaults to the ``cosmological_parameters`` section. Each
    head takes the parameters that probe depends on, minus whatever the trunk
    already carries.
    """
    names = [str(p) for p in model_parameters]
    index_of = {p: i for i, p in enumerate(names)}

    if trunk_params is None:
        trunk = default_trunk_parameters(names)
    else:
        trunk = [str(p) for p in trunk_params]
        unknown = [p for p in trunk if p not in index_of]
        if unknown:
            raise ValueError(
                f"trunk_params lists parameter(s) that are not varied by this "
                f"pipeline: {unknown}. Varied parameters are: {names}"
            )
    if not trunk:
        raise ValueError(
            "SharedTrunkMLP: no parameters were routed to the shared trunk. "
            f"No varied parameter is in the '{COSMOLOGY_SECTION}' section; set "
            "trunk_params explicitly to choose the shared inputs."
        )

    trunk_set = set(trunk)
    head_parameters: List[List[str]] = []
    for probe in probes:
        probe_params = spectrum_model_parameters(str(probe), names)
        head_parameters.append([p for p in probe_params if p not in trunk_set])

    return {
        "trunk_parameters": trunk,
        "trunk_param_index": np.asarray(
            [index_of[p] for p in trunk], dtype=np.int64
        ),
        "head_parameters": head_parameters,
        "head_param_index": [
            np.asarray([index_of[p] for p in head], dtype=np.int64)
            for head in head_parameters
        ],
    }


def build_shared_trunk_spec(
    metadata: Dict[str, Any],
    n_modes: int,
    model_parameters: Sequence[str],
    spectra: Optional[Any] = None,
    trunk_params: Optional[Sequence[str]] = None,
    bin_embedding: bool = False,
) -> Dict[str, Any]:
    """Build the probe partition and parameter routing for a shared trunk.

    Returns a dict with:

    ``probes``            probe names, in head order
    ``probe_mode_index``  per-probe flat-vector indices (ascending)
    ``gather_index``      ``(n_modes,)`` index into the concatenated head output
    ``trunk_param_index`` / ``head_param_index``  columns of the parameter
                          vector feeding the trunk and each head

    With ``bin_embedding=True`` (``SharedTrunkEmbMLP``) each head predicts one
    bin pair's ell nodes rather than the probe's whole block, so the spec also
    carries the per-probe bin-pair layout and ``gather_index`` indexes the
    concatenated ``(pair, ell)`` outputs instead.
    """
    model = "SharedTrunkEmbMLP" if bin_embedding else "SharedTrunkMLP"
    if not metadata:
        raise ValueError(
            f"nn_model={model} requires fiducial vector metadata "
            "(name/bin1/bin2, or angle+bin1+bin2 for name inference)."
        )
    mode_indices = spectrum_mode_indices(metadata, spectra=spectra)
    if not mode_indices:
        raise ValueError(
            f"{model}: no probe blocks found in the vector metadata"
        )

    probes = list(mode_indices.keys())
    covered = int(sum(idx.size for idx in mode_indices.values()))
    if covered != int(n_modes):
        raise ValueError(
            f"{model}: probe blocks cover {covered} modes but the "
            f"training target has {n_modes}; metadata and theory vector "
            "disagree."
        )
    if len(probes) == 1:
        logger.warning(
            "%s: only one probe (%s) is present, so the trunk is shared with "
            "nothing and the model reduces to a single-probe network.",
            model,
            probes[0],
        )

    gather_index = np.full(int(n_modes), -1, dtype=np.int64)
    probe_mode_index: List[np.ndarray] = []
    offset = 0
    for probe in probes:
        idx = np.asarray(mode_indices[probe], dtype=np.int64).ravel()
        gather_index[idx] = offset + np.arange(idx.size, dtype=np.int64)
        probe_mode_index.append(idx)
        offset += int(idx.size)
    if np.any(gather_index < 0):
        raise ValueError(f"{model}: some modes were not assigned a probe")

    routing = split_trunk_head_parameters(
        model_parameters, probes, trunk_params=trunk_params
    )

    spec: Dict[str, Any] = {
        "probes": probes,
        "probe_mode_index": probe_mode_index,
        "gather_index": gather_index,
        "n_modes": int(n_modes),
        "parameters": [str(p) for p in model_parameters],
    }
    spec.update(routing)

    if bin_embedding:
        from .bin_embedding import build_per_probe_pair_spec

        pair_spec = build_per_probe_pair_spec(
            metadata, int(n_modes), probes, spectra=spectra
        )
        # Heads now emit (pair, ell) nodes, so the pair layout defines where
        # each flat mode is read from.
        spec["gather_index"] = pair_spec["gather_index"]
        spec["per_probe_pairs"] = pair_spec["per_probe"]
        spec["table_size"] = pair_spec["table_size"]
        spec["source_bins"] = pair_spec["source_bins"]
        spec["lens_bins"] = pair_spec["lens_bins"]

    logger.info(
        "%s layout: %d modes -> %d probes %s; trunk sees %d "
        "parameter(s) %s",
        model,
        int(n_modes),
        len(probes),
        [f"{p}:{idx.size}" for p, idx in zip(probes, probe_mode_index)],
        len(spec["trunk_parameters"]),
        spec["trunk_parameters"],
    )
    for probe, head in zip(probes, spec["head_parameters"]):
        logger.info(
            "  [%s] head sees %d extra parameter(s): %s",
            probe,
            len(head),
            head if head else "(trunk latent only)",
        )
    return spec


def spec_from_state(state: Any) -> Dict[str, Any]:
    """Rebuild a spec loaded from ``.npz`` (arrays come back as objects)."""
    if isinstance(state, np.ndarray):
        state = state.item()
    spec = dict(state)
    spec["probes"] = [str(p) for p in spec["probes"]]
    spec["parameters"] = [str(p) for p in spec.get("parameters", [])]
    spec["gather_index"] = np.asarray(spec["gather_index"], dtype=np.int64)
    spec["n_modes"] = int(spec["n_modes"])
    spec["probe_mode_index"] = [
        np.asarray(idx, dtype=np.int64) for idx in spec["probe_mode_index"]
    ]
    spec["trunk_param_index"] = np.asarray(
        spec["trunk_param_index"], dtype=np.int64
    )
    spec["head_param_index"] = [
        np.asarray(idx, dtype=np.int64) for idx in spec["head_param_index"]
    ]
    spec["trunk_parameters"] = [str(p) for p in spec["trunk_parameters"]]
    spec["head_parameters"] = [
        [str(p) for p in head] for head in spec["head_parameters"]
    ]
    if "per_probe_pairs" in spec:
        spec["per_probe_pairs"] = [
            {
                "name": str(p["name"]),
                "slot_rows": np.asarray(p["slot_rows"], dtype=np.int64),
                "n_pairs": int(p["n_pairs"]),
                "n_ell_grid": int(p["n_ell_grid"]),
                "output_offset": int(p["output_offset"]),
                "pairs": [list(t) for t in p["pairs"]],
            }
            for p in spec["per_probe_pairs"]
        ]
        spec["table_size"] = int(spec["table_size"])
    return spec
