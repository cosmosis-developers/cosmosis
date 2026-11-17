"""Tomographic bin-pair embedding layout for ``nn_model = EmbMLP``.

The flat 3x2pt theory vector is a concatenation of ``(name, bin1, bin2)``
blocks, each spanning several ell nodes. ``EmbMLP`` conditions the network on
bin-pair identity instead of giving every mode its own output weight, so it
needs a map from flat mode index to ``(pair, ell-slot)``.

Bin identity is categorical (5 tomographic bins per side), so it enters through
learned lookup tables. Source and lens identity use disjoint row ranges of one
table -- they share no parameters, but a single ``gather`` covers both.

Row layout of the combined table::

    [ 0, n_probes )                                probe identity
    [ n_probes, n_probes + n_source )              source-bin identity
    [ n_probes + n_source, + n_lens )              lens-bin identity

Each pair contributes three rows: its probe, then two slots whose meaning
depends on the probe::

    shear_cl         slot1 = source(bin1), slot2 = source(bin2)
    galaxy_cl        slot1 = lens(bin1),   slot2 = lens(bin2)
    galaxy_shear_cl  slot1 = lens(bin1),   slot2 = source(bin2)

C_ell^ij = C_ell^ji holds for the two auto-probes, so their pairs are
canonicalised to ``(min, max)``. Scale-cut files normally store only the upper
triangle, but if both orderings are present they collapse onto the same pair
and therefore receive an identical prediction by construction. The cross
spectrum is not symmetric and keeps its ``(lens, source)`` ordering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .amplitude_prefactor import GC_SPECTRA, WL_SPECTRA, XC_SPECTRA
from .vector_blocks import iter_bin_pairs

logger = logging.getLogger(__name__)

SOURCE = "source"
LENS = "lens"


def _slot_sides(spectrum: str) -> Tuple[str, str]:
    """Which bin type each of the two slots refers to, for one spectrum."""
    if spectrum in WL_SPECTRA:
        return (SOURCE, SOURCE)
    if spectrum in GC_SPECTRA:
        return (LENS, LENS)
    if spectrum in XC_SPECTRA:
        return (LENS, SOURCE)
    raise ValueError(
        f"EmbMLP: unknown spectrum {spectrum!r}; expected one of "
        f"{sorted(set(WL_SPECTRA) | set(GC_SPECTRA) | set(XC_SPECTRA))}"
    )


def _is_symmetric(spectrum: str) -> bool:
    """True when C^ij = C^ji, i.e. both bins index the same tomographic set."""
    return spectrum in WL_SPECTRA or spectrum in GC_SPECTRA


def _ell_grid(angles: Optional[np.ndarray], n_modes: int) -> Optional[np.ndarray]:
    """Sorted unique ell nodes, or None when angle metadata is unusable."""
    if angles is None:
        return None
    arr = np.asarray(angles, dtype=float).ravel()
    if arr.size != n_modes or not np.all(np.isfinite(arr)):
        return None
    # Angles repeat exactly across bin pairs (same ell array per spectrum);
    # round only to absorb float round-trips through the metadata.
    return np.unique(np.round(arr, 9))


def build_bin_pair_spec(
    metadata: Dict[str, Any],
    n_modes: int,
    spectra: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the flat-mode -> (pair, ell-slot) layout used by ``EmbMLP``.

    Returns a dict with:

    ``slot_rows``     ``(n_pairs, 3)`` int rows into the embedding table
    ``gather_index``  ``(n_modes,)`` int index into a flattened
                      ``(n_pairs, n_ell_grid)`` network output
    ``table_size``    number of embedding rows
    ``n_pairs`` / ``n_ell_grid``  network output layout
    ``pairs``         per-pair ``(spectrum, bin1, bin2)`` for logging
    """
    if not metadata:
        raise ValueError(
            "nn_model=EmbMLP requires fiducial vector metadata "
            "(name/bin1/bin2, or angle+bin1+bin2 for name inference)."
        )
    blocks = iter_bin_pairs(metadata, spectra=spectra)
    if not blocks:
        raise ValueError("EmbMLP: no bin-pair blocks found in vector metadata")

    covered = int(sum(b["indices"].size for b in blocks))
    if covered != n_modes:
        raise ValueError(
            f"EmbMLP: bin-pair blocks cover {covered} modes but the training "
            f"target has {n_modes}; metadata and theory vector disagree."
        )

    grid = _ell_grid(metadata.get("angle"), n_modes)
    if grid is None:
        logger.warning(
            "EmbMLP: no usable 'angle' metadata; falling back to positional "
            "ell slots within each bin pair. This is only correct if every "
            "pair starts at the same ell_min."
        )
        angles = None
    else:
        angles = np.asarray(metadata["angle"], dtype=float).ravel()

    probes: List[str] = []
    source_bins: List[int] = []
    lens_bins: List[int] = []
    for block in blocks:
        spectrum = str(block["name"])
        if spectrum not in probes:
            probes.append(spectrum)
        side1, side2 = _slot_sides(spectrum)
        for side, b in ((side1, int(block["bin1"])), (side2, int(block["bin2"]))):
            target = source_bins if side == SOURCE else lens_bins
            if b not in target:
                target.append(b)
    source_bins.sort()
    lens_bins.sort()

    probe_row = {p: i for i, p in enumerate(probes)}
    source_row = {b: len(probes) + i for i, b in enumerate(source_bins)}
    lens_row = {
        b: len(probes) + len(source_bins) + i for i, b in enumerate(lens_bins)
    }
    table_size = len(probes) + len(source_bins) + len(lens_bins)

    def _rows(spectrum: str, bin1: int, bin2: int) -> Tuple[int, int, int]:
        side1, side2 = _slot_sides(spectrum)
        lookup = {SOURCE: source_row, LENS: lens_row}
        return (
            probe_row[spectrum],
            lookup[side1][bin1],
            lookup[side2][bin2],
        )

    n_ell_grid = int(grid.size) if grid is not None else max(
        int(b["indices"].size) for b in blocks
    )

    pair_rows: List[Tuple[int, int, int]] = []
    pair_tags: List[Tuple[str, int, int]] = []
    pair_of_key: Dict[Tuple[str, int, int], int] = {}
    gather_index = np.full(n_modes, -1, dtype=np.int64)

    for block in blocks:
        spectrum = str(block["name"])
        b1, b2 = int(block["bin1"]), int(block["bin2"])
        if _is_symmetric(spectrum) and b1 > b2:
            b1, b2 = b2, b1
        key = (spectrum, b1, b2)
        if key not in pair_of_key:
            pair_of_key[key] = len(pair_rows)
            pair_rows.append(_rows(spectrum, b1, b2))
            pair_tags.append(key)
        p = pair_of_key[key]

        idx = np.asarray(block["indices"], dtype=int)
        if angles is not None:
            slots = np.searchsorted(grid, np.round(angles[idx], 9))
        else:
            slots = np.arange(idx.size, dtype=int)
        if np.any(slots >= n_ell_grid):
            raise ValueError(
                f"EmbMLP: ell slot out of range for {key} "
                "(inconsistent angle metadata)"
            )
        gather_index[idx] = p * n_ell_grid + slots

    if np.any(gather_index < 0):
        raise ValueError("EmbMLP: some modes were not assigned a bin pair")

    duplicates = int(n_modes - np.unique(gather_index).size)
    if duplicates:
        logger.info(
            "EmbMLP: %d modes share a bin pair with another mode (symmetric "
            "orderings present); they are predicted identically by construction.",
            duplicates,
        )

    logger.info(
        "EmbMLP layout: %d modes -> %d pairs x %d ell nodes; embedding table "
        "%d rows (%d probes, %d source bins, %d lens bins)",
        n_modes, len(pair_rows), n_ell_grid, table_size,
        len(probes), len(source_bins), len(lens_bins),
    )

    return {
        "slot_rows": np.asarray(pair_rows, dtype=np.int64),
        "gather_index": gather_index,
        "table_size": int(table_size),
        "n_pairs": int(len(pair_rows)),
        "n_ell_grid": int(n_ell_grid),
        "n_modes": int(n_modes),
        "probes": list(probes),
        "source_bins": list(source_bins),
        "lens_bins": list(lens_bins),
        "pairs": [list(t) for t in pair_tags],
    }


def build_per_probe_pair_spec(
    metadata: Dict[str, Any],
    n_modes: int,
    probe_order: List[str],
    spectra: Optional[Any] = None,
) -> Dict[str, Any]:
    """Bin-pair layout split by probe, for ``nn_model = SharedTrunkEmbMLP``.

    Differs from :func:`build_bin_pair_spec` in two ways. Probe identity is not
    embedded, because each probe has its own head; and each probe gets its own
    pair list and ell grid, since scale cuts leave the probes on different ell
    ranges. The source/lens lookup table stays global, so source bin 3 shares
    one embedding row between the shear and cross heads -- it is the same
    redshift kernel in both.

    Probe heads are laid out in ``probe_order`` (the caller's head order); each
    contributes ``n_pairs * n_ell_grid`` outputs and ``gather_index`` maps the
    flat theory vector into that concatenation.
    """
    blocks = iter_bin_pairs(metadata, spectra=spectra)
    if not blocks:
        raise ValueError(
            "SharedTrunkEmbMLP: no bin-pair blocks found in vector metadata"
        )
    covered = int(sum(b["indices"].size for b in blocks))
    if covered != int(n_modes):
        raise ValueError(
            f"SharedTrunkEmbMLP: bin-pair blocks cover {covered} modes but the "
            f"training target has {n_modes}; metadata and theory vector "
            "disagree."
        )

    if _ell_grid(metadata.get("angle"), n_modes) is None:
        logger.warning(
            "SharedTrunkEmbMLP: no usable 'angle' metadata; falling back to "
            "positional ell slots within each bin pair. This is only correct "
            "if every pair of a probe starts at the same ell_min."
        )
        angles = None
    else:
        angles = np.asarray(metadata["angle"], dtype=float).ravel()

    by_probe: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        by_probe.setdefault(str(block["name"]), []).append(block)
    missing = [p for p in probe_order if p not in by_probe]
    if missing:
        raise ValueError(
            f"SharedTrunkEmbMLP: no bin pairs found for probe(s) {missing}"
        )
    unexpected = [p for p in by_probe if p not in probe_order]
    if unexpected:
        raise ValueError(
            f"SharedTrunkEmbMLP: bin pairs found for probe(s) {unexpected} "
            "that have no head; probe ordering is inconsistent."
        )

    # One lookup table across probes: bin identity is the same physical kernel
    # wherever it appears, even though the heads that consume it are separate.
    source_bins: List[int] = []
    lens_bins: List[int] = []
    for block in blocks:
        side1, side2 = _slot_sides(str(block["name"]))
        for side, b in ((side1, int(block["bin1"])), (side2, int(block["bin2"]))):
            target = source_bins if side == SOURCE else lens_bins
            if b not in target:
                target.append(b)
    source_bins.sort()
    lens_bins.sort()
    source_row = {b: i for i, b in enumerate(source_bins)}
    lens_row = {b: len(source_bins) + i for i, b in enumerate(lens_bins)}
    table_size = len(source_bins) + len(lens_bins)

    gather_index = np.full(int(n_modes), -1, dtype=np.int64)
    per_probe: List[Dict[str, Any]] = []
    offset = 0
    for probe in probe_order:
        probe_blocks = by_probe[probe]
        if angles is not None:
            probe_idx = np.concatenate(
                [np.asarray(b["indices"], dtype=int) for b in probe_blocks]
            )
            grid = np.unique(np.round(angles[probe_idx], 9))
            n_ell_grid = int(grid.size)
        else:
            grid = None
            n_ell_grid = max(int(b["indices"].size) for b in probe_blocks)

        pair_rows: List[Tuple[int, int]] = []
        pair_tags: List[Tuple[str, int, int]] = []
        pair_of_key: Dict[Tuple[str, int, int], int] = {}
        for block in probe_blocks:
            spectrum = str(block["name"])
            b1, b2 = int(block["bin1"]), int(block["bin2"])
            if _is_symmetric(spectrum) and b1 > b2:
                b1, b2 = b2, b1
            key = (spectrum, b1, b2)
            if key not in pair_of_key:
                side1, side2 = _slot_sides(spectrum)
                lookup = {SOURCE: source_row, LENS: lens_row}
                pair_of_key[key] = len(pair_rows)
                pair_rows.append((lookup[side1][b1], lookup[side2][b2]))
                pair_tags.append(key)
            p = pair_of_key[key]

            idx = np.asarray(block["indices"], dtype=int)
            if grid is not None:
                slots = np.searchsorted(grid, np.round(angles[idx], 9))
            else:
                slots = np.arange(idx.size, dtype=int)
            if np.any(slots >= n_ell_grid):
                raise ValueError(
                    f"SharedTrunkEmbMLP: ell slot out of range for {key} "
                    "(inconsistent angle metadata)"
                )
            gather_index[idx] = offset + p * n_ell_grid + slots

        n_pairs = len(pair_rows)
        per_probe.append(
            {
                "name": probe,
                "slot_rows": np.asarray(pair_rows, dtype=np.int64),
                "n_pairs": int(n_pairs),
                "n_ell_grid": int(n_ell_grid),
                "output_offset": int(offset),
                "pairs": [list(t) for t in pair_tags],
            }
        )
        offset += n_pairs * n_ell_grid

    if np.any(gather_index < 0):
        raise ValueError("SharedTrunkEmbMLP: some modes were not assigned a bin pair")

    duplicates = int(n_modes - np.unique(gather_index).size)
    if duplicates:
        logger.info(
            "SharedTrunkEmbMLP: %d modes share a bin pair with another mode "
            "(symmetric orderings present); they are predicted identically by "
            "construction.",
            duplicates,
        )

    logger.info(
        "SharedTrunkEmbMLP layout: %d modes -> %s; embedding table %d rows "
        "(%d source bins, %d lens bins)",
        n_modes,
        [f"{p['name']}:{p['n_pairs']}x{p['n_ell_grid']}" for p in per_probe],
        table_size,
        len(source_bins),
        len(lens_bins),
    )

    return {
        "per_probe": per_probe,
        "gather_index": gather_index,
        "table_size": int(table_size),
        "total_output": int(offset),
        "source_bins": list(source_bins),
        "lens_bins": list(lens_bins),
    }


def spec_from_state(state: Any) -> Dict[str, Any]:
    """Rebuild a spec dict loaded from ``.npz`` (arrays come back as objects)."""
    if isinstance(state, np.ndarray):
        state = state.item()
    spec = dict(state)
    spec["slot_rows"] = np.asarray(spec["slot_rows"], dtype=np.int64)
    spec["gather_index"] = np.asarray(spec["gather_index"], dtype=np.int64)
    for key in ("table_size", "n_pairs", "n_ell_grid", "n_modes"):
        spec[key] = int(spec[key])
    return spec
