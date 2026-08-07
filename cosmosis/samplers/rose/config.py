"""
Configuration methods for ROSE sampler.

This module contains methods for reading and validating configuration
parameters from ini files.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np

from .utils import SAVE_NONE, SAVE_MODEL, SAVE_ALL, mkdir

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _parse_per_likelihood(
    raw: str,
    value_type: Callable[[str], Any] = str,
    default: Any = None,
) -> Union[Any, Dict[str, Any]]:
    """Parse a setting that may be either global or per-likelihood.

    Accepted syntaxes (whitespace-separated tokens)::

        "log_norm"
            → "log_norm"  (single value applied to every likelihood)

        "lsst:signed_log_norm desi_bao:log_norm"
            → {"lsst": "signed_log_norm", "desi_bao": "log_norm"}

        "lsst:signed_log_norm default:log_norm"
            → {"lsst": "signed_log_norm", "__default__": "log_norm"}
            (``default:`` sets a fallback used by any likelihood not listed)

    Values are run through ``value_type`` (e.g. ``int``, ``float``) so numeric
    settings like ``n_pca`` or ``batch_sizes`` still type-check.
    """
    if raw is None or not str(raw).strip():
        return default
    tokens = str(raw).split()
    if len(tokens) == 1 and ":" not in tokens[0]:
        return value_type(tokens[0])
    result: Dict[str, Any] = {}
    for tok in tokens:
        if ":" not in tok:
            raise ValueError(
                f"Per-likelihood setting token '{tok}' is missing a ':'. "
                "Use either a single value (applied globally) or "
                "'name:value' pairs, e.g. 'lsst:log_norm desi_bao:norm'."
            )
        name, val = tok.split(":", 1)
        name = name.strip()
        key = "__default__" if name.lower() in ("default", "*") else name
        result[key] = value_type(val.strip())
    return result


def _parse_number_list(
    raw: str,
    value_type: Callable[[str], T] = float,
    default: Optional[List[T]] = None,
) -> Optional[List[T]]:
    """Parse a space- or comma-separated list of numbers from an ini string.

    Accepts forms like ``512 512 512``, ``512,512,512``, or ``[512, 512, 512]``.
    An empty / missing string returns ``default`` (typically ``None``, meaning
    the caller should fall back to its built-in schedule).
    """
    if raw is None or not str(raw).strip():
        return default
    cleaned = str(raw).strip().strip("[]()")
    cleaned = cleaned.replace(",", " ")
    parts = cleaned.split()
    if not parts:
        return default
    return [value_type(x) for x in parts]


def default_resample_hpd_fractions(n_resample: int) -> List[float]:
    """Default HPD fractions for each training resample step.

    ``n_resample = iterations - 1`` (initial LH is separate). Matches::

        4 iterations (3 resamples): 0.50, 0.60, 0.75
        5 iterations (4 resamples): 0.50, 0.60, 0.75, 0.90

    Otherwise linearly spaced from 0.50 → 0.90.
    """
    n = int(n_resample)
    if n <= 0:
        return []
    if n == 1:
        return [0.75]
    if n == 2:
        return [0.50, 0.75]
    if n == 3:
        return [0.50, 0.60, 0.75]
    if n == 4:
        return [0.50, 0.60, 0.75, 0.90]
    return [float(x) for x in np.linspace(0.50, 0.90, n)]


def _parse_number_list_setting(
    raw: str,
    value_type: Callable[[str], T] = float,
    default: Any = None,
) -> Union[Optional[List[T]], Dict[str, List[T]]]:
    """Parse a number-list setting that may be global or per-likelihood.

    Global forms (applied to every likelihood)::

        "512 512 512"
        "512,512,512"
        "[512, 512, 512]"

    Per-likelihood forms use whitespace-separated ``name:v1,v2,...`` tokens
    (commas inside the value; avoid spaces inside a token)::

        "lsst:512,512,512 desi_bao:256,256,256,256"
        "lsst:1e-2,1e-3,1e-4 default:1e-2,1e-3,1e-4,1e-5"

    An empty / missing string returns ``default``.
    """
    if raw is None or not str(raw).strip():
        return default
    tokens = str(raw).split()
    if any(":" in tok for tok in tokens):
        result: Dict[str, List[T]] = {}
        for tok in tokens:
            if ":" not in tok:
                raise ValueError(
                    f"Per-likelihood list token '{tok}' is missing a ':'. "
                    "Use a global list (e.g. '512 512 512') or "
                    "'name:v1,v2,...' tokens (commas inside the value)."
                )
            name, val = tok.split(":", 1)
            name = name.strip()
            key = "__default__" if name.lower() in ("default", "*") else name
            parsed = _parse_number_list(val, value_type)
            if not parsed:
                raise ValueError(
                    f"Empty number list for likelihood '{name}' in setting "
                    f"token '{tok}'"
                )
            result[key] = parsed
        return result
    return _parse_number_list(raw, value_type, default=default)


def _setting_values(setting: Any) -> list:
    """Return the list of concrete values inside a setting.

    Works for both scalars and per-likelihood dicts so validation code can
    iterate uniformly.
    """
    if isinstance(setting, dict):
        return list(setting.values())
    return [setting]


class RoseConfigMixin:
    """Mixin class providing configuration methods for RoseSampler."""
    
    def _configure_output_saving(self) -> None:
        """Configure output saving options."""
        save_dir = getattr(self.output, "filename_base", "")
        if save_dir:
            self.save_dir = self.read_ini("save_dir", str, save_dir + ".rose_output")
        else:
            self.save_dir = self.read_ini("save_dir", str, "")

        if self.save_dir:
            save_output = self.read_ini_choices("save_output", str, ["", "all", "last", "none"], "")
            if save_output == "" or save_output == "none":
                self.save_output = SAVE_NONE
            elif save_output == "last":
                mkdir(self.save_dir)
            elif save_output == "all":
                # must be save_output == "all" or would have failed above
                mkdir(self.save_dir)
                self.save_output = SAVE_ALL
            else:
                raise ValueError(f"Unknown save_output option '{save_output}' - "
                               "should be 'last', 'all', 'none' or empty")
        else:
            self.save_output = SAVE_NONE
            # Set a default directory for model saving (even when not saving outputs)
            # This is needed because train_emulator always needs a model_filename
            import tempfile
            import os
            self.save_dir = os.path.join(tempfile.gettempdir(), "rose_emulator")
            mkdir(self.save_dir)
            logger.warning("No outputs will be saved (save_output not specified), "
                         f"but models will be saved to temporary directory: {self.save_dir}")
    
    def _configure_training_parameters(self) -> None:
        """Configure neural network training parameters.

        ``batch_sizes`` and ``n_cycles_per_training`` may be specified globally
        or with per-likelihood overrides, e.g.::

            batch_sizes = 128
            batch_sizes = 64 64 64 64 64
            batch_sizes = lsst:64,64,64 desi_bao:32
            n_cycles_per_training = lsst:8 default:5

        A single ``batch_sizes`` value is broadcast across all training cycles.
        Values are for the first training iteration; later iterations scale every
        entry by ``n_train / n_train_first`` as the training set grows.
        """
        self.max_iterations = self.read_ini("iterations", int, 4)
        self.initial_size = self.read_ini("initial_size", int, 9600)
        self.resample_size = self.read_ini("resample_size", int, 4800)
        self.chi2_cut_off = self.read_ini("chi2_cut_off", float, 1e6)
        # Number of test points collected once, in the last training iteration,
        # from the 1-sigma or 2-sigma (credible) region of the one-before-last MCMC chain.
        self.final_test_size = self.read_ini(
            "final_test_size", int, int(0.8 * self.resample_size)
        )
        # Fraction of the chain (ranked by posterior) that defines the
        # tempered HPD credible region used for training/test resampling.
        # Use 0.68 for 1-sigma or 0.95 for 2-sigma.
        self.test_credible_fraction = self.read_ini(
            "test_credible_fraction", float, 0.95 
        )
        # HPD fill design: split the tempered HPD into this many equal-mass
        # log-posterior shells and draw volume-uniform Latin-hypercube points
        # inside each shell's whitened ellipsoid (not farthest-point).
        self.resample_hpd_nshells = self.read_ini("resample_hpd_nshells", int, 4)
        # Mahalanobis-radius quantile of shell points used as the ellipsoid
        # boundary (1.0 = max distance in the shell).
        self.resample_hpd_radius_quantile = self.read_ini(
            "resample_hpd_radius_quantile", float, 0.95
        )
        # Resample mixture: fraction of new training points from tempered HPD
        # (rest = HPD base with weak-param coordinates redrawn from the prior).
        # Blank → automatic schedule from ``iterations`` (see
        # ``default_resample_hpd_fractions``). Override with an explicit list
        # of length ``iterations - 1``, e.g. ``0.5 0.6 0.75``.
        self.resample_hpd_fractions = _parse_number_list(
            self.read_ini("resample_hpd_fractions", str, ""), float, default=None
        )
        # Varied parameters whose unit-cube coordinates are redrawn from the
        # prior on the explore half of each resample (space-separated
        # ``section--name`` or unique ``name``). Empty → 100% HPD (legacy).
        weak_raw = self.read_ini("resample_weak_params", str, "")
        self.resample_weak_params = [
            p.strip() for p in weak_raw.replace(",", " ").split() if p.strip()
        ]
        self.resample_weak_param_indices = self._resolve_resample_weak_param_indices(
            self.resample_weak_params
        )
        if self.resample_hpd_fractions is None:
            n_resample = max(int(self.max_iterations) - 1, 0)
            self.resample_hpd_fractions = default_resample_hpd_fractions(n_resample)
        if self.resample_weak_param_indices:
            logger.info(
                "Resample mixture enabled: weak params %s (indices %s); "
                "HPD fractions per resample step: %s",
                [str(self.pipeline.varied_params[i]) for i in self.resample_weak_param_indices],
                self.resample_weak_param_indices,
                [f"{x:.2f}" for x in self.resample_hpd_fractions],
            )
        # Optional: before training the final emulator, drop early training
        # points whose *true* pipeline chi2 is much worse than the last
        # training iteration (``chi2 > outlier_chi2_factor * max(chi2_last)``).
        # Only the initial prior draw and the first resampled iteration are
        # eligible; later iterations (already near the tempered posterior)
        # are kept. Does not use the emulated-chain HPD geometry.
        self.remove_outliers = self.read_ini("remove_outliers", bool, False)
        self.outlier_chi2_factor = self.read_ini("outlier_chi2_factor", float, 2.5)
        # How many leading ``points_per_iteration`` slices may be pruned
        # (1 = initial only, 2 = initial + first iteration, …).
        self.outlier_prune_n_early = self.read_ini("outlier_prune_n_early", int, 2)
        # Legacy knob (emulated-chain HPD box). Ignored by the chi2 prune;
        # kept so old inis still parse.
        self.outlier_nsigma = self.read_ini("outlier_nsigma", float, 3.0)
        # Final-stage convergence (opt-in via kl_convergence = T).
        # The held-out test set is NEVER folded into training. After the last
        # tempered train, ROSE scores Δχ² on that holdout and stops when
        # MAD(Δχ² − median) <= delta_chi2_mad_threshold. If not, it adds new
        # HPD training points and retrains up to kl_max_retrain times.
        # With kl_convergence_chain = T, each (re)trained emulator also gets an
        # untempered final_sampler chain; Jeffreys KL on kl_params vs the
        # previous chain can stop the loop when < kl_threshold.
        # Passive IW KL (all + selected) is logged to rose_kl.txt only.
        self.kl_convergence = self.read_ini("kl_convergence", bool, False)
        self.kl_convergence_chain = self.read_ini("kl_convergence_chain", bool, False)
        self.kl_threshold = self.read_ini("kl_threshold", float, 0.1)
        self.kl_max_retrain = self.read_ini("kl_max_retrain", int, 5)
        self.kl_extra_size = self.read_ini("kl_extra_size", int, self.resample_size)
        self.kl_n_samples = self.read_ini("kl_n_samples", int, 7000)
        self.delta_chi2_mad_threshold = self.read_ini(
            "delta_chi2_mad_threshold", float, 1.0
        )
        # Neighbour order for the per-iteration sample-based (k-NN) KL
        # estimator. k=1 is the classic Wang–Kulkarni–Verdú estimator; k=3–5
        # is usually less noisy in moderate dimension (e.g. ~6 cosmological
        # parameters).
        self.kl_knn_k = self.read_ini("kl_knn_k", int, 10)
        # Subtract a same-distribution "null" k-NN KL (split-sample self-KL)
        # from the cross-chain estimate. This removes most of the positive
        # finite-sample bias of the Wang–Kulkarni–Verdú estimator in moderate
        # dimension, so kl_knn is closer in scale to kl_gaussian when the
        # posteriors are similar.
        self.kl_knn_debias = self.read_ini("kl_knn_debias", bool, True)
        # Parameters to drop from the default "selected" KL subspace. Blank
        # defaults to ``resample_weak_params``. Set ``none`` to keep all.
        kl_excl_raw = self.read_ini("kl_exclude_params", str, "")
        kl_excl_tok = kl_excl_raw.replace(",", " ").split()
        if kl_excl_tok and kl_excl_tok[0].lower() in ("none", "off", "false", "f"):
            self.kl_exclude_params: List[str] = []
            self.kl_exclude_param_indices: List[int] = []
        elif kl_excl_tok:
            self.kl_exclude_params = [p.strip() for p in kl_excl_tok if p.strip()]
            self.kl_exclude_param_indices = self._resolve_resample_weak_param_indices(
                self.kl_exclude_params
            )
        else:
            self.kl_exclude_params = list(self.resample_weak_params)
            self.kl_exclude_param_indices = list(self.resample_weak_param_indices)
        # Optional include-list for the selected KL subspace / chain-KL stop.
        # When blank, selected = all varied minus kl_exclude_params.
        kl_incl_raw = self.read_ini("kl_params", str, "")
        kl_incl_tok = [p.strip() for p in kl_incl_raw.replace(",", " ").split() if p.strip()]
        if kl_incl_tok:
            self.kl_params = kl_incl_tok
            self.kl_param_indices = self._resolve_resample_weak_param_indices(
                self.kl_params
            )
        else:
            self.kl_params = []
            self.kl_param_indices = [
                i for i in range(self.ndim)
                if i not in set(self.kl_exclude_param_indices)
            ]
        if not self.kl_param_indices:
            raise ValueError(
                "kl_params / kl_exclude_params leave no parameters for the "
                "selected KL subspace; unset some exclusions or set "
                "kl_exclude_params = none"
            )
        logger.info(
            "KL selected subspace (%d params): %s",
            len(self.kl_param_indices),
            [str(self.pipeline.varied_params[i]) for i in self.kl_param_indices],
        )
        if self.kl_exclude_param_indices and not kl_incl_tok:
            logger.info(
                "KL default exclusions %s (indices %s)",
                [str(self.pipeline.varied_params[i]) for i in self.kl_exclude_param_indices],
                self.kl_exclude_param_indices,
            )

        self.n_cycles_per_training = _parse_per_likelihood(
            self.read_ini("n_cycles_per_training", str, "5"), int
        )
        # Per-stage batch sizes for the *first* training iteration. A single
        # value (default 32) is broadcast to every training cycle; otherwise
        # the list length must match ``n_cycles_per_training``. In later
        # iterations each entry is multiplied by sqrt(n_train / n_train_first).
        # Accepts the same global / per-likelihood list syntax as
        # ``learning_rates``.
        self.batch_sizes = _parse_number_list_setting(
            self.read_ini("batch_sizes", str, "32"), int
        )
        # Optional CosmoPowerNN training-schedule overrides. Leave blank to keep
        # the defaults derived from n_cycles_per_training
        # (learning rates 1e-2, 1e-3, ...; patience=100; max_epochs=1000;
        # gradient_accumulation_steps=1 per cycle).
        #
        # Global list (spaces or commas)::
        #   learning_rates = 1e-2 1e-3 1e-4 1e-5 1e-6
        # Per-likelihood (commas inside each token)::
        #   learning_rates = lsst:1e-2,1e-3,1e-4 desi_bao:1e-2,1e-3 default:1e-2,1e-3,1e-4,1e-5
        #
        # Schedule list lengths must match that likelihood's
        # ``n_cycles_per_training`` (except ``batch_sizes``, where a single
        # value is broadcast).
        self.validation_split = _parse_per_likelihood(
            self.read_ini("validation_split", str, "0.1"), float
        )
        self.learning_rates = _parse_number_list_setting(
            self.read_ini("learning_rates", str, ""), float
        )
        self.gradient_accumulation_steps = _parse_number_list_setting(
            self.read_ini("gradient_accumulation_steps", str, ""), int
        )
        self.patience_values = _parse_number_list_setting(
            self.read_ini("patience_values", str, ""), int
        )
        self.max_epochs = _parse_number_list_setting(
            self.read_ini("max_epochs", str, ""), int
        )
        
        # Validate training parameters
        if self.max_iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.initial_size < 10:
            raise ValueError("initial_size must be >= 10")
        if self.resample_size < 1:
            raise ValueError("resample_size must be >= 1")
        if self.final_test_size < 0:
            raise ValueError("final_test_size must be >= 0 (use 0 to disable test collection)")
        if not (0.0 < self.test_credible_fraction <= 1.0):
            raise ValueError("test_credible_fraction must be in (0, 1]")
        if self.resample_hpd_nshells < 1:
            raise ValueError("resample_hpd_nshells must be >= 1")
        if not (0.0 < self.resample_hpd_radius_quantile <= 1.0):
            raise ValueError("resample_hpd_radius_quantile must be in (0, 1]")
        n_resample = max(int(self.max_iterations) - 1, 0)
        if len(self.resample_hpd_fractions) != n_resample:
            raise ValueError(
                f"resample_hpd_fractions length ({len(self.resample_hpd_fractions)}) "
                f"must equal iterations-1 ({n_resample})"
            )
        if any(not (0.0 <= float(f) <= 1.0) for f in self.resample_hpd_fractions):
            raise ValueError("resample_hpd_fractions entries must be in [0, 1]")
        if self.outlier_chi2_factor <= 1.0:
            raise ValueError(
                "outlier_chi2_factor must be > 1 (threshold is "
                "factor * max(chi2 of last training iteration))"
            )
        if self.outlier_prune_n_early < 1:
            raise ValueError("outlier_prune_n_early must be >= 1")
        if self.outlier_nsigma <= 0:
            raise ValueError("outlier_nsigma must be > 0")
        if self.remove_outliers and self.outlier_nsigma != 3.0:
            logger.warning(
                "outlier_nsigma is deprecated and ignored; remove_outliers now "
                "uses true pipeline chi2 (outlier_chi2_factor / "
                "outlier_prune_n_early)."
            )
        if self.kl_threshold <= 0:
            raise ValueError("kl_threshold must be > 0")
        if self.kl_max_retrain < 0:
            raise ValueError("kl_max_retrain must be >= 0")
        if self.kl_extra_size < 1:
            raise ValueError("kl_extra_size must be >= 1")
        if self.kl_n_samples < 10:
            raise ValueError("kl_n_samples must be >= 10")
        if self.kl_knn_k < 1:
            raise ValueError("kl_knn_k must be >= 1")
        if self.delta_chi2_mad_threshold <= 0:
            raise ValueError("delta_chi2_mad_threshold must be > 0")
        if self.kl_convergence_chain and not self.kl_convergence:
            logger.warning(
                "kl_convergence_chain=T has no effect unless kl_convergence=T"
            )
        if any(v < 1 for v in _setting_values(self.n_cycles_per_training)):
            raise ValueError("n_cycles_per_training must be >= 1 (for every likelihood)")
        for sizes in _setting_values(self.batch_sizes):
            if not sizes or any(v < 1 for v in sizes):
                raise ValueError(
                    "batch_sizes must be a non-empty list of integers >= 1 "
                    "(for every likelihood)"
                )
        if any(not (0.0 < v < 1.0) for v in _setting_values(self.validation_split)):
            raise ValueError("validation_split must be in (0, 1) (for every likelihood)")
        self._validate_training_schedule_lists()
    
    def _resolve_resample_weak_param_indices(self, names: List[str]) -> List[int]:
        """Map ``resample_weak_params`` names to indices in ``varied_params``."""
        if not names:
            return []
        varied = list(self.pipeline.varied_params)
        full = [str(p) for p in varied]
        short = [p.name for p in varied]
        indices: List[int] = []
        for raw in names:
            key = str(raw).strip()
            if key in full:
                indices.append(full.index(key))
                continue
            hits = [i for i, n in enumerate(short) if n == key]
            if len(hits) == 1:
                indices.append(hits[0])
                continue
            if len(hits) > 1:
                raise ValueError(
                    f"resample_weak_params entry '{key}' matches multiple varied "
                    f"parameters {[full[i] for i in hits]}; use section--name"
                )
            raise ValueError(
                f"resample_weak_params entry '{key}' not found in varied parameters "
                f"{full}"
            )
        # Stable unique order
        seen = set()
        unique: List[int] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                unique.append(i)
        return unique

    def _configure_mcmc_parameters(self) -> None:
        """Configure MCMC sampling parameters."""
        self.emcee_walkers = self.read_ini("emcee_walkers", int)
        self.emcee_samples = self.read_ini("emcee_samples", int)
        self.emcee_burn = self.read_ini("emcee_burn", float, 0.3)
        self.emcee_thin = self.read_ini("emcee_thin", int, 1)
        
        # Validate MCMC parameters
        if self.emcee_walkers >= self.initial_size:
            raise ValueError("emcee_walkers must be < initial_size for proper initialization")
        if self.emcee_walkers < 2 * self.ndim:
            logger.warning(f"emcee_walkers ({self.emcee_walkers}) < 2*ndim ({2*self.ndim}) "
                          "may lead to poor sampling")
        if self.emcee_samples < 100:
            logger.warning(f"emcee_samples ({self.emcee_samples}) is very small")
        if not (0 <= self.emcee_burn <= 1) and self.emcee_burn >= self.emcee_samples:
            raise ValueError("emcee_burn must be fraction in [0,1] or integer < emcee_samples")
        if self.emcee_thin < 1:
            raise ValueError("emcee_thin must be >= 1")
    
    def _configure_nautilus_parameters(self) -> None:
        """Configure Nautilus sampling parameters for final iteration."""
        self.nautilus_n_live = self.read_ini("nautilus_n_live", int, 2000)
        self.nautilus_n_update = self.read_ini("nautilus_n_update", int, self.nautilus_n_live)
        self.nautilus_enlarge_per_dim = self.read_ini("nautilus_enlarge_per_dim", float, 1.1)
        self.nautilus_n_points_min = self.read_ini("nautilus_n_points_min", int, self.ndim + 50)
        self.nautilus_split_threshold = self.read_ini("nautilus_split_threshold", float, 100.0)
        self.nautilus_n_networks = self.read_ini("nautilus_n_networks", int, 4)
        self.nautilus_n_batch = self.read_ini("nautilus_n_batch", int, 100)
        self.nautilus_f_live = self.read_ini("nautilus_f_live", float, 0.01)
        self.nautilus_n_shell = self.read_ini("nautilus_n_shell", int, self.nautilus_n_batch)
        self.nautilus_n_eff = self.read_ini("nautilus_n_eff", float, 10000.0)
        self.nautilus_n_like_max = self.read_ini("nautilus_n_like_max", int, 10000000000000000000000)
        self.nautilus_discard_exploration = self.read_ini("nautilus_discard_exploration", bool, True)
        
        # Validate nautilus parameters
        if self.nautilus_n_live < 2 * self.ndim:
            logger.warning(f"nautilus_n_live ({self.nautilus_n_live}) < 2*ndim ({2*self.ndim}) "
                          "may lead to poor sampling")
        if self.nautilus_n_points_min < self.ndim:
            raise ValueError("nautilus_n_points_min must be >= ndim")
        if self.nautilus_n_eff < 100:
            logger.warning(f"nautilus_n_eff ({self.nautilus_n_eff}) is very small")
        
        logger.info(f"Nautilus configured for final iteration: n_live={self.nautilus_n_live}, "
                   f"n_eff={self.nautilus_n_eff}")
    
    def _configure_nuts_parameters(self) -> None:
        """Configure NUTS sampling parameters for final iteration.

        Default diagonal-mass mode uses TFP's Stan-style
        ``windowed_sampling.make_windowed_adapt_kernel`` (expanding fast/slow
        windows for step size + diagonal mass). Prefer
        ``nuts_sample_unit_space=T`` so all parameters share a common scale;
        in physical space the diagonal mass prior uses pilot-chain variances
        or ``(prior_width/4)^2`` instead of unit variance.
        """
        self.nuts_step_size = self.read_ini("nuts_step_size", float, 0.05)
        self.nuts_use_fixed_step_size = self.read_ini("nuts_use_fixed_step_size", bool, False)
        self.nuts_fixed_step_size = self.read_ini("nuts_fixed_step_size", float, 0.3)
        self.nuts_max_tree_depth = self.read_ini("nuts_max_tree_depth", int, 12)
        self.nuts_max_energy_diff = self.read_ini("nuts_max_energy_diff", float, 1000.0)
        # TFP NUTS: compile N leapfrog steps into one TF op (usually leave at 1).
        self.nuts_unrolled_leapfrog_steps = self.read_ini("nuts_unrolled_leapfrog_steps", int, 1)
        # TFP while_loop parallel_iterations for the NUTS tree build.
        self.nuts_parallel_iterations = self.read_ini("nuts_parallel_iterations", int, 10)
        self.nuts_num_adaptation_steps = self.read_ini("nuts_num_adaptation_steps", int, 1000)
        self.nuts_num_burnin_steps = self.read_ini("nuts_num_burnin_steps", int, 1000)
        self.nuts_num_results = self.read_ini("nuts_num_results", int, 2000)
        self.nuts_num_chains = self.read_ini("nuts_num_chains", int, 1)
        # Stan / TFP windowed default is ~0.8–0.85; 0.75 explores a bit more.
        self.nuts_target_accept_prob = self.read_ini("nuts_target_accept_prob", float, 0.8)
        self.nuts_sample_unit_space = self.read_ini("nuts_sample_unit_space", bool, True)
        self.nuts_progress_interval = self.read_ini("nuts_progress_interval", int, 100)
        # Mass-matrix / momentum preconditioning:
        #   none      — identity mass + DualAveraging step size
        #   diagonal  — Stan-style windowed DualAveraging + diagonal mass (default)
        #   dense     — fixed dense mass from previous ROSE chain + DualAveraging
        #               step size (windowed mass adapt is diagonal-only)
        self.nuts_mass_matrix = self.read_ini_choices(
            "nuts_mass_matrix", str, ["none", "diagonal", "dense"], "diagonal"
        )
        # Optional CosmoSIS text chain used for NUTS init + mass pilot when
        # ``trained_before=T`` (no in-memory tempered chain). Leave blank to
        # skip; then init falls back to randomized / pipeline starts.
        self.nuts_pilot_chain = self.read_ini("nuts_pilot_chain", str, "")
        
        # Validate NUTS parameters
        if self.nuts_step_size <= 0:
            raise ValueError("nuts_step_size must be > 0")
        if self.nuts_fixed_step_size <= 0:
            raise ValueError("nuts_fixed_step_size must be > 0")
        if self.nuts_max_tree_depth < 1:
            raise ValueError("nuts_max_tree_depth must be >= 1")
        if self.nuts_unrolled_leapfrog_steps < 1:
            raise ValueError("nuts_unrolled_leapfrog_steps must be >= 1")
        if self.nuts_parallel_iterations < 1:
            raise ValueError("nuts_parallel_iterations must be >= 1")
        if self.nuts_num_results < 100:
            logger.warning(f"nuts_num_results ({self.nuts_num_results}) is very small")
        if self.nuts_num_chains < 1:
            raise ValueError("nuts_num_chains must be >= 1")
        if not (0.5 <= self.nuts_target_accept_prob <= 0.99):
            raise ValueError("nuts_target_accept_prob should be in (0.5, 0.99), e.g. 0.65-0.8 for better exploration")
        if self.nuts_progress_interval < 0:
            raise ValueError(
                "nuts_progress_interval must be >= 0 "
                "(chunk size for the tqdm bar; 0 disables the bar)"
            )
        if self.nuts_num_adaptation_steps > self.nuts_num_burnin_steps:
            logger.warning(
                f"nuts_num_adaptation_steps ({self.nuts_num_adaptation_steps}) > "
                f"nuts_num_burnin_steps ({self.nuts_num_burnin_steps}); "
                "mass/step adaptation will continue into the retained samples. "
                "Prefer burnin >= adaptation so windowed warmup is discarded."
            )
        if self.nuts_use_fixed_step_size and self.nuts_num_adaptation_steps > 0:
            logger.info(
                "nuts_use_fixed_step_size=T: DualAveraging / windowed adaptation "
                f"disabled; using fixed step_size={self.nuts_fixed_step_size}"
            )
        if (
            not self.nuts_sample_unit_space
            and self.nuts_mass_matrix == "diagonal"
            and not self.nuts_use_fixed_step_size
        ):
            logger.info(
                "nuts_sample_unit_space=F: diagonal mass prior will use pilot "
                "variances or (prior_width/4)^2 — not unit variance"
            )
        
        logger.info(
            f"NUTS configured for final iteration: step_size={self.nuts_step_size}, "
            f"fixed_step={self.nuts_use_fixed_step_size}/{self.nuts_fixed_step_size}, "
            f"max_tree_depth={self.nuts_max_tree_depth}, "
            f"unrolled_leapfrog={self.nuts_unrolled_leapfrog_steps}, "
            f"parallel_iterations={self.nuts_parallel_iterations}, "
            f"num_results={self.nuts_num_results}, "
            f"adaptation/burnin={self.nuts_num_adaptation_steps}/"
            f"{self.nuts_num_burnin_steps}, "
            f"target_accept_prob={self.nuts_target_accept_prob}, "
            f"sample_unit_space={self.nuts_sample_unit_space}, "
            f"mass_matrix={self.nuts_mass_matrix} "
            f"({'windowed Stan schedule' if self.nuts_mass_matrix == 'diagonal' and not self.nuts_use_fixed_step_size else 'non-windowed'}); "
            f"multi-chain uses CosmoSIS pool.map when pool is available"
        )

    def _configure_numpyro_parameters(self) -> None:
        """Configure NumPyro NUTS (JAX) for the final iteration.

        The tempered posterior is still the TF autodiff likelihood used by TFP
        NUTS; ``jax.experimental.jax2tf.call_tf`` wraps it so NumPyro can
        differentiate through TF. Requires ``numpyro``, ``jax``, and a
        TF/JAX pair compatible with ``jax2tf`` (see ROSE docs).
        """
        self.numpyro_num_warmup = self.read_ini("numpyro_num_warmup", int, 1000)
        self.numpyro_num_samples = self.read_ini("numpyro_num_samples", int, 2000)
        self.numpyro_num_chains = self.read_ini("numpyro_num_chains", int, 4)
        self.numpyro_target_accept_prob = self.read_ini(
            "numpyro_target_accept_prob", float, 0.8
        )
        self.numpyro_max_tree_depth = self.read_ini("numpyro_max_tree_depth", int, 10)
        self.numpyro_sample_unit_space = self.read_ini(
            "numpyro_sample_unit_space", bool, True
        )
        self.numpyro_progress_bar = self.read_ini("numpyro_progress_bar", bool, True)
        self.numpyro_chain_method = self.read_ini_choices(
            "numpyro_chain_method",
            str,
            ["parallel", "sequential", "vectorized"],
            "parallel",
        )
        # Pilot chain for init / mass matrix when there is no in-memory ROSE
        # chain (typical with trained_before=T). With trained_before=F the
        # previous tempered emcee chain is used automatically; if that is
        # missing, the latest *_tempering_*_iteration_*.txt next to the
        # CosmoSIS output is auto-selected. Leave blank unless overriding.
        # Reuse nuts_pilot_chain if numpyro_pilot_chain is blank.
        self.numpyro_pilot_chain = self.read_ini("numpyro_pilot_chain", str, "")
        if not str(self.numpyro_pilot_chain).strip():
            self.numpyro_pilot_chain = self.read_ini("nuts_pilot_chain", str, "")

        if self.numpyro_num_warmup < 0:
            raise ValueError("numpyro_num_warmup must be >= 0")
        if self.numpyro_num_samples < 100:
            logger.warning(
                f"numpyro_num_samples ({self.numpyro_num_samples}) is very small"
            )
        if self.numpyro_num_chains < 1:
            raise ValueError("numpyro_num_chains must be >= 1")
        if not (0.5 <= self.numpyro_target_accept_prob <= 0.99):
            raise ValueError("numpyro_target_accept_prob should be in (0.5, 0.99)")
        if self.numpyro_max_tree_depth < 1:
            raise ValueError("numpyro_max_tree_depth must be >= 1")

        # Alias pilot path onto nuts_pilot_chain so _maybe_load_nuts_pilot_chain works.
        self.nuts_pilot_chain = str(self.numpyro_pilot_chain)
        # Alias unit-space flag for shared init helpers.
        self.nuts_sample_unit_space = bool(self.numpyro_sample_unit_space)

        logger.info(
            f"NumPyro configured: warmup={self.numpyro_num_warmup}, "
            f"samples={self.numpyro_num_samples}, chains={self.numpyro_num_chains}, "
            f"target_accept={self.numpyro_target_accept_prob}, "
            f"max_tree_depth={self.numpyro_max_tree_depth}, "
            f"unit_space={self.numpyro_sample_unit_space}, "
            f"chain_method={self.numpyro_chain_method}"
        )

    def _configure_advanced_options(self) -> None:
        """Configure advanced and experimental options."""
        # Pipeline emulation settings
        self.last_emulated_module = self.read_ini("last_emulated_module", str, "")
        
        # Prior module settings
        self.prior_module = self.read_ini("prior_module", str, "")

        # Tempering settings
        tempering = self.read_ini("tempering", str, "")
        if tempering == "":
            self.tempering = np.linspace(0.05, 1.0, self.max_iterations)
        else:
            self.tempering = _parse_number_list(tempering, float)
        if len(self.tempering) != self.max_iterations:
            raise ValueError(f"Tempering param must have same number of values as max_iterations ({self.max_iterations})")
        if self.tempering[-1] != 1.0:
            raise ValueError("Final tempering value must be 1. You gave: " + tempering)
   
        logger.info(f"Tempering schedule: {self.tempering}")
        
        # Random seed
        self.seed = self.read_ini("seed", int, 0)
        if self.seed == 0:
            self.seed = None  # Use random seed
        
        # Neural network options. Each of these accepts either a single
        # value applied to all likelihoods, or per-likelihood overrides using
        # 'name:value' tokens, e.g.
        #   data_transformation = lsst:signed_log_norm desi_bao:log_norm
        #   n_pca      = lsst:64 default:32
        #   loss_function = lsst:weighted_w_cov desi_bao:standard
        # weighted_w_cov: LINNA Auxilleryfunc (feature-space χ²(M,NN)/χ²(M,d);
        #   needs data_transformation in {weighted_norm, weighted_median_norm,
        #   norm}. Incompatible with amplitude_prefactor=T and with PCA /
        #   PCA_per_bin (θ-dependent amp divide / reduced feature space).
        # Supported transforms: log_norm, signed_log_norm, norm,
        # weighted_norm, weighted_median_norm (both need inv_cov / full
        # data vector; rescale by sqrt(diag(C)) then mean/std or median/MAD),
        # PCA, PCA_per_bin (PCA with n_pca components per redshift-bin pair,
        # concatenated into one CosmoPower-like target; use with
        # amplitude_prefactor=T/F and amplitude_prefactor_spectra for 3x2pt).
        self.data_transformation = _parse_per_likelihood(
            self.read_ini("data_transformation", str, "log_norm"), str
        )
        # Optional block-wise 3x2pt amplitude prefactor applied *before*
        # data_transformation (and undone after backtransform). Values:
        #   F / none  → off (default)
        #   T / 3x2pt → on (WL/~S8^2, XC/~b_i S8^2, GC/~b_i b_j sigma8^2;
        #                   As_1e9*Omega_m/0.3 proxy when S8/sigma8 are not varied;
        #                   As_1e9 = As/1e-9)
        # Per-likelihood: ``amplitude_prefactor = lsst:T desi_bao:F``
        self.amplitude_prefactor = _parse_per_likelihood(
            self.read_ini("amplitude_prefactor", str, "F"), str
        )
        # Split a 3x2pt likelihood into per-spectrum emulators (default F =
        # one combined emulator). Spectra order comes from
        # amplitude_prefactor_spectra / data_sets. Hard-coded param exclusions:
        #   shear_cl: drop bin_bias, photoz_lens_errors
        #   galaxy_cl: drop shear_calibration_parameters, photoz_source_errors,
        #              intrinsic_alignment_parameters
        #   galaxy_shear_cl: all varied params
        # Per-likelihood: ``spectrum_emulators = lsst:T desi_bao:F``
        self.spectrum_emulators = _parse_per_likelihood(
            self.read_ini("spectrum_emulators", str, "F"), str
        )
        # Probe names in data_sets order, used only when CosmoSIS does not
        # store data_vector/<like>_name (≤3.19). Default matches LSST 3x2pt.
        # Space-separated global list, or per-likelihood with commas inside:
        #   amplitude_prefactor_spectra = shear_cl galaxy_shear_cl galaxy_cl
        #   amplitude_prefactor_spectra = lsst:shear_cl,galaxy_shear_cl,galaxy_cl
        _amp_spec_raw = self.read_ini(
            "amplitude_prefactor_spectra",
            str,
            "shear_cl galaxy_shear_cl galaxy_cl",
        )
        if _amp_spec_raw and any(":" in tok for tok in str(_amp_spec_raw).split()):
            self.amplitude_prefactor_spectra = _parse_per_likelihood(
                _amp_spec_raw, str
            )
        else:
            self.amplitude_prefactor_spectra = str(_amp_spec_raw).strip()
        # For data_transformation=PCA: total PCA components.
        # For PCA_per_bin: components per redshift-bin pair (clamped to n_modes).
        self.n_pca = _parse_per_likelihood(
            self.read_ini("n_pca", str, "32"), int
        )
        self.loss_function = _parse_per_likelihood(
            self.read_ini("loss_function", str, "standard"), str
        )
        self._validate_loss_transform_compatibility()

        if any(
            isinstance(v, str) and v.startswith("weighted")
            for v in _setting_values(self.loss_function)
        ) and self.keys:
            raise ValueError("Weighted loss function can only be used with full data vector "
                           "(empty keys parameter)")
        _weighted_transforms = ("weighted_norm", "weighted_median_norm")
        if any(
            isinstance(v, str) and v in _weighted_transforms
            for v in _setting_values(self.data_transformation)
        ) and self.keys:
            raise ValueError(
                "weighted_norm / weighted_median_norm can only be used with the "
                "full data vector (empty keys parameter); they need the "
                "likelihood inverse covariance"
            )
        available_final_samplers = ["emcee", "nautilus", "nuts", "numpyro"]
        self.final_sampler = self.read_ini_choices("final_sampler", str, available_final_samplers, "emcee")

        # Nautilus configuration for final iteration
        if self.final_sampler == "nautilus":
            self._configure_nautilus_parameters()
            # Add log_weight column to output when using nautilus (for all iterations)
            if self.output is not None:
                self.output.add_column("log_weight", float)
        
        # NUTS (TensorFlow Probability) configuration for final iteration
        elif self.final_sampler == "nuts":
            self._configure_nuts_parameters()
        # NumPyro NUTS: TF emulator log-prob via jax2tf.call_tf
        elif self.final_sampler == "numpyro":
            self._configure_numpyro_parameters()
        else:
        # If the final sampler is emcee then it just inherits
        # the sampler settings used for all the intermediate
        # sampling cycles
            logger.info(f"Final sampler is emcee, so using the same settings as the intermediate sampling cycles")

        
        # Neural network architecture (scalar or per-likelihood dict)
        self.nn_model = _parse_per_likelihood(
            self.read_ini("nn_model", str, "MLP"), str
        )
        # Hidden-layer widths for the MLP. Global: ``n_hidden = 512 512 512``.
        # Per-likelihood: ``n_hidden = lsst:512,512,512 desi_bao:256,256,256,256``.
        # Defaults to four layers of 512 (the previous hard-coded ROSE layout).
        self.n_hidden = _parse_number_list_setting(
            self.read_ini("n_hidden", str, "512 512 512 512"), int
        )
        for widths in _setting_values(self.n_hidden):
            if not widths or any(n < 1 for n in widths):
                raise ValueError(
                    "n_hidden must be a non-empty list of integers >= 1 "
                    "(for every likelihood)"
                )

    def _peek_training_setting(self, setting: Any, likelihood_name: str) -> Any:
        """Resolve a setting value for ``likelihood_name`` without getattr."""
        if isinstance(setting, dict):
            if likelihood_name in setting:
                return setting[likelihood_name]
            if "__default__" in setting:
                return setting["__default__"]
            raise KeyError(
                f"No value for likelihood '{likelihood_name}' in setting {setting!r}"
            )
        return setting

    def _validate_loss_transform_compatibility(self) -> None:
        """Reject incompatible loss_function / data_transformation / amplitude_prefactor combos.

        ``weighted_w_cov`` assumes a single global map from physical data + C
        into NN feature space. That breaks for:

        * ``amplitude_prefactor=T`` (θ-dependent amplitude divide before
          transform; physical ``d`` / ``C^{-1}`` are not in feature space)
        * ``PCA`` / ``PCA_per_bin`` (reduced feature space; LINNA cov transform
          is not implemented for PCA coefficients)
        """
        from .amplitude_prefactor import parse_amplitude_prefactor

        ok_transforms = ("weighted_norm", "weighted_median_norm", "norm")
        pca_transforms = ("PCA", "PCA_per_bin")

        like_names: List[str] = []
        pipe_names = getattr(getattr(self, "pipeline", None), "likelihood_names", None)
        if pipe_names and pipe_names != "no_likelihood_names_sentinel":
            like_names = [str(n) for n in pipe_names]
        for attr in (
            "loss_function", "data_transformation", "amplitude_prefactor"
        ):
            setting = getattr(self, attr)
            if isinstance(setting, dict):
                like_names.extend(
                    k for k in setting.keys() if k != "__default__"
                )
        like_names = sorted(set(like_names))
        if not like_names:
            like_names = ["*"]

        for name in like_names:
            try:
                loss = str(
                    self._peek_training_setting(self.loss_function, name)
                ).strip()
                transform = str(
                    self._peek_training_setting(self.data_transformation, name)
                ).strip()
                amp_raw = self._peek_training_setting(self.amplitude_prefactor, name)
            except KeyError:
                # Per-likelihood dict without this name and without __default__:
                # leave for train-time resolution.
                continue

            label = "all likelihoods" if name == "*" else f"likelihood '{name}'"
            if loss != "weighted_w_cov":
                continue

            if parse_amplitude_prefactor(amp_raw):
                raise ValueError(
                    f"Incompatible ROSE settings for {label}: "
                    "loss_function=weighted_w_cov cannot be combined with "
                    "amplitude_prefactor=T. The LINNA cov-weighted loss maps the "
                    "physical data vector / C^{-1} into feature space, but "
                    "amplitude_prefactor applies a θ-dependent divide before "
                    "data_transformation. Use loss_function=standard with "
                    "amplitude_prefactor, or turn amplitude_prefactor off."
                )
            if transform in pca_transforms:
                raise ValueError(
                    f"Incompatible ROSE settings for {label}: "
                    f"loss_function=weighted_w_cov cannot be combined with "
                    f"data_transformation={transform}. weighted_w_cov currently "
                    f"supports only {list(ok_transforms)}. Use "
                    "loss_function=standard with PCA / PCA_per_bin."
                )
            if transform not in ok_transforms:
                raise ValueError(
                    f"Incompatible ROSE settings for {label}: "
                    f"loss_function=weighted_w_cov requires data_transformation "
                    f"in {list(ok_transforms)}; got {transform!r}."
                )

    def _validate_training_schedule_lists(self) -> None:
        """Ensure optional schedule lists agree with n_cycles_per_training.

        Supports global and per-likelihood forms. Remaining per-likelihood
        length checks that cannot be resolved at config time (e.g. a schedule
        entry whose likelihood is not yet listed in ``n_cycles_per_training``)
        are re-checked when training each emulator.
        """
        schedule_attrs = (
            ("learning_rates", self.learning_rates, lambda v: v > 0, False),
            ("batch_sizes", self.batch_sizes, lambda v: v >= 1, True),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps,
             lambda v: v >= 1, False),
            ("patience_values", self.patience_values, lambda v: v >= 1, False),
            ("max_epochs", self.max_epochs, lambda v: v >= 1, False),
        )
        for name, setting, ok, allow_broadcast in schedule_attrs:
            if setting is None:
                continue
            for values in _setting_values(setting):
                if not values:
                    raise ValueError(f"{name} must be a non-empty list when set")
                if any(not ok(v) for v in values):
                    raise ValueError(f"{name} contains an invalid value: {values}")

            items = (
                list(setting.items()) if isinstance(setting, dict)
                else [("*", setting)]
            )
            for label, values in items:
                expected_lengths: List[int] = []
                if label == "*":
                    expected_lengths = list(
                        _setting_values(self.n_cycles_per_training)
                    )
                elif label == "__default__":
                    if isinstance(self.n_cycles_per_training, dict):
                        if "__default__" in self.n_cycles_per_training:
                            expected_lengths = [
                                self.n_cycles_per_training["__default__"]
                            ]
                    else:
                        expected_lengths = [self.n_cycles_per_training]
                else:
                    try:
                        expected_lengths = [
                            self._peek_training_setting(
                                self.n_cycles_per_training, label
                            )
                        ]
                    except KeyError:
                        expected_lengths = []
                for n in expected_lengths:
                    if allow_broadcast and len(values) == 1:
                        continue
                    if len(values) != n:
                        where = (
                            "globally" if label == "*"
                            else f"for '{label}'"
                        )
                        raise ValueError(
                            f"{name} {where} has length {len(values)} but "
                            f"n_cycles_per_training is {n}; schedule "
                            "lists must match n_cycles_per_training"
                            + (" (or be a single value to broadcast)"
                               if allow_broadcast else "")
                        )

