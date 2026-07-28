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
        # tempered HPD credible region used for both training resampling
        # (homogeneous / farthest-point in unit space) and final test points
        # (uniform draw). Use 0.68 for 1-sigma or 0.95 for 2-sigma.
        self.test_credible_fraction = self.read_ini(
            "test_credible_fraction", float, 0.95 
        )
        # Optional: before training the final emulator, drop early training
        # points that lie outside the n-sigma HPD region of the last tempered
        # MCMC chain (same 68/95/99.7 convention as elsewhere in ROSE; the cut
        # is the unit-prior bounding box of that HPD cloud). Can improve local
        # accuracy near the posterior, but leave False when the final sampler
        # (e.g. Nautilus) starts from the prior and benefits from retaining
        # broad prior-volume coverage in the training set.
        self.remove_outliers = self.read_ini("remove_outliers", bool, False)
        self.outlier_nsigma = self.read_ini("outlier_nsigma", float, 3.0)
        # Convergence (KL divergence) settings. This test is opt-in: set
        # kl_convergence = T in the ini file to enable it. When enabled, before
        # the final sampling stage the test points are folded into the training
        # set, the emulator is retrained, and the KL divergence between the
        # emulated posteriors of two consecutive iterations is computed. If it
        # exceeds kl_threshold, extra training points are added and the emulator
        # is retrained until the criterion is met or kl_max_retrain attempts are
        # exhausted.
        self.kl_convergence = self.read_ini("kl_convergence", bool, False)
        self.kl_threshold = self.read_ini("kl_threshold", float, 0.1)
        self.kl_max_retrain = self.read_ini("kl_max_retrain", int, 5)
        self.kl_extra_size = self.read_ini("kl_extra_size", int, self.resample_size)
        self.kl_n_samples = self.read_ini("kl_n_samples", int, 7000)
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

        self.n_cycles_per_training = _parse_per_likelihood(
            self.read_ini("n_cycles_per_training", str, "5"), int
        )
        # Per-stage batch sizes for the *first* training iteration. A single
        # value (default 32) is broadcast to every training cycle; otherwise
        # the list length must match ``n_cycles_per_training``. In later
        # iterations each entry is multiplied by n_train / n_train_first.
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
        if self.outlier_nsigma <= 0:
            raise ValueError("outlier_nsigma must be > 0")
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
        """Configure NUTS sampling parameters for final iteration."""
        self.nuts_step_size = self.read_ini("nuts_step_size", float, 0.05)
        self.nuts_use_fixed_step_size = self.read_ini("nuts_use_fixed_step_size", bool, False)
        self.nuts_fixed_step_size = self.read_ini("nuts_fixed_step_size", float, 0.3)
        self.nuts_max_tree_depth = self.read_ini("nuts_max_tree_depth", int, 12)
        self.nuts_max_energy_diff = self.read_ini("nuts_max_energy_diff", float, 1000.0)
        self.nuts_unrolled_leapfrog_steps = self.read_ini("nuts_unrolled_leapfrog_steps", int, 1) #do not increase it unless profiling shows benefit
        self.nuts_parallel_iterations = self.read_ini("nuts_parallel_iterations", int, 10)
        self.nuts_num_adaptation_steps = self.read_ini("nuts_num_adaptation_steps", int, 1000)
        self.nuts_num_burnin_steps = self.read_ini("nuts_num_burnin_steps", int, 1000)
        self.nuts_num_results = self.read_ini("nuts_num_results", int, 2000)
        self.nuts_num_chains = self.read_ini("nuts_num_chains", int, 1)
        self.nuts_target_accept_prob = self.read_ini("nuts_target_accept_prob", float, 0.75)
        self.nuts_sample_unit_space = self.read_ini("nuts_sample_unit_space", bool, True)
        self.nuts_progress_interval = self.read_ini("nuts_progress_interval", int, 500)
        
        # Validate NUTS parameters
        if self.nuts_step_size <= 0:
            raise ValueError("nuts_step_size must be > 0")
        if self.nuts_max_tree_depth < 1:
            raise ValueError("nuts_max_tree_depth must be >= 1")
        if self.nuts_num_results < 100:
            logger.warning(f"nuts_num_results ({self.nuts_num_results}) is very small")
        if self.nuts_num_chains < 1:
            raise ValueError("nuts_num_chains must be >= 1")
        if not (0.5 <= self.nuts_target_accept_prob <= 0.99):
            raise ValueError("nuts_target_accept_prob should be in (0.5, 0.99), e.g. 0.65-0.8 for better exploration")
        if self.nuts_progress_interval < 0:
            raise ValueError("nuts_progress_interval must be >= 0 (use 0 to disable progress output)")
        
        logger.info(f"NUTS configured for final iteration: step_size={self.nuts_step_size}, "
                   f"max_tree_depth={self.nuts_max_tree_depth}, num_results={self.nuts_num_results}, "
                   f"target_accept_prob={self.nuts_target_accept_prob}, sample_unit_space={self.nuts_sample_unit_space}")
    
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
        #   loss_function = lsst:weighted_mse desi_bao:standard
        self.data_transformation = _parse_per_likelihood(
            self.read_ini("data_transformation", str, "log_norm"), str
        )
        self.n_pca = _parse_per_likelihood(
            self.read_ini("n_pca", str, "32"), int
        )
        self.loss_function = _parse_per_likelihood(
            self.read_ini("loss_function", str, "standard"), str
        )

        if any(
            isinstance(v, str) and v.startswith("weighted")
            for v in _setting_values(self.loss_function)
        ) and self.keys:
            raise ValueError("Weighted loss function can only be used with full data vector "
                           "(empty keys parameter)")
        available_final_samplers = ["emcee", "nautilus", "nuts"]
        self.final_sampler = self.read_ini_choices("final_sampler", str, available_final_samplers, "emcee")

        # Nautilus configuration for final iteration
        if self.final_sampler == "nautilus":
            self._configure_nautilus_parameters()
            # Add log_weight column to output when using nautilus (for all iterations)
            if self.output is not None:
                self.output.add_column("log_weight", float)
        
        # NUTS configuration for final iteration
        elif self.final_sampler == "nuts":
            self._configure_nuts_parameters()
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

