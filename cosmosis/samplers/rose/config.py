"""
Configuration methods for ROSE sampler.

This module contains methods for reading and validating configuration
parameters from ini files.
"""

import logging
from typing import Any, Callable, Dict, Union

import numpy as np

from .utils import SAVE_NONE, SAVE_MODEL, SAVE_ALL, mkdir

logger = logging.getLogger(__name__)


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
    settings like ``n_pca`` or ``batch_size`` still type-check.
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
        save_outputs = self.read_ini("save_outputs", str, "")
        
        if save_outputs:
            self.save_outputs_dir = self.read_ini("save_outputs_dir", str, "")
            if not self.save_outputs_dir:
                raise ValueError("save_outputs_dir must be specified when save_outputs is set")
                
            mkdir(self.save_outputs_dir)
            
            if save_outputs == "model":
                self.save_outputs = SAVE_MODEL
            elif save_outputs == "all":
                self.save_outputs = SAVE_ALL
            else:
                raise ValueError(f"Unknown save_outputs option '{save_outputs}' - "
                               "should be 'model', 'all', or empty")
        else:
            self.save_outputs = SAVE_NONE
            # Set a default directory for model saving (even when not saving outputs)
            # This is needed because train_emulator always needs a model_filename
            import tempfile
            import os
            self.save_outputs_dir = os.path.join(tempfile.gettempdir(), "rose_emulator")
            mkdir(self.save_outputs_dir)
            logger.warning("No outputs will be saved (save_outputs not specified), "
                         f"but models will be saved to temporary directory: {self.save_outputs_dir}")
    
    def _configure_training_parameters(self) -> None:
        """Configure neural network training parameters.

        ``batch_size`` and ``training_iterations`` may be specified either as a
        single integer (applied to all likelihoods) or with per-likelihood
        overrides, e.g. ``batch_size = lsst:64 desi_bao:32`` or
        ``training_iterations = lsst:8 default:5``.
        """
        self.max_iterations = self.read_ini("iterations", int, 4)
        self.initial_size = self.read_ini("initial_size", int, 9600)
        self.resample_size = self.read_ini("resample_size", int, 4800)
        self.chi2_cut_off = self.read_ini("chi2_cut_off", float)
        # Number of test points collected once, in the last training iteration,
        # from the 1-sigma or 2-sigma (credible) region of the one-before-last MCMC chain.
        self.final_test_size = self.read_ini(
            "final_test_size", int, int(0.5 * self.resample_size)
        )
        # Fraction of the chain (ranked by posterior) that defines the
        # "1-sigma"/68% credible region from which test points are drawn;
        # should be 0.95 for 2-sigma/95% credible region.
        self.test_credible_fraction = self.read_ini(
            "test_credible_fraction", float, 0.95 
        )
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
        self.kl_n_samples = self.read_ini("kl_n_samples", int, 2000)
        self.batch_size = _parse_per_likelihood(
            self.read_ini("batch_size", str, "32"), int
        )
        self.training_iterations = _parse_per_likelihood(
            self.read_ini("training_iterations", str, "5"), int
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
        if self.kl_threshold <= 0:
            raise ValueError("kl_threshold must be > 0")
        if self.kl_max_retrain < 0:
            raise ValueError("kl_max_retrain must be >= 0")
        if self.kl_extra_size < 1:
            raise ValueError("kl_extra_size must be >= 1")
        if self.kl_n_samples < 10:
            raise ValueError("kl_n_samples must be >= 10")
        if any(v < 1 for v in _setting_values(self.batch_size)):
            raise ValueError("batch_size must be >= 1 (for every likelihood)")
        if any(v < 1 for v in _setting_values(self.training_iterations)):
            raise ValueError("training_iterations must be >= 1 (for every likelihood)")
    
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
        tempering = self.read_ini("tempering", float, 0.05)
        self.tempering = np.full(self.max_iterations, tempering)
        
        tempering_file = self.read_ini("tempering_file", str, "")
        if tempering_file:
            try:
                custom_tempering = np.genfromtxt(tempering_file)
                if len(custom_tempering) < self.max_iterations:
                    logger.warning(f"Tempering file has {len(custom_tempering)} values but "
                                 f"{self.max_iterations} iterations requested")
                self.tempering = custom_tempering[:self.max_iterations]
            except Exception as e:
                raise ValueError(f"Failed to read tempering file {tempering_file}: {e}")
        
        logger.info(f"Tempering schedule: {self.tempering}")
        
        # Random seed
        self.seed = self.read_ini("seed", int, 0)
        if self.seed == 0:
            self.seed = None  # Use random seed
        
        # Neural network options. Each of these accepts either a single
        # value applied to all likelihoods, or per-likelihood overrides using
        # 'name:value' tokens, e.g.
        #   data_trafo = lsst:signed_log_norm desi_bao:log_norm
        #   n_pca      = lsst:64 default:32
        #   loss_function = lsst:weighted_mse desi_bao:standard
        self.data_trafo = _parse_per_likelihood(
            self.read_ini("data_trafo", str, "log_norm"), str
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
        
        # Nautilus configuration for final iteration
        self.use_nautilus_final = self.read_ini("use_nautilus_final", bool, False)
        if self.use_nautilus_final:
            self._configure_nautilus_parameters()
            # Add log_weight column to output when using nautilus (for all iterations)
            if self.output is not None:
                self.output.add_column("log_weight", float)
        
        # NUTS configuration for final iteration
        self.use_nuts_final = self.read_ini("use_nuts_final", bool, False)
        if self.use_nuts_final:
            self._configure_nuts_parameters()
        
        # Neural network architecture (scalar or per-likelihood dict)
        self.nn_model = _parse_per_likelihood(
            self.read_ini("nn_model", str, "MLP"), str
        )

