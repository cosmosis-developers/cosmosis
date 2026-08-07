"""
MCMC sampling methods for ROSE sampler.

This module contains methods for running MCMC sampling using emcee, nautilus,
TFP NUTS, and NumPyro NUTS (via jax2tf.call_tf), and processing the resulting
chains.
"""

import logging
import os
from functools import partial
from timeit import default_timer
from typing import Any, Optional, Tuple

import numpy as np

from .utils import log_probability_function, SAVE_ALL, log_probability_function_nautilus, prior_transform
import cosmosis.samplers.rose.utils as utils_module
from ...runtime.prior import GaussianPrior, UniformPrior

logger = logging.getLogger(__name__)

s = 1e-7
class RoseSamplingMixin:
    """Mixin class providing MCMC sampling methods for RoseSampler."""
    
    def _get_current_tempering(self) -> float:
        """Get tempering factor for current iteration."""
        if self.iterations < self.max_iterations - 1:
            tempering = self.tempering[self.iterations]
            logger.info(f"Running MCMC with tempering {tempering} (iteration {self.iterations + 1})")
        else:
            tempering = 1.0
            logger.info(f"Running final MCMC without tempering (iteration {self.iterations + 1})")
        
        return tempering

    def _run_emcee_sampling(self, tempering: float) -> float:
        """Run emcee MCMC sampling.

        Returns:
            Wall-clock seconds spent in the MCMC loop only (excludes chain I/O).
        """
        import emcee
        # Ensure emu_pipeline is set up on all processes before MCMC starts
        # This is critical for MPI parallelization where worker processes need
        # access to emu_pipeline for log probability evaluation
        if self.emu_pipeline is None:
            logger.warning("emu_pipeline is None, setting it up now (this should happen in execute())")
            self.compute_fiducial_setup_emu_pipeline()
            # Update global sampler reference after setup
            utils_module._sampler = self
        
        # Use module-level function (can be pickled for MPI) and pass tempering
        # plus the current emulator model path via args. emcee will call:
        # log_probability_function(u, tempering, model_path). The model path lets
        # worker processes (which never run execute()) load/reload the SAME
        # emulator the master just trained, instead of a stale cached one.
        model_path = self._worker_emu_model_path()
        emcee_sampler = emcee.EnsembleSampler(
            self.emcee_walkers,
            self.ndim,
            log_probability_function,
            args=[tempering, model_path],
            pool=self.pool,
        )
        
        # Get starting positions (emcee samples in unit cube [0,1]^ndim)
        if self.trained_before:
            start_pos = [
                self.pipeline.normalize_vector_to_prior(self.pipeline.randomized_start())
                for _ in range(self.emcee_walkers)
            ]
        else:
            start_pos = self.get_emcee_start()
        
        logger.info(f"Starting MCMC with {len(start_pos)} walkers")
        # Run MCMC
        start_time = default_timer()
        #emcee_sampler.run_mcmc(start_pos, self.emcee_samples, progress=True)
        #end_time = default_timer()


        # Track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(self.emcee_samples)
        tau_all_params = np.empty([int(self.emcee_samples/100),self.ndim])
        old_tau = np.inf

        print("Running production...")
        # Sample up to niter steps
        for sample in emcee_sampler.sample(start_pos, iterations=self.emcee_samples, progress=True, store=True):
            # Check convergence (compute tau) every 100 steps
            if emcee_sampler.iteration % 100: continue

            # Compute the autocorrelation time tau
            tau = emcee_sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            tau_all_params[index] = tau
            index += 1

            # Check convergence, code stops if converged
            converged = np.all(tau * 50 < emcee_sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

        end_time = default_timer()
        time_sampling_s = end_time - start_time
        logger.info('tau: %s', tau_all_params)
        logger.info(f"MCMC sampling took {time_sampling_s:.1f} seconds")
        
        # Process MCMC results (chain file I/O — not counted in sampling time)
        self._process_mcmc_results(emcee_sampler, tempering)
        return time_sampling_s

    def _run_nautilus_sampling(self, tempering: float) -> float:
        """Run nautilus sampling for final iteration.

        Returns:
            Wall-clock seconds spent in nautilus.run only (excludes chain I/O).
        """
        from nautilus import Sampler

        if self.emu_pipeline is None:
            logger.warning("emu_pipeline is None, setting it up now (this should happen in execute())")
            self.compute_fiducial_setup_emu_pipeline()
            # Update global sampler reference after setup
            utils_module._sampler = self
        
        # # Set up resume filepath if available
        try:
            resume_filepath = self.output.name_for_sampler_resume_info()
            if resume_filepath is not None:
                resume_filepath = resume_filepath + "_nautilus.hdf5"
            else:
                resume_filepath = None
        except NotImplementedError:
            resume_filepath = None
        
        logger.info(f"Starting Nautilus sampling with n_live={self.nautilus_n_live}")
        
        # Capture current model directory so workers (which never run execute()) load the same
        # per-likelihood emulators when the pool pickles these callables. Using the
        # shared helper keeps this in sync with the emcee path and with
        # utils._ensure_emulator, so workers reload after e.g. a final-step KL
        # retrain instead of serving a stale cached emulator.
        current_model_path = self._worker_emu_model_path()
        # When using a pool, workers load from disk; ensure every per-likelihood
        # emulator is on disk even if save_output is not "all". (Only relevant
        # for from-scratch runs; a pre-trained run has current_model_path=None
        # and workers load from load_emu_filename.)
        if current_model_path is not None and self.pool is not None and self.emulator:
            emulators_dict = (
                self.emulator if isinstance(self.emulator, dict)
                else {self.likelihood_names[0]: self.emulator}
            )
            os.makedirs(current_model_path, exist_ok=True)
            for name, emu in emulators_dict.items():
                info_file = os.path.join(current_model_path, name) + ".npz"
                if not os.path.exists(info_file):
                    logger.info(
                        f"Saving emulator for '{name}' to disk so workers can load it"
                    )
                    emu.save_to(os.path.join(current_model_path, name))
        # Module-level callables + partial(model_path=...) pickle for MPI; nested defs do not.
        prior_transform_with_path = partial(
            prior_transform, model_path=current_model_path
        )
        log_prob_nautilus_with_path = partial(
            log_probability_function_nautilus, model_path=current_model_path
        )

        # Create nautilus sampler
        nautilus_sampler = Sampler(
            prior_transform_with_path,
            log_prob_nautilus_with_path,
            self.ndim,
            n_live=self.nautilus_n_live,
            n_update=self.nautilus_n_update,
            enlarge_per_dim=self.nautilus_enlarge_per_dim,
            n_points_min=self.nautilus_n_points_min,
            split_threshold=self.nautilus_split_threshold,
            n_networks=self.nautilus_n_networks,
            n_batch=self.nautilus_n_batch,
            seed=self.seed,
            filepath=resume_filepath,
            resume=True,
            pool=self.pool,
            blobs_dtype=float
        )
        
        # Run nautilus
        start_time = default_timer()
        nautilus_sampler.run(
            f_live=self.nautilus_f_live,
            n_shell=self.nautilus_n_shell,
            n_eff=self.nautilus_n_eff,
            n_like_max=self.nautilus_n_like_max,
            discard_exploration=self.nautilus_discard_exploration,
            verbose=True
        )
        end_time = default_timer()
        time_sampling_s = end_time - start_time
        
        logger.info(f"Nautilus sampling took {time_sampling_s:.1f} seconds")
        
        # Process nautilus results (chain file I/O — not counted in sampling time)
        self._process_nautilus_results(nautilus_sampler, tempering)
        return time_sampling_s

    def _nn_emulator_tf_bundle(self, emulator, varied_names, DTYPE):
        """Pack NNEmulator TF constants and param indices into the full varied vector."""
        import tensorflow as tf

        missing = [p for p in emulator.model_parameters if p not in varied_names]
        if missing:
            raise ValueError(
                f"Emulator model_parameters not in varied params: {missing}"
            )
        param_indices = [varied_names.index(p) for p in emulator.model_parameters]
        return {
            "emulator": emulator,
            "param_indices": tf.constant(param_indices, dtype=tf.int32),
            "X_mean": tf.constant(
                [emulator.X_mean[key] for key in emulator.model_parameters], dtype=DTYPE
            ),
            "X_std": tf.constant(
                [emulator.X_std[key] for key in emulator.model_parameters], dtype=DTYPE
            ),
            "y_mean": tf.constant(emulator.y_mean, dtype=DTYPE),
            "y_std": tf.constant(emulator.y_std, dtype=DTYPE),
        }

    def _build_nuts_likelihood_contexts(self, data_vectors, data_inv_covariance, DTYPE):
        """Build differentiable likelihood contexts for all trained emulators.

        Supports:
        - one or more CosmoSIS likelihood entries in ``self.emulator``
        - ``NNEmulator`` (single network per likelihood)
        - ``CompositeSpectrumEmulator`` (``spectrum_emulators=T``): the three
          probe nets (WL/XC/GC) belong to *one* likelihood. Predictions are
          scattered into the full theory vector and chi2 uses the *full*
          inverse covariance (including cross-probe blocks). Per-spectrum
          likelihoods are never formed or summed.
        """
        import tensorflow as tf
        from .nn_emulator import NNEmulator
        from .spectrum_emulator import CompositeSpectrumEmulator

        if isinstance(self.emulator, dict):
            emulators = self.emulator
        else:
            name = (
                self.likelihood_names[0]
                if getattr(self, "likelihood_names", None)
                else "likelihood"
            )
            emulators = {name: self.emulator}

        varied_names = [str(p) for p in self.pipeline.varied_params]
        contexts = []
        for name, emu in emulators.items():
            if name not in data_vectors or name not in data_inv_covariance:
                logger.warning(
                    f"NUTS: skipping likelihood '{name}' — missing data vector "
                    "or inverse covariance for autodiff chi2"
                )
                continue
            dv = tf.constant(np.atleast_1d(data_vectors[name]), dtype=DTYPE)
            icov = tf.constant(np.atleast_2d(data_inv_covariance[name]), dtype=DTYPE)

            if isinstance(emu, CompositeSpectrumEmulator):
                subs = []
                for spectrum in emu.spectrum_order:
                    sub = emu.emulators[spectrum]
                    if not isinstance(sub, NNEmulator):
                        raise TypeError(
                            f"Composite sub-emulator '{name}/{spectrum}' must be NNEmulator"
                        )
                    bundle = self._nn_emulator_tf_bundle(sub, varied_names, DTYPE)
                    idx = np.asarray(emu.mode_indices[spectrum], dtype=np.int32).ravel()
                    bundle["mode_indices"] = tf.constant(idx.reshape(-1, 1), dtype=tf.int32)
                    bundle["spectrum"] = spectrum
                    subs.append(bundle)
                contexts.append(
                    {
                        "type": "composite",
                        "name": name,
                        "subs": subs,
                        "n_modes": int(emu.n_modes),
                        "data_vector": dv,
                        "inv_covariance": icov,
                    }
                )
                logger.info(
                    f"NUTS autodiff: composite '{name}' spectra "
                    f"{[s['spectrum'] for s in subs]} assembled into one "
                    f"{emu.n_modes}-mode theory vector; chi2 uses full inv-cov "
                    f"(cross-probe blocks included, not summed per spectrum)"
                )
            elif isinstance(emu, NNEmulator) or hasattr(emu, "cp_nn"):
                bundle = self._nn_emulator_tf_bundle(emu, varied_names, DTYPE)
                contexts.append(
                    {
                        "type": "nn",
                        "name": name,
                        "bundle": bundle,
                        "data_vector": dv,
                        "inv_covariance": icov,
                    }
                )
                logger.info(
                    f"NUTS autodiff: single NN emulator '{name}' "
                    f"({len(emu.model_parameters)} params, "
                    f"{int(np.asarray(emu.y_mean).size)} modes)"
                )
            else:
                raise TypeError(
                    f"Unsupported emulator type for NUTS autodiff: "
                    f"{type(emu).__name__} ('{name}')"
                )

        if not contexts:
            logger.warning(
                "NUTS: no differentiable likelihood contexts built; "
                "will fall back to non-TF pipeline likelihood"
            )
        elif len(contexts) > 1:
            logger.info(
                f"NUTS autodiff will sum chi2 over {len(contexts)} likelihoods: "
                f"{[c['name'] for c in contexts]}"
            )
        return contexts

    def _nn_emulator_predict_tf(self, physical_params_tf, bundle, DTYPE):
        """Differentiable NNEmulator theory prediction from the full param vector."""
        import tensorflow as tf

        emulator = bundle["emulator"]
        params_sub = tf.gather(physical_params_tf, bundle["param_indices"])
        params_norm = (params_sub - bundle["X_mean"]) / bundle["X_std"]
        params_nn_norm = (
            params_norm - emulator.cp_nn.parameters_mean
        ) / emulator.cp_nn.parameters_std

        if len(params_nn_norm.shape) == 1:
            params_nn_norm = tf.expand_dims(params_nn_norm, 0)

        layers = [params_nn_norm]
        if emulator.cp_nn.architecture_type == "MLP":
            for i in range(emulator.cp_nn.n_layers - 1):
                linear_out = (
                    tf.matmul(layers[-1], emulator.cp_nn.W[i]) + emulator.cp_nn.b[i]
                )
                activated = emulator.cp_nn.activation(
                    linear_out, emulator.cp_nn.alphas[i], emulator.cp_nn.betas[i]
                )
                layers.append(activated)
            output = tf.matmul(layers[-1], emulator.cp_nn.W[-1]) + emulator.cp_nn.b[-1]
        else:
            raise NotImplementedError(
                f"NUTS autodiff not implemented for architecture "
                f"{emulator.cp_nn.architecture_type}"
            )

        pred_norm = output * emulator.cp_nn.features_std + emulator.cp_nn.features_mean
        if len(pred_norm.shape) == 2:
            pred_norm = pred_norm[0]

        pred_intermediate = pred_norm * bundle["y_std"] + bundle["y_mean"]
        predictions = emulator.backtransform_tf(pred_intermediate, dtype=DTYPE)

        if getattr(emulator, "amplitude_prefactor", None) is not None:
            param_index = {p: i for i, p in enumerate(emulator.model_parameters)}
            amp = emulator.amplitude_prefactor.factors_tf(params_sub, param_index, DTYPE)
            predictions = predictions * amp
        return predictions

    def _nuts_theory_vector_tf(self, physical_params_tf, context, DTYPE):
        """Build the full theory vector for one CosmoSIS likelihood.

        For ``spectrum_emulators=T``, probe networks only supply disjoint
        slices of that vector; they do not define separate likelihoods.
        """
        import tensorflow as tf

        if context["type"] == "nn":
            return self._nn_emulator_predict_tf(
                physical_params_tf, context["bundle"], DTYPE
            )

        # Composite: predict each spectrum slice, scatter into one DV.
        predictions = tf.zeros([context["n_modes"]], dtype=DTYPE)
        for sub in context["subs"]:
            pred_s = self._nn_emulator_predict_tf(physical_params_tf, sub, DTYPE)
            predictions = tf.tensor_scatter_nd_update(
                predictions, sub["mode_indices"], pred_s
            )
        return predictions

    def _verify_nuts_gradients(self, test_params: np.ndarray, tempering: float,
                               contexts, DTYPE) -> None:
        """Verify gradient computation by comparing autodiff with finite differences."""
        import tensorflow as tf

        logger.info("Verifying NUTS gradient computation...")

        test_params_tf = tf.constant(test_params, dtype=DTYPE)

        with tf.GradientTape() as tape:
            tape.watch(test_params_tf)
            log_prob_tf = self._log_prob_nuts_impl(
                test_params_tf, contexts, tempering, DTYPE
            )

        grad_autodiff = tape.gradient(log_prob_tf, test_params_tf)

        if grad_autodiff is None:
            logger.error("Autodiff gradient is None! Gradient computation may be broken.")
            return

        grad_autodiff_np = grad_autodiff.numpy()

        eps = 1e-5
        grad_finite_diff = np.zeros_like(test_params)

        log_prob_base = self._log_prob_nuts_impl(
            test_params_tf, contexts, tempering, DTYPE, TF=True
        ).numpy()

        log_prob_base_no_tf = self._log_prob_nuts_impl(
            test_params_tf, contexts, tempering, DTYPE, TF=False
        ).numpy()

        for i in range(len(test_params)):
            params_forward = test_params.copy()
            params_forward[i] += eps
            params_forward_tf = tf.constant(params_forward, dtype=DTYPE)

            log_prob_forward = self._log_prob_nuts_impl(
                params_forward_tf, contexts, tempering, DTYPE, False
            ).numpy()

            params_backward = test_params.copy()
            params_backward[i] -= eps
            params_backward_tf = tf.constant(params_backward, dtype=DTYPE)

            log_prob_backward = self._log_prob_nuts_impl(
                params_backward_tf, contexts, tempering, DTYPE, False
            ).numpy()

            grad_finite_diff[i] = (log_prob_forward - log_prob_backward) / (2 * eps)

        logger.info("=" * 60)
        logger.info("Gradient Verification Results")
        logger.info("=" * 60)
        logger.info(f"Test point: {test_params}")
        logger.info(f"Log probability: {log_prob_base:.6f}")
        logger.info(f"Log probability no TF: {log_prob_base_no_tf:.6f}")
        logger.info("")
        logger.info("Parameter | Autodiff Gradient | Finite Diff Gradient | Relative Error")
        logger.info("-" * 70)

        max_rel_error = 0.0
        for i, param_name in enumerate(self.pipeline.varied_params):
            autodiff_val = grad_autodiff_np[i]
            finite_diff_val = grad_finite_diff[i]

            if abs(autodiff_val) > 1e-10 or abs(finite_diff_val) > 1e-10:
                rel_error = abs(autodiff_val - finite_diff_val) / max(
                    abs(autodiff_val), abs(finite_diff_val), 1e-10
                )
            else:
                rel_error = abs(autodiff_val - finite_diff_val)

            max_rel_error = max(max_rel_error, rel_error)

            logger.info(
                f"{param_name.name:20s} | {autodiff_val:15.6e} | "
                f"{finite_diff_val:15.6e} | {rel_error:10.2e}"
            )

        logger.info("-" * 70)
        logger.info(f"Maximum relative error: {max_rel_error:.2e}")

        if max_rel_error > 0.1:
            logger.warning(
                f"Large gradient discrepancy detected! Max relative error: {max_rel_error:.2e}"
            )
            logger.warning("This may cause poor NUTS exploration. Check gradient computation.")
        elif max_rel_error > 0.01:
            logger.warning(
                f"Moderate gradient discrepancy. Max relative error: {max_rel_error:.2e}"
            )
        else:
            logger.info("Gradients match well. Gradient computation appears correct.")

        logger.info("=" * 60)

    def _nuts_pilot_samples_for_mass(self) -> Optional[np.ndarray]:
        """Return previous ROSE samples in the NUTS coordinate space, if available."""
        self._maybe_load_nuts_pilot_chain()
        if self.nuts_sample_unit_space:
            samples = getattr(self, "unit_chain", None)
        else:
            samples = getattr(self, "chain", None)
        if samples is None:
            return None
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < max(20, arr.shape[1] + 5):
            return None
        return arr

    def _has_in_memory_tempered_chain(self, *, min_rows: Optional[int] = None) -> bool:
        """True if ``unit_chain`` + ``chain_logpost`` from a prior ROSE MCMC exist."""
        unit = getattr(self, "unit_chain", None)
        logpost = getattr(self, "chain_logpost", None)
        if unit is None or logpost is None:
            return False
        n = len(unit)
        if n == 0 or n != len(logpost):
            return False
        if min_rows is not None and n < int(min_rows):
            return False
        return True

    def _find_latest_tempered_chain_file(self) -> str:
        """Find the highest-iteration ``*_tempering_*_iteration_*.txt`` near the output.

        Used when ``trained_before=F`` so final NumPyro/NUTS can recover the last
        emcee tempered chain if it is not already in memory (or as a file fallback).
        """
        import glob
        import re

        candidates = []
        search_dirs = []
        out = getattr(self, "output", None)
        base = getattr(out, "filename_base", None) if out is not None else None
        if base:
            search_dirs.append(os.path.dirname(os.path.abspath(base)) or ".")
        save_dir = str(getattr(self, "save_dir", "") or "").strip()
        if save_dir:
            # Chains live next to rose_outputs/, not inside it.
            parent = os.path.dirname(os.path.abspath(save_dir))
            if parent:
                search_dirs.append(parent)
            search_dirs.append(os.path.abspath(save_dir))
        search_dirs.append(os.path.abspath("."))

        seen = set()
        pat = re.compile(
            r"tempering_([0-9eE.+-]+)_iteration_(\d+)", re.IGNORECASE
        )
        for d in search_dirs:
            if not d or d in seen or not os.path.isdir(d):
                continue
            seen.add(d)
            for path in glob.glob(os.path.join(d, "*tempering_*_iteration_*.txt")):
                m = pat.search(os.path.basename(path))
                if not m:
                    continue
                try:
                    temper = float(m.group(1))
                    iteration = int(m.group(2))
                except ValueError:
                    continue
                candidates.append((iteration, temper, path))

        if not candidates:
            return ""
        # Prefer highest ROSE iteration; tie-break on largest tempering.
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[-1][2]

    def _load_chain_file_for_nuts(self, path: str, *, source: str) -> bool:
        """Load a CosmoSIS text chain into ``chain`` / ``unit_chain`` / ``chain_logpost``."""
        try:
            with open(path) as f:
                header = None
                for line in f:
                    if line.startswith("#") and "--" in line and "prior" in line:
                        header = line[1:].strip().split()
                        break
            if not header:
                raise ValueError("no parameter header found")
            data = np.loadtxt(path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            cols = {name: i for i, name in enumerate(header)}
            varied = [str(p) for p in self.pipeline.varied_params]
            missing = [p for p in varied if p not in cols]
            if missing:
                raise ValueError(f"pilot chain missing columns: {missing[:5]}...")
            physical = np.column_stack([data[:, cols[p]] for p in varied])
            if "post" in cols:
                logpost = np.asarray(data[:, cols["post"]], dtype=float)
            elif "tempered_post" in cols:
                logpost = np.asarray(data[:, cols["tempered_post"]], dtype=float)
            else:
                raise ValueError("pilot chain needs a 'post' (or tempered_post) column")
            unit = np.array(
                [self.pipeline.normalize_vector_to_prior(p) for p in physical]
            )
            self.chain = physical
            self.unit_chain = unit
            self.chain_logpost = logpost
            # Unknown file tempering; keep any previous value or None.
            if not hasattr(self, "chain_tempering") or self.chain_tempering is None:
                self.chain_tempering = float("nan")
            logger.info(
                f"Loaded {source} ({len(physical)} samples) from {path} "
                f"for NUTS/NumPyro init / mass matrix"
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to load {source} '{path}': {exc}")
            return False

    def _maybe_load_nuts_pilot_chain(self) -> None:
        """Ensure a tempered pilot chain is available for NUTS/NumPyro init.

        Priority:
        1. In-memory ``unit_chain`` / ``chain_logpost`` from the previous ROSE
           emcee iteration (``trained_before=F`` path).
        2. Explicit ``numpyro_pilot_chain`` / ``nuts_pilot_chain`` file.
        3. If ``trained_before=F``, auto-discover the latest
           ``*_tempering_*_iteration_*.txt`` next to the CosmoSIS output.
        """
        if getattr(self, "_nuts_pilot_chain_loaded", False):
            return

        if self._has_in_memory_tempered_chain():
            t = getattr(self, "chain_tempering", None)
            t_str = f"{float(t):.4g}" if t is not None and np.isfinite(float(t)) else "?"
            logger.info(
                f"NUTS/NumPyro pilot: using in-memory ROSE chain "
                f"(n={len(self.unit_chain)}, tempering={t_str})"
            )
            self._nuts_pilot_chain_loaded = True
            return

        path = str(getattr(self, "nuts_pilot_chain", "") or "").strip()
        source = "nuts_pilot_chain / numpyro_pilot_chain"
        if path and not os.path.exists(path):
            logger.warning(
                f"{source} not found: {path} "
                "(check spelling; will try latest tempered chain if available)"
            )
            path = ""

        if not path and not getattr(self, "trained_before", False):
            path = self._find_latest_tempered_chain_file()
            source = "latest tempered ROSE chain"
            if path:
                logger.info(
                    f"trained_before=F: auto-selected pilot chain for NumPyro/NUTS: {path}"
                )

        if not path:
            self._nuts_pilot_chain_loaded = True
            return

        ok = self._load_chain_file_for_nuts(path, source=source)
        self._nuts_pilot_chain_loaded = True
        if not ok and not getattr(self, "trained_before", False):
            # Last resort: another tempered file if the explicit path failed.
            alt = self._find_latest_tempered_chain_file()
            if alt and os.path.abspath(alt) != os.path.abspath(path):
                self._load_chain_file_for_nuts(
                    alt, source="latest tempered ROSE chain"
                )

    def _nuts_prior_widths(self) -> np.ndarray:
        """Physical prior widths ``high - low`` for each varied parameter."""
        return np.asarray(
            [float(p.limits[1] - p.limits[0]) for p in self.pipeline.varied_params],
            dtype=np.float64,
        )

    @staticmethod
    def _clip_unit_cube(unit: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Clamp CosmoSIS unit-cube coordinates into the open interval ``(eps, 1-eps)``.

        NUTS / NumPyro treat the state as unconstrained reals, so leapfrog steps
        routinely produce values like ``-1e-7`` or ``1+1e-7``. CosmoSIS
        ``denormalize_from_prior`` requires ``[0, 1]`` strictly.
        """
        return np.clip(np.asarray(unit, dtype=np.float64), eps, 1.0 - eps)

    def _denormalize_unit_vector(self, unit: np.ndarray) -> np.ndarray:
        """``denormalize_vector_from_prior`` after clipping to the unit cube."""
        return self.pipeline.denormalize_vector_from_prior(self._clip_unit_cube(unit))

    @staticmethod
    def _logit_unit_cube(unit: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Map unit-cube coords to unconstrained reals (inverse sigmoid)."""
        u = np.clip(np.asarray(unit, dtype=np.float64), eps, 1.0 - eps)
        return np.log(u) - np.log1p(-u)

    def _unit_to_physical_tf(self, unit_tf, DTYPE):
        """Map CosmoSIS unit-cube coordinates → physical via prior CDF inverses.

        Matches ``pipeline.denormalize_vector_from_prior``. A linear
        ``low + u*(high-low)`` map is **wrong** for Gaussian / truncated-Gaussian
        nuisance parameters (photo-z, m-cal, …): unit ``u≈0.84`` is +1σ under
        the prior CDF, but the linear map sends it near the hard prior edge and
        destroys the likelihood (typical symptom: NumPyro ``post ~ -10^2…-10^3``
        while the emcee pilot on the same emulator is ``post ~ +O(10)``).
        """
        import tensorflow as tf
        from ...runtime.prior import (
            GaussianPrior,
            TruncatedGaussianPrior,
            UniformPrior,
        )

        unit = tf.reshape(unit_tf, [-1])
        # Stay inside open (0,1) for Uniform edges and erfinv stability.
        # NUTS may propose values slightly outside [0, 1].
        u_eps = tf.constant(1e-7, dtype=DTYPE)
        one = tf.constant(1.0, dtype=DTYPE)
        unit = tf.clip_by_value(unit, u_eps, one - u_eps)
        sqrt2 = tf.constant(np.sqrt(2.0), dtype=DTYPE)

        def _normal_ppf(u):
            return sqrt2 * tf.math.erfinv(2.0 * u - 1.0)

        def _normal_cdf(z):
            return 0.5 * (1.0 + tf.math.erf(z / sqrt2))

        pieces = []
        for i, param in enumerate(self.pipeline.varied_params):
            u_i = unit[i]
            prior = param.prior
            if isinstance(prior, UniformPrior):
                a = tf.constant(float(prior.a), dtype=DTYPE)
                b = tf.constant(float(prior.b), dtype=DTYPE)
                pieces.append(a + u_i * (b - a))
            elif isinstance(prior, TruncatedGaussianPrior):
                mu = tf.constant(float(prior.mu), dtype=DTYPE)
                sigma = tf.constant(float(prior.sigma), dtype=DTYPE)
                a = tf.constant(float(prior.a), dtype=DTYPE)
                b = tf.constant(float(prior.b), dtype=DTYPE)
                cdf_a = _normal_cdf(a)
                cdf_b = _normal_cdf(b)
                p = cdf_a + u_i * (cdf_b - cdf_a)
                z = _normal_ppf(p)
                pieces.append(mu + sigma * z)
            elif isinstance(prior, GaussianPrior):
                mu = tf.constant(float(prior.mu), dtype=DTYPE)
                sigma = tf.constant(float(prior.sigma), dtype=DTYPE)
                pieces.append(mu + sigma * _normal_ppf(u_i))
            else:
                # Fall back to hard limits (same as the old linear map).
                lo = tf.constant(float(param.limits[0]), dtype=DTYPE)
                hi = tf.constant(float(param.limits[1]), dtype=DTYPE)
                pieces.append(lo + u_i * (hi - lo))
        return tf.stack(pieces, axis=0)

    def _nuts_initial_mass_mean_variance(self, ndim: int, DTYPE, tf, initial_state):
        """Weakly informative RunningVariance prior for diagonal mass adaptation.

        Unit-cube sampling → unit variance (natural scale).
        Physical-space sampling → prefer diagonal of the tempered pilot chain;
        otherwise ``(prior_width / 4)^2`` so DualAveraging is not started with
        a unit-variance mass that is wildly mismatched to Ωm vs w vs biases.
        """
        mean0 = tf.reshape(tf.cast(initial_state, DTYPE), [ndim])
        pilot = self._nuts_pilot_samples_for_mass()
        if pilot is not None:
            max_rows = 5000
            if pilot.shape[0] > max_rows:
                rng = np.random.default_rng(getattr(self, "seed", None) or 0)
                pilot = pilot[
                    rng.choice(pilot.shape[0], size=max_rows, replace=False)
                ]
            var = np.var(pilot, axis=0)
            var = np.maximum(var, 1e-12)
            mean_np = np.mean(pilot, axis=0)
            logger.info(
                "Diagonal mass prior from pilot chain "
                f"(med σ={float(np.median(np.sqrt(var))):.3g})"
            )
            return (
                tf.constant(mean_np, dtype=DTYPE),
                tf.constant(var, dtype=DTYPE),
            )

        if self.nuts_sample_unit_space:
            var = tf.ones([ndim], dtype=DTYPE)
            logger.info(
                "Diagonal mass prior: unit variance (sampling in unit cube)"
            )
            return mean0, var

        widths = self._nuts_prior_widths()
        # ~Uniform[a,b] std is width/√12; use width/4 as a slightly broader start.
        var = tf.constant(np.maximum((widths / 4.0) ** 2, 1e-12), dtype=DTYPE)
        logger.info(
            "Diagonal mass prior: (prior_width/4)^2 per parameter "
            "(physical space; no pilot chain)"
        )
        return mean0, var

    def _nuts_initial_states(self, n_chains: int, DTYPE, tf):
        """Initial NUTS states in the sampling coordinate (unit or physical).

        Prefer the highest-posterior points from the previous tempered ROSE
        chain when available — starting from a random / last-training point is
        a common way for HMC to stick in a terrible basin. Falls back to
        pipeline starts with small jitter.
        """
        self._maybe_load_nuts_pilot_chain()
        ndim = len(self.pipeline.varied_params)
        logpost = getattr(self, "chain_logpost", None)
        if self.nuts_sample_unit_space:
            pool = getattr(self, "unit_chain", None)
        else:
            pool = getattr(self, "chain", None)

        starts = []
        if (
            pool is not None
            and logpost is not None
            and len(pool) == len(logpost)
            and len(pool) > 0
        ):
            order = np.argsort(np.asarray(logpost, dtype=float))[::-1]
            rng = np.random.default_rng(getattr(self, "seed", None) or 0)
            for i in range(n_chains):
                base = np.asarray(pool[order[i % len(order)]], dtype=np.float64)
                # Tiny jitter so multi-chain starts are not identical.
                if self.nuts_sample_unit_space:
                    jitter = 1e-3 * rng.normal(size=ndim)
                    starts.append(np.clip(base + jitter, 1e-6, 1.0 - 1e-6))
                else:
                    widths = self._nuts_prior_widths()
                    jitter = 1e-3 * widths * rng.normal(size=ndim)
                    starts.append(base + jitter)
            t = getattr(self, "chain_tempering", None)
            t_str = (
                f"{float(t):.4g}"
                if t is not None and np.isfinite(float(t))
                else "?"
            )
            logger.info(
                f"NUTS/NumPyro init: {n_chains} start(s) from top points of "
                f"latest tempered chain (tempering={t_str}, n={len(pool)}, "
                f"max logpost={float(logpost[order[0]]):.3g})"
            )
        else:
            if self.trained_before:
                logger.warning(
                    "NUTS/NumPyro init: no tempered chain available "
                    "(trained_before without a previous ROSE chain / "
                    "numpyro_pilot_chain); using randomized / pipeline starts"
                )
            else:
                logger.warning(
                    "NUTS/NumPyro init: no in-memory or saved tempered chain "
                    "from the previous ROSE iteration; using training/pipeline "
                    "starts. For trained_before=F this usually means the last "
                    "emcee chain was empty or not stored."
                )
            for i in range(n_chains):
                # CosmoSIS randomized_start() returns *physical* prior draws
                # (Parameter.random_point), not unit-cube values.
                if self.trained_before:
                    physical = np.asarray(
                        self.pipeline.randomized_start(), dtype=np.float64
                    )
                    if self.nuts_sample_unit_space:
                        starts.append(
                            np.asarray(
                                self.pipeline.normalize_vector_to_prior(physical),
                                dtype=np.float64,
                            )
                        )
                    else:
                        starts.append(physical)
                    continue
                if len(getattr(self, "unit_sample", [])) > 0:
                    unit = np.asarray(self.unit_sample[-(i + 1)], dtype=np.float64)
                else:
                    unit = np.asarray(
                        self.pipeline.normalize_vector_to_prior(
                            np.array(
                                [p.start for p in self.pipeline.varied_params],
                                dtype=float,
                            )
                        ),
                        dtype=np.float64,
                    )
                if self.nuts_sample_unit_space:
                    starts.append(unit)
                else:
                    starts.append(
                        np.asarray(
                            self._denormalize_unit_vector(unit),
                            dtype=np.float64,
                        )
                    )
            logger.info(
                f"NUTS init: {n_chains} start(s) from training/pipeline "
                f"(no usable tempered posterior ranking)"
            )

        return [
            tf.constant(np.asarray(s, dtype=np.float32), dtype=DTYPE) for s in starts
        ]

    def _nuts_dense_momentum_distribution(self, ndim: int, DTYPE, tfp, tf):
        """Build a fixed dense momentum distribution from the previous ROSE chain.

        Optimal HMC mass is ≈ Cov(θ)^{-1}, so momentum ~ N(0, Cov(θ)^{-1}).
        Falls back to None if no usable pilot samples exist.
        """
        tfd = tfp.distributions
        pilot = self._nuts_pilot_samples_for_mass()
        if pilot is None:
            return None

        # Subsample for stable/fast cov estimate if the tempered chain is huge.
        max_rows = 5000
        if pilot.shape[0] > max_rows:
            rng = np.random.default_rng(getattr(self, "seed", None) or 0)
            idx = rng.choice(pilot.shape[0], size=max_rows, replace=False)
            pilot = pilot[idx]

        cov = np.cov(pilot, rowvar=False)
        cov = np.atleast_2d(cov)
        # Ridge for PSD / invertibility (important in high-D 3x2pt).
        eps = 1e-3 * float(np.mean(np.diag(cov))) + 1e-8
        cov = cov + eps * np.eye(cov.shape[0])
        try:
            mom_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("Dense mass: covariance inversion failed; falling back")
            return None

        # Symmetrize numerical noise.
        mom_cov = 0.5 * (mom_cov + mom_cov.T)
        scale_tril = tf.linalg.cholesky(tf.constant(mom_cov, dtype=DTYPE))
        momentum_distribution = tfd.MultivariateNormalTriL(
            loc=tf.zeros([ndim], dtype=DTYPE),
            scale_tril=scale_tril,
        )
        cond = float(np.linalg.cond(cov))
        logger.info(
            f"Dense mass matrix from {pilot.shape[0]} pilot samples "
            f"(position cov cond≈{cond:.2e}, ridge={eps:.2e})"
        )
        return momentum_distribution

    def _build_nuts_adaptive_kernel(self, log_prob_nuts, initial_state, DTYPE, tfp, tf):
        """Build NUTS with Stan-style windowed step-size / mass adaptation.

        Default (``nuts_mass_matrix=diagonal``) uses TFP's
        ``windowed_sampling.make_windowed_adapt_kernel`` — the same expanding
        fast/slow window schedule as ``windowed_adaptive_nuts`` / Stan — rather
        than a hand-nested DualAveraging → DiagonalMassMatrix stack (easy to
        mis-time so the mass freezes on transient burn-in).

        ``nuts_mass_matrix``:
          - ``none``: identity mass, DualAveraging step-size only
          - ``diagonal``: windowed DualAveraging + diagonal mass (default)
          - ``dense``: fixed dense mass from previous ROSE chain + DualAveraging
            step-size (windowed mass adapt is diagonal-only in this TFP API)
        """
        from tensorflow_probability.python.experimental.mcmc import (
            windowed_sampling as windowed_mcmc,
        )

        ndim = int(np.prod(initial_state.shape))
        # initial_state may be [ndim] or [n_chains, ndim]; mass is per-event dim.
        if len(initial_state.shape) > 1:
            ndim = int(initial_state.shape[-1])
        mass_mode = getattr(self, "nuts_mass_matrix", "diagonal")
        step_size = (
            self.nuts_fixed_step_size
            if self.nuts_use_fixed_step_size
            else self.nuts_step_size
        )

        nuts_kwargs = dict(
            target_log_prob_fn=log_prob_nuts,
            step_size=step_size,
            max_tree_depth=self.nuts_max_tree_depth,
            max_energy_diff=self.nuts_max_energy_diff,
            unrolled_leapfrog_steps=self.nuts_unrolled_leapfrog_steps,
            parallel_iterations=self.nuts_parallel_iterations,
            name="nuts_kernel",
        )

        # ----- fixed step size: no DualAveraging / windowed schedule -----
        if self.nuts_use_fixed_step_size:
            if mass_mode == "dense":
                mom = self._nuts_dense_momentum_distribution(ndim, DTYPE, tfp, tf)
                if mom is not None:
                    nuts_kwargs["momentum_distribution"] = mom
                    logger.info(
                        f"Fixed-step preconditioned NUTS (dense mass), "
                        f"step_size={step_size}"
                    )
                    return tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
                        **nuts_kwargs
                    )
                mass_mode = "diagonal"
            if mass_mode == "none":
                logger.info(f"Fixed-step NUTS (identity mass), step_size={step_size}")
                return tfp.mcmc.NoUTurnSampler(**nuts_kwargs)
            logger.info(
                f"Fixed-step preconditioned NUTS (identity mass start), "
                f"step_size={step_size}"
            )
            return tfp.experimental.mcmc.PreconditionedNoUTurnSampler(**nuts_kwargs)

        # ----- dense: fixed pilot mass + step-size adaptation only -----
        if mass_mode == "dense":
            mom = self._nuts_dense_momentum_distribution(ndim, DTYPE, tfp, tf)
            if mom is not None:
                nuts_kwargs["momentum_distribution"] = mom
                kernel = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
                    **nuts_kwargs
                )
                logger.info(
                    f"Dense-mass NUTS + DualAveraging step size "
                    f"({self.nuts_num_adaptation_steps} steps, "
                    f"target_accept={self.nuts_target_accept_prob})"
                )
                return tfp.mcmc.DualAveragingStepSizeAdaptation(
                    inner_kernel=kernel,
                    num_adaptation_steps=self.nuts_num_adaptation_steps,
                    target_accept_prob=self.nuts_target_accept_prob,
                )
            logger.warning(
                "nuts_mass_matrix=dense requested but no usable previous "
                "chain; falling back to windowed diagonal adaptation"
            )
            mass_mode = "diagonal"

        # ----- none: step-size adaptation, identity mass -----
        if mass_mode == "none":
            kernel = tfp.mcmc.NoUTurnSampler(**nuts_kwargs)
            logger.info(
                f"NUTS (identity mass) + DualAveraging step size "
                f"({self.nuts_num_adaptation_steps} steps, "
                f"target_accept={self.nuts_target_accept_prob})"
            )
            return tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=self.nuts_num_adaptation_steps,
                target_accept_prob=self.nuts_target_accept_prob,
            )

        # ----- diagonal (default): Stan-style windowed adaptation -----
        mean0, var0 = self._nuts_initial_mass_mean_variance(
            ndim, DTYPE, tf, initial_state
        )
        initial_running_variance = tfp.experimental.stats.RunningVariance.from_stats(
            num_samples=tf.constant(float(max(ndim, 10)), dtype=DTYPE),
            mean=mean0,
            variance=var0,
        )
        dual_averaging_kwargs = {
            "num_adaptation_steps": int(self.nuts_num_adaptation_steps),
            "target_accept_prob": float(self.nuts_target_accept_prob),
        }
        logger.info(
            f"Windowed adaptive NUTS (Stan schedule via "
            f"tfp.experimental.mcmc.windowed_sampling): "
            f"adaptation={self.nuts_num_adaptation_steps}, "
            f"target_accept={self.nuts_target_accept_prob}, "
            f"step_size={step_size}, max_tree_depth={self.nuts_max_tree_depth}"
        )
        return windowed_mcmc.make_windowed_adapt_kernel(
            kind="nuts",
            proposal_kernel_kwargs=nuts_kwargs,
            dual_averaging_kwargs=dual_averaging_kwargs,
            initial_running_variance=initial_running_variance,
            chain_axis_names=None,
            shard_axis_names=None,
        )

    def _run_nuts_sampling(self, tempering: float) -> float:
        """Run NUTS sampling using TensorFlow Probability.

        Returns:
            Wall-clock seconds spent in NUTS sampling (excludes chain file I/O).
        """
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
        except ImportError as e:
            raise ImportError(
                "TensorFlow and TensorFlow Probability are required for NUTS sampling. "
                f"Install with: pip install tensorflow tensorflow-probability. Error: {e}"
            )
        
        if self.emu_pipeline is None:
            logger.warning("emu_pipeline is None, setting it up now (this should happen in execute())")
            self.compute_fiducial_setup_emu_pipeline()
            # Update global sampler reference after setup
            utils_module._sampler = self
        
        logger.info(
            f"Starting NUTS sampling with step_size={self.nuts_step_size}, "
            f"max_tree_depth={self.nuts_max_tree_depth}, "
            f"num_results={self.nuts_num_results}, "
            f"adaptation={self.nuts_num_adaptation_steps}, "
            f"burnin={self.nuts_num_burnin_steps}, "
            f"mass_matrix={self.nuts_mass_matrix}, "
            f"unit_space={self.nuts_sample_unit_space}"
        )

        if not self.emulator:
            raise RuntimeError("Emulators not set. This should be set during training or loading.")
        DTYPE = tf.float32

        # Starts in the NUTS coordinate (unit cube or physical). Prefer
        # high-posterior tempered-chain points when available.
        chain_starts = self._nuts_initial_states(self.nuts_num_chains, DTYPE, tf)
        initial_state = chain_starts[0]
        if self.nuts_sample_unit_space:
            initial_state_physical = self._denormalize_unit_vector(
                initial_state.numpy()
            )
        else:
            initial_state_physical = np.asarray(initial_state.numpy(), dtype=float)
        logger.info(f"Initial physical state for NUTS: {initial_state_physical}")

        # Differentiable log-prob: one context per CosmoSIS likelihood.
        # spectrum_emulators=T → CompositeSpectrumEmulator under that likelihood:
        # assemble WL/XC/GC into one theory vector, then one full-cov chi2.
        contexts = []
        data_vectors = None
        data_inv_covariance = None
        try:
            like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, block = utils_module.task(
                initial_state_physical, self, True
            )
            logger.info(
                f"Extracted data vectors for likelihoods: {list(data_vectors.keys())} "
                f"and covariance matrices for: {list(data_inv_covariance.keys())}"
            )
            contexts = self._build_nuts_likelihood_contexts(
                data_vectors, data_inv_covariance, DTYPE
            )
        except Exception as e:
            logger.warning(f"Could not get sample run for NUTS data extraction: {e}")
            logger.warning("NUTS will use fallback likelihood computation (slower, no gradients)")

        if not data_vectors or not data_inv_covariance:
            raise RuntimeError(
                "NUTS requires data vectors and inverse covariances extracted "
                "from the pipeline (needed for autodiff chi2 and MPI workers). "
                f"Extraction failed or returned empty (keys={list((data_vectors or {}).keys())})."
            )

        # Verify gradient computation by comparing with finite differences
        self._verify_nuts_gradients(initial_state_physical, tempering, contexts, DTYPE)

        # Ensure workers can load the same emulator from disk (emcee/nautilus pattern).
        model_path = self._worker_emu_model_path()
        if self.pool is not None and self.emulator and model_path is not None:
            for name, emu in (
                self.emulator.items()
                if isinstance(self.emulator, dict)
                else [("likelihood", self.emulator)]
            ):
                if not getattr(emu, "trained", False):
                    continue
                save_base = os.path.join(model_path, name)
                if not os.path.exists(save_base + ".npz"):
                    logger.info(
                        f"Saving emulator for '{name}' to disk so NUTS workers can load it"
                    )
                    if hasattr(emu, "save_to"):
                        emu.save_to(save_base)

        # Pilot samples for mass-matrix adaptation (subsample for pickling/MPI).
        pilot_unit = None
        pilot_physical = None
        pilot_logpost = None
        self._maybe_load_nuts_pilot_chain()
        pilot = self._nuts_pilot_samples_for_mass()
        if pilot is not None:
            max_rows = 5000
            if pilot.shape[0] > max_rows:
                rng = np.random.default_rng(getattr(self, "seed", None) or 0)
                idx = rng.choice(pilot.shape[0], size=max_rows, replace=False)
            else:
                idx = np.arange(pilot.shape[0])
            if self.nuts_sample_unit_space:
                pilot_unit = np.asarray(self.unit_chain, dtype=np.float64)[idx]
                pilot_physical = np.asarray(self.chain, dtype=np.float64)[idx]
            else:
                pilot_physical = np.asarray(self.chain, dtype=np.float64)[idx]
                pilot_unit = np.asarray(self.unit_chain, dtype=np.float64)[idx]
            if getattr(self, "chain_logpost", None) is not None:
                pilot_logpost = np.asarray(self.chain_logpost, dtype=float)[idx]

        # Pack likelihood data as plain numpy (picklable for MPI).
        data_vectors_np = {
            k: np.asarray(v, dtype=np.float64) for k, v in data_vectors.items()
        }
        inv_cov_np = {
            k: np.asarray(v, dtype=np.float64) for k, v in data_inv_covariance.items()
        }

        tasks = []
        for chain_idx, start in enumerate(chain_starts):
            seed_i = None
            if self.seed is not None:
                seed_i = int(self.seed) + int(chain_idx)
            tasks.append(
                {
                    "chain_idx": int(chain_idx),
                    "initial_state": np.asarray(start.numpy(), dtype=np.float32),
                    "tempering": float(tempering),
                    "model_path": model_path,
                    "data_vectors": data_vectors_np,
                    "data_inv_covariance": inv_cov_np,
                    "pilot_unit": pilot_unit,
                    "pilot_physical": pilot_physical,
                    "pilot_logpost": pilot_logpost,
                    "seed": seed_i,
                    # Progress bar on every rank/chain unless nuts_progress_interval=0.
                    "enable_progress": True,
                }
            )

        use_pool = (
            self.pool is not None
            and self.nuts_num_chains > 1
        )
        start_time = default_timer()
        if use_pool:
            pool_size = getattr(self.pool, "size", None)
            logger.info(
                f"Dispatching {self.nuts_num_chains} NUTS chain(s) via CosmoSIS "
                f"pool.map (pool size={pool_size})"
            )
            results = self.pool.map(utils_module.nuts_chain_worker, tasks)
        else:
            if self.pool is not None and self.nuts_num_chains == 1:
                logger.info(
                    "NUTS: single chain — running on master only "
                    "(no pool.map overhead)"
                )
            results = [self._execute_one_nuts_chain(t) for t in tasks]

        # Combine chains in chain_idx order (pool.map preserves task order).
        results = sorted(results, key=lambda r: int(r["chain_idx"]))
        all_chains = [r["samples"] for r in results]
        all_log_probs = [r["log_probs"] for r in results]
        all_blobs = [r["blobs"] for r in results]

        if self.nuts_num_chains > 1:
            self.chain = np.vstack(all_chains)
            self.nuts_logp = np.concatenate(all_log_probs)
            self.blobs = [item for sublist in all_blobs for item in sublist]
        else:
            self.chain = all_chains[0] if all_chains else np.array([])
            self.nuts_logp = np.array(all_log_probs[0]) if all_log_probs else np.array([])
            self.blobs = all_blobs[0] if all_blobs else []

        self.unit_chain = np.array([
            self.pipeline.normalize_vector_to_prior(p) for p in self.chain
        ])

        end_time = default_timer()
        time_sampling_s = end_time - start_time
        logger.info(
            f"NUTS sampling took {time_sampling_s:.1f} seconds "
            f"({self.nuts_num_chains} chain(s)"
            + (f", pool size={getattr(self.pool, 'size', '?')}" if use_pool else ", serial")
            + ")"
        )

        self._process_nuts_results(tempering)
        return time_sampling_s

    def _run_numpyro_sampling(self, tempering: float) -> float:
        """Run NumPyro NUTS with the TF emulator log-prob via ``jax2tf.call_tf``.

        Reuses the same differentiable TF likelihood as TFP NUTS
        (:meth:`_log_prob_nuts_impl`); JAX only sees a wrapped callable, so the
        CosmoPower / spectrum emulators stay in TensorFlow.
        """
        try:
            import numpyro
            from numpyro.infer import MCMC, NUTS as NumPyroNUTS
        except ImportError as e:
            raise ImportError(
                "NumPyro is required for final_sampler=numpyro. "
                "Install a version compatible with your JAX (e.g. "
                "`pip install 'numpyro==0.13.2'` with jax 0.4.x). "
                f"Original error: {e}"
            )
        try:
            # Host-device count must be set before JAX initializes devices.
            n_chains = int(self.numpyro_num_chains)
            if n_chains > 1 and self.numpyro_chain_method == "parallel":
                numpyro.set_host_device_count(n_chains)
            import jax
            import jax.numpy as jnp
            from jax.experimental import jax2tf
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "JAX + jax2tf are required for final_sampler=numpyro. "
                f"Original error: {e}"
            )

        if self.emu_pipeline is None:
            self.compute_fiducial_setup_emu_pipeline()
            utils_module._sampler = self

        if not self.emulator:
            raise RuntimeError(
                "Emulators not set. Train or load them before NumPyro sampling."
            )

        DTYPE = tf.float32
        ndim = int(self.ndim)
        # Reuse NUTS unit-space / pilot helpers (aliases set in config).
        self.nuts_sample_unit_space = bool(self.numpyro_sample_unit_space)
        chain_starts = self._nuts_initial_states(n_chains, DTYPE, tf)
        initial_state = chain_starts[0]
        if self.numpyro_sample_unit_space:
            initial_physical = self._denormalize_unit_vector(initial_state.numpy())
        else:
            initial_physical = np.asarray(initial_state.numpy(), dtype=float)

        logger.info(f"NumPyro initial physical state: {initial_physical}")

        try:
            (
                _like,
                _theory,
                data_vectors,
                data_inv_covariance,
                _err,
                _block,
            ) = utils_module.task(initial_physical, self, True)
        except Exception as e:
            raise RuntimeError(
                "NumPyro requires data vectors / inv-cov from the pipeline "
                f"for the TF likelihood. Extraction failed: {e}"
            ) from e

        contexts = self._build_nuts_likelihood_contexts(
            data_vectors, data_inv_covariance, DTYPE
        )
        self._verify_nuts_gradients(initial_physical, tempering, contexts, DTYPE)

        # Concrete TF log-prob with fixed signature for jax2tf.
        if self.numpyro_sample_unit_space:
            # autograph=False: AutoGraph can misbind NNEmulator methods
            # (e.g. backtransform_tf → _linna_feature_space_loss_arrays) when
            # jax2tf.call_tf traces this graph. Match the TFP NUTS path.
            # Unit → physical via prior CDF (not linear limits): required for
            # Gaussian nuisance parameters. No extra (high-low) Jacobian —
            # CosmoSIS emcee also evaluates the full prior in physical space.
            @tf.function(
                input_signature=[tf.TensorSpec([ndim], tf.float32)],
                autograph=False,
            )
            def log_prob_tf(state):
                physical = self._unit_to_physical_tf(state, DTYPE)
                return self._log_prob_nuts_impl(
                    physical, contexts, tempering, DTYPE
                )
        else:

            @tf.function(
                input_signature=[tf.TensorSpec([ndim], tf.float32)],
                autograph=False,
            )
            def log_prob_tf(state):
                return self._log_prob_nuts_impl(
                    state, contexts, tempering, DTYPE
                )

        log_prob_jax = jax2tf.call_tf(log_prob_tf)

        # NumPyro NUTS treats the state as unconstrained ℝⁿ. For unit-cube
        # sampling, work in logit space and map with sigmoid so leapfrog
        # cannot leave (0,1) (avoids CosmoSIS denormalize crashes and most
        # boundary divergences). Include the sigmoid Jacobian so the target
        # remains the CosmoSIS measure on the unit cube.
        if self.numpyro_sample_unit_space:

            def potential_fn(z):
                z32 = jnp.asarray(z, dtype=jnp.float32)
                u = jax.nn.sigmoid(z32)
                log_jac = jnp.sum(jnp.log(u) + jnp.log1p(-u))
                return -log_prob_jax(u) - log_jac

            init_stack = np.stack(
                [
                    self._logit_unit_cube(np.asarray(s.numpy(), dtype=np.float64))
                    for s in chain_starts
                ],
                axis=0,
            ).astype(np.float32)
        else:

            def potential_fn(z):
                z32 = jnp.asarray(z, dtype=jnp.float32)
                return -log_prob_jax(z32)

            init_stack = np.stack(
                [np.asarray(s.numpy(), dtype=np.float32) for s in chain_starts],
                axis=0,
            )
        init_params = jnp.asarray(init_stack)

        seed = int(self.seed) if self.seed is not None else 0
        rng_key = jax.random.PRNGKey(seed)

        kernel = NumPyroNUTS(
            potential_fn=potential_fn,
            target_accept_prob=float(self.numpyro_target_accept_prob),
            max_tree_depth=int(self.numpyro_max_tree_depth),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=int(self.numpyro_num_warmup),
            num_samples=int(self.numpyro_num_samples),
            num_chains=n_chains,
            progress_bar=bool(self.numpyro_progress_bar),
            chain_method=str(self.numpyro_chain_method),
        )

        logger.info(
            f"Starting NumPyro NUTS: chains={n_chains}, "
            f"warmup={self.numpyro_num_warmup}, samples={self.numpyro_num_samples}, "
            f"chain_method={self.numpyro_chain_method}"
        )
        start_time = default_timer()
        mcmc.run(rng_key, init_params=init_params)
        end_time = default_timer()
        time_sampling_s = end_time - start_time

        # Diagnostics (R-hat / n_eff) to the ROSE logger.
        try:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                mcmc.print_summary(exclude_deterministic=False)
            summary = buf.getvalue().strip()
            if summary:
                logger.info("NumPyro summary:\n%s", summary)
        except Exception as exc:
            logger.warning("NumPyro print_summary failed: %s", exc)

        raw_samples = mcmc.get_samples(group_by_chain=True)
        if isinstance(raw_samples, dict):
            # Older NumPyro may name unconstrained sites (e.g. "Param:0").
            if len(raw_samples) != 1:
                raise RuntimeError(
                    f"NumPyro returned multiple sample sites: {list(raw_samples)}"
                )
            raw_samples = next(iter(raw_samples.values()))
        by_chain = np.asarray(raw_samples, dtype=np.float64)
        # Expect (n_chains, n_samples, ndim); fall back if flat.
        if by_chain.ndim == 2:
            samples_coord = by_chain
        elif by_chain.ndim == 3:
            samples_coord = by_chain.reshape(-1, by_chain.shape[-1])
        else:
            raise RuntimeError(
                f"Unexpected NumPyro sample shape {by_chain.shape}"
            )

        if self.numpyro_sample_unit_space:
            # Samples are in logit / unconstrained space → unit cube → physical.
            unit = 1.0 / (1.0 + np.exp(-samples_coord))
            samples_physical = np.array(
                [self._denormalize_unit_vector(unit[i]) for i in range(len(unit))]
            )
        else:
            samples_physical = samples_coord

        chain_log_probs = []
        chain_blobs = []
        for sample in samples_physical:
            try:
                r = self.emu_pipeline.run_results(sample)
                chain_log_probs.append(r.post * tempering)
                chain_blobs.append((r.prior, r.extra))
            except Exception:
                chain_log_probs.append(-np.inf)
                chain_blobs.append(
                    (-np.inf, [np.nan] * self.pipeline.number_extra)
                )

        self.chain = np.asarray(samples_physical, dtype=np.float64)
        self.nuts_logp = np.asarray(chain_log_probs, dtype=float)
        self.blobs = chain_blobs
        self.unit_chain = np.array(
            [self.pipeline.normalize_vector_to_prior(p) for p in self.chain]
        )

        logger.info(
            f"NumPyro sampling took {time_sampling_s:.1f} seconds "
            f"({len(self.chain)} samples from {n_chains} chain(s))"
        )
        self._process_nuts_results(tempering)
        return time_sampling_s

    def _execute_one_nuts_chain(self, task: dict) -> dict:
        """Run a single NUTS chain; safe for CosmoSIS ``pool.map`` workers.

        Builds TF contexts / adaptive kernel locally (not picklable across MPI).
        """
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
        except ImportError as e:
            raise ImportError(
                "TensorFlow and TensorFlow Probability are required for NUTS sampling. "
                f"Error: {e}"
            )

        chain_idx = int(task["chain_idx"])
        tempering = float(task["tempering"])
        DTYPE = tf.float32
        initial_state = tf.constant(
            np.asarray(task["initial_state"], dtype=np.float32), dtype=DTYPE
        )

        # Install pilot chain on this process for mass-matrix / dense prior.
        if task.get("pilot_unit") is not None:
            self.unit_chain = np.asarray(task["pilot_unit"], dtype=np.float64)
        if task.get("pilot_physical") is not None:
            self.chain = np.asarray(task["pilot_physical"], dtype=np.float64)
        if task.get("pilot_logpost") is not None:
            self.chain_logpost = np.asarray(task["pilot_logpost"], dtype=float)

        if self.emu_pipeline is None:
            self.compute_fiducial_setup_emu_pipeline()

        contexts = self._build_nuts_likelihood_contexts(
            task["data_vectors"], task["data_inv_covariance"], DTYPE
        )

        if self.nuts_sample_unit_space:

            def log_prob_nuts(state_tf):
                physical = self._unit_to_physical_tf(state_tf, DTYPE)
                return self._log_prob_nuts_impl(
                    physical, contexts, tempering, DTYPE
                )
        else:

            def log_prob_nuts(physical_params_tf):
                return self._log_prob_nuts_impl(
                    physical_params_tf, contexts, tempering, DTYPE
                )

        adaptive_kernel = self._build_nuts_adaptive_kernel(
            log_prob_nuts=log_prob_nuts,
            initial_state=initial_state,
            DTYPE=DTYPE,
            tfp=tfp,
            tf=tf,
        )

        seed = task.get("seed")
        show_progress = bool(task.get("enable_progress", True)) and (
            int(getattr(self, "nuts_progress_interval", 0) or 0) > 0
        )
        logger.info(
            f"NUTS chain {chain_idx + 1}: sampling "
            f"(results={self.nuts_num_results}, burnin={self.nuts_num_burnin_steps})"
        )

        samples_np = self._nuts_run_sample_chain(
            kernel=adaptive_kernel,
            initial_state=initial_state,
            num_results=int(self.nuts_num_results),
            num_burnin_steps=int(self.nuts_num_burnin_steps),
            seed=seed,
            chain_idx=chain_idx,
            show_progress=show_progress,
            tfp=tfp,
            tf=tf,
        )
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(-1, 1)

        if self.nuts_sample_unit_space:
            samples_physical = np.array(
                [self._denormalize_unit_vector(samples_np[i]) for i in range(len(samples_np))]
            )
        else:
            samples_physical = samples_np

        chain_log_probs = []
        chain_blobs = []
        for sample in samples_physical:
            try:
                r = self.emu_pipeline.run_results(sample)
                chain_log_probs.append(r.post * tempering)
                chain_blobs.append((r.prior, r.extra))
            except Exception:
                chain_log_probs.append(-np.inf)
                chain_blobs.append(
                    (-np.inf, [np.nan] * self.pipeline.number_extra)
                )

        logger.info(
            f"NUTS chain {chain_idx + 1}: finished "
            f"({len(samples_physical)} samples, "
            f"med post={float(np.median(np.asarray(chain_log_probs) / tempering)):.3g})"
        )
        return {
            "chain_idx": chain_idx,
            "samples": np.asarray(samples_physical, dtype=np.float64),
            "log_probs": np.asarray(chain_log_probs, dtype=float),
            "blobs": chain_blobs,
        }

    def _nuts_run_sample_chain(
        self,
        *,
        kernel,
        initial_state,
        num_results: int,
        num_burnin_steps: int,
        seed,
        chain_idx: int,
        show_progress: bool,
        tfp,
        tf,
    ) -> np.ndarray:
        """Run ``tfp.mcmc.sample_chain`` with an optional tqdm progress bar.

        A single compiled ``sample_chain`` call cannot update a Python progress
        bar mid-run, so we advance in chunks of ``nuts_progress_interval``
        steps, carrying ``previous_kernel_results`` so windowed adaptation
        continues correctly across chunk boundaries.
        """
        from tqdm.auto import tqdm

        chunk = int(getattr(self, "nuts_progress_interval", 0) or 0)
        total_steps = int(num_burnin_steps) + int(num_results)
        if (not show_progress) or chunk <= 0 or total_steps <= 0:
            @tf.function(autograph=False)
            def _run_all():
                return tfp.mcmc.sample_chain(
                    num_results=num_results,
                    num_burnin_steps=num_burnin_steps,
                    current_state=initial_state,
                    kernel=kernel,
                    trace_fn=None,
                    seed=seed,
                )

            samples, _ = _run_all()
            return np.asarray(samples.numpy())

        # Chunked path with tqdm. Kernel results carry adaptation state across chunks.
        state = initial_state
        pkr = None
        kept = []
        remaining_burnin = int(num_burnin_steps)
        remaining_keep = int(num_results)
        rng = np.random.default_rng(int(seed) if seed is not None else None)

        desc = f"NUTS chain {chain_idx + 1}"
        position = int(chain_idx) if self.pool is None else None
        pbar = tqdm(
            total=total_steps,
            desc=desc,
            unit="step",
            position=position,
            leave=True,
            dynamic_ncols=True,
        )
        try:
            while remaining_burnin > 0 or remaining_keep > 0:
                if remaining_burnin > 0:
                    n = min(chunk, remaining_burnin)
                    keep_chunk = False
                else:
                    n = min(chunk, remaining_keep)
                    keep_chunk = True

                chunk_seed = int(rng.integers(0, 2**31 - 1))
                kwargs = dict(
                    num_results=n,
                    num_burnin_steps=0,
                    current_state=state,
                    kernel=kernel,
                    trace_fn=None,
                    return_final_kernel_results=True,
                    seed=chunk_seed,
                )
                if pkr is not None:
                    kwargs["previous_kernel_results"] = pkr

                states, _trace, pkr = tfp.mcmc.sample_chain(**kwargs)
                state = states[-1]
                if keep_chunk:
                    kept.append(np.asarray(states.numpy()))
                    remaining_keep -= n
                else:
                    remaining_burnin -= n

                pbar.update(n)
                pbar.set_postfix(
                    phase=("burnin" if not keep_chunk else "sample"),
                    kept=int(num_results) - max(remaining_keep, 0),
                    refresh=False,
                )
        finally:
            pbar.close()

        if not kept:
            raise RuntimeError("NUTS produced no retained samples")
        return np.concatenate(kept, axis=0)

    def _log_prob_nuts_impl(self, physical_params_tf, contexts, tempering, DTYPE,
                            TF=True):
        """Differentiable (tempered) log posterior for NUTS.

        Each entry in ``contexts`` is one CosmoSIS likelihood. For
        ``spectrum_emulators=T``, that likelihood's context is composite: the
        probe nets fill one theory vector and chi2 uses the full inv-cov
        (cross-covariance included). Independent chi2 terms are only summed
        across separate CosmoSIS likelihoods, never across spectra.
        """
        import tensorflow as tf
        from ...runtime.prior import UniformPrior, GaussianPrior, TruncatedGaussianPrior

        if len(physical_params_tf.shape) > 1:
            physical_params_tf = tf.reshape(physical_params_tf, [-1])

        if contexts and TF:
            log_like_terms = []
            for context in contexts:
                predictions = self._nuts_theory_vector_tf(
                    physical_params_tf, context, DTYPE
                )
                diff = predictions - context["data_vector"]
                # Full Gaussian chi2 for this likelihood (includes cross-probe
                # blocks when the theory vector is a composite 3x2pt DV).
                chi2 = tf.einsum("i,ij,j->", diff, context["inv_covariance"], diff)
                log_like_terms.append(tf.reshape(-0.5 * chi2, []))
            log_like = tf.add_n(log_like_terms)
        else:
            # Fallback: pipeline likelihood (breaks gradient flow).
            def get_likelihood(params_tf):
                try:
                    params_np = (
                        params_tf.numpy()
                        if hasattr(params_tf, "numpy")
                        else np.array(params_tf)
                    )
                    r = self.emu_pipeline.run_results(params_np)
                    return float(r.like)
                except Exception:
                    return -np.inf

            log_like = tf.py_function(
                func=get_likelihood,
                inp=[physical_params_tf],
                Tout=tf.float32,
            )
            log_like.set_shape([])

        # Differentiable prior over varied parameters
        log_prior_terms = []

        for i, param in enumerate(self.pipeline.varied_params):
            param_value = physical_params_tf[i]
            prior = param.prior
            limits = param.limits

            in_bounds = tf.logical_and(
                param_value >= limits[0],
                param_value <= limits[1],
            )

            if isinstance(prior, UniformPrior):
                prior_norm_tf = tf.constant(prior.norm, dtype=DTYPE)
                prior_log_prob = tf.where(
                    in_bounds,
                    prior_norm_tf,
                    tf.constant(-np.inf, dtype=DTYPE),
                )
                log_prior_terms.append(prior_log_prob)

            elif isinstance(prior, (GaussianPrior, TruncatedGaussianPrior)):
                mu = tf.constant(prior.mu, dtype=DTYPE)
                sigma2 = tf.constant(prior.sigma2, dtype=DTYPE)
                norm = tf.constant(prior.norm, dtype=DTYPE)
                gaussian_log_prob = (
                    -0.5 * tf.square(param_value - mu) / sigma2 - norm
                )
                prior_log_prob = tf.where(
                    in_bounds,
                    gaussian_log_prob,
                    tf.constant(-np.inf, dtype=DTYPE),
                )
                log_prior_terms.append(prior_log_prob)

            else:
                def get_single_prior(params_tf, param_idx):
                    try:
                        params_np = (
                            params_tf.numpy()
                            if hasattr(params_tf, "numpy")
                            else np.array(params_tf)
                        )
                        param_val = params_np[param_idx]
                        param_obj = self.pipeline.varied_params[param_idx]
                        return float(param_obj.evaluate_prior(param_val))
                    except Exception:
                        return -np.inf

                prior_log_prob = tf.py_function(
                    func=lambda p, idx=i: get_single_prior(p, idx),
                    inp=[physical_params_tf],
                    Tout=tf.float32,
                )
                prior_log_prob.set_shape([])
                prior_log_prob = tf.stop_gradient(prior_log_prob)
                log_prior_terms.append(prior_log_prob)

        if log_prior_terms:
            log_prior = tf.add_n(log_prior_terms)
        else:
            log_prior = tf.constant(0.0, dtype=DTYPE)

        log_prior = tf.reshape(log_prior, [])
        log_post = (log_like + log_prior) * tempering
        return tf.reshape(log_post, [])

    def _process_nuts_results(self, tempering: float) -> None:
        """Process NUTS results and update output chains."""
        logp = self.nuts_logp
        self.blobs = self.blobs

        # Store the (untempered) posterior per chain point for credible-region
        # test-point selection in the next iteration.
        self.chain_logpost = np.asarray(logp) / tempering
        # NUTS samples are equal-weight (log-weight 0).
        self.chain_log_weights = np.zeros(len(self.chain))
        self.chain_tempering = float(tempering)

        self._begin_chain_output_file()
        for params, tempered_post, blob in zip(self.chain, logp, self.blobs):
            prior, extra = blob
            post = tempered_post / tempering
            self.output.parameters(params, extra, prior, tempered_post, post)
        self._finalize_chain_output_file(tempering, logp, nautilus_weights=False)
        logger.info(f"Generated {len(self.chain)} NUTS samples")

    def _begin_chain_output_file(self) -> None:
        """Archive/reset the CosmoSIS output file before writing a new chain.

        Convergence-loop untempered chains set ``_chain_save_suffix`` (e.g.
        ``_from_emumodel_4``). Those must not reuse the tempered
        ``_tempering_*_iteration_*`` path, which would overwrite the last
        tempered chain. Any tempered chain still sitting in the main file is
        archived once first.
        """
        suffix = getattr(self, "_chain_save_suffix", None)
        if suffix:
            if (
                self.save_output == SAVE_ALL
                and 0 < self.iterations < self.max_iterations
                and not getattr(self, "_tempered_chain_archived", False)
            ):
                tsuffix = (
                    f"_tempering_{self.tempering[self.iterations - 1]}"
                    f"_iteration_{self.iterations}"
                )
                self.output.save_and_reset_to_chain_start(tsuffix)
                self._tempered_chain_archived = True
            else:
                self.output.reset_to_chain_start()
            return

        if self.save_output == SAVE_ALL and 0 < self.iterations < self.max_iterations:
            tsuffix = (
                f"_tempering_{self.tempering[self.iterations - 1]}"
                f"_iteration_{self.iterations}"
            )
            self.output.save_and_reset_to_chain_start(tsuffix)
            self._tempered_chain_archived = True
        else:
            self.output.reset_to_chain_start()

    def _finalize_chain_output_file(
        self,
        tempering: float,
        logp: np.ndarray,
        nautilus_weights: bool = False,
        log_weights: Optional[np.ndarray] = None,
    ) -> None:
        """After writing samples, archive convergence chains as ``_from_emumodel_*``."""
        suffix = getattr(self, "_chain_save_suffix", None)
        if not suffix:
            return
        # Rename the just-written main file to ..._from_emumodel_N.txt
        self.output.save_and_reset_to_chain_start(suffix)
        logger.info("Saved chain to output file with suffix %s", suffix)
        # Keep the same samples in the main chain file as the latest posterior.
        self.output.reset_to_chain_start()
        blobs = getattr(self, "blobs", None)
        if blobs is None:
            return
        for i, (params, tempered_post, blob) in enumerate(
            zip(self.chain, logp, blobs)
        ):
            prior, extra = blob
            post = tempered_post / tempering
            if nautilus_weights:
                w = 0.0 if log_weights is None else float(log_weights[i])
                self.output.parameters(params, extra, prior, tempered_post, post, w)
            else:
                self.output.parameters(params, extra, prior, tempered_post, post)

    def _process_mcmc_results(self, emcee_sampler: Any, tempering: float) -> None:
        """Process MCMC results and update output chains."""
        # Base the burn-in on the *actual* number of iterations run, not the
        # configured maximum. Otherwise an early-converged chain (e.g. emcee
        # stopping at 1500 / 5000 steps) would have every sample discarded
        # when using a fractional burn-in, leaving `self.chain` empty and
        # breaking downstream resampling.
        n_iterations = emcee_sampler.get_chain().shape[0]
        if self.emcee_burn < 1:
            burn = int(self.emcee_burn * n_iterations)
        else:
            burn = int(self.emcee_burn)
        # Always keep at least one walker-step so the chain isn't empty.
        burn = min(burn, max(0, n_iterations - 1))

        # Extract chains
        self.unit_chain = emcee_sampler.get_chain(discard=burn, thin=self.emcee_thin, flat=True)
        logp = emcee_sampler.get_log_prob(discard=burn, thin=self.emcee_thin, flat=True)
        self.blobs = emcee_sampler.get_blobs(discard=burn, thin=self.emcee_thin, flat=True)
        if len(self.unit_chain) == 0:
            raise RuntimeError(
                f"emcee produced an empty chain after burn-in "
                f"(iterations={n_iterations}, burn={burn}, thin={self.emcee_thin}). "
                "Reduce emcee_burn or increase emcee_samples."
            )
        
        # Transform to physical parameters
        self.chain = np.array([
            self.pipeline.denormalize_vector_from_prior(p) for p in self.unit_chain
        ])

        # Store the (untempered) posterior per chain point, aligned with
        # self.chain/self.unit_chain, so the next iteration can select test
        # points from the 1-sigma credible region of this chain.
        self.chain_logpost = np.asarray(logp) / tempering
        # emcee samples are equal-weight (log-weight 0); recorded so the
        # per-iteration KL diagnostic can treat all samplers uniformly.
        self.chain_log_weights = np.zeros(len(self.chain))
        self.chain_tempering = float(tempering)

        self._begin_chain_output_file()
        for params, tempered_post, extra in zip(self.chain, logp, self.blobs):
            prior, extra = extra
            post = tempered_post / tempering
            if self.final_sampler == "nautilus":
                self.output.parameters(params, extra, prior, tempered_post, post, 0.0)
            else:
                self.output.parameters(params, extra, prior, tempered_post, post)
        self._finalize_chain_output_file(tempering, logp, nautilus_weights=False)
        logger.info(f"Generated {len(self.chain)} chain samples")

    def _process_nautilus_results(self, nautilus_sampler: Any, tempering: float) -> None:
        """Process nautilus results and update output chains."""
        try:
            # Try to get posterior samples with blobs
            results = nautilus_sampler.posterior(return_blobs=True)
            has_blobs = True
        except ValueError as e:
            if "No blobs have been calculated" in str(e):
                logger.warning("Nautilus did not calculate blobs, computing them manually")
                # Get posterior samples without blobs
                results = nautilus_sampler.posterior(return_blobs=False)
                has_blobs = False
            else:
                raise e
        
        # Extract results
        samples = results[0]  # Physical parameter samples
        log_weights = results[1]  # Log weights
        log_likelihoods = results[2]  # Log likelihoods
        
        if has_blobs and len(results) > 3:
            blobs = results[3]  # Blobs (prior, extra)
            # Handle blobs - nautilus returns flattened format
            if isinstance(blobs[0], (int, float)):
                # Single scalar per sample (just prior)
                priors = blobs
                extras = [None] * len(blobs)
            else:
                # Tuple of scalars per sample (prior + extra data)
                priors = np.array([r[0] for r in blobs])
                extras = []
                for r in blobs:
                    if len(r) > 1:
                        # Convert extra data to list
                        extra_data = list(r[1:]) if len(r) > 1 else None
                        extras.append(extra_data)
                    else:
                        extras.append(None)
        else:
            # Compute priors manually
            logger.info("Computing priors manually for nautilus samples")
            priors = []
            extras = []
            for sample in samples:
                # Convert to unit cube for prior calculation
                unit_sample = self.pipeline.normalize_vector_to_prior(sample)
                prior = self.pipeline.prior(unit_sample)
                priors.append(prior)
                extras.append(None)  # No extra data for nautilus
        
        # Calculate posterior probabilities
        posts = log_likelihoods + priors
        
        # Store results in the same format as emcee for compatibility
        self.chain = samples
        self.unit_chain = np.array([
            self.pipeline.normalize_vector_to_prior(p) for p in samples
        ])

        # Store the (untempered) posterior per chain point for credible-region
        # test-point selection in the next iteration.
        self.chain_logpost = np.asarray(posts)
        # Nautilus returns IMPORTANCE-WEIGHTED samples: the raw sample cloud
        # includes a broad low-weight exploration tail. Storing the log-weights
        # lets the per-iteration KL diagnostic weight the samples correctly
        # (otherwise the unweighted covariance is hugely inflated and the
        # Gaussian KL blows up).
        self.chain_log_weights = np.asarray(log_weights, dtype=float)
        self.chain_tempering = float(tempering)
        
        # Create log probability array (tempered)
        tempered_posts = posts * tempering
        logp = tempered_posts
        
        # Store blobs in emcee-compatible format
        self.blobs = list(zip(priors, extras))

        self._begin_chain_output_file()
        for params, tempered_post, blob, log_weight in zip(
            self.chain, logp, self.blobs, log_weights
        ):
            prior, extra = blob
            post = tempered_post / tempering
            if extra is None:
                extra = []
            self.output.parameters(params, extra, prior, tempered_post, post, log_weight)
        self._finalize_chain_output_file(
            tempering, logp, nautilus_weights=True, log_weights=log_weights
        )
        logger.info(f"Generated {len(self.chain)} nautilus samples with weights")

    def get_emcee_start(self) -> np.ndarray:
        """Get starting positions for MCMC walkers.
        
        Returns:
            Array of starting positions in unit hypercube
            
        TODO: Improve by selecting high-likelihood samples and adding noise
        """
        if len(self.unit_sample) < self.emcee_walkers:
            raise RuntimeError(f"Not enough training samples ({len(self.unit_sample)}) "
                             f"for {self.emcee_walkers} walkers")
        
        # For now, just take the last N samples
        # TODO: Select best samples and add small random perturbations
        return self.unit_sample[-self.emcee_walkers:]

