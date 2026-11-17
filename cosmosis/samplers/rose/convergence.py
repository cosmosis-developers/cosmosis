"""
Convergence tests for the ROSE sampler.

Just before the final sampling stage, the sampler collects a set of test points
from the tempered HPD of the one-before-last MCMC chain (see
``data_processing.generate_updated_sample``). This module runs convergence
diagnostics on those test points using the freshly trained emulator:

(a) Emulator accuracy on all emulated modes: relative-error percentiles between
    the emulated data vectors and the true full-pipeline data vectors, plus a
    covariance-normalized residual ``|emu - truth| / sqrt(diag(C))``.
(b) Delta chi^2 on the held-out test set, including debiased scatter metrics
    ``MAD(Δχ² − median)``, ``std(Δχ² − median)``, ``max|Δχ² − median|``, and
    ``frac(|Δχ² − median| < 2)``, written to ``rose_convergence.txt``.
    ``mad_pass`` requires ``MAD(r) <= delta_chi2_mad_threshold`` and, when
    ``delta_chi2_max_abs_r > 0``, also ``max|r| <= delta_chi2_max_abs_r``.
(c) Optional final-stage retrain loop (``kl_convergence = T``): the test set
    stays held out; new HPD training points are added until the MAD criterion
    (and optionally chain-based KL on ``kl_params``) is met. Passive
    importance-reweighted KL between consecutive emulators is logged to
    ``rose_kl.txt`` only (no ``kl_convergence.png``).

Separately, a lightweight per-iteration KL diagnostic is computed after every
MCMC stage (see ``_record_iteration_kl``): symmetric Gaussian and k-NN KL
between consecutive tempered chains after untempering, reported for all
parameters and for the selected ``kl_params`` subspace in ``rose_kl.txt``.

All numeric results are written to ``{save_dir}/convergence`` and, when
matplotlib is available, summary figures are saved alongside them.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Percentile levels reported for the emulator accuracy test.
_ACCURACY_PERCENTILES = (68.0, 95.0, 99.0, 99.9)


class RoseConvergenceMixin:
    """Mixin providing convergence diagnostics for :class:`RoseSampler`."""

    def run_convergence_tests(self) -> None:
        """Run holdout diagnostics and optional final-stage retrain loop.

        Called in the last training iteration after the emulator has been
        trained but before (or instead of) the final sampling stage. No-op when
        no test points were collected (e.g. ``final_test_size = 0``).
        """
        self._final_chain_already_sampled = False
        sample_test = getattr(self, "sample_test", None)
        if sample_test is None or len(sample_test) == 0:
            logger.warning(
                "No test points available; skipping convergence tests. "
                "Set final_test_size > 0 to enable them."
            )
            return

        if not getattr(self, "emulator", None):
            logger.warning("No trained emulator available; skipping convergence tests.")
            return

        logger.info(
            "Running convergence tests on %d held-out test points",
            len(sample_test),
        )

        self.convergence_dir = os.path.join(self.save_dir, "convergence")
        os.makedirs(self.convergence_dir, exist_ok=True)

        self.convergence_results: Dict[str, Any] = {}
        # Stash tempered HPD state before any untempered diagnostic chains.
        self._stash_tempered_hpd_state()

        emu_version = int(getattr(self, "_current_emu_version", self.iterations + 1))
        self.convergence_results["accuracy"] = self._test_emulator_accuracy(
            emu_version=emu_version
        )
        dchi2 = self._test_delta_chi2(emu_version=emu_version)
        self.convergence_results["delta_chi2"] = dchi2

        mad_pass = bool(dchi2.get("mad_pass", False))
        if mad_pass:
            logger.info(
                "Debiased Δχ² MAD=%.4g <= threshold %.4g; emulator accurate enough "
                "(no extra retrain).",
                dchi2["mad_r"], float(self.delta_chi2_mad_threshold),
            )
            # Still log IW KL vs previous on-disk emulator for the record.
            self._log_iw_emu_for_version(emu_version)
            self._append_rose_convergence_row(
                emu_version=emu_version,
                dchi2=dchi2,
                kl_all=float("nan"),
                kl_sel=float("nan"),
                chain_kl_pass=False,
                n_train_added=0,
            )
            self.kl_converged = True
            self.convergence_results["kl"] = {
                "converged": True,
                "reason": "mad_pass",
                "n_retrain": 0,
            }
        elif getattr(self, "kl_convergence", False):
            logger.info(
                "Final-stage convergence loop enabled (kl_convergence = T, "
                "kl_convergence_chain = %s)",
                bool(getattr(self, "kl_convergence_chain", False)),
            )
            # Convergence / KL rows (including the initial emu version) are
            # written inside the loop so baseline chain KL is not left as nan.
            self.convergence_results["kl"] = self._run_final_convergence_loop(
                initial_dchi2=dchi2,
            )
        else:
            logger.info(
                "kl_convergence = F; skipping extra retrain "
                "(MAD not passed: MAD=%.4g > %.4g).",
                dchi2.get("mad_r", float("nan")),
                float(self.delta_chi2_mad_threshold),
            )
            self._log_iw_emu_for_version(emu_version)
            self._append_rose_convergence_row(
                emu_version=emu_version,
                dchi2=dchi2,
                kl_all=float("nan"),
                kl_sel=float("nan"),
                chain_kl_pass=False,
                n_train_added=0,
            )
            self.kl_converged = False
            self.convergence_results["kl"] = {
                "converged": False,
                "reason": "kl_convergence_disabled",
                "n_retrain": 0,
            }

        self._save_convergence_results()

    def _stash_tempered_hpd_state(self) -> None:
        """Preserve tempered-chain arrays used for HPD resampling."""
        chain = getattr(self, "chain", None)
        if chain is None:
            self._hpd_stash = None
            return
        self._hpd_stash = {
            "chain": np.asarray(chain, dtype=float).copy(),
            "unit_chain": (
                np.asarray(self.unit_chain, dtype=float).copy()
                if getattr(self, "unit_chain", None) is not None else None
            ),
            "chain_logpost": (
                np.asarray(self.chain_logpost, dtype=float).copy()
                if getattr(self, "chain_logpost", None) is not None else None
            ),
            "chain_log_weights": (
                np.asarray(self.chain_log_weights, dtype=float).copy()
                if getattr(self, "chain_log_weights", None) is not None else None
            ),
            "chain_tempering": float(getattr(self, "chain_tempering", 1.0)),
        }

    def _restore_tempered_hpd_state(self) -> None:
        """Restore tempered HPD arrays before drawing new training points."""
        stash = getattr(self, "_hpd_stash", None)
        if not stash:
            return
        self.chain = stash["chain"]
        if stash["unit_chain"] is not None:
            self.unit_chain = stash["unit_chain"]
        if stash["chain_logpost"] is not None:
            self.chain_logpost = stash["chain_logpost"]
        if stash["chain_log_weights"] is not None:
            self.chain_log_weights = stash["chain_log_weights"]
        self.chain_tempering = stash["chain_tempering"]

    def _emu_version_tag(self, emu_version: Optional[int] = None) -> str:
        v = int(
            emu_version if emu_version is not None
            else getattr(self, "_current_emu_version", self.iterations + 1)
        )
        return f"emumodel_{v}"

    # ------------------------------------------------------------------
    # (a) Emulator accuracy on all emulated modes
    # ------------------------------------------------------------------
    def _test_emulator_accuracy(self, emu_version: Optional[int] = None) -> Dict[str, Any]:
        """Relative-error percentiles between emulated and true data vectors."""
        param_names = [str(p) for p in self.pipeline.varied_params]
        X = {name: self.sample_test[:, i] for i, name in enumerate(param_names)}
        tag = self._emu_version_tag(emu_version)

        results: Dict[str, Any] = {}
        for name in self.likelihood_names:
            emu = self.emulator[name]
            predictions = np.atleast_2d(emu.predict(X))
            truth = np.asarray(self.sample_data_vectors_test[name])

            if predictions.shape != truth.shape:
                logger.warning(
                    "[%s] Emulator prediction shape %s does not match test data "
                    "vector shape %s; skipping accuracy test for this likelihood.",
                    name, predictions.shape, truth.shape,
                )
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                rel_error = np.abs((predictions - truth) / truth)
            rel_error = np.where(np.isfinite(rel_error), rel_error, np.nan)

            sigma = self._data_vector_sigma(name, n_modes=truth.shape[1])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_cov_error = np.abs((predictions - truth) / sigma)
            rel_cov_error = np.where(
                np.isfinite(rel_cov_error), rel_cov_error, np.nan
            )

            percentiles = np.nanpercentile(rel_error, _ACCURACY_PERCENTILES, axis=0)
            modes = np.asarray(getattr(emu, "modes", np.arange(truth.shape[1])))

            results[name] = {
                "modes": modes,
                "percentile_levels": np.array(_ACCURACY_PERCENTILES),
                "percentiles": percentiles,
                "median_rel_error": float(np.nanmedian(rel_error)),
                "max_rel_error": float(np.nanmax(rel_error)),
            }

            cov_percentiles = np.nanpercentile(rel_cov_error, _ACCURACY_PERCENTILES, axis=0)
            results[name]["cov_percentiles"] = cov_percentiles
            results[name]["median_rel_cov_error"] = float(np.nanmedian(rel_cov_error))
            results[name]["max_rel_cov_error"] = float(np.nanmax(rel_cov_error))

            logger.info(
                "[%s] Accuracy on %d test points (%s): median rel. error=%.3g, "
                "68%%=%.3g, 95%%=%.3g, 99%%=%.3g (max mode-wise)",
                name, len(truth), tag,
                results[name]["median_rel_error"],
                float(np.nanmax(percentiles[0])),
                float(np.nanmax(percentiles[1])),
                float(np.nanmax(percentiles[2])),
            )
            logger.info(
                "[%s] Cov-normalized accuracy (%s): median=%.3g, "
                "68%%=%.3g, 95%%=%.3g, 99%%=%.3g (max mode-wise)",
                name, tag,
                results[name]["median_rel_cov_error"],
                float(np.nanmax(results[name]["cov_percentiles"][0])),
                float(np.nanmax(results[name]["cov_percentiles"][1])),
                float(np.nanmax(results[name]["cov_percentiles"][2])),
            )

            self._plot_accuracy(name, results[name], emu_version=emu_version)
            self._plot_accuracy_cov(name, results[name], emu_version=emu_version)

        return results

    def _data_vector_sigma(self, name: str, n_modes: int) -> Optional[np.ndarray]:
        """Return ``sqrt(diag(C))`` for likelihood ``name``, or ``None`` if unknown."""
        errors = getattr(self, "fiducial_errors", None) or {}
        sigma = errors.get(name)
        if sigma is None:
            inv_cov = (getattr(self, "inv_cov", None) or {}).get(name)
            if inv_cov is None:
                return None
            cov = np.linalg.inv(np.atleast_2d(np.asarray(inv_cov, dtype=float)))
            sigma = np.sqrt(np.diag(cov))

        sigma = np.asarray(sigma, dtype=float).ravel()
        if sigma.shape[0] != n_modes:
            logger.warning(
                "[%s] Data-vector sigma length %d does not match %d modes; "
                "skipping covariance-normalized accuracy.",
                name, sigma.shape[0], n_modes,
            )
            return None
        return sigma

    # ------------------------------------------------------------------
    # (b) Delta chi^2 between true and emulated likelihoods
    # ------------------------------------------------------------------
    def _test_delta_chi2(self, emu_version: Optional[int] = None) -> Dict[str, Any]:
        """Delta chi^2 and debiased scatter on the held-out test set."""
        if self.emu_pipeline is None:
            self.compute_fiducial_setup_emu_pipeline()

        true_like = np.asarray(self.sample_likes_test, dtype=float)

        emu_like = np.full(len(self.sample_test), np.nan)
        for i, p in enumerate(self.sample_test):
            try:
                r = self.emu_pipeline.run_results(p)
                emu_like[i] = r.like
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Emulated pipeline failed on test point %d: %s", i, exc)

        valid = np.isfinite(emu_like) & np.isfinite(true_like)
        chi2_true = -2.0 * true_like
        chi2_emu = -2.0 * emu_like
        delta_chi2 = chi2_emu - chi2_true

        valid_delta = delta_chi2[valid]
        median = float(np.median(valid_delta)) if valid_delta.size else float("nan")
        if valid_delta.size:
            resid = valid_delta - median
            mad_r = float(np.median(np.abs(resid)))
            std_r = float(np.std(resid))
            frac_abs_r_lt_2 = float(np.mean(np.abs(resid) < 2.0))
            max_abs_r = float(np.max(np.abs(resid)))
        else:
            mad_r = std_r = frac_abs_r_lt_2 = max_abs_r = float("nan")

        thresh = float(getattr(self, "delta_chi2_mad_threshold", 1.0))
        max_cap = float(getattr(self, "delta_chi2_max_abs_r", 0.0))
        mad_ok = bool(np.isfinite(mad_r) and mad_r <= thresh)
        # Optional hard cap: catch rare catastrophic outliers MAD can miss.
        # <= 0 disables the cap (backward-compatible default).
        if max_cap > 0:
            max_ok = bool(np.isfinite(max_abs_r) and max_abs_r <= max_cap)
        else:
            max_ok = True
        mad_pass = bool(mad_ok and max_ok)

        results = {
            "delta_chi2": delta_chi2,
            "chi2_true": chi2_true,
            "chi2_emu": chi2_emu,
            "n_valid": int(valid.sum()),
            "mean": float(np.mean(valid_delta)) if valid_delta.size else float("nan"),
            "std": float(np.std(valid_delta)) if valid_delta.size else float("nan"),
            "median": median,
            "abs_median": float(np.median(np.abs(valid_delta))) if valid_delta.size else float("nan"),
            "abs_max": float(np.max(np.abs(valid_delta))) if valid_delta.size else float("nan"),
            "mad_r": mad_r,
            "std_r": std_r,
            "frac_abs_r_lt_2": frac_abs_r_lt_2,
            "max_abs_r": max_abs_r,
            "mad_threshold": thresh,
            "max_abs_r_threshold": max_cap,
            "mad_pass": mad_pass,
        }

        tag = self._emu_version_tag(emu_version)
        if max_cap > 0:
            logger.info(
                "Delta chi^2 over %d valid test points (%s): mean=%.3g, "
                "median=%.3g, MAD(r)=%.3g (thresh %.3g), max|r|=%.3g "
                "(cap %.3g), std(r)=%.3g, frac(|r|<2)=%.3g, mad_pass=%s",
                results["n_valid"], tag, results["mean"], median,
                mad_r, thresh, max_abs_r, max_cap, std_r, frac_abs_r_lt_2, mad_pass,
            )
        else:
            logger.info(
                "Delta chi^2 over %d valid test points (%s): mean=%.3g, "
                "median=%.3g, MAD(r)=%.3g, max|r|=%.3g, std(r)=%.3g, "
                "frac(|r|<2)=%.3g, mad_pass=%s",
                results["n_valid"], tag, results["mean"], median,
                mad_r, max_abs_r, std_r, frac_abs_r_lt_2, mad_pass,
            )

        self._plot_delta_chi2(results, emu_version=emu_version)
        return results

    def _append_rose_convergence_row(
        self,
        emu_version: int,
        dchi2: Dict[str, Any],
        kl_all: float,
        kl_sel: float,
        chain_kl_pass: bool,
        n_train_added: int,
    ) -> None:
        """Append one row to ``{save_dir}/rose_convergence.txt``."""
        path = os.path.join(self.save_dir, "rose_convergence.txt")
        write_header = not os.path.isfile(path)
        mad_pass = bool(dchi2.get("mad_pass", False))
        with open(path, "a") as f:
            if write_header:
                f.write(
                    "emu_version\tn_test\tmad_r\tstd_r\tfrac_abs_r_lt_2\t"
                    "max_abs_r\tmedian_dchi2\tmad_pass\tkl_all\tkl_sel\t"
                    "chain_kl_pass\tn_train_added\n"
                )
            f.write(
                f"{emu_version}\t{int(dchi2.get('n_valid', 0))}\t"
                f"{float(dchi2.get('mad_r', float('nan'))):.6e}\t"
                f"{float(dchi2.get('std_r', float('nan'))):.6e}\t"
                f"{float(dchi2.get('frac_abs_r_lt_2', float('nan'))):.6e}\t"
                f"{float(dchi2.get('max_abs_r', float('nan'))):.6e}\t"
                f"{float(dchi2.get('median', float('nan'))):.6e}\t"
                f"{int(mad_pass)}\t"
                f"{float(kl_all):.6e}\t{float(kl_sel):.6e}\t"
                f"{int(bool(chain_kl_pass))}\t{int(n_train_added)}\n"
            )
        logger.info("Appended convergence diagnostics to %s", path)

    # ------------------------------------------------------------------
    # (c) Final-stage MAD (+ optional chain KL) retrain loop
    # ------------------------------------------------------------------
    def _run_final_convergence_loop(self, initial_dchi2: Dict[str, Any]) -> Dict[str, Any]:
        """Retrain until MAD and/or selected chain-KL criteria are met.

        The held-out test set is never folded into training. New points are
        drawn from the stashed tempered HPD. Optional untempered
        ``final_sampler`` chains drive a Jeffreys KL stop on ``kl_params``.
        Passive IW KL between consecutive emulators is logged to ``rose_kl.txt``.

        For the first final-stage emulator (after the last tempered train):
        - ``iw_emu`` compares it to the previous on-disk emulator
        - ``untempered_chain`` KL (if enabled) is vs the last tempered chain
          (importance-reweighted to the untempered posterior)
        """
        threshold = float(self.kl_threshold)
        max_retrain = int(self.kl_max_retrain)
        use_chains = bool(getattr(self, "kl_convergence_chain", False))
        timing_iteration = int(self.max_iterations)
        emu_version = int(getattr(self, "_current_emu_version", self.iterations + 1))

        theta_ref, logq = self._kl_reference_samples_from_hpd_stash()
        # IW: current emu vs previous on-disk model (natural pair for this version).
        logp_curr = self._log_iw_emu_for_version(
            emu_version, theta_ref=theta_ref, logq=logq,
        )
        logp_prev = logp_curr

        prev_chain = None
        prev_logw = None
        kl_all = kl_sel = float("nan")
        if use_chains:
            logger.info(
                "Running baseline untempered %s chain on %s for chain-KL",
                self.final_sampler, self._emu_version_tag(emu_version),
            )
            t_samp = self._run_untempered_convergence_chain(emu_version)
            self._append_timing_row(timing_iteration, 0.0, 0.0, t_samp)
            curr_chain = np.asarray(self.chain, dtype=float).copy()
            self._final_chain_already_sampled = True

            # Baseline KL vs last tempered chain (untempered via importance weights).
            tempered_chain, tempered_logw = self._tempered_stash_chain_and_logw()
            kl_knn_all = kl_knn_sel = float("nan")
            if (
                tempered_chain is not None
                and len(tempered_chain) > 1
                and len(curr_chain) > 1
            ):
                (
                    kl_all, kl_knn_all, kl_sel, kl_knn_sel, _
                ) = self._chain_kl_metrics_between(
                    tempered_chain, curr_chain,
                    logw_a=tempered_logw, logw_b=None,
                )
                logger.info(
                    "Baseline chain Jeffreys KL (%s vs last tempered): "
                    "all=%.4g, sel=%.4g",
                    self._emu_version_tag(emu_version), kl_all, kl_sel,
                )
            else:
                logger.warning(
                    "No usable tempered stash for baseline chain KL on %s",
                    self._emu_version_tag(emu_version),
                )
            self._append_rose_kl_row(
                iteration=emu_version,
                source="untempered_chain",
                kl_gauss_all=kl_all,
                kl_knn_all=kl_knn_all,
                kl_gauss_sel=kl_sel,
                kl_knn_sel=kl_knn_sel,
                ess=float(len(curr_chain)),
                tempering=1.0,
            )
            prev_chain = curr_chain
            prev_logw = None

        self._append_rose_convergence_row(
            emu_version=emu_version,
            dchi2=initial_dchi2,
            kl_all=kl_all,
            kl_sel=kl_sel,
            chain_kl_pass=False,
            n_train_added=0,
        )

        attempt = 0
        converged = False
        reason = "max_retrain"
        dchi2 = initial_dchi2
        mad_history = [float(initial_dchi2.get("mad_r", float("nan")))]
        kl_sel_history = []

        while attempt < max_retrain:
            attempt += 1
            self._restore_tempered_hpd_state()
            t0 = time.perf_counter()
            n_extra = self._add_training_points(self.kl_extra_size)
            time_training_set_s = time.perf_counter() - t0
            emu_version += 1
            logger.info(
                "Convergence retrain %d/%d: added %d training points (%s)",
                attempt, max_retrain, n_extra, self._emu_version_tag(emu_version),
            )
            t1 = time.perf_counter()
            self.train_emulator(model_version=emu_version)
            time_train_emulator_s = time.perf_counter() - t1
            self._append_timing_row(
                timing_iteration, time_training_set_s, time_train_emulator_s, 0.0,
            )

            self.convergence_results["accuracy"] = self._test_emulator_accuracy(
                emu_version=emu_version
            )
            dchi2 = self._test_delta_chi2(emu_version=emu_version)
            self.convergence_results["delta_chi2"] = dchi2
            mad_history.append(float(dchi2.get("mad_r", float("nan"))))

            # Passive IW KL (all + selected) -> rose_kl.txt
            if theta_ref is not None and logp_prev is not None:
                logp_curr = self._emulated_log_posterior(theta_ref)
                kl_iw_all, ess_all = self._gaussian_kl_via_importance(
                    theta_ref, logp_prev, logq, logp_curr,
                    param_indices=self._kl_all_param_indices(),
                )
                kl_iw_sel, ess_sel = self._gaussian_kl_via_importance(
                    theta_ref, logp_prev, logq, logp_curr,
                    param_indices=self._kl_selected_param_indices(),
                )
                self._append_rose_kl_row(
                    iteration=emu_version,
                    source="iw_emu",
                    kl_gauss_all=kl_iw_all,
                    kl_knn_all=float("nan"),
                    kl_gauss_sel=kl_iw_sel,
                    kl_knn_sel=float("nan"),
                    ess=min(ess_all, ess_sel),
                    tempering=1.0,
                )
                logp_prev = logp_curr
                logger.info(
                    "Passive IW KL after %s: all=%.4g, sel=%.4g (ESS~%.0f)",
                    self._emu_version_tag(emu_version), kl_iw_all, kl_iw_sel,
                    min(ess_all, ess_sel),
                )
            elif theta_ref is not None:
                # Fallback if baseline IW failed (e.g. missing previous model).
                logp_prev = self._emulated_log_posterior(theta_ref)

            kl_all = kl_sel = float("nan")
            chain_kl_pass = False
            if use_chains:
                logger.info(
                    "Running untempered %s chain on %s",
                    self.final_sampler, self._emu_version_tag(emu_version),
                )
                t_samp = self._run_untempered_convergence_chain(emu_version)
                self._append_timing_row(timing_iteration, 0.0, 0.0, t_samp)
                curr_chain = np.asarray(self.chain, dtype=float).copy()
                self._final_chain_already_sampled = True
                kl_knn_all = kl_knn_sel = float("nan")
                if prev_chain is not None and len(prev_chain) > 1 and len(curr_chain) > 1:
                    (
                        kl_all, kl_knn_all, kl_sel, kl_knn_sel, _
                    ) = self._chain_kl_metrics_between(
                        prev_chain, curr_chain,
                        logw_a=prev_logw, logw_b=None,
                    )
                    chain_kl_pass = bool(np.isfinite(kl_sel) and kl_sel < threshold)
                    kl_sel_history.append(kl_sel)
                    logger.info(
                        "Chain Jeffreys KL (%s vs previous): all=%.4g, sel=%.4g "
                        "(threshold %.4g, pass=%s)",
                        self._emu_version_tag(emu_version), kl_all, kl_sel,
                        threshold, chain_kl_pass,
                    )
                self._append_rose_kl_row(
                    iteration=emu_version,
                    source="untempered_chain",
                    kl_gauss_all=kl_all,
                    kl_knn_all=kl_knn_all,
                    kl_gauss_sel=kl_sel,
                    kl_knn_sel=kl_knn_sel,
                    ess=float(len(curr_chain)),
                    tempering=1.0,
                )
                prev_chain = curr_chain
                prev_logw = None

            self._append_rose_convergence_row(
                emu_version=emu_version,
                dchi2=dchi2,
                kl_all=kl_all,
                kl_sel=kl_sel,
                chain_kl_pass=chain_kl_pass,
                n_train_added=n_extra,
            )

            mad_pass = bool(dchi2.get("mad_pass", False))
            if mad_pass:
                converged = True
                reason = "mad_pass"
                break
            if use_chains and chain_kl_pass:
                converged = True
                reason = "chain_kl_pass"
                break

        if converged:
            logger.info(
                "Final-stage convergence reached (%s) after %d retrain(s).",
                reason, attempt,
            )
        else:
            logger.warning(
                "Final-stage convergence NOT reached after %d retrain(s) "
                "(last MAD=%.4g, last kl_sel=%.4g). Proceeding with last emulator.",
                attempt,
                dchi2.get("mad_r", float("nan")),
                kl_sel if np.isfinite(kl_sel) else float("nan"),
            )

        self.kl_converged = bool(converged)
        return {
            "converged": bool(converged),
            "reason": reason,
            "n_retrain": int(attempt),
            "mad_history": np.array(mad_history, dtype=float),
            "kl_sel_history": np.array(kl_sel_history, dtype=float),
            "threshold": threshold,
            "mad_threshold": float(self.delta_chi2_mad_threshold),
        }

    def _run_untempered_convergence_chain(self, emu_version: int) -> float:
        """Run untempered ``final_sampler`` and save as ``_from_emumodel_{v}``."""
        self._chain_save_suffix = f"_from_emumodel_{int(emu_version)}"
        try:
            return float(self._run_untempered_final_sampler())
        finally:
            self._chain_save_suffix = None

    def _run_untempered_final_sampler(self) -> float:
        """Run ``final_sampler`` at tempering=1 and return sampling wall time."""
        tempering = 1.0
        sampler = getattr(self, "final_sampler", "emcee")
        if sampler == "nuts":
            return float(self._run_nuts_sampling(tempering))
        if sampler == "numpyro":
            return float(self._run_numpyro_sampling(tempering))
        if sampler == "nautilus":
            return float(self._run_nautilus_sampling(tempering))
        return float(self._run_emcee_sampling(tempering))

    def _tempered_stash_chain_and_logw(self):
        """Return (chain, untempered log-weights) from the tempered HPD stash."""
        stash = getattr(self, "_hpd_stash", None)
        if not stash or stash.get("chain") is None or len(stash["chain"]) < 2:
            return None, None
        chain = np.asarray(stash["chain"], dtype=float)
        n = len(chain)
        logw = stash.get("chain_log_weights")
        if logw is None or len(logw) != n:
            logw = np.zeros(n, dtype=float)
        else:
            logw = np.asarray(logw, dtype=float).copy()
        tempering = float(stash.get("chain_tempering", 1.0))
        logpost = stash.get("chain_logpost")
        if abs(tempering - 1.0) > 1e-12:
            if logpost is None or len(logpost) != n:
                logger.warning(
                    "Stashed chain_logpost unavailable; cannot reweight "
                    "tempering=%.4g for baseline chain KL.",
                    tempering,
                )
            else:
                logw = logw + (1.0 - tempering) * np.asarray(logpost, dtype=float)
        return chain, logw

    def _emulated_log_posterior_at_version(
        self, theta: np.ndarray, emu_version: int
    ) -> Optional[np.ndarray]:
        """Evaluate emulated log-posterior under a previously saved emulator.

        Temporarily loads ``emumodel_{emu_version}`` from ``save_dir``, then
        restores the currently loaded emulator in memory.
        """
        path = os.path.join(self.save_dir, f"emumodel_{int(emu_version)}")
        if not os.path.isdir(path):
            logger.warning(
                "Cannot evaluate log-posterior for %s: directory not found",
                path,
            )
            return None
        saved_emu = getattr(self, "emulator", None)
        saved_indices = getattr(self, "emulator_output_indices", None)
        saved_loaded_path = getattr(self, "_loaded_emu_path", None)
        if saved_emu is None:
            logger.warning(
                "No current emulator to restore after loading %s; skipping",
                path,
            )
            return None
        try:
            self.load_emulator(path)
            return self._emulated_log_posterior(theta)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to load/evaluate %s for IW KL: %s", path, exc,
            )
            return None
        finally:
            self.emulator = saved_emu
            if saved_indices is not None:
                self.emulator_output_indices = saved_indices
            self._loaded_emu_path = saved_loaded_path
            if getattr(self, "emu_module", None) is not None:
                self.emu_module.data.set_emulator(saved_emu)
                if saved_indices is not None:
                    self.emu_module.data.output_indices = saved_indices

    def _log_iw_emu_for_version(
        self,
        emu_version: int,
        theta_ref: Optional[np.ndarray] = None,
        logq: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Log passive IW KL of ``emumodel_{v}`` vs ``emumodel_{v-1}``.

        Returns the current emulator's log-posterior on ``theta_ref`` (for
        chaining into the next IW comparison), or ``None`` if unavailable.
        """
        if theta_ref is None or logq is None:
            theta_ref, logq = self._kl_reference_samples_from_hpd_stash()
        if theta_ref is None or logq is None:
            logger.warning(
                "No tempered reference samples; skipping IW KL for %s",
                self._emu_version_tag(emu_version),
            )
            return None

        logp_curr = self._emulated_log_posterior(theta_ref)
        prev_version = int(emu_version) - 1
        if prev_version < 1:
            logger.info(
                "No previous emulator before %s; skipping IW KL",
                self._emu_version_tag(emu_version),
            )
            return logp_curr

        logp_prev = self._emulated_log_posterior_at_version(theta_ref, prev_version)
        if logp_prev is None:
            logger.warning(
                "Skipping IW KL for %s (could not evaluate previous emulator)",
                self._emu_version_tag(emu_version),
            )
            return logp_curr

        kl_iw_all, ess_all = self._gaussian_kl_via_importance(
            theta_ref, logp_prev, logq, logp_curr,
            param_indices=self._kl_all_param_indices(),
        )
        kl_iw_sel, ess_sel = self._gaussian_kl_via_importance(
            theta_ref, logp_prev, logq, logp_curr,
            param_indices=self._kl_selected_param_indices(),
        )
        self._append_rose_kl_row(
            iteration=int(emu_version),
            source="iw_emu",
            kl_gauss_all=kl_iw_all,
            kl_knn_all=float("nan"),
            kl_gauss_sel=kl_iw_sel,
            kl_knn_sel=float("nan"),
            ess=min(ess_all, ess_sel),
            tempering=1.0,
        )
        logger.info(
            "Passive IW KL for %s vs %s: all=%.4g, sel=%.4g (ESS~%.0f)",
            self._emu_version_tag(emu_version),
            self._emu_version_tag(prev_version),
            kl_iw_all, kl_iw_sel, min(ess_all, ess_sel),
        )
        return logp_curr

    def _kl_reference_samples_from_hpd_stash(self):
        """IW reference samples from the stashed tempered chain (not overwritten)."""
        stash = getattr(self, "_hpd_stash", None)
        if not stash or stash.get("chain") is None or len(stash["chain"]) < 2:
            return None, None
        chain = np.asarray(stash["chain"], dtype=float)
        logpost = stash.get("chain_logpost")
        if logpost is None or len(logpost) != len(chain):
            logpost = np.zeros(len(chain))
            tempering_prev = 1.0
        else:
            logpost = np.asarray(logpost, dtype=float)
            tempering_prev = float(stash.get("chain_tempering", 1.0))
        n = len(chain)
        n_sub = min(int(self.kl_n_samples), n)
        idx = np.random.choice(n, size=n_sub, replace=False)
        return chain[idx], tempering_prev * logpost[idx]

    def _chain_kl_metrics_between(
        self,
        chain_a: np.ndarray,
        chain_b: np.ndarray,
        logw_a: Optional[np.ndarray] = None,
        logw_b: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float, float]:
        """Gaussian + kNN Jeffreys KL (all + selected) between two chains.

        Optional log-weights reweight each chain to the untempered posterior
        (used when comparing a tempered stash to an untempered chain).
        Returns ``(kl_gauss_all, kl_knn_all, kl_gauss_sel, kl_knn_sel, ess)``.
        """
        chain_a = np.asarray(chain_a, dtype=float)
        chain_b = np.asarray(chain_b, dtype=float)
        n_a, n_b = len(chain_a), len(chain_b)
        if n_a < 2 or n_b < 2:
            return (
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            )

        if logw_a is None or len(logw_a) != n_a:
            logw_a = np.zeros(n_a, dtype=float)
        else:
            logw_a = np.asarray(logw_a, dtype=float)
        if logw_b is None or len(logw_b) != n_b:
            logw_b = np.zeros(n_b, dtype=float)
        else:
            logw_b = np.asarray(logw_b, dtype=float)

        def _one(indices: Sequence[int]) -> Tuple[float, float, float]:
            idx = np.asarray(indices, dtype=int)
            a = chain_a[:, idx]
            b = chain_b[:, idx]
            moments_a = self._weighted_moments(a, logw_a)
            moments_b = self._weighted_moments(b, logw_b)
            if moments_a is None or moments_b is None:
                return float("nan"), float("nan"), float("nan")
            mean_a, cov_a, ess_a = moments_a
            mean_b, cov_b, ess_b = moments_b
            kl_gauss = 0.5 * (
                _gaussian_kl(mean_a, cov_a, mean_b, cov_b)
                + _gaussian_kl(mean_b, cov_b, mean_a, cov_a)
            )
            n_sub = min(int(getattr(self, "kl_n_samples", 2000)), n_a, n_b)
            samp_a = self._resample_equal_weight(a, logw_a, cov_a, n_sub)
            samp_b = self._resample_equal_weight(b, logw_b, cov_b, n_sub)
            kl_knn = _knn_kl_symmetric(
                samp_a, samp_b,
                k=int(getattr(self, "kl_knn_k", 3)),
                debias=bool(getattr(self, "kl_knn_debias", True)),
            )
            return float(kl_gauss), float(kl_knn), float(min(ess_a, ess_b))

        kl_g_all, kl_k_all, ess = _one(self._kl_all_param_indices())
        kl_g_sel, kl_k_sel, _ = _one(self._kl_selected_param_indices())
        return kl_g_all, kl_k_all, kl_g_sel, kl_k_sel, ess

    def _jeffreys_kl_between_chains(
        self,
        chain_a: np.ndarray,
        chain_b: np.ndarray,
        param_indices: Sequence[int],
    ) -> float:
        """Symmetrized Gaussian KL between two equal-weight MCMC chains."""
        idx = np.asarray(param_indices, dtype=int)
        a = np.asarray(chain_a, dtype=float)[:, idx]
        b = np.asarray(chain_b, dtype=float)[:, idx]
        if len(a) < idx.size + 2 or len(b) < idx.size + 2:
            return float("nan")
        mean_a, cov_a = a.mean(axis=0), np.cov(a.T, ddof=1)
        mean_b, cov_b = b.mean(axis=0), np.cov(b.T, ddof=1)
        cov_a = np.atleast_2d(cov_a) + np.eye(len(idx)) * 1e-12
        cov_b = np.atleast_2d(cov_b) + np.eye(len(idx)) * 1e-12
        return 0.5 * (
            _gaussian_kl(mean_a, cov_a, mean_b, cov_b)
            + _gaussian_kl(mean_b, cov_b, mean_a, cov_a)
        )

    # ------------------------------------------------------------------
    # Per-iteration KL divergence between consecutive MCMC chains
    # ------------------------------------------------------------------
    def _untempered_log_weights(self, chain: np.ndarray) -> np.ndarray:
        """Log importance weights that map the current chain to the untempered posterior."""
        n = len(chain)
        logw = getattr(self, "chain_log_weights", None)
        if logw is None or len(logw) != n:
            logw = np.zeros(n, dtype=float)
        else:
            logw = np.asarray(logw, dtype=float).copy()

        tempering = float(getattr(self, "chain_tempering", 1.0))
        logpost = getattr(self, "chain_logpost", None)
        if abs(tempering - 1.0) > 1e-12:
            if logpost is None or len(logpost) != n:
                logger.warning(
                    "chain_logpost unavailable; cannot reweight tempering=%.4g "
                    "to the untempered posterior for the per-iteration KL.",
                    tempering,
                )
            else:
                logpost = np.asarray(logpost, dtype=float)
                logw = logw + (1.0 - tempering) * logpost
        return logw

    def _resample_equal_weight(self, chain: np.ndarray, logw: np.ndarray,
                               cov: np.ndarray, n_sub: int) -> np.ndarray:
        """Build an equal-weight subsample for the k-NN KL estimator."""
        finite = np.isfinite(logw)
        weighted = bool(finite.any() and np.ptp(logw[finite]) > 0)
        n_sub = min(n_sub, len(chain))
        if not weighted:
            idx = np.random.choice(len(chain), size=n_sub, replace=False)
            return chain[idx]

        w = logw - np.max(logw[finite])
        w = np.where(np.isfinite(w), np.exp(w), 0.0)
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            idx = np.random.choice(len(chain), size=n_sub, replace=False)
            return chain[idx]

        idx = np.random.choice(len(chain), size=n_sub, replace=True, p=w / w_sum)
        samples = chain[idx].astype(float)
        jitter_scale = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        jitter_scale = np.where(jitter_scale > 0, jitter_scale, 1.0)
        samples = samples + np.random.normal(size=samples.shape) * (1e-3 * jitter_scale)
        return samples

    def _record_iteration_kl(self):
        """Compute and persist per-iteration chain-to-chain KL divergences.

        Reports Gaussian and k-NN Jeffreys KL for all parameters and for the
        selected ``kl_params`` subspace. Appends one row to ``rose_kl.txt``.
        """
        chain = getattr(self, "chain", None)
        if chain is None or len(chain) < 2:
            logger.warning(
                "No usable MCMC chain available; skipping per-iteration KL divergence."
            )
            return float("nan"), float("nan")

        chain = np.asarray(chain, dtype=float)
        logw = self._untempered_log_weights(chain)
        tempering = float(getattr(self, "chain_tempering", 1.0))

        kl_gauss_all, kl_knn_all, ess = self._chain_kl_pair(
            chain, logw, self._kl_all_param_indices(), store_key="all"
        )
        kl_gauss_sel, kl_knn_sel, _ = self._chain_kl_pair(
            chain, logw, self._kl_selected_param_indices(), store_key="sel"
        )

        self._append_rose_kl_row(
            iteration=self.iterations + 1,
            source="tempered_chain",
            kl_gauss_all=kl_gauss_all,
            kl_knn_all=kl_knn_all,
            kl_gauss_sel=kl_gauss_sel,
            kl_knn_sel=kl_knn_sel,
            ess=ess,
            tempering=tempering,
        )

        logger.info(
            "Iteration %d chain-to-chain KL (untempered): "
            "all Gaussian=%.4g kNN=%.4g; sel Gaussian=%.4g kNN=%.4g; "
            "ESS=%.0f, T=%.4g",
            self.iterations + 1, kl_gauss_all, kl_knn_all,
            kl_gauss_sel, kl_knn_sel, ess, tempering,
        )
        return float(kl_gauss_sel), float(kl_knn_sel)

    def _chain_kl_pair(
        self,
        chain: np.ndarray,
        logw: np.ndarray,
        param_indices: Sequence[int],
        store_key: str,
    ) -> Tuple[float, float, float]:
        """Gaussian + kNN KL vs previous stored moments for one subspace."""
        idx = np.asarray(param_indices, dtype=int)
        chain_kl = chain[:, idx]
        moments = self._weighted_moments(chain_kl, logw)
        if moments is None:
            return float("nan"), float("nan"), float("nan")
        mean, cov, ess = moments
        n_sub = min(int(getattr(self, "kl_n_samples", 2000)), len(chain_kl))
        chain_samples = self._resample_equal_weight(chain_kl, logw, cov, n_sub)

        prev_attr = f"_kl_prev_chain_moments_{store_key}"
        prev_samp_attr = f"_kl_prev_chain_samples_{store_key}"
        prev_moments = getattr(self, prev_attr, None)
        prev_samples = getattr(self, prev_samp_attr, None)

        kl_gauss = float("nan")
        kl_knn = float("nan")
        if prev_moments is not None:
            mean_prev, cov_prev = prev_moments
            if mean_prev.shape == mean.shape:
                kl_gauss = 0.5 * (
                    _gaussian_kl(mean_prev, cov_prev, mean, cov)
                    + _gaussian_kl(mean, cov, mean_prev, cov_prev)
                )
        if prev_samples is not None and prev_samples.shape[1] == chain_samples.shape[1]:
            kl_knn = _knn_kl_symmetric(
                prev_samples, chain_samples,
                k=int(getattr(self, "kl_knn_k", 3)),
                debias=bool(getattr(self, "kl_knn_debias", True)),
            )

        setattr(self, prev_attr, (mean, cov))
        setattr(self, prev_samp_attr, chain_samples)
        return float(kl_gauss), float(kl_knn), float(ess)

    def _append_rose_kl_row(
        self,
        iteration: int,
        source: str,
        kl_gauss_all: float,
        kl_knn_all: float,
        kl_gauss_sel: float,
        kl_knn_sel: float,
        ess: float,
        tempering: float,
    ) -> None:
        kl_file = os.path.join(self.save_dir, "rose_kl.txt")
        write_header = not os.path.isfile(kl_file)
        with open(kl_file, "a") as f:
            if write_header:
                f.write(
                    "iteration\tsource\tkl_gaussian_all\tkl_knn_all\t"
                    "kl_gaussian_sel\tkl_knn_sel\tess\ttempering\n"
                )
            f.write(
                f"{iteration}\t{source}\t{kl_gauss_all:.6e}\t{kl_knn_all:.6e}\t"
                f"{kl_gauss_sel:.6e}\t{kl_knn_sel:.6e}\t"
                f"{ess:.1f}\t{tempering:.6g}\n"
            )

    def _kl_all_param_indices(self) -> np.ndarray:
        return np.arange(len(self.pipeline.varied_params), dtype=int)

    def _kl_selected_param_indices(self) -> np.ndarray:
        idx = getattr(self, "kl_param_indices", None)
        if idx is None or len(idx) == 0:
            return self._kl_active_param_indices()
        return np.asarray(idx, dtype=int)

    def _kl_active_param_indices(self) -> np.ndarray:
        """Indices kept after ``kl_exclude_params`` (legacy default selected set)."""
        ndim = len(self.pipeline.varied_params)
        exclude = set(getattr(self, "kl_exclude_param_indices", None) or [])
        keep = [i for i in range(ndim) if i not in exclude]
        if not keep:
            raise ValueError(
                "No parameters remain for KL after kl_exclude_params / "
                "resample_weak_params exclusions"
            )
        return np.asarray(keep, dtype=int)

    def _project_theta_for_kl(
        self, theta: np.ndarray, param_indices: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Project ``theta`` onto KL subspace indices."""
        theta = np.asarray(theta, dtype=float)
        idx = (
            np.asarray(param_indices, dtype=int)
            if param_indices is not None
            else self._kl_selected_param_indices()
        )
        if theta.ndim == 1:
            return theta[idx]
        if theta.shape[1] == len(self.pipeline.varied_params):
            return theta[:, idx]
        if theta.shape[1] == len(idx):
            return theta
        raise ValueError(
            f"Cannot project theta of shape {theta.shape} for KL "
            f"(ndim={len(self.pipeline.varied_params)}, keep={len(idx)})"
        )

    def _kl_reference_samples(self):
        """Return (theta, logq) reference samples for importance reweighting."""
        chain = getattr(self, "chain", None)
        if chain is None or len(chain) < 2:
            return None, None
        chain = np.asarray(chain)

        logpost = getattr(self, "chain_logpost", None)
        if logpost is None or len(logpost) != len(chain):
            logger.warning(
                "chain_logpost unavailable; assuming a uniform sampling density "
                "for the KL importance weights (results may be less accurate)."
            )
            logpost = np.zeros(len(chain))
            tempering_prev = 1.0
        else:
            logpost = np.asarray(logpost, dtype=float)
            prev_idx = max(0, self.iterations - 1)
            try:
                tempering_prev = float(self.tempering[prev_idx])
            except (IndexError, TypeError):
                tempering_prev = 1.0

        n = len(chain)
        n_sub = min(int(self.kl_n_samples), n)
        idx = np.random.choice(n, size=n_sub, replace=False)
        theta = chain[idx]
        logq = tempering_prev * logpost[idx]
        return theta, logq

    def _emulated_log_posterior(self, theta: np.ndarray) -> np.ndarray:
        """Emulated (untempered) log-posterior at each row of ``theta``."""
        if self.emu_pipeline is None:
            self.compute_fiducial_setup_emu_pipeline()
        logp = np.full(len(theta), -np.inf)
        for i, p in enumerate(theta):
            try:
                r = self.emu_pipeline.run_results(p)
                logp[i] = r.post
            except Exception:  # pragma: no cover - defensive
                pass
        return logp

    def _append_test_points_to_training(self) -> int:
        """Append the already-evaluated test points to the training arrays.

        Kept for optional manual use; the final-stage convergence loop no longer
        folds the held-out test set into training.
        """
        n_added = len(self.sample_test)
        if n_added == 0:
            return 0
        self.sample = np.vstack([self.sample, self.sample_test])
        self.unit_sample = np.vstack([self.unit_sample, self.unit_sample_test])
        for name in self.sample_data_vectors:
            self.sample_data_vectors[name] = np.vstack(
                [self.sample_data_vectors[name], self.sample_data_vectors_test[name]]
            )
        self.sample_likes = np.concatenate([self.sample_likes, self.sample_likes_test])
        self.sample_priors = np.concatenate([self.sample_priors, self.sample_priors_test])
        self.sample_posts = np.concatenate([self.sample_posts, self.sample_posts_test])
        self.points_per_iteration = np.concatenate(
            [self.points_per_iteration, np.array([n_added])]
        )
        return n_added

    def _add_training_points(self, n_points: int) -> int:
        """Draw, evaluate, and append ``n_points`` new training points from tempered HPD."""
        from .data_processing import _task_wrapper
        from .utils import task

        sample, unit_sample = self._sample_hpd_stratified_volume_lh(
            n_points,
            seed=None if self.seed is None else int(self.seed) + 4242 + int(
                getattr(self, "_current_emu_version", 0)
            ),
        )

        if self.pool:
            results = self.pool.map(_task_wrapper, sample)
        else:
            results = [task(p, self) for p in sample]

        before = len(self.sample)
        self._update_training_set(results, sample, unit_sample)
        return len(self.sample) - before

    @staticmethod
    def _weighted_moments(theta: np.ndarray, logw: np.ndarray):
        """Return (mean, cov, ess) of ``theta`` under self-normalized weights."""
        logw = np.asarray(logw, dtype=float)
        finite = np.isfinite(logw)
        if finite.sum() < theta.shape[1] + 1:
            return None
        theta = theta[finite]
        logw = logw[finite]
        logw -= np.max(logw)
        w = np.exp(logw)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            return None
        w /= w_sum
        ess = 1.0 / np.sum(w ** 2)
        mean = w @ theta
        diff = theta - mean
        cov = (diff * w[:, None]).T @ diff
        denom = 1.0 - np.sum(w ** 2)
        if denom > 0:
            cov /= denom
        cov += np.eye(cov.shape[0]) * 1e-12
        return mean, cov, ess

    def _gaussian_kl_via_importance(
        self, theta, logp_a, logq, logp_b, param_indices: Optional[Sequence[int]] = None
    ):
        """Symmetrized Gaussian KL between two emulated posteriors via IW."""
        theta_kl = self._project_theta_for_kl(theta, param_indices=param_indices)
        moments_a = self._weighted_moments(theta_kl, logp_a - logq)
        moments_b = self._weighted_moments(theta_kl, logp_b - logq)
        if moments_a is None or moments_b is None:
            return float("inf"), 0.0

        mean_a, cov_a, ess_a = moments_a
        mean_b, cov_b, ess_b = moments_b

        kl_sym = 0.5 * (
            _gaussian_kl(mean_a, cov_a, mean_b, cov_b)
            + _gaussian_kl(mean_b, cov_b, mean_a, cov_a)
        )
        min_ess = min(ess_a, ess_b)
        if min_ess < 50:
            logger.warning(
                "Low effective sample size (%.0f) in KL importance reweighting; "
                "the KL estimate may be noisy.",
                min_ess,
            )
        return float(kl_sym), float(min_ess)

    # ------------------------------------------------------------------
    # Saving / plotting helpers
    # ------------------------------------------------------------------
    def _save_convergence_results(self) -> None:
        """Persist convergence diagnostics to an npz file."""
        save_dict: Dict[str, Any] = {}

        accuracy = self.convergence_results.get("accuracy", {})
        for name, res in accuracy.items():
            save_dict[f"accuracy/{name}/modes"] = res["modes"]
            save_dict[f"accuracy/{name}/percentile_levels"] = res["percentile_levels"]
            save_dict[f"accuracy/{name}/percentiles"] = res["percentiles"]
            save_dict[f"accuracy/{name}/median_rel_error"] = res["median_rel_error"]
            save_dict[f"accuracy/{name}/max_rel_error"] = res["max_rel_error"]
            if "cov_percentiles" in res:
                save_dict[f"accuracy/{name}/cov_percentiles"] = res["cov_percentiles"]
                save_dict[f"accuracy/{name}/median_rel_cov_error"] = res[
                    "median_rel_cov_error"
                ]
                save_dict[f"accuracy/{name}/max_rel_cov_error"] = res[
                    "max_rel_cov_error"
                ]

        dchi2 = self.convergence_results.get("delta_chi2", {})
        for key in (
            "delta_chi2", "chi2_true", "chi2_emu",
            "mean", "std", "median", "abs_median", "abs_max", "n_valid",
            "mad_r", "std_r", "frac_abs_r_lt_2", "max_abs_r",
            "mad_threshold", "max_abs_r_threshold", "mad_pass",
        ):
            if key in dchi2:
                save_dict[f"delta_chi2/{key}"] = dchi2[key]

        kl = self.convergence_results.get("kl", {})
        for key in (
            "converged", "reason", "n_retrain", "threshold", "mad_threshold",
            "mad_history", "kl_sel_history",
        ):
            if key in kl:
                save_dict[f"kl/{key}"] = kl[key]

        out_path = os.path.join(self.convergence_dir, "convergence_tests.npz")
        np.savez(out_path, **save_dict)
        logger.info("Saved convergence diagnostics to %s", out_path)

    def _plot_accuracy(
        self, name: str, result: Dict[str, Any], emu_version: Optional[int] = None
    ) -> None:
        """Save an accuracy percentile figure (mirrors salmon_plot.py)."""
        plt = _get_pyplot()
        if plt is None:
            return

        modes = result["modes"]
        percentiles = result["percentiles"]
        tag = self._emu_version_tag(emu_version)

        fig = plt.figure(figsize=(6, 5))
        plt.fill_between(modes, 0, percentiles[2], color="salmon", label="99%", alpha=0.8)
        plt.fill_between(modes, 0, percentiles[1], color="red", label="95%", alpha=0.7)
        plt.fill_between(modes, 0, percentiles[0], color="darkred", label="68%", alpha=1.0)
        plt.axhline(y=0.01, color="grey", linestyle="--")
        plt.legend(frameon=False, fontsize=16, loc="upper left")
        plt.ylabel(r"$\frac{|\mathrm{emu} - \mathrm{test}|}{\mathrm{test}}$", fontsize=24)
        plt.xlabel("modes", fontsize=16)
        plt.title(f"Emulator accuracy: {name} ({tag})", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, f"accuracy_{name}_{tag}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved accuracy figure to %s", out_path)

    def _plot_accuracy_cov(
        self, name: str, result: Dict[str, Any], emu_version: Optional[int] = None
    ) -> None:
        """Save covariance-normalized accuracy percentiles ``|emu-test|/σ``."""
        plt = _get_pyplot()
        if plt is None:
            return

        modes = result["modes"]
        percentiles = result["cov_percentiles"]
        tag = self._emu_version_tag(emu_version)

        fig = plt.figure(figsize=(6, 5))
        plt.fill_between(modes, 0, percentiles[2], color="salmon", label="99%", alpha=0.8)
        plt.fill_between(modes, 0, percentiles[1], color="red", label="95%", alpha=0.7)
        plt.fill_between(modes, 0, percentiles[0], color="darkred", label="68%", alpha=1.0)
        plt.axhline(y=1.0, color="grey", linestyle="--", label=r"$1\sigma$")
        plt.legend(frameon=False, fontsize=16, loc="upper left")
        plt.ylabel(
            r"$\frac{|\mathrm{emu} - \mathrm{test}|}{\sqrt{\mathrm{diag}(C)}}$",
            fontsize=24,
        )
        plt.xlabel("modes", fontsize=16)
        plt.title(f"Emulator accuracy (cov-normalized): {name} ({tag})", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, f"accuracy_cov_{name}_{tag}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cov-normalized accuracy figure to %s", out_path)

    def _plot_delta_chi2(
        self, result: Dict[str, Any], emu_version: Optional[int] = None
    ) -> None:
        """Save a histogram of the per-test-point Delta chi^2."""
        plt = _get_pyplot()
        if plt is None:
            return

        delta = result["delta_chi2"]
        delta = delta[np.isfinite(delta)]
        if delta.size == 0:
            return

        tag = self._emu_version_tag(emu_version)
        fig = plt.figure(figsize=(6, 5))
        plt.hist(delta, bins=min(50, max(10, delta.size // 5)), color="steelblue", alpha=0.85)
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
        plt.xlabel(r"$\Delta \chi^2 = \chi^2_{\mathrm{emu}} - \chi^2_{\mathrm{true}}$", fontsize=16)
        plt.ylabel("count", fontsize=16)
        plt.title(
            rf"{tag}: MAD(r)={result.get('mad_r', float('nan')):.2g}, "
            rf"max$|r|$={result.get('max_abs_r', float('nan')):.2g}, "
            rf"med$\Delta\chi^2$={result['median']:.2g}",
            fontsize=13,
        )
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, f"delta_chi2_{tag}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Delta chi^2 figure to %s", out_path)

def _gaussian_kl(mean0: np.ndarray, cov0: np.ndarray,
                 mean1: np.ndarray, cov1: np.ndarray) -> float:
    """Closed-form KL divergence KL(N0 || N1) between two Gaussians."""
    d = len(mean0)
    cov1_inv = np.linalg.pinv(cov1)
    diff = mean1 - mean0
    trace_term = np.trace(cov1_inv @ cov0)
    quad_term = diff @ cov1_inv @ diff
    _, logdet0 = np.linalg.slogdet(cov0)
    _, logdet1 = np.linalg.slogdet(cov1)
    return float(0.5 * (trace_term - d + quad_term + (logdet1 - logdet0)))


def _knn_distances(query: np.ndarray, reference: np.ndarray, k: int,
                   exclude_self: bool) -> np.ndarray:
    """Distance from each ``query`` point to its ``k``-th neighbour in ``reference``.

    When ``exclude_self`` is True the query points are assumed to be a subset of
    ``reference`` (self-distances of zero are discarded), so the k-th neighbour
    is found among the remaining points.
    """
    n_needed = k + 1 if exclude_self else k
    from scipy.spatial import cKDTree
    tree = cKDTree(reference)
    dist, _ = tree.query(query, k=n_needed)
    dist = np.asarray(dist, dtype=float)
    # query() returns a 1D array when n_needed == 1; make it (n, 1).
    if dist.ndim == 1:
        dist = dist[:, None]
    return dist[:, -1]
    # # Brute-force fallback (O(n*m)) when scipy is unavailable.
    # diff = query[:, None, :] - reference[None, :, :]
    # dmat = np.sqrt(np.sum(diff * diff, axis=2))
    # dmat.sort(axis=1)
    # return dmat[:, n_needed - 1]


def _knn_kl_divergence(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Sample-based estimate of KL(P || Q) from samples ``x``~P and ``y``~Q.

    Implements the k-nearest-neighbour estimator of Wang, Kulkarni & Verdu
    (2009), "Divergence estimation for multidimensional densities via
    k-nearest-neighbor distances":

        KL(P||Q) ~= (d / n) * sum_i log(nu_k(i) / rho_k(i)) + log(m / (n - 1))

    where ``rho_k(i)`` is the distance from ``x_i`` to its k-th neighbour within
    ``x`` (excluding itself) and ``nu_k(i)`` is the distance from ``x_i`` to its
    k-th neighbour in ``y``. No Gaussianity assumption is made.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n, d = x.shape
    m = y.shape[0]
    if n < k + 1 or m < k:
        return float("nan")

    rho = _knn_distances(x, x, k, exclude_self=True)
    nu = _knn_distances(x, y, k, exclude_self=False)

    valid = (rho > 0) & (nu > 0) & np.isfinite(rho) & np.isfinite(nu)
    if valid.sum() < 1:
        return float("nan")
    rho = rho[valid]
    nu = nu[valid]

    kl = (d / rho.size) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1.0))
    return float(kl)


def _whiten_pair(sample_a: np.ndarray, sample_b: np.ndarray):
    """Whiten two sample sets with the pooled covariance (Mahalanobis metric).

    Cosmological posteriors are strongly correlated; isotropic Euclidean k-NN
    after only per-axis rescaling still stretches the metric along the
    principal correlations and inflates the KL bias. Full whitening puts all
    directions on an equal footing.
    """
    a = np.atleast_2d(np.asarray(sample_a, dtype=float))
    b = np.atleast_2d(np.asarray(sample_b, dtype=float))
    pooled = np.vstack([a, b])
    cov = np.atleast_2d(np.cov(pooled, rowvar=False))
    cov = cov + np.eye(cov.shape[0]) * 1e-12
    try:
        # L L^T = cov  =>  whitened = L^{-1} x
        chol = np.linalg.cholesky(cov)
        a_w = np.linalg.solve(chol, a.T).T
        b_w = np.linalg.solve(chol, b.T).T
        return a_w, b_w
    except np.linalg.LinAlgError:
        scale = np.std(pooled, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return a / scale, b / scale


def _knn_kl_raw_symmetric(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Symmetrized k-NN KL on already-whitened samples (no debiasing)."""
    kl_ab = _knn_kl_divergence(a, b, k=k)
    kl_ba = _knn_kl_divergence(b, a, k=k)
    if not (np.isfinite(kl_ab) and np.isfinite(kl_ba)):
        return float("nan")
    return float(max(0.0, 0.5 * (kl_ab + kl_ba)))


def _knn_self_kl_null(samples: np.ndarray, k: int) -> float:
    """Same-distribution null k-NN KL from a random 50/50 split of ``samples``.

    For identical distributions the Wang–Kulkarni–Verdú estimator is biased
    *positive* in finite samples; this null estimates that floor so it can be
    subtracted from a cross-chain KL.
    """
    n = len(samples)
    if n < 2 * (k + 1):
        return 0.0
    idx = np.random.permutation(n)
    half = n // 2
    return _knn_kl_raw_symmetric(samples[idx[:half]], samples[idx[half:]], k)


def _knn_kl_symmetric(sample_a: np.ndarray, sample_b: np.ndarray,
                      k: int = 3, debias: bool = True) -> float:
    """Symmetrized (Jeffreys) k-NN KL divergence between two sample sets.

    Steps:
      1. Whiten both clouds with the pooled covariance (Mahalanobis metric)
         so correlations do not inflate nearest-neighbour distances.
      2. Compute the raw symmetrized Wang–Kulkarni–Verdú KL.
      3. If ``debias`` (ini ``kl_knn_debias``), subtract the average
         same-distribution null KL of each cloud. This removes most of the
         positive finite-sample bias in moderate dimension.

    ``k`` is the neighbour order (ini option ``kl_knn_k``; default 3).
    """
    a = np.atleast_2d(np.asarray(sample_a, dtype=float))
    b = np.atleast_2d(np.asarray(sample_b, dtype=float))
    if a.shape[1] != b.shape[1]:
        return float("nan")

    a, b = _whiten_pair(a, b)
    kl_raw = _knn_kl_raw_symmetric(a, b, k)
    if not np.isfinite(kl_raw):
        return float("nan")
    if not debias:
        return float(kl_raw)

    null_a = _knn_self_kl_null(a, k)
    null_b = _knn_self_kl_null(b, k)
    null = 0.0
    n_null = 0
    if np.isfinite(null_a):
        null += null_a
        n_null += 1
    if np.isfinite(null_b):
        null += null_b
        n_null += 1
    if n_null > 0:
        null /= n_null
        kl_raw = kl_raw - null
    return float(max(0.0, kl_raw))


def _get_pyplot():
    """Return a headless-safe pyplot module, or ``None`` if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover - matplotlib optional
        logger.warning("matplotlib unavailable (%s); skipping convergence plots.", exc)
        return None
