"""
Convergence tests for the ROSE sampler.

Just before the final sampling stage, the sampler collects a set of test points
from the 1-sigma (credible) region of the one-before-last MCMC chain (see
``data_processing.generate_updated_sample``). This module runs convergence
diagnostics on those test points using the freshly trained emulator:

(a) Emulator accuracy on all emulated modes: relative-error percentiles between
    the emulated data vectors and the true full-pipeline data vectors (the same
    diagnostic as ``rose_test/salmon_plot.py``, but computed in-memory), plus a
    covariance-normalized residual
    ``|emu - truth| / sqrt(diag(C))``.
(b) Delta chi^2: for every test point, the difference between the true
    full-pipeline log-likelihood and the emulated log-likelihood.
(c) KL divergence between consecutive-iteration likelihood estimates
    (implemented separately; see ``_test_kl_divergence``).

Separately, a lightweight per-iteration KL diagnostic is computed after every
MCMC stage (see ``_record_iteration_kl``): the symmetric KL divergence between
the current and previous iteration's MCMC chains after importance-reweighting
both to the *untempered* posterior (so tempering changes no longer dominate the
signal). Two estimators are reported side by side -- a closed-form Gaussian KL
(``kl_gaussian``) and a non-parametric, sample-based k-nearest-neighbour KL
(``kl_knn``; neighbour order set by ``kl_knn_k``). One row per iteration is
appended to ``{save_dir}/rose_kl.txt`` (mirroring ``rose_timing.txt``).

All numeric results are written to ``{save_dir}/convergence`` and, when
matplotlib is available, summary figures are saved alongside them.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Percentile levels reported for the emulator accuracy test.
_ACCURACY_PERCENTILES = (68.0, 95.0, 99.0, 99.9)


class RoseConvergenceMixin:
    """Mixin providing convergence diagnostics for :class:`RoseSampler`."""

    def run_convergence_tests(self) -> None:
        """Run all convergence tests on the collected test points.

        This is intended to be called in the last training iteration, after the
        emulator has been trained but before the final sampling stage. It is a
        no-op when no test points were collected (e.g. ``final_test_size = 0``).
        """
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
            "Running convergence tests on %d test points from the 1-sigma "
            "credible region of the one-before-last MCMC chain",
            len(sample_test),
        )

        self.convergence_dir = os.path.join(self.save_dir, "convergence")
        os.makedirs(self.convergence_dir, exist_ok=True)

        self.convergence_results: Dict[str, Any] = {}
        # (a) and (b) use the emulator trained WITHOUT the test points, so the
        # test points remain genuinely held-out for these accuracy diagnostics.
        self.convergence_results["accuracy"] = self._test_emulator_accuracy()
        self.convergence_results["delta_chi2"] = self._test_delta_chi2()

        # (c) is opt-in (kl_convergence = T). It folds the test points into the
        # training set, retrains, and drives the KL-divergence convergence loop.
        # This updates self.emulator, so it runs after the held-out diagnostics
        # above.
        if getattr(self, "kl_convergence", False):
            logger.info("KL-divergence convergence test enabled (kl_convergence = T)")
            self.convergence_results["kl"] = self._test_kl_divergence()
        else:
            logger.info(
                "KL-divergence convergence test disabled; set kl_convergence = T "
                "to enable it."
            )

        self._save_convergence_results()

    # ------------------------------------------------------------------
    # (a) Emulator accuracy on all emulated modes
    # ------------------------------------------------------------------
    def _test_emulator_accuracy(self) -> Dict[str, Any]:
        """Relative-error percentiles between emulated and true data vectors.

        For each likelihood the emulator is evaluated on the test parameters and
        compared, mode by mode, to the true full-pipeline data vectors. We report
        the 68/95/99/99.9 percentiles of (i) the absolute relative error and
        (ii) the absolute covariance-normalized residual
        ``|emu - truth| / sqrt(diag(C))`` across the test set for every emulated
        mode.
        """
        param_names = [str(p) for p in self.pipeline.varied_params]
        X = {name: self.sample_test[:, i] for i, name in enumerate(param_names)}

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
                "[%s] Accuracy on %d test points: median rel. error=%.3g, "
                "68%%=%.3g, 95%%=%.3g, 99%%=%.3g (max mode-wise)",
                name, len(truth),
                results[name]["median_rel_error"],
                float(np.nanmax(percentiles[0])),
                float(np.nanmax(percentiles[1])),
                float(np.nanmax(percentiles[2])),
            )
            if rel_cov_error is not None:
                logger.info(
                    "[%s] Cov-normalized accuracy: median=%.3g, "
                    "68%%=%.3g, 95%%=%.3g, 99%%=%.3g (max mode-wise, in units of "
                    "sqrt(diag(C)))",
                    name,
                    results[name]["median_rel_cov_error"],
                    float(np.nanmax(results[name]["cov_percentiles"][0])),
                    float(np.nanmax(results[name]["cov_percentiles"][1])),
                    float(np.nanmax(results[name]["cov_percentiles"][2])),
                )

            self._plot_accuracy(name, results[name])
            self._plot_accuracy_cov(name, results[name])

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
    def _test_delta_chi2(self) -> Dict[str, Any]:
        """Delta chi^2 between the true and emulated likelihoods per test point.

        The true log-likelihoods were computed by the full pipeline when the test
        points were collected (``self.sample_likes_test``). The emulated
        log-likelihoods are obtained by running the emulated pipeline on the same
        test parameters. We report ``delta_chi2 = chi2_emu - chi2_true`` where
        ``chi2 = -2 * log_like``.
        """
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
        results = {
            "delta_chi2": delta_chi2,
            "chi2_true": chi2_true,
            "chi2_emu": chi2_emu,
            "n_valid": int(valid.sum()),
            "mean": float(np.mean(valid_delta)) if valid_delta.size else float("nan"),
            "std": float(np.std(valid_delta)) if valid_delta.size else float("nan"),
            "median": float(np.median(valid_delta)) if valid_delta.size else float("nan"),
            "abs_median": float(np.median(np.abs(valid_delta))) if valid_delta.size else float("nan"),
            "abs_max": float(np.max(np.abs(valid_delta))) if valid_delta.size else float("nan"),
        }

        logger.info(
            "Delta chi^2 over %d valid test points: mean=%.3g, std=%.3g, "
            "median|dchi2|=%.3g, max|dchi2|=%.3g",
            results["n_valid"], results["mean"], results["std"],
            results["abs_median"], results["abs_max"],
        )

        self._plot_delta_chi2(results)
        return results

    # ------------------------------------------------------------------
    # (c) KL-divergence convergence loop
    # ------------------------------------------------------------------
    def _test_kl_divergence(self) -> Dict[str, Any]:
        """Drive the KL-divergence convergence loop.

        Steps:
          1. Evaluate the emulated posterior of the current emulator (iteration
             N, trained without the test points) on a fixed reference set of
             chain samples.
          2. Fold the test points into the training set and retrain -> N+1.
          3. Estimate the KL divergence between the N and N+1 emulated posteriors
             via importance reweighting of the one-before-last chain (Gaussian
             approximation with the closed-form KL).
          4. While KL >= kl_threshold and attempts remain, add more training
             points, retrain, and recompute the KL against the previous
             iteration until the criterion is met.

        The (possibly repeatedly) retrained emulator is left in place and used
        for the subsequent final sampling stage.
        """
        theta, logq = self._kl_reference_samples()
        if theta is None:
            logger.warning("KL convergence test skipped (no usable reference chain).")
            return {"converged": False, "kl_history": [], "reason": "no_reference_chain"}

        threshold = float(self.kl_threshold)
        max_retrain = int(self.kl_max_retrain)

        # Posterior of iteration N (emulator without test points). Each KL
        # retrain writes a brand-new emumodel_{version} directory (rather than
        # overwriting the current one) so the freshly trained model is both kept
        # in memory and, being on disk, picked up by the MPI worker processes via
        # _worker_emu_model_path()/_ensure_emulator.
        logp_prev = self._emulated_log_posterior(theta)
        emu_version = getattr(self, "_current_emu_version", self.iterations + 1)

        # All KL-loop timing rows reuse the final-iteration label (max_iterations),
        # so rose_timing.txt can show multiple rows for that iteration when
        # kl_convergence = T. Sampling is not run in this loop, so time_sampling_s=0.
        timing_iteration = int(self.max_iterations)

        # Fold the held-out test points into the training set and retrain -> N+1.
        t0 = time.perf_counter()
        n_added = self._append_test_points_to_training()
        time_training_set_s = time.perf_counter() - t0
        emu_version += 1
        logger.info(
            "Folded %d test points into the training set; retraining emulator "
            "(emumodel_%d)", n_added, emu_version,
        )
        t1 = time.perf_counter()
        self.train_emulator(model_version=emu_version)
        time_train_emulator_s = time.perf_counter() - t1
        self._append_timing_row(
            timing_iteration, time_training_set_s, time_train_emulator_s, 0.0,
        )
        logp_curr = self._emulated_log_posterior(theta)

        kl, ess = self._gaussian_kl_via_importance(theta, logp_prev, logq, logp_curr)
        kl_history = [kl]
        ess_history = [ess]
        logger.info(
            "KL divergence after folding test points: %.4g (threshold %.4g, ESS~%.0f)",
            kl, threshold, ess,
        )

        attempt = 0
        converged = np.isfinite(kl) and kl < threshold
        while not converged and attempt < max_retrain:
            attempt += 1
            t0 = time.perf_counter()
            n_extra = self._add_training_points(self.kl_extra_size)
            time_training_set_s = time.perf_counter() - t0
            emu_version += 1
            logger.info(
                "KL not converged (%.4g >= %.4g); retrain attempt %d/%d after "
                "adding %d training points (emumodel_%d)",
                kl, threshold, attempt, max_retrain, n_extra, emu_version,
            )
            logp_prev = logp_curr
            t1 = time.perf_counter()
            self.train_emulator(model_version=emu_version)
            time_train_emulator_s = time.perf_counter() - t1
            self._append_timing_row(
                timing_iteration, time_training_set_s, time_train_emulator_s, 0.0,
            )
            logp_curr = self._emulated_log_posterior(theta)
            kl, ess = self._gaussian_kl_via_importance(theta, logp_prev, logq, logp_curr)
            kl_history.append(kl)
            ess_history.append(ess)
            logger.info("KL divergence after retrain attempt %d: %.4g (ESS~%.0f)", attempt, kl, ess)
            converged = np.isfinite(kl) and kl < threshold

        if converged:
            logger.info(
                "KL convergence reached (KL=%.4g < %.4g) after %d extra retrain(s). "
                "Proceeding to final sampling with the converged emulator.",
                kl, threshold, attempt,
            )
        else:
            logger.warning(
                "KL convergence NOT reached after %d retrain attempts (final KL=%.4g >= %.4g). "
                "Proceeding to final sampling with the last emulator; consider increasing "
                "kl_max_retrain, kl_extra_size, or the training budget.",
                attempt, kl, threshold,
            )

        self.kl_converged = bool(converged)
        results = {
            "converged": bool(converged),
            "kl": float(kl),
            "threshold": threshold,
            "n_retrain": int(attempt),
            "kl_history": np.array(kl_history, dtype=float),
            "ess_history": np.array(ess_history, dtype=float),
        }
        self._plot_kl_history(results)
        return results

    # ------------------------------------------------------------------
    # Per-iteration KL divergence between consecutive MCMC chains
    # ------------------------------------------------------------------
    def _untempered_log_weights(self, chain: np.ndarray) -> np.ndarray:
        """Log importance weights that map the current chain to the untempered posterior.

        Emcee/NUTS draw from the tempered posterior
        ``q(θ) ∝ exp(T * log π(θ))``. Reweighting to the untempered target
        ``π(θ)`` therefore contributes ``(1 - T) * log π(θ)``.

        Nautilus (and any future weighted sampler) may already carry its own
        importance weights in ``chain_log_weights``; those are added on top.
        When ``T = 1`` the tempering term vanishes and only the sampler weights
        remain.
        """
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
                # w ∝ π / q ∝ exp((1 - T) * log π)
                logw = logw + (1.0 - tempering) * logpost
        return logw

    def _resample_equal_weight(self, chain: np.ndarray, logw: np.ndarray,
                               cov: np.ndarray, n_sub: int) -> np.ndarray:
        """Build an equal-weight subsample for the k-NN KL estimator.

        Multinomial-resamples by ``logw`` when the weights are non-uniform
        (tempering and/or sampler weights), with a tiny jitter to break the
        exact duplicate ties that replacement creates. Falls back to plain
        subsampling without replacement for equal-weight chains.
        """
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

        Called once per iteration right after the MCMC stage. Both the current
        and previous chains are importance-reweighted to the *untempered*
        posterior before the KL is computed, so changes in the tempering
        schedule do not dominate the diagnostic. Sampler-native weights
        (e.g. nautilus importance weights) are included as well.

        Two symmetric (Jeffreys) KL divergences are reported:

        * ``kl_gaussian`` -- closed-form Gaussian KL from weighted mean/cov.
        * ``kl_knn`` -- sample-based k-NN KL (neighbour order ``kl_knn_k``)
          on equal-weight resamples of the reweighted clouds. Uses Mahalanobis
          whitening and, when ``kl_knn_debias = T``, subtracts a same-
          distribution null to reduce the positive finite-sample bias.

        One row per iteration is appended to ``{save_dir}/rose_kl.txt``.
        The first iteration has no previous chain, so both KL entries are ``nan``.

        Returns ``(kl_gaussian, kl_knn)``.
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

        moments = self._weighted_moments(chain, logw)
        if moments is None:
            logger.warning(
                "Could not compute tempering-reweighted chain moments "
                "(tempering=%.4g); skipping per-iteration KL.",
                tempering,
            )
            return float("nan"), float("nan")
        mean, cov, ess = moments
        if ess < 50:
            logger.warning(
                "Low ESS (%.0f) after tempering reweight (T=%.4g) for "
                "per-iteration KL; the estimate may be noisy.",
                ess, tempering,
            )

        n_sub = min(int(getattr(self, "kl_n_samples", 2000)), len(chain))
        chain_samples = self._resample_equal_weight(chain, logw, cov, n_sub)

        kl_gauss = float("nan")
        kl_knn = float("nan")
        prev_moments = getattr(self, "_kl_prev_chain_moments", None)
        prev_samples = getattr(self, "_kl_prev_chain_samples", None)
        if prev_moments is not None:
            mean_prev, cov_prev = prev_moments
            if mean_prev.shape == mean.shape:
                kl_ab = _gaussian_kl(mean_prev, cov_prev, mean, cov)
                kl_ba = _gaussian_kl(mean, cov, mean_prev, cov_prev)
                kl_gauss = 0.5 * (kl_ab + kl_ba)
        if prev_samples is not None and prev_samples.shape[1] == chain_samples.shape[1]:
            kl_knn = _knn_kl_symmetric(
                prev_samples, chain_samples,
                k=int(getattr(self, "kl_knn_k", 3)),
                debias=bool(getattr(self, "kl_knn_debias", True)),
            )

        self._kl_prev_chain_moments = (mean, cov)
        self._kl_prev_chain_samples = chain_samples

        kl_file = os.path.join(self.save_dir, "rose_kl.txt")
        write_header = not os.path.isfile(kl_file)
        with open(kl_file, "a") as f:
            if write_header:
                f.write("iteration\tkl_gaussian\tkl_knn\tess\ttempering\n")
            f.write(
                f"{self.iterations + 1}\t{kl_gauss:.6e}\t{kl_knn:.6e}\t"
                f"{ess:.1f}\t{tempering:.6g}\n"
            )

        if np.isfinite(kl_gauss) or np.isfinite(kl_knn):
            logger.info(
                "Iteration %d chain-to-chain KL (untempered): Gaussian=%.4g, "
                "k-NN=%.4g, ESS=%.0f, T=%.4g (saved to %s)",
                self.iterations + 1, kl_gauss, kl_knn, ess, tempering, kl_file,
            )
        else:
            logger.info(
                "Iteration %d chain-to-chain KL: nan "
                "(no previous chain; T=%.4g, ESS=%.0f; saved to %s)",
                self.iterations + 1, tempering, ess, kl_file,
            )
        return float(kl_gauss), float(kl_knn)

    def _kl_reference_samples(self):
        """Return (theta, logq) reference samples for importance reweighting.

        ``theta`` is a (subsampled) copy of the one-before-last MCMC chain and
        ``logq`` is the log sampling density of those samples (up to an additive
        constant), taken as the tempered emulated posterior of the previous
        iteration stored in ``self.chain_logpost``.
        """
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
            # Tempering used to produce the one-before-last chain.
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
        """Append the already-evaluated test points to the training arrays."""
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
        """Draw, evaluate, and append ``n_points`` new training points.

        Points are drawn homogeneously from the tempered HPD credible region of
        the one-before-last MCMC chain (``self.chain``), mirroring
        ``generate_updated_sample`` but without collecting test points.
        """
        from .data_processing import _task_wrapper
        from .utils import task

        idx = self._select_credible_region_indices(n_points, homogeneous=True)
        sample = self.chain[idx]
        unit_sample = self.unit_chain[idx]

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
        # Regularize for numerical stability of the KL closed form.
        cov += np.eye(cov.shape[0]) * 1e-12
        return mean, cov, ess

    def _gaussian_kl_via_importance(self, theta, logp_a, logq, logp_b):
        """Symmetrized Gaussian KL between two emulated posteriors.

        Each posterior is approximated by a Gaussian fit to the reference
        samples reweighted by ``exp(logp - logq)``. Returns ``(kl, min_ess)``
        where ``kl`` is the symmetrized (Jeffreys) divergence and ``min_ess`` is
        the smaller effective sample size of the two reweightings.
        """
        moments_a = self._weighted_moments(theta, logp_a - logq)
        moments_b = self._weighted_moments(theta, logp_b - logq)
        if moments_a is None or moments_b is None:
            return float("inf"), 0.0

        mean_a, cov_a, ess_a = moments_a
        mean_b, cov_b, ess_b = moments_b

        kl_ab = _gaussian_kl(mean_a, cov_a, mean_b, cov_b)
        kl_ba = _gaussian_kl(mean_b, cov_b, mean_a, cov_a)
        kl_sym = 0.5 * (kl_ab + kl_ba)

        min_ess = min(ess_a, ess_b)
        if min_ess < 50:
            logger.warning(
                "Low effective sample size (%.0f) in KL importance reweighting; "
                "the KL estimate may be noisy. Consider a less aggressive "
                "tempering schedule for the one-before-last chain.",
                min_ess,
            )
        return float(kl_sym), float(min_ess)

    def _plot_kl_history(self, result: Dict[str, Any]) -> None:
        """Save a plot of the KL divergence versus retrain attempt."""
        plt = _get_pyplot()
        if plt is None:
            return
        history = np.asarray(result.get("kl_history", []), dtype=float)
        if history.size == 0:
            return
        fig = plt.figure(figsize=(6, 5))
        plt.plot(np.arange(len(history)), history, "o-", color="darkgreen")
        plt.axhline(result["threshold"], color="grey", linestyle="--",
                    label=f"threshold={result['threshold']:.3g}")
        plt.yscale("log")
        plt.xlabel("retrain attempt", fontsize=16)
        plt.ylabel("symmetric KL divergence", fontsize=16)
        plt.title("KL convergence" + (" (converged)" if result["converged"] else " (not converged)"),
                  fontsize=13)
        plt.legend(frameon=False, fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, "kl_convergence.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved KL convergence figure to %s", out_path)

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
        for key in ("delta_chi2", "chi2_true", "chi2_emu"):
            if key in dchi2:
                save_dict[f"delta_chi2/{key}"] = dchi2[key]
        for key in ("mean", "std", "median", "abs_median", "abs_max", "n_valid"):
            if key in dchi2:
                save_dict[f"delta_chi2/{key}"] = dchi2[key]

        kl = self.convergence_results.get("kl", {})
        for key in ("converged", "kl", "threshold", "n_retrain", "kl_history", "ess_history"):
            if key in kl:
                save_dict[f"kl/{key}"] = kl[key]

        out_path = os.path.join(self.convergence_dir, "convergence_tests.npz")
        np.savez(out_path, **save_dict)
        logger.info("Saved convergence diagnostics to %s", out_path)

    def _plot_accuracy(self, name: str, result: Dict[str, Any]) -> None:
        """Save an accuracy percentile figure (mirrors salmon_plot.py)."""
        plt = _get_pyplot()
        if plt is None:
            return

        modes = result["modes"]
        percentiles = result["percentiles"]

        fig = plt.figure(figsize=(6, 5))
        plt.fill_between(modes, 0, percentiles[2], color="salmon", label="99%", alpha=0.8)
        plt.fill_between(modes, 0, percentiles[1], color="red", label="95%", alpha=0.7)
        plt.fill_between(modes, 0, percentiles[0], color="darkred", label="68%", alpha=1.0)
        plt.axhline(y=0.01, color="grey", linestyle="--")
        plt.legend(frameon=False, fontsize=16, loc="upper left")
        plt.ylabel(r"$\frac{|\mathrm{emu} - \mathrm{test}|}{\mathrm{test}}$", fontsize=24)
        plt.xlabel("modes", fontsize=16)
        plt.title(f"Emulator accuracy: {name}", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, f"accuracy_{name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved accuracy figure to %s", out_path)

    def _plot_accuracy_cov(self, name: str, result: Dict[str, Any]) -> None:
        """Save covariance-normalized accuracy percentiles ``|emu-test|/σ``."""
        plt = _get_pyplot()
        if plt is None:
            return

        modes = result["modes"]
        percentiles = result["cov_percentiles"]

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
        plt.title(f"Emulator accuracy (cov-normalized): {name}", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, f"accuracy_cov_{name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cov-normalized accuracy figure to %s", out_path)

    def _plot_delta_chi2(self, result: Dict[str, Any]) -> None:
        """Save a histogram of the per-test-point Delta chi^2."""
        plt = _get_pyplot()
        if plt is None:
            return

        delta = result["delta_chi2"]
        delta = delta[np.isfinite(delta)]
        if delta.size == 0:
            return

        fig = plt.figure(figsize=(6, 5))
        plt.hist(delta, bins=min(50, max(10, delta.size // 5)), color="steelblue", alpha=0.85)
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
        plt.xlabel(r"$\Delta \chi^2 = \chi^2_{\mathrm{emu}} - \chi^2_{\mathrm{true}}$", fontsize=16)
        plt.ylabel("count", fontsize=16)
        plt.title(
            rf"$\langle\Delta\chi^2\rangle$={result['mean']:.2g}, "
            rf"med$|\Delta\chi^2|$={result['abs_median']:.2g}",
            fontsize=13,
        )
        plt.tight_layout()
        out_path = os.path.join(self.convergence_dir, "delta_chi2.png")
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
