"""
Data processing and sample generation for ROSE sampler.

This module contains methods for generating training samples, processing results,
and managing training/test datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from timeit import default_timer

import numpy as np

from .utils import task, SAVE_ALL

logger = logging.getLogger(__name__)

# Global sampler reference for picklable task wrapper
_sampler = None


def _task_wrapper(p: np.ndarray) -> Any:
    """Wrapper function for task that can be pickled for MPI.
    
    This function uses the global _sampler variable set by RoseSampler.config().
    It's needed because lambda functions cannot be pickled for MPI communication.
    
    Args:
        p: Parameter vector to evaluate
        
    Returns:
        Result from task(p, _sampler)
    """
    global _sampler
    if _sampler is None:
        raise RuntimeError("Global sampler not set. This should be set in RoseSampler.config()")
    return task(p, _sampler)


class RoseDataProcessingMixin:
    """Mixin class providing data processing methods for RoseSampler."""
    
    def generate_initial_sample(self) -> None:
        """Generate initial training sample using Latin Hypercube sampling.
        
        This method creates the initial training dataset by:
        1. Sampling parameters from priors using Latin Hypercube
        2. Running the full pipeline for each parameter set
        3. Filtering out poor fits based on chi2 cutoff
        4. Storing results for emulator training
        """
        import scipy.stats
        
        logger.info("Generating initial training sample")
        
        # Generate Latin Hypercube sample in unit cube
        hypercube = scipy.stats.qmc.LatinHypercube(self.ndim, seed=self.seed)
        unit_sample = hypercube.random(n=self.initial_size)
        
        # Transform to physical parameter space
        sample = np.array([
            self.pipeline.denormalize_vector_from_prior(p) for p in unit_sample
        ])
        
        logger.info(f"Generated {len(sample)} parameter combinations")
        logger.info(f"Parallelization enabled: {self.pool is not None}")
        
        # Run pipeline for all samples
        start_time = default_timer()
        if self.pool:
            sample_results = self.pool.map(_task_wrapper, sample)
        else:
            sample_results = [task(p, self) for p in sample]
        
        end_time = default_timer()
        logger.info(f"Initial sample evaluation took {end_time - start_time:.1f} seconds")
        
        # Process results and apply chi2 cutoff
        self._process_initial_results(sample_results, sample, unit_sample)
        
        # Initialize test sample storage
        self._initialize_test_storage()

    def _process_initial_results(self, sample_results: List, sample: np.ndarray, 
                                unit_sample: np.ndarray) -> None:
        """Process initial sample results and apply chi2 filtering."""
        # Extract successful results
        valid_results = [s for s in sample_results if s is not None]
        
        if not valid_results:
            raise RuntimeError("No valid pipeline evaluations in initial sample")
        
        # Extract components
        sample_likes = np.array([s[0] for s in valid_results])
        sample_data_vectors = self._stack_per_likelihood([s[1] for s in valid_results])
        sample_priors = np.array([s[2] for s in valid_results])
        sample_posts = np.array([s[3] for s in valid_results])
        
        # Apply chi2 cutoff
        chi2_values = -2 * sample_likes
        cut = chi2_values < self.chi2_cut_off

        shapes = {name: v.shape for name, v in sample_data_vectors.items()}
        logger.info(f"Data vector shapes per likelihood: {shapes}")
        logger.info(f"Chi2 range: [{chi2_values.min():.1f}, {chi2_values.max():.1f}]")
        
        # Filter arrays
        self.sample_likes = sample_likes[cut]
        self.sample_priors = sample_priors[cut]
        self.sample_posts = sample_posts[cut]
        self.sample_data_vectors = {name: v[cut] for name, v in sample_data_vectors.items()}

        
        # Filter corresponding parameter arrays
        valid_indices = np.arange(len(sample_results))[np.array([s is not None for s in sample_results])]
        filtered_indices = valid_indices[cut]
        self.sample = sample[filtered_indices]
        self.unit_sample = unit_sample[filtered_indices]
        
        n_kept = len(self.sample_likes)
        n_total = len(sample_results)
        logger.info(f"Kept {n_kept}/{n_total} samples after chi2 < {self.chi2_cut_off} filter")
        self.points_per_iteration = np.array([n_kept])
        # For the case of rejected points due to additional priors, for instance, w0+wa<0
        if n_kept < 0.8*self.initial_size:
            #raise RuntimeError(f"Only {n_kept} samples passed chi2 filter, but need "
            #                 f"at least {0.8*self.initial_size} (80% of initial size) --> increase the initial size")
            logger.warning(f"Only {n_kept} samples passed chi2 filter, but need "
                           f"at least {0.8*self.initial_size} (80% of initial size) --> increase the initial size")
        
        
        self.initial_size_cut = n_kept

    def _initialize_test_storage(self) -> None:
        """Initialize storage for test samples."""
        self.sample_test = np.array([]).reshape(0, self.ndim)
        self.sample_data_vectors_test = {
            name: np.empty((0, v.shape[1])) for name, v in self.sample_data_vectors.items()
        }
        self.unit_sample_test = np.array([]).reshape(0, self.ndim)
        self.sample_likes_test = np.array([])
        self.sample_priors_test = np.array([])
        self.sample_posts_test = np.array([])
        self.points_per_iteration_test = np.array([])

    @staticmethod
    def _stack_per_likelihood(theory_dicts: List[dict]) -> dict:
        """Stack a list of per-likelihood theory dicts into a dict of 2D arrays.

        Assumes all dicts have the same set of likelihood keys in the same
        (insertion) order, which is guaranteed by utils.task scanning block
        entries deterministically.
        """
        if not theory_dicts:
            return {}
        names = list(theory_dicts[0].keys())
        return {
            name: np.array([np.asarray(d[name]) for d in theory_dicts])
            for name in names
        }

    def generate_updated_sample(self) -> None:
        """Generate additional training samples from the tempered MCMC chain.

        HPD fills use stratified whitened volume-uniform Latin hypercube
        sampling (``_sample_hpd_stratified_volume_lh``). When
        ``resample_weak_params`` is set, a tempering-dependent mixture also
        redraws those weak-parameter coordinates from a prior Latin hypercube.
        """
        sample, unit_sample = self._select_mixture_training_samples(self.resample_size)

        # Test points are collected only once, during the last training
        # iteration (i.e. just before the final sampling stage), drawn from the
        # credible region of the one-before-last MCMC chain. In all earlier
        # iterations no test points are evaluated.
        collect_test = (
            self.iterations == (self.max_iterations - 1)
            and self.final_test_size > 0
        )
        sample_test = None
        unit_sample_test = None
        if collect_test:
            sample_test, unit_sample_test = self._sample_hpd_stratified_volume_lh(
                self.final_test_size,
                seed=(
                    None if self.seed is None
                    else int(self.seed) + int(self.iterations) + 101
                ),
            )
            logger.info(
                f"Collecting {len(sample_test)} stratified volume-LH test points "
                f"from the {self.test_credible_fraction:.0%} tempered HPD "
                "(final training iteration)"
            )

        n_test = len(sample_test) if collect_test else 0
        logger.info(f"Running exact pipeline on {len(sample)} training + {n_test} test samples")
        
        # Evaluate samples with exact pipeline
        start_time = default_timer()
        sample_results_test = None
        if self.pool:
            sample_results = self.pool.map(_task_wrapper, sample)
            if collect_test:
                sample_results_test = self.pool.map(_task_wrapper, sample_test)
        else:
            sample_results = [task(p, self) for p in sample]
            if collect_test:
                sample_results_test = [task(p, self) for p in sample_test]
        
        end_time = default_timer()
        logger.info(f"Sample evaluation took {end_time - start_time:.1f} seconds")
        
        # Update training set (every iteration) and test set (final iteration only)
        self._update_training_set(sample_results, sample, unit_sample)
        if collect_test:
            self._update_test_set(sample_results_test, sample_test, unit_sample_test)

        # Final training iteration: optionally prune early points whose true
        # chi2 is much worse than the last iteration (see _remove_training_outliers).
        if (
            self.iterations == (self.max_iterations - 1)
            and getattr(self, "remove_outliers", False)
        ):
            self._remove_training_outliers()
        
        # Save datasets if requested
        if self.save_output == SAVE_ALL and self.iterations == (self.max_iterations - 1):
            self._save_datasets()

    def _current_resample_hpd_fraction(self) -> float:
        """HPD fraction for this ``generate_updated_sample`` call.

        ``self.iterations`` is 1 on the first resample (after T=tempering[0]),
        so the schedule index is ``iterations - 1``.
        """
        fracs = getattr(self, "resample_hpd_fractions", None)
        if not fracs:
            return 1.0
        idx = int(self.iterations) - 1
        if idx < 0:
            idx = 0
        if idx >= len(fracs):
            idx = len(fracs) - 1
        return float(fracs[idx])

    def _select_mixture_training_samples(self, n_select: int):
        """Return ``(physical, unit)`` training arrays of length ``n_select``.

        HPD fills use stratified whitened volume-uniform Latin hypercube
        sampling (``_sample_hpd_stratified_volume_lh``). When weak params are
        configured, a tempering-dependent fraction keeps pure HPD fills and the
        rest redraw those weak coordinates from a prior Latin hypercube.
        """
        chain_length = len(self.chain)
        if chain_length == 0:
            raise RuntimeError(
                "Cannot resample training points: the MCMC chain is empty. "
                "This usually means the sampler converged before the configured "
                "emcee_samples were reached and the burn-in then discarded "
                "everything. Lower emcee_burn or increase emcee_samples."
            )

        weak_idx = list(getattr(self, "resample_weak_param_indices", []) or [])
        f_hpd = self._current_resample_hpd_fraction()
        if not weak_idx:
            f_hpd = 1.0

        n_hpd = int(round(f_hpd * n_select))
        n_hpd = min(max(n_hpd, 0), n_select)
        n_exp = n_select - n_hpd

        prev_T = None
        try:
            if self.iterations >= 1 and self.iterations - 1 < len(self.tempering):
                prev_T = float(self.tempering[self.iterations - 1])
        except Exception:
            prev_T = None
        T_msg = f" (chain from T={prev_T:g})" if prev_T is not None else ""

        seed_base = None if self.seed is None else int(self.seed) + int(self.iterations)

        if n_exp == 0 or not weak_idx:
            logger.info(
                f"Selecting {n_select} stratified volume-LH HPD training samples "
                f"({self.test_credible_fraction:.0%} tempered region, "
                f"nshells={getattr(self, 'resample_hpd_nshells', 4)}){T_msg}"
            )
            return self._sample_hpd_stratified_volume_lh(
                n_select, seed=None if seed_base is None else seed_base + 3
            )

        weak_names = [str(self.pipeline.varied_params[i]) for i in weak_idx]
        logger.info(
            f"Selecting {n_select} mixed training samples{T_msg}: "
            f"{n_hpd} stratified HPD ({f_hpd:.0%}) + {n_exp} HPD⊕prior on {weak_names}"
        )

        if n_hpd > 0:
            phys_hpd, unit_hpd = self._sample_hpd_stratified_volume_lh(
                n_hpd, seed=None if seed_base is None else seed_base + 3
            )
        else:
            unit_hpd = np.empty((0, self.ndim), dtype=float)
            phys_hpd = np.empty((0, self.ndim), dtype=float)

        phys_exp, unit_exp = self._sample_hpd_stratified_volume_lh(
            n_exp, seed=None if seed_base is None else seed_base + 17
        )
        import scipy.stats
        lh = scipy.stats.qmc.LatinHypercube(
            len(weak_idx), seed=None if seed_base is None else seed_base + 19
        )
        weak_u = lh.random(n=n_exp)
        unit_exp = np.asarray(unit_exp, dtype=float).copy()
        for j, dim in enumerate(weak_idx):
            unit_exp[:, dim] = weak_u[:, j]
        phys_exp = np.array([
            self.pipeline.denormalize_vector_from_prior(u) for u in unit_exp
        ])

        if n_hpd == 0:
            return phys_exp, unit_exp
        return np.vstack([phys_hpd, phys_exp]), np.vstack([unit_hpd, unit_exp])

    def _sample_hpd_stratified_volume_lh(
        self, n_select: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw ``n_select`` points in the tempered HPD via stratified volume LH.

        1. Rank the tempered chain into the HPD (``test_credible_fraction``).
        2. Split that cloud into equal-mass log-posterior shells.
        3. In each shell, draw volume-uniform Latin-hypercube points inside the
           shell's whitened Mahalanobis ellipsoid (continuous unit-cube coords,
           clipped to ``[0, 1]``), then denormalize to physical parameters.

        Returns:
            ``(physical, unit)`` arrays of shape ``(n_select, ndim)``.
        """
        import scipy.stats

        if n_select <= 0:
            return (
                np.empty((0, self.ndim), dtype=float),
                np.empty((0, self.ndim), dtype=float),
            )

        region_indices = np.asarray(self._credible_region_indices(), dtype=int)
        if region_indices.size == 0:
            raise RuntimeError(
                "Tempered HPD region is empty; cannot sample training points"
            )

        unit_region = np.asarray(self.unit_chain[region_indices], dtype=float)
        logpost = getattr(self, "chain_logpost", None)
        if logpost is not None and len(logpost) == len(self.chain):
            lp_region = np.asarray(logpost, dtype=float)[region_indices]
        else:
            lp_region = np.zeros(len(region_indices), dtype=float)

        finite = np.isfinite(lp_region) & np.all(np.isfinite(unit_region), axis=1)
        unit_region = unit_region[finite]
        lp_region = lp_region[finite]
        if len(unit_region) < 2:
            raise RuntimeError(
                "Too few finite HPD points to build a whitened volume-LH design"
            )

        ndim = int(unit_region.shape[1])
        nshells = int(getattr(self, "resample_hpd_nshells", 4))
        min_per_shell = max(ndim + 2, 8)
        nshells = max(1, min(nshells, n_select, max(1, len(unit_region) // min_per_shell)))
        radius_q = float(getattr(self, "resample_hpd_radius_quantile", 0.95))

        order = np.argsort(lp_region)[::-1]
        shell_orders = [
            np.asarray(s, dtype=int) for s in np.array_split(order, nshells) if len(s)
        ]
        nshells = len(shell_orders)

        counts = [n_select // nshells] * nshells
        for i in range(n_select % nshells):
            counts[i] += 1

        global_mu, global_L, global_radius = self._hpd_whitening(
            unit_region, radius_q=radius_q
        )

        unit_parts = []
        for s_i, (local_order, n_s) in enumerate(zip(shell_orders, counts)):
            if n_s <= 0:
                continue
            pts = unit_region[local_order]
            if len(pts) >= min_per_shell:
                mu, L, radius = self._hpd_whitening(pts, radius_q=radius_q)
            else:
                mu = pts.mean(axis=0)
                L, radius = global_L, global_radius
            shell_seed = None if seed is None else int(seed) + 1000 * (s_i + 1)
            unit_parts.append(
                self._volume_uniform_ellipsoid_lh(
                    n_s, mu=mu, chol=L, radius=radius, seed=shell_seed
                )
            )

        unit = np.clip(np.vstack(unit_parts), 0.0, 1.0)
        if len(unit) < n_select:
            pad = self._volume_uniform_ellipsoid_lh(
                n_select - len(unit),
                mu=global_mu,
                chol=global_L,
                radius=global_radius,
                seed=None if seed is None else int(seed) + 7,
            )
            unit = np.clip(np.vstack([unit, pad]), 0.0, 1.0)
        elif len(unit) > n_select:
            unit = unit[:n_select]

        phys = np.array([
            self.pipeline.denormalize_vector_from_prior(u) for u in unit
        ])
        return phys, unit

    def _hpd_whitening(self, points: np.ndarray, radius_q: float = 0.95):
        """Return ``(mean, chol, radius)`` for a Mahalanobis ball around ``points``.

        ``chol`` satisfies ``cov ≈ chol @ chol.T``. ``radius`` is the
        ``radius_q`` quantile of Mahalanobis distances of ``points`` to the mean
        (at least a small positive floor).
        """
        pts = np.asarray(points, dtype=float)
        mu = pts.mean(axis=0)
        ndim = pts.shape[1]
        if len(pts) == 1:
            cov = np.eye(ndim) * 1e-4
        else:
            cov = np.cov(pts, rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]])
            cov = np.asarray(cov, dtype=float)
        diag = np.clip(np.diag(cov), 1e-12, None)
        cov = cov + np.eye(ndim) * (1e-8 + 1e-4 * float(np.mean(diag)))
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = cov + np.eye(ndim) * 1e-3 * float(np.mean(diag))
            chol = np.linalg.cholesky(cov)

        diff = pts - mu
        try:
            white = np.linalg.solve(chol, diff.T).T
            d = np.linalg.norm(white, axis=1)
        except np.linalg.LinAlgError:
            d = np.linalg.norm(diff, axis=1)
        d = d[np.isfinite(d)]
        if d.size == 0:
            radius = 1.0
        else:
            radius = max(float(np.quantile(d, radius_q)), 1e-3)
        return mu, chol, radius

    def _volume_uniform_ellipsoid_lh(
        self,
        n: int,
        mu: np.ndarray,
        chol: np.ndarray,
        radius: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Volume-uniform Latin-hypercube samples in a Mahalanobis ball.

        Direction from inverse-normal LH coords (normalized); radius
        ``U^{1/N} * radius`` so the fill is uniform in ellipsoid volume, not on
        the surface.
        """
        import scipy.stats

        if n <= 0:
            return np.empty((0, len(mu)), dtype=float)

        ndim = int(len(mu))
        lh = scipy.stats.qmc.LatinHypercube(ndim + 1, seed=seed)
        u = lh.random(n=n)
        g = scipy.stats.norm.ppf(np.clip(u[:, :ndim], 1e-8, 1.0 - 1e-8))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        directions = g / norms
        r = (u[:, ndim] ** (1.0 / float(ndim))) * float(radius)
        z = directions * r[:, None]
        return np.asarray(mu, dtype=float) + z @ np.asarray(chol, dtype=float).T

    def _select_non_overlapping_indices(self, chain_length: int, 
                                      excluded_indices: np.ndarray, 
                                      n_select: int) -> np.ndarray:
        """Select indices that don't overlap with excluded set."""
        available_indices = np.setdiff1d(np.arange(chain_length), excluded_indices)
        
        if len(available_indices) < n_select:
            logger.warning(f"Only {len(available_indices)} non-overlapping indices available, "
                          f"requested {n_select}")
            n_select = len(available_indices)
        
        return np.random.choice(available_indices, size=n_select, replace=False)

    def _credible_region_indices(self, fraction: float = None) -> np.ndarray:
        """Indices of chain points in the tempered HPD credible region.

        The current MCMC chain was drawn from the tempered posterior
        ``q(θ) ∝ π(θ)^T``. Ranking by ``chain_logpost`` (untempered) is
        equivalent to ranking by the tempered log-density for fixed ``T > 0``,
        so the top ``fraction`` of points is the tempered highest-posterior-
        density region of that chain.

        Args:
            fraction: HPD mass to keep. Defaults to ``test_credible_fraction``
                (0.95 ≈ 2-sigma). Use ``erf(n/sqrt(2))`` for an n-sigma mass
                (0.68 / 0.95 / 0.997 for 1 / 2 / 3 sigma).
        """
        if fraction is None:
            fraction = self.test_credible_fraction
        logpost = getattr(self, "chain_logpost", None)
        if logpost is None or len(logpost) != len(self.chain):
            logger.warning(
                "Chain posterior values unavailable; using the full chain "
                "instead of the tempered credible region."
            )
            return np.arange(len(self.chain))

        logpost = np.asarray(logpost)
        finite = np.isfinite(logpost)
        n_finite = int(finite.sum())
        if n_finite == 0:
            logger.warning(
                "No finite posterior values in chain; using full chain for "
                "credible-region selection."
            )
            return np.arange(len(self.chain))

        # Rank by posterior (descending) and keep the top fraction.
        order = np.argsort(logpost)[::-1][:n_finite]
        n_region = max(1, int(np.ceil(float(fraction) * n_finite)))
        return order[:n_region]

    def _select_homogeneous_indices(
        self, region_indices: np.ndarray, n_select: int
    ) -> np.ndarray:
        """Legacy farthest-point subset of ``region_indices`` (unit cube).

        Prefer ``_sample_hpd_stratified_volume_lh`` for new training/test fills.
        Kept for callers that still need chain indices.
        """
        region_indices = np.asarray(region_indices)
        n_region = len(region_indices)
        if n_region == 0:
            raise RuntimeError("Cannot select homogeneous points from an empty region")

        if n_region < n_select:
            logger.warning(
                f"Credible region has only {n_region} points, fewer than "
                f"requested {n_select}; sampling with replacement after "
                "taking the full region."
            )
            extra = np.random.choice(
                region_indices, size=n_select - n_region, replace=True
            )
            return np.concatenate([region_indices, extra])

        if n_region == n_select:
            return region_indices.copy()

        coords = np.asarray(self.unit_chain[region_indices], dtype=float)
        # Seed with the point closest to the region's centroid so the first
        # pick is central and stable across minor chain reorderings.
        centroid = coords.mean(axis=0)
        first = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
        selected_local = [first]
        min_sqdist = np.sum((coords - coords[first]) ** 2, axis=1)
        min_sqdist[first] = -np.inf

        for _ in range(1, n_select):
            nxt = int(np.argmax(min_sqdist))
            selected_local.append(nxt)
            d_new = np.sum((coords - coords[nxt]) ** 2, axis=1)
            min_sqdist = np.minimum(min_sqdist, d_new)
            min_sqdist[nxt] = -np.inf

        return region_indices[np.asarray(selected_local, dtype=int)]

    def _select_credible_region_indices(
        self, n_select: int, homogeneous: bool = False
    ) -> np.ndarray:
        """Select chain indices inside the tempered HPD credible region.

        The MCMC chain is ranked by posterior and the highest-posterior
        fraction (``test_credible_fraction``, default 0.95) is taken as the
        tempered HPD credible region. For an equal-weight MCMC sample, keeping
        the top 68% / 95% of points by posterior density is the 1-sigma /
        2-sigma HPD region in any dimension.

        Args:
            n_select: Number of points to draw from the credible region.
            homogeneous: If True, use legacy farthest-point sampling on chain
                indices. Prefer ``_sample_hpd_stratified_volume_lh`` for
                volume-uniform stratified HPD fills. If False, draw uniformly
                at random from the region.

        Returns:
            Array of indices into ``self.chain`` / ``self.unit_chain``.
        """
        region_indices = self._credible_region_indices()

        if homogeneous:
            return self._select_homogeneous_indices(region_indices, n_select)

        replace = len(region_indices) < n_select
        if replace:
            logger.warning(
                f"Credible region has only {len(region_indices)} points, fewer "
                f"than requested {n_select}; sampling with replacement."
            )
        return np.random.choice(region_indices, size=n_select, replace=replace)

    def _update_training_set(self, sample_results: List, sample: np.ndarray, 
                           unit_sample: np.ndarray) -> None:
        """Update training set with new samples."""
        valid_results = [s for s in sample_results if s is not None]
        
        #if not valid_results:
        #    logger.warning("No valid results in training sample update")
        #    return
        
        # For the case of rejected points due to additional priors, for instance, w0+wa<0
        if len(valid_results) < 0.8*self.resample_size:
            logger.warning(f"Only {len(valid_results)} samples passed chi2 filter, but need "
                           f"at least {0.8*self.resample_size} (80% of resample size) --> increase the resample size")
        if not valid_results:
            return
        new_data_vectors = self._stack_per_likelihood([s[1] for s in valid_results])
        new_likes = np.array([s[0] for s in valid_results])
        new_priors = np.array([s[2] for s in valid_results])
        new_posts = np.array([s[3] for s in valid_results])
        
        # Filter for valid indices
        valid_mask = np.array([s is not None for s in sample_results])
        valid_sample = sample[valid_mask]
        valid_unit_sample = unit_sample[valid_mask]
        
        # Append to existing arrays
        self.sample = np.vstack([self.sample, valid_sample])
        for name, block in new_data_vectors.items():
            self.sample_data_vectors[name] = np.vstack([self.sample_data_vectors[name], block])
        self.unit_sample = np.vstack([self.unit_sample, valid_unit_sample])
        self.sample_likes = np.concatenate([self.sample_likes, new_likes])
        self.sample_priors = np.concatenate([self.sample_priors, new_priors])
        self.sample_posts = np.concatenate([self.sample_posts, new_posts])
        self.points_per_iteration = np.concatenate([self.points_per_iteration, np.array([len(valid_results)])])

    def _remove_training_outliers(self) -> None:
        """Drop early training points with true chi2 far above the last iteration.

        Called once on the final training iteration when ``remove_outliers`` is
        True, after the last resample has been appended. Uses *true* pipeline
        likelihoods already stored on the training set (``chi2 = -2 log L``):

        - Let ``chi2_last`` be the chi2 values of the most recent
          ``points_per_iteration`` slice (last training iteration).
        - Threshold ``T = outlier_chi2_factor * max(chi2_last)`` (default
          factor 2.5).
        - Among the first ``outlier_prune_n_early`` slices only (default 2 =
          initial prior draw + first resampled iteration), remove points with
          ``chi2 > T``. Later iterations are never pruned.

        This avoids the old emulated-chain HPD / AABB cut, which can delete
        true-posterior support when the tempered emulator chain is biased or
        non-Gaussian. Points are stashed on ``self._pruned_training`` and
        written under ``pruned/...`` in ``total_training_set.npz``.

        Leave ``remove_outliers=False`` when the final sampler explores from
        the prior (e.g. Nautilus) and needs broad training coverage.
        """
        self._pruned_training = None
        n_before = len(self.unit_sample)
        if n_before == 0:
            return

        ppi = np.asarray(self.points_per_iteration, dtype=int)
        if ppi.size < 2 or int(ppi.sum()) != n_before:
            logger.warning(
                "remove_outliers: points_per_iteration (sum=%s, n=%d) inconsistent "
                "with training size (%d), or fewer than 2 iterations; skipping.",
                int(ppi.sum()) if ppi.size else None,
                int(ppi.size),
                n_before,
            )
            return

        factor = float(getattr(self, "outlier_chi2_factor", 2.5))
        n_early = int(getattr(self, "outlier_prune_n_early", 2))
        n_early = min(n_early, len(ppi) - 1)  # never treat the last slice as early
        if n_early < 1:
            logger.warning(
                "remove_outliers: no early iterations eligible to prune; skipping."
            )
            return

        likes = np.asarray(self.sample_likes, dtype=float)
        chi2 = -2.0 * likes
        iter_ids = np.repeat(np.arange(len(ppi)), ppi)

        last_id = len(ppi) - 1
        last_chi2 = chi2[iter_ids == last_id]
        if last_chi2.size == 0 or not np.any(np.isfinite(last_chi2)):
            logger.warning(
                "remove_outliers: no finite chi2 in last training iteration; skipping."
            )
            return

        chi2_ref = float(np.nanmax(last_chi2))
        if not np.isfinite(chi2_ref):
            logger.warning("remove_outliers: max(chi2_last) is non-finite; skipping.")
            return

        threshold = factor * chi2_ref
        early = iter_ids < n_early
        prune = early & np.isfinite(chi2) & (chi2 > threshold)
        keep = ~prune
        n_keep = int(keep.sum())
        n_pruned = int(prune.sum())

        if n_pruned == 0:
            logger.info(
                f"remove_outliers: no early points exceed chi2 > {factor:g} * "
                f"max(chi2_last)={chi2_ref:.3g} (threshold={threshold:.3g}); "
                f"kept all {n_before}."
            )
            return

        ndim = int(np.asarray(self.unit_sample).shape[1])
        min_keep = max(ndim + 1, 10)
        if n_keep < min_keep:
            logger.warning(
                f"remove_outliers would leave only {n_keep} training points "
                f"(need >= {min_keep}); skipping prune. Consider raising "
                f"outlier_chi2_factor (now {factor:g})."
            )
            return

        self._pruned_training = {
            "sample": self.sample[prune],
            "unit_sample": self.unit_sample[prune],
            "sample_data_vectors": {
                name: v[prune] for name, v in self.sample_data_vectors.items()
            },
            "sample_likes": self.sample_likes[prune],
            "sample_priors": self.sample_priors[prune],
            "sample_posts": self.sample_posts[prune],
            "chi2": chi2[prune],
            "chi2_threshold": threshold,
            "chi2_ref_last_max": chi2_ref,
            "outlier_chi2_factor": factor,
            "outlier_prune_n_early": n_early,
            "points_per_iteration_before_prune": ppi.copy(),
        }

        new_ppi = np.bincount(iter_ids[keep], minlength=len(ppi)).astype(int)

        self.sample = self.sample[keep]
        self.unit_sample = self.unit_sample[keep]
        self.sample_likes = self.sample_likes[keep]
        self.sample_priors = self.sample_priors[keep]
        self.sample_posts = self.sample_posts[keep]
        self.sample_data_vectors = {
            name: v[keep] for name, v in self.sample_data_vectors.items()
        }
        self.points_per_iteration = new_ppi

        early_before = int(np.sum(ppi[:n_early]))
        early_after = int(np.sum(new_ppi[:n_early]))
        logger.info(
            f"remove_outliers: pruned {n_pruned}/{early_before} early points "
            f"(iterations 0..{n_early - 1}) with chi2 > {factor:g} * "
            f"max(chi2_last)={chi2_ref:.3g} (threshold={threshold:.3g}); "
            f"kept {n_keep}/{n_before} total "
            f"(early {early_before}->{early_after}). "
            f"points_per_iteration: {ppi.tolist()} -> {new_ppi.tolist()}. "
            "Stashed outliers for total_training_set.npz."
        )

    def _update_test_set(self, sample_results_test: List, sample_test: np.ndarray,
                        unit_sample_test: np.ndarray) -> None:
        """Update test set with new samples."""
        valid_results = [s for s in sample_results_test if s is not None]
        
        if not valid_results:
            logger.warning("No valid results in test sample update")
            return
        
        new_data_vectors = self._stack_per_likelihood([s[1] for s in valid_results])
        new_likes = np.array([s[0] for s in valid_results])
        new_priors = np.array([s[2] for s in valid_results])
        new_posts = np.array([s[3] for s in valid_results])
        
        # Filter for valid indices
        valid_mask = np.array([s is not None for s in sample_results_test])
        valid_sample = sample_test[valid_mask]
        valid_unit_sample = unit_sample_test[valid_mask]

        # Append to test arrays
        self.sample_test = np.vstack([self.sample_test, valid_sample])
        for name, block in new_data_vectors.items():
            self.sample_data_vectors_test[name] = np.vstack(
                [self.sample_data_vectors_test[name], block]
            )
        self.unit_sample_test = np.vstack([self.unit_sample_test, valid_unit_sample])
        self.sample_likes_test = np.concatenate([self.sample_likes_test, new_likes])
        self.sample_priors_test = np.concatenate([self.sample_priors_test, new_priors])
        self.sample_posts_test = np.concatenate([self.sample_posts_test, new_posts])
        self.points_per_iteration_test = np.concatenate([self.points_per_iteration_test, np.array([len(valid_results)])])

    def _save_datasets(self) -> None:
        """Save training and test datasets."""
        logger.info("Saving training and test datasets")
        
        # Training set (active points used for the final emulator)
        training_dict = self._build_dataset_dict(
            self.sample, self.unit_sample, self.sample_data_vectors,
            self.sample_likes, self.sample_priors, self.sample_posts,
            self.points_per_iteration
        )
        pruned = getattr(self, "_pruned_training", None)
        if pruned is not None:
            training_dict.update(self._build_pruned_dataset_entries(pruned))
            logger.info(
                f"Including {len(pruned['sample_likes'])} pruned outlier "
                "points under pruned/* keys in total_training_set.npz"
            )
        np.savez(f'{self.save_dir}/total_training_set.npz', **training_dict)
        
        # Test set
        test_dict = self._build_dataset_dict(
            self.sample_test, self.unit_sample_test, self.sample_data_vectors_test,
            self.sample_likes_test, self.sample_priors_test, self.sample_posts_test,
            self.points_per_iteration_test
        )
        np.savez(f'{self.save_dir}/total_testing_set.npz', **test_dict)

    def _build_pruned_dataset_entries(self, pruned: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize stashed outlier points under ``pruned/...`` keys."""
        param_names = [str(param) for param in self.pipeline.varied_params]
        sample = pruned["sample"]
        unit_sample = pruned["unit_sample"]
        data_vectors = pruned["sample_data_vectors"]
        chi2 = pruned.get("chi2")
        if chi2 is None:
            chi2 = -2.0 * np.asarray(pruned["sample_likes"], dtype=float)
        entries: Dict[str, Any] = {
            "pruned/chi2": np.asarray(chi2, dtype=float),
            "pruned/likes": pruned["sample_likes"],
            "pruned/priors": pruned["sample_priors"],
            "pruned/posts": pruned["sample_posts"],
            "pruned/n_points": np.asarray(len(pruned["sample_likes"])),
        }
        if "chi2_threshold" in pruned:
            entries["pruned/chi2_threshold"] = np.asarray(pruned["chi2_threshold"])
        if "chi2_ref_last_max" in pruned:
            entries["pruned/chi2_ref_last_max"] = np.asarray(pruned["chi2_ref_last_max"])
        if "outlier_chi2_factor" in pruned:
            entries["pruned/outlier_chi2_factor"] = np.asarray(pruned["outlier_chi2_factor"])
        if "outlier_prune_n_early" in pruned:
            entries["pruned/outlier_prune_n_early"] = np.asarray(
                pruned["outlier_prune_n_early"]
            )
        # Legacy keys from the old HPD-box prune (harmless if absent).
        if "outlier_nsigma" in pruned:
            entries["pruned/outlier_nsigma"] = np.asarray(pruned["outlier_nsigma"])
        if "mahalanobis_d" in pruned:
            entries["pruned/mahalanobis_d"] = pruned["mahalanobis_d"]
        if "points_per_iteration_before_prune" in pruned:
            entries["points_per_iteration_before_prune"] = pruned[
                "points_per_iteration_before_prune"
            ]
        for i, param in enumerate(param_names):
            entries[f"pruned/{param}"] = sample[:, i]
            entries[f"pruned/{param}--norm"] = unit_sample[:, i]

        likelihood_names = list(data_vectors.keys())
        for name, block in data_vectors.items():
            entries[f"pruned/features/{name}"] = block
        concatenated = (
            np.concatenate([data_vectors[n] for n in likelihood_names], axis=1)
            if likelihood_names and all(len(data_vectors[n]) for n in likelihood_names)
            else np.empty((0, 0))
        )
        entries["pruned/features"] = concatenated
        entries["pruned/likelihood_names"] = likelihood_names
        return entries

    def _build_dataset_dict(self, sample: np.ndarray, unit_sample: np.ndarray,
                           data_vectors: Dict[str, np.ndarray], likes: np.ndarray,
                           priors: np.ndarray, posts: np.ndarray,
                           points_per_iteration: np.ndarray) -> Dict[str, Any]:
        """Build dataset dictionary for saving.

        ``data_vectors`` is a per-likelihood dict. We save each likelihood's
        2D array under ``features/{likelihood_name}`` and also write the
        concatenated features for convenience, plus per-likelihood sizes.
        """
        param_names = [str(param) for param in self.pipeline.varied_params]
        
        dataset_dict = {}
        
        for i, param in enumerate(param_names):
            dataset_dict[param] = sample[:, i]
            dataset_dict[f"{param}--norm"] = unit_sample[:, i]

        likelihood_names = list(data_vectors.keys())
        concatenated = (
            np.concatenate([data_vectors[n] for n in likelihood_names], axis=1)
            if likelihood_names and all(len(data_vectors[n]) for n in likelihood_names)
            else np.empty((0, 0))
        )
        for name, block in data_vectors.items():
            dataset_dict[f"features/{name}"] = block

        dataset_dict.update({
            'fixed_keys': [str(key) for key in self.fixed_keys] if self.fixed_keys else '',
            'fixed_features': self.fixed_vector,
            'output_keys': [str(key) for key in self.keys] if self.keys else 'data_vector',
            'likelihood_names': likelihood_names,
            'features_size': np.array(
                [self.data_vector_sizes[n] for n in likelihood_names], dtype=int
            ),
            'features': concatenated,
            'chi2': likes,
            'priors': priors,
            'posts': posts,
            'points_per_iteration': points_per_iteration
        })
        
        return dataset_dict


