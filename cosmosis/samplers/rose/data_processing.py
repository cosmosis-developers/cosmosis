"""
Data processing and sample generation for ROSE sampler.

This module contains methods for generating training samples, processing results,
and managing training/test datasets.
"""

import logging
from typing import List, Dict, Any
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
        """Generate additional training samples from MCMC chain.
        
        Training points are drawn from the tempered HPD credible region of the
        current MCMC chain (``test_credible_fraction``, default 95% / 2-sigma)
        and spaced as evenly as possible in unit-prior parameter space so the
        emulator is not biased toward high-density posterior pockets. Test
        points (final training iteration only) use the same region with a
        simpler uniform draw.
        """
        logger.info(
            f"Selecting {self.resample_size} homogeneous training samples "
            f"from the {self.test_credible_fraction:.0%} tempered credible "
            "region of the MCMC chain"
        )
        
        chain_length = len(self.chain)
        if chain_length == 0:
            raise RuntimeError(
                "Cannot resample training points: the MCMC chain is empty. "
                "This usually means the sampler converged before the configured "
                "emcee_samples were reached and the burn-in then discarded "
                "everything. Lower emcee_burn or increase emcee_samples."
            )

        train_indices = self._select_credible_region_indices(
            self.resample_size, homogeneous=True
        )
        unit_sample = self.unit_chain[train_indices]
        sample = self.chain[train_indices]

        # Test points are collected only once, during the last training
        # iteration (i.e. just before the final sampling stage), drawn from the
        # 1-sigma / credible region of the one-before-last MCMC chain (the chain
        # currently stored in self.chain). In all earlier iterations no test
        # points are evaluated.
        collect_test = (
            self.iterations == (self.max_iterations - 1)
            and self.final_test_size > 0
        )
        sample_test = None
        unit_sample_test = None
        if collect_test:
            test_indices = self._select_credible_region_indices(
                self.final_test_size, homogeneous=False
            )
            unit_sample_test = self.unit_chain[test_indices]
            sample_test = self.chain[test_indices]
            logger.info(
                f"Collecting {len(sample_test)} test points from the "
                f"{self.test_credible_fraction:.0%} credible region of the "
                "one-before-last MCMC chain (final training iteration)"
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

        # Final training iteration: optionally prune early exploration points
        # that fall outside the n-sigma region of the last tempered chain before
        # the final emulator is trained (and before datasets are saved).
        if (
            self.iterations == (self.max_iterations - 1)
            and getattr(self, "remove_outliers", False)
        ):
            self._remove_training_outliers()
        
        # Save datasets if requested
        if self.save_output == SAVE_ALL and self.iterations == (self.max_iterations - 1):
            self._save_datasets()

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
        """Space-filling subset of ``region_indices`` in unit-parameter space.

        Uses greedy farthest-point (maximin) sampling on ``self.unit_chain`` so
        selected points cover the credible region as evenly as possible and the
        training set is not concentrated where the MCMC density is highest.
        Distances are Euclidean in the unit-prior cube, which puts every varied
        parameter on a common [0, 1] scale.
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
            homogeneous: If True, use farthest-point sampling in unit-parameter
                space (preferred for training). If False, draw uniformly at
                random from the region (sufficient for testing).

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
        """Move out-of-region training points aside for the final emulator.

        Called once, on the final training iteration, when ``remove_outliers`` is
        True. ``outlier_nsigma`` is interpreted with the same HPD convention used
        elsewhere in ROSE (1 / 2 / 3 sigma ↔ 68% / 95% / 99.7% highest-posterior
        mass via ``erf(n/sqrt(2))``). Training points that fall outside the
        axis-aligned bounding box of that tempered HPD region in unit-prior
        space are removed from the active training set but retained on
        ``self._pruned_training`` and written into ``total_training_set.npz``
        under ``pruned/...`` keys so no evaluated points are lost.
        ``points_per_iteration`` is recomputed from the keep mask so each
        entry still counts how many points from that iteration survived.

        A fixed Mahalanobis ``d <= n`` cut is *not* used: in moderate dimension
        (e.g. 6 parameters) that ellipsoid is much smaller than the n-sigma HPD
        region and can wipe out most of the training set.

        Leave ``remove_outliers=False`` when the final sampler (e.g. Nautilus)
        explores from the prior and benefits from retaining those broader
        training points in the emulator itself.
        """
        import math

        self._pruned_training = None
        unit_chain = getattr(self, "unit_chain", None)
        if unit_chain is None or len(unit_chain) == 0:
            logger.warning(
                "remove_outliers=True but no tempered chain is available; "
                "skipping outlier removal."
            )
            return

        n_before = len(self.unit_sample)
        if n_before == 0:
            return

        nsigma = float(self.outlier_nsigma)
        # 1D Gaussian mass for n-sigma; matches ROSE's 68/95/99.7 HPD labels.
        fraction = math.erf(nsigma / math.sqrt(2.0))
        fraction = min(max(fraction, 1e-6), 1.0)
        region_indices = self._credible_region_indices(fraction=fraction)
        region_unit = np.asarray(self.unit_chain[region_indices], dtype=float)
        if len(region_unit) == 0:
            logger.warning(
                "remove_outliers: empty credible region; skipping prune."
            )
            return

        lo = region_unit.min(axis=0)
        hi = region_unit.max(axis=0)
        # Tiny padding so points sitting on the HPD boundary are not clipped
        # by floating-point noise.
        pad = 1e-8 * np.maximum(hi - lo, 1e-12)
        train = np.asarray(self.unit_sample, dtype=float)
        keep = np.all((train >= (lo - pad)) & (train <= (hi + pad)), axis=1)
        n_keep = int(keep.sum())

        if n_keep == n_before:
            logger.info(
                f"remove_outliers: all {n_before} training points already lie "
                f"within the {nsigma:g}-sigma ({fraction:.1%}) HPD box of the "
                "last tempered chain."
            )
            return

        ndim = train.shape[1]
        min_keep = max(ndim + 1, 10)
        if n_keep < min_keep:
            logger.warning(
                f"remove_outliers would leave only {n_keep} training points "
                f"(need >= {min_keep}); skipping prune to avoid an unusable "
                "training set. Consider a larger outlier_nsigma."
            )
            return

        pruned = ~keep
        # Diagnostic: Mahalanobis distance to the HPD cloud (not used for the cut).
        mu = region_unit.mean(axis=0)
        cov = np.cov(region_unit, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.array([[float(cov)]])
        cov = np.asarray(cov, dtype=float) + np.eye(ndim) * 1e-12
        try:
            prec = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(cov)
        diff = train[pruned] - mu
        d2 = np.einsum("ni,ij,nj->n", diff, prec, diff)

        self._pruned_training = {
            "sample": self.sample[pruned],
            "unit_sample": self.unit_sample[pruned],
            "sample_data_vectors": {
                name: v[pruned] for name, v in self.sample_data_vectors.items()
            },
            "sample_likes": self.sample_likes[pruned],
            "sample_priors": self.sample_priors[pruned],
            "sample_posts": self.sample_posts[pruned],
            "mahalanobis_d": np.sqrt(np.maximum(d2, 0.0)),
            "outlier_nsigma": nsigma,
            "credible_fraction": fraction,
            "unit_lo": lo,
            "unit_hi": hi,
            "points_per_iteration_before_prune": np.asarray(
                self.points_per_iteration, dtype=int
            ).copy(),
        }

        # Recompute per-iteration counts on the pruned (still chronologically
        # ordered) arrays so progression plots can still slice by iteration.
        # Collapsing to ``[n_keep]`` used to wipe the history (e.g. turning
        # ``[80, 40, 40, 40, 40]`` into ``[140]``).
        ppi = np.asarray(self.points_per_iteration, dtype=int)
        if ppi.size > 0 and int(ppi.sum()) == n_before:
            iter_ids = np.repeat(np.arange(len(ppi)), ppi)
            new_ppi = np.bincount(iter_ids[keep], minlength=len(ppi)).astype(int)
        else:
            logger.warning(
                "remove_outliers: points_per_iteration (sum=%s) inconsistent "
                "with training size (%d); falling back to a single count.",
                int(ppi.sum()) if ppi.size else None,
                n_before,
            )
            new_ppi = np.array([n_keep], dtype=int)

        self.sample = self.sample[keep]
        self.unit_sample = self.unit_sample[keep]
        self.sample_likes = self.sample_likes[keep]
        self.sample_priors = self.sample_priors[keep]
        self.sample_posts = self.sample_posts[keep]
        self.sample_data_vectors = {
            name: v[keep] for name, v in self.sample_data_vectors.items()
        }
        self.points_per_iteration = new_ppi

        logger.info(
            f"remove_outliers: kept {n_keep}/{n_before} training points inside "
            f"the {nsigma:g}-sigma ({fraction:.1%} HPD) unit-space box of the "
            f"last tempered chain; stashed {n_before - n_keep} outliers for "
            f"total_training_set.npz. points_per_iteration: {ppi.tolist()} -> "
            f"{new_ppi.tolist()}."
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
        entries: Dict[str, Any] = {
            "pruned/outlier_nsigma": np.asarray(pruned["outlier_nsigma"]),
            "pruned/mahalanobis_d": pruned["mahalanobis_d"],
            "pruned/chi2": pruned["sample_likes"],
            "pruned/priors": pruned["sample_priors"],
            "pruned/posts": pruned["sample_posts"],
            "pruned/n_points": np.asarray(len(pruned["sample_likes"])),
        }
        if "credible_fraction" in pruned:
            entries["pruned/credible_fraction"] = np.asarray(pruned["credible_fraction"])
        if "unit_lo" in pruned:
            entries["pruned/unit_lo"] = pruned["unit_lo"]
            entries["pruned/unit_hi"] = pruned["unit_hi"]
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


