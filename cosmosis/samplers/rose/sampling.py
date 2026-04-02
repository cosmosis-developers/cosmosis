"""
MCMC sampling methods for ROSE sampler.

This module contains methods for running MCMC sampling using emcee, nautilus, and NUTS,
and processing the resulting chains.
"""

import logging
import os
from functools import partial
from timeit import default_timer
from typing import Any, Tuple

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

    def _run_emcee_sampling(self, tempering: float) -> None:
        """Run emcee MCMC sampling."""
        import emcee
        # Ensure emu_pipeline is set up on all processes before MCMC starts
        # This is critical for MPI parallelization where worker processes need
        # access to emu_pipeline for log probability evaluation
        if self.emu_pipeline is None:
            logger.warning("emu_pipeline is None, setting it up now (this should happen in execute())")
            self.compute_fiducial_setup_emu_pipeline()
            # Update global sampler reference after setup
            utils_module._sampler = self
        
        # Use module-level function (can be pickled for MPI) and pass sampler/tempering via args
        # emcee will call: log_probability_function(u, tempering)
        emcee_sampler = emcee.EnsembleSampler(
            self.emcee_walkers,
            self.ndim,
            log_probability_function,
            args=[tempering],
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
        logger.info('tau: %s', tau_all_params)
        logger.info(f"MCMC sampling took {end_time - start_time:.1f} seconds")
        
        # Process MCMC results
        self._process_mcmc_results(emcee_sampler, tempering)

    def _run_nautilus_sampling(self, tempering: float) -> None:
        """Run nautilus sampling for final iteration."""
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
        
        # Capture current model path so workers (which never run execute()) load the same emulator
        # when the pool pickles these callables.
        current_model_path = os.path.join(self.save_outputs_dir, f'emumodel_{self.iterations + 1}')
        # When using a pool, workers load from disk; ensure the current model is saved there
        # even if save_outputs is not "all" (train_emulator always saves, but this covers
        # any case where the file might be missing).
        if self.pool is not None and self.emulator is not None:
            info_file = current_model_path + ".npz"
            if not os.path.exists(info_file):
                logger.info("Saving current emulator to disk so worker processes can load it")
                self.emulator.save_to(current_model_path)
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
            resume=False,  # Don't resume for ROSE
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
        
        logger.info(f"Nautilus sampling took {end_time - start_time:.1f} seconds")
        
        # Process nautilus results
        self._process_nautilus_results(nautilus_sampler, tempering)

    def _verify_nuts_gradients(self, test_params: np.ndarray, tempering: float,
                               emulator, X_mean_tf, X_std_tf, y_mean_tf, y_std_tf,
                               data_vector_tf, inv_covariance_tf, DTYPE) -> None:
        """Verify gradient computation by comparing autodiff with finite differences."""
        import tensorflow as tf
        
        logger.info("Verifying NUTS gradient computation...")
        
        # Convert test params to TensorFlow
        test_params_tf = tf.constant(test_params, dtype=DTYPE)
        
        # Compute gradient using TensorFlow autodiff
        with tf.GradientTape() as tape:
            tape.watch(test_params_tf)
            log_prob_tf = self._log_prob_nuts_impl(
                test_params_tf, emulator, X_mean_tf, X_std_tf,
                y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
                tempering, DTYPE
            )
        
        grad_autodiff = tape.gradient(log_prob_tf, test_params_tf)
        
        if grad_autodiff is None:
            logger.error("Autodiff gradient is None! Gradient computation may be broken.")
            return
        
        grad_autodiff_np = grad_autodiff.numpy()
        
        # Compute gradient using finite differences
        eps = 1e-5
        grad_finite_diff = np.zeros_like(test_params)
        
        # Get baseline log probability
        log_prob_base = self._log_prob_nuts_impl(
            test_params_tf, emulator, X_mean_tf, X_std_tf,
            y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
            tempering, DTYPE, TF=True
        ).numpy()

        # Get baseline log probability
        log_prob_base_no_tf = self._log_prob_nuts_impl(
            test_params_tf, emulator, X_mean_tf, X_std_tf,
            y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
            tempering, DTYPE, TF=False
        ).numpy()


        for i in range(len(test_params)):
            # Forward difference
            params_forward = test_params.copy()
            params_forward[i] += eps
            params_forward_tf = tf.constant(params_forward, dtype=DTYPE)
            
            log_prob_forward = self._log_prob_nuts_impl(
                params_forward_tf, emulator, X_mean_tf, X_std_tf,
                y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
                tempering, DTYPE, False
            ).numpy()
            
            # Central difference (more accurate)
            params_backward = test_params.copy()
            params_backward[i] -= eps
            params_backward_tf = tf.constant(params_backward, dtype=DTYPE)
            
            log_prob_backward = self._log_prob_nuts_impl(
                params_backward_tf, emulator, X_mean_tf, X_std_tf,
                y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
                tempering, DTYPE, False
            ).numpy()
            
            # Central difference gradient
            #print(f"Difference in logP: {(log_prob_forward - log_prob_backward)}")
            grad_finite_diff[i] = (log_prob_forward - log_prob_backward) / (2 * eps)
        
        # Compare gradients
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
            
            # Compute relative error (handle zero gradients)
            if abs(autodiff_val) > 1e-10 or abs(finite_diff_val) > 1e-10:
                rel_error = abs(autodiff_val - finite_diff_val) / max(abs(autodiff_val), abs(finite_diff_val), 1e-10)
            else:
                rel_error = abs(autodiff_val - finite_diff_val)
            
            max_rel_error = max(max_rel_error, rel_error)
            
            logger.info(f"{param_name.name:20s} | {autodiff_val:15.6e} | {finite_diff_val:15.6e} | {rel_error:10.2e}")
        
        logger.info("-" * 70)
        logger.info(f"Maximum relative error: {max_rel_error:.2e}")
        
        if max_rel_error > 0.1:
            logger.warning(f"Large gradient discrepancy detected! Max relative error: {max_rel_error:.2e}")
            logger.warning("This may cause poor NUTS exploration. Check gradient computation.")
        elif max_rel_error > 0.01:
            logger.warning(f"Moderate gradient discrepancy. Max relative error: {max_rel_error:.2e}")
        else:
            logger.info("Gradients match well. Gradient computation appears correct.")
        
        logger.info("=" * 60)

    def _run_nuts_sampling(self, tempering: float) -> None:
        """Run NUTS sampling using TensorFlow Probability."""
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
        
        logger.info(f"Starting NUTS sampling with step_size={self.nuts_step_size}, "
                   f"max_tree_depth={self.nuts_max_tree_depth}, num_results={self.nuts_num_results}")
        
        # Get initial state in unit hypercube
        if self.trained_before:
            # Use randomized start from pipeline
            initial_state_unit = self.pipeline.randomized_start()
        else:
            # Use last training sample
            if len(self.unit_sample) == 0:
                raise RuntimeError("No training samples available for NUTS initialization")
            initial_state_unit = self.unit_sample[-1]
        
        # Convert to physical space for initial state
        try:
            initial_state_physical = self.pipeline.denormalize_vector_from_prior(initial_state_unit)
        except ValueError:
            # If denormalization fails, try to get a valid starting point
            logger.warning("Initial state denormalization failed, using pipeline default")
            initial_state_physical = np.array([
                #self.pipeline.prior_info[i].start for i in range(self.ndim)
                self.pipeline.varied_params[i].start for i in range(self.ndim)
            ])
            initial_state_unit = self.pipeline.normalize_vector_to_prior(initial_state_physical)
        
        # Create TensorFlow-compatible log probability function with autodiff
        # Similar to compute_gradients, but for log posterior
        # Access emulator (stored in both self.emulator and self.emu_module.data.emulator)
        emulator = self.emulator
        if emulator is None:
            raise RuntimeError("Emulator not set. This should be set during training or loading.")
        DTYPE = tf.float32
        
        # Get normalization constants
        X_mean_tf = tf.constant([emulator.X_mean[key] for key in emulator.model_parameters], dtype=DTYPE)
        X_std_tf = tf.constant([emulator.X_std[key] for key in emulator.model_parameters], dtype=DTYPE)
        y_mean_tf = tf.constant(emulator.y_mean, dtype=DTYPE)
        y_std_tf = tf.constant(emulator.y_std, dtype=DTYPE)
        
        # Get data vector and inverse covariance from pipeline for likelihood computation
        # Extract these from a sample run to use as TensorFlow constants
        data_vector_tf = None
        inv_covariance_tf = None
        logger.info(f"Initial state for NUTS: {initial_state_physical}")
        try:
            like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, block = utils_module.task(initial_state_physical, self, True)
            logger.info(f"Extracted {len(data_vectors)} data vectors and {len(data_inv_covariance)} covariance matrices")
            # For now let's test just one probe's data vector and inverse covariance
            # TODO: Handle multiple probes properly
            data_vector_tf = tf.constant(np.atleast_1d(data_vectors[0]), dtype=DTYPE)
            inv_covariance_tf = tf.constant(np.atleast_2d(data_inv_covariance[0]), dtype=DTYPE)
            logger.info(f"Data vector shape: {data_vector_tf.shape}, Inv covariance shape: {inv_covariance_tf.shape}")
        except Exception as e:
            logger.warning(f"Could not get sample run for NUTS data extraction: {e}")
            logger.warning("NUTS will use fallback likelihood computation (slower, no gradients)")
        
        # Verify gradient computation by comparing with finite differences
        # This helps diagnose if poor exploration is due to incorrect gradients
        self._verify_nuts_gradients(initial_state_physical, tempering, emulator, 
                                     X_mean_tf, X_std_tf, y_mean_tf, y_std_tf, 
                                     data_vector_tf, inv_covariance_tf, DTYPE)

        # Optionally sample in unit (prior) space for better exploration of Omega_m, w, wa.
        # In unit space all parameters are on [0,1], so geometry is better for HMC/NUTS.
        if self.nuts_sample_unit_space:
            param_low = np.array([p.limits[0] for p in self.pipeline.varied_params], dtype=np.float32)
            param_high = np.array([p.limits[1] for p in self.pipeline.varied_params], dtype=np.float32)
            low_tf = tf.constant(param_low, dtype=DTYPE)
            high_tf = tf.constant(param_high, dtype=DTYPE)
            log_jacobian_const = tf.constant(
                np.sum(np.log(param_high - param_low)), dtype=DTYPE
            )
            logger.info("NUTS sampling in unit (prior) space for better exploration of correlated parameters")

            def log_prob_nuts(state_tf):
                # state_tf is in [0,1]^d; transform to physical and add log Jacobian
                state_1d = tf.reshape(state_tf, [-1])
                physical = low_tf + state_1d * (high_tf - low_tf)
                log_prob_physical = self._log_prob_nuts_impl(
                    physical, emulator, X_mean_tf, X_std_tf,
                    y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf, tempering, DTYPE
                )
                return log_prob_physical + log_jacobian_const
        else:
            def log_prob_nuts(physical_params_tf):
                return self._log_prob_nuts_impl(
                    physical_params_tf, emulator, X_mean_tf, X_std_tf,
                    y_mean_tf, y_std_tf, data_vector_tf,
                    inv_covariance_tf, tempering, DTYPE
                )
    
        # Initialize NUTS kernel: use unit or physical state depending on option
        if self.nuts_sample_unit_space:
            initial_state = tf.constant(np.asarray(initial_state_unit, dtype=np.float32), dtype=DTYPE)
        else:
            initial_state = tf.constant(initial_state_physical, dtype=DTYPE)
    
        # Create NUTS kernel (use configured values so ini tuning has effect)
        nuts_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=log_prob_nuts,
            step_size=self.nuts_step_size,
            max_tree_depth=self.nuts_max_tree_depth,
            max_energy_diff=self.nuts_max_energy_diff,
            name='nuts_kernel'
        )
        logger.info(f"NUTS kernel initialized with step_size={self.nuts_step_size}, "
                   f"max_tree_depth={self.nuts_max_tree_depth}")
        
        # Adaptive step size: lower target_accept_prob (e.g. 0.65–0.75) often improves exploration
        adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=self.nuts_num_adaptation_steps,
            target_accept_prob=self.nuts_target_accept_prob,
        )
        logger.info(f"Adaptive kernel: {self.nuts_num_adaptation_steps} adaptation steps, "
                   f"target_accept_prob={self.nuts_target_accept_prob}")
        # Run sampling
        start_time = default_timer()

        # Progress reporting: trace_fn is called every step; we print every N steps so the run is not silent.
        # Without this, the first @tf.function trace runs the full chain with no output (appears to freeze).
        progress_interval = max(1, int(self.nuts_progress_interval)) if self.nuts_progress_interval > 0 else 0
        step_counter = tf.Variable(0, dtype=tf.int32) if progress_interval > 0 else None

        if progress_interval > 0:
            def trace_fn(_, pkr):
                step_counter.assign_add(1)
                tf.cond(
                    tf.equal(tf.math.mod(step_counter, progress_interval), 0),
                    lambda: (tf.print("ROSE NUTS progress:", step_counter, "steps"), tf.constant(0, dtype=tf.int32))[1],
                    lambda: tf.constant(0, dtype=tf.int32),
                )
                return pkr
        else:
            def trace_fn(_, pkr):
                return pkr

        # Define once outside the loop so we only trace the graph once (first call is slow, then reuse).
        @tf.function
        def run_chain(chain_initial, chain_idx_tensor):
            return tfp.mcmc.sample_chain(
                num_results=self.nuts_num_results,
                num_burnin_steps=self.nuts_num_burnin_steps,
                current_state=chain_initial,
                kernel=adaptive_kernel,
                trace_fn=trace_fn,
                seed=self.seed + int(chain_idx_tensor) if self.seed is not None else None,
            )

        if progress_interval > 0:
            logger.info(
                f"Running {self.nuts_num_chains} chain(s) with {self.nuts_num_results} samples each "
                f"(burnin={self.nuts_num_burnin_steps}). First run may take 5–15 min while TensorFlow compiles; "
                f"progress every {progress_interval} steps."
            )
        else:
            logger.info(
                f"Running {self.nuts_num_chains} chain(s) with {self.nuts_num_results} samples each "
                f"(burnin={self.nuts_num_burnin_steps}). First run may take 5–15 min while TensorFlow compiles."
            )
        
        # Run multiple chains if requested
        all_chains = []
        all_log_probs = []
        all_blobs = []
        logger.info(f"Running {self.nuts_num_chains} chain(s) with {self.nuts_num_results} samples each")

        for chain_idx in range(self.nuts_num_chains):
            if self.nuts_num_chains > 1:
                # Slightly perturb initial state for each chain
                #perturbation_scale = 0.0001 * tf.reduce_max(tf.abs(initial_state)) + 1e-8
                perturbation_scale = 0.1 * tf.reduce_max(tf.abs(initial_state)) 
                chain_initial = initial_state + tf.random.normal(
                    shape=initial_state.shape,
                    stddev=perturbation_scale,
                    dtype=tf.float32,
                    seed=self.seed + chain_idx if self.seed is not None else None
                )
                logger.info(f"Chain {chain_idx + 1}: perturbed initial state by ~{float(perturbation_scale):.4f}")
            else:
                chain_initial = initial_state
                logger.info(f"Chain {chain_idx + 1}: using initial state")

            if step_counter is not None:
                step_counter.assign(0)
            samples, trace = run_chain(chain_initial, tf.constant(chain_idx, dtype=tf.int32))

            # Convert to numpy (per chain)
            samples_np = samples.numpy()
            if len(samples_np.shape) == 1:
                samples_np = samples_np.reshape(-1, 1)

            # If we sampled in unit space, convert to physical for run_results and storage
            if self.nuts_sample_unit_space:
                samples_physical = np.array([
                    self.pipeline.denormalize_vector_from_prior(samples_np[i])
                    for i in range(len(samples_np))
                ])
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
                    chain_blobs.append((-np.inf, [np.nan] * self.pipeline.number_extra))

            all_chains.append(samples_physical)
            all_log_probs.append(chain_log_probs)
            all_blobs.append(chain_blobs)
        
        # Combine chains (after loop)
        if self.nuts_num_chains > 1:
            self.chain = np.vstack(all_chains)
            self.nuts_logp = np.concatenate(all_log_probs)
            self.blobs = [item for sublist in all_blobs for item in sublist]
        else:
            # For single chain, use the results from the loop
            self.chain = all_chains[0] if all_chains else np.array([])
            self.nuts_logp = np.array(all_log_probs[0]) if all_log_probs else np.array([])
            self.blobs = all_blobs[0] if all_blobs else []
        
        # Convert to unit cube for output consistency
        self.unit_chain = np.array([
            self.pipeline.normalize_vector_to_prior(p) for p in self.chain
        ])
        
        end_time = default_timer()
        logger.info(f"NUTS sampling took {end_time - start_time:.1f} seconds")
        
        # Process NUTS results
        self._process_nuts_results(tempering)

    
    def _log_prob_nuts_impl(self, physical_params_tf, emulator, X_mean_tf, X_std_tf,
                            y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf, 
                            tempering, DTYPE, TF=True):
        """Implementation of log probability function for NUTS."""
        import tensorflow as tf
        from ...runtime.prior import UniformPrior, GaussianPrior, TruncatedGaussianPrior
        
        # Ensure 1D
        if len(physical_params_tf.shape) > 1:
            physical_params_tf = tf.reshape(physical_params_tf, [-1])
        
        # Compute predictions through emulator (fully differentiable)
        # Use the input tensor directly - TensorFlow will track gradients automatically
        params_norm = (physical_params_tf - X_mean_tf) / X_std_tf
        params_nn_norm = (params_norm - emulator.cp_nn.parameters_mean) / emulator.cp_nn.parameters_std
        
        if len(params_nn_norm.shape) == 1:
            params_nn_norm = tf.expand_dims(params_nn_norm, 0)
        
        x = params_nn_norm
        layers = [x]
        
        if emulator.cp_nn.architecture_type == "MLP":
            for i in range(emulator.cp_nn.n_layers - 1):
                linear_out = tf.matmul(layers[-1], emulator.cp_nn.W[i]) + emulator.cp_nn.b[i]
                activated = emulator.cp_nn.activation(linear_out, emulator.cp_nn.alphas[i], emulator.cp_nn.betas[i])
                layers.append(activated)
            output = tf.matmul(layers[-1], emulator.cp_nn.W[-1]) + emulator.cp_nn.b[-1]
        
        pred_norm = output * emulator.cp_nn.features_std + emulator.cp_nn.features_mean
        if len(pred_norm.shape) == 2:
            pred_norm = pred_norm[0]
        
        pred_intermediate = pred_norm * y_std_tf + y_mean_tf
        
        if emulator.data_trafo == 'log_norm':
            predictions = tf.pow(10.0, pred_intermediate)
        elif emulator.data_trafo == 'signed_log_norm':
            predictions = tf.sign(pred_intermediate) * tf.multiply(s, tf.subtract(tf.pow(10.0, tf.abs(pred_intermediate)), 1.0))
        elif emulator.data_trafo == 'norm':
            predictions = pred_intermediate
        elif emulator.data_trafo == 'PCA':
            pca_matrix_tf = tf.constant(emulator.pca_transform_matrix, dtype=DTYPE)
            features_std_tf = tf.constant(emulator.features_std, dtype=DTYPE)
            features_mean_tf = tf.constant(emulator.features_mean, dtype=DTYPE)
            pca_reconstructed = tf.matmul(tf.expand_dims(pred_intermediate, 0), pca_matrix_tf)
            predictions = (pca_reconstructed * features_std_tf + features_mean_tf)[0]
        else:
            predictions = pred_intermediate
        
        # Compute log_likelihood from predictions using TensorFlow operations
        if data_vector_tf is not None and inv_covariance_tf is not None and TF:
            # Gaussian likelihood: -0.5 * (x - mu)^T * C^-1 * (x - mu)
            diff = predictions - data_vector_tf
            # Compute chi-squared: diff^T * inv_cov * diff
            # Use einsum with explicit scalar output ('->' means output is scalar)
            chi2 = tf.einsum('i,ij,j->', diff, inv_covariance_tf, diff)
            log_like = -0.5 * chi2
            # Ensure it's a scalar (explicit reshape to empty shape)
            log_like = tf.reshape(log_like, [])
            #print(f"Log like: {log_like.numpy()} with TF={TF} for params={physical_params_tf.numpy()}")
        else:
            # Fallback: use py_function to get likelihood from pipeline
            # This breaks gradient flow, but is a fallback
            def get_likelihood(params_tf):
                try:
                    # Convert TensorFlow tensor to NumPy array
                    params_np = params_tf.numpy() if hasattr(params_tf, 'numpy') else np.array(params_tf)
                    r = self.emu_pipeline.run_results(params_np)
                    return float(r.like)
                except:
                    return -np.inf
            
            log_like = tf.py_function(
                func=get_likelihood,
                inp=[physical_params_tf],
                Tout=tf.float32
            )
            log_like.set_shape([])
            #print(f"Log like: {log_like} with TF={TF} for params={physical_params_tf.numpy()}")
        
        # Compute log_prior directly in TensorFlow for full differentiability
        # This is critical for NUTS to work correctly with priors
        log_prior_terms = []
        
        for i, param in enumerate(self.pipeline.varied_params):
            param_value = physical_params_tf[i]
            prior = param.prior
            limits = param.limits
            
            # Check bounds first (applies to all prior types)
            # Use a large negative value instead of -inf for gradient flow
            # We'll handle -inf properly by checking bounds
            in_bounds = tf.logical_and(
                param_value >= limits[0],
                param_value <= limits[1]
            )
            
            if isinstance(prior, UniformPrior):
                # Uniform prior: log(1/(b-a)) = -log(b-a) if in bounds, else -inf
                # For uniform priors, the gradient is 0 (constant function)
                # Use tf.where for cleaner conditional logic
                prior_norm_tf = tf.constant(prior.norm, dtype=DTYPE)  # prior.norm = -log(b-a)
                # Use -inf for out-of-bounds (mathematically correct, TensorFlow handles it)
                prior_log_prob = tf.where(
                    in_bounds,
                    prior_norm_tf,
                    tf.constant(-np.inf, dtype=DTYPE)
                )
                log_prior_terms.append(prior_log_prob)
                
            elif isinstance(prior, (GaussianPrior, TruncatedGaussianPrior)):
                # Gaussian prior: -0.5 * (x-mu)^2 / sigma^2 - norm
                # This has non-zero gradients: d/dx = -(x-mu) / sigma^2
                mu = tf.constant(prior.mu, dtype=DTYPE)
                sigma2 = tf.constant(prior.sigma2, dtype=DTYPE)
                norm = tf.constant(prior.norm, dtype=DTYPE)
                
                # Compute Gaussian log probability (fully differentiable)
                # TensorFlow will automatically compute gradients: d/dx = -(x-mu) / sigma^2
                gaussian_log_prob = -0.5 * tf.square(param_value - mu) / sigma2 - norm
                
                # Apply bounds check: use -inf if out of bounds
                # This ensures gradients flow correctly for in-bounds values
                prior_log_prob = tf.where(
                    in_bounds,
                    gaussian_log_prob,
                    tf.constant(-np.inf, dtype=DTYPE)
                )
                
                log_prior_terms.append(prior_log_prob)
                
            else:
                # For other prior types (Exponential, etc.), fall back to py_function
                # This breaks gradients but handles unsupported prior types
                def get_single_prior(params_tf, param_idx):
                    try:
                        params_np = params_tf.numpy() if hasattr(params_tf, 'numpy') else np.array(params_tf)
                        param_val = params_np[param_idx]
                        param_obj = self.pipeline.varied_params[param_idx]
                        return float(param_obj.evaluate_prior(param_val))
                    except:
                        return -np.inf
                
                prior_log_prob = tf.py_function(
                    func=lambda p: get_single_prior(p, i),
                    inp=[physical_params_tf],
                    Tout=tf.float32
                )
                prior_log_prob.set_shape([])
                # Stop gradients for unsupported prior types
                prior_log_prob = tf.stop_gradient(prior_log_prob)
                log_prior_terms.append(prior_log_prob)
        
        # Sum all prior terms to get total log prior
        if log_prior_terms:
            log_prior = tf.add_n(log_prior_terms)
        else:
            log_prior = tf.constant(0.0, dtype=DTYPE)
        
        # Ensure it's a scalar
        log_prior = tf.reshape(log_prior, [])
        
        # Log posterior with tempering
        log_post = (log_like + log_prior) * tempering
        # Ensure final result is a scalar
        log_post = tf.reshape(log_post, [])
        return log_post
        
        

    def _process_nuts_results(self, tempering: float) -> None:
        """Process NUTS results and update output chains."""
        # Calculate burn-in (use same logic as emcee)
        if hasattr(self, 'emcee_burn') and self.emcee_burn < 1:
            burn = int(self.emcee_burn * len(self.chain))
        elif hasattr(self, 'emcee_burn'):
            burn = int(self.emcee_burn)
        else:
            burn = 0
        
        # Apply burn-in and thinning
        if hasattr(self, 'emcee_thin'):
            thin = self.emcee_thin
        else:
            thin = 1
        
        self.chain = self.chain[burn::thin]
        self.unit_chain = self.unit_chain[burn::thin]
        logp = self.nuts_logp[burn::thin]
        self.blobs = self.blobs[burn::thin]
        
        # Handle output file management
        if self.save_outputs == SAVE_ALL and 0 < self.iterations < self.max_iterations:
            suffix = f'_tempering_{self.tempering[self.iterations-1]}_iteration_{self.iterations}'
            self.output.save_and_reset_to_chain_start(suffix)
        else:
            self.output.reset_to_chain_start()
        
        # Output chain points (similar to emcee format)
        for params, tempered_post, blob in zip(self.chain, logp, self.blobs):
            prior, extra = blob
            post = tempered_post / tempering
            
            # NUTS doesn't use log weights (unlike nautilus)
            self.output.parameters(params, extra, prior, tempered_post, post)
        
        logger.info(f"Generated {len(self.chain)} NUTS samples")

    def _process_mcmc_results(self, emcee_sampler: Any, tempering: float) -> None:
        """Process MCMC results and update output chains."""
        # Calculate burn-in
        if self.emcee_burn < 1:
            burn = int(self.emcee_burn * self.emcee_samples)
        else:
            burn = int(self.emcee_burn)
        
        # Ensure we keep at least one sample (e.g. when convergence broke early)
        #n_iterations = emcee_sampler.get_chain().shape[0]
        #burn = min(burn, max(0, n_iterations - 1))
        
        # Extract chains
        self.unit_chain = emcee_sampler.get_chain(discard=burn, thin=self.emcee_thin, flat=True)
        logp = emcee_sampler.get_log_prob(discard=burn, thin=self.emcee_thin, flat=True)
        self.blobs = emcee_sampler.get_blobs(discard=burn, thin=self.emcee_thin, flat=True)
        
        # Transform to physical parameters
        self.chain = np.array([
            self.pipeline.denormalize_vector_from_prior(p) for p in self.unit_chain
        ])
        
        # Final iteration (e.g. trained_before=True or last of multi-iteration): write to main file only.
        # Intermediate iterations: save current chain to suffixed file and reset main for next chain.
        #is_final_chain = (self.iterations == self.max_iterations - 1)
        if self.save_outputs == SAVE_ALL and 0 < self.iterations < self.max_iterations:
            suffix = f'_tempering_{self.tempering[self.iterations-1]}_iteration_{self.iterations}'
            self.output.save_and_reset_to_chain_start(suffix)
        else:
            self.output.reset_to_chain_start()
        
        # Output chain points
        for params, tempered_post, extra in zip(self.chain, logp, self.blobs):
            prior, extra = extra
            post = tempered_post / tempering
            
            # If using nautilus for final iteration, always pass log_weight (0.0 for emcee)
            if self.use_nautilus_final:
                self.output.parameters(params, extra, prior, tempered_post, post, 0.0)
            else:
                self.output.parameters(params, extra, prior, tempered_post, post)
            # Always save log_weight = 0.0 for emcee
            #self.output.parameters(params, extra, prior, tempered_post, post, 0.0)
        
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
        
        # Create log probability array (tempered)
        tempered_posts = posts * tempering
        logp = tempered_posts
        
        # Store blobs in emcee-compatible format
        self.blobs = list(zip(priors, extras))
        
        # Handle output file management
        if self.save_outputs == SAVE_ALL and 0 < self.iterations < self.max_iterations:
            #suffix = f'_nautilus_iteration_{self.iterations}'
            suffix = f'_tempering_{self.tempering[self.iterations-1]}_iteration_{self.iterations}'
            self.output.save_and_reset_to_chain_start(suffix)
        else:
            self.output.reset_to_chain_start()
        
        # Output chain points - include log weights for nautilus
        for params, tempered_post, blob, log_weight in zip(self.chain, logp, self.blobs, log_weights):
            prior, extra = blob
            post = tempered_post / tempering
            
            # Ensure extra is always a list (not None)
            if extra is None:
                extra = []
            
            # Use nautilus format: params + extra, prior, tempered_post, post, log_weight
            # The log_weight column was added first, so it comes before the other sampler outputs
            self.output.parameters(params, extra, prior, tempered_post, post, log_weight)
        
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

