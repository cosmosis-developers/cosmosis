# Create a version that properly computes gradients using TensorFlow autodiff
    # # Similar to compute_gradients but for log probability
    # def log_prob_with_gradients(physical_params_tf):
    #     """Compute log probability with proper gradient tracking using TensorFlow autodiff."""
    #     # Ensure 1D
    #     if len(physical_params_tf.shape) > 1:
    #         physical_params_tf = tf.reshape(physical_params_tf, [-1])
        
    #     # Use Variable for gradient tracking
    #     params_var = tf.Variable(physical_params_tf, trainable=True)
        
    #     with tf.GradientTape() as tape:
    #         # Compute predictions through emulator (fully differentiable, same as compute_gradients)
    #         params_norm = (params_var - X_mean_tf) / X_std_tf
    #         params_nn_norm = (params_norm - emulator.cp_nn.parameters_mean) / emulator.cp_nn.parameters_std
            
    #         if len(params_nn_norm.shape) == 1:
    #             params_nn_norm = tf.expand_dims(params_nn_norm, 0)
            
    #         x = params_nn_norm
    #         layers = [x]
            
    #         if emulator.cp_nn.architecture_type == "MLP":
    #             for i in range(emulator.cp_nn.n_layers - 1):
    #                 linear_out = tf.matmul(layers[-1], emulator.cp_nn.W[i]) + emulator.cp_nn.b[i]
    #                 activated = emulator.cp_nn.activation(linear_out, emulator.cp_nn.alphas[i], emulator.cp_nn.betas[i])
    #                 layers.append(activated)
    #             output = tf.matmul(layers[-1], emulator.cp_nn.W[-1]) + emulator.cp_nn.b[-1]
    #         else:
    #             raise NotImplementedError(f"NUTS not implemented for architecture: {emulator.cp_nn.architecture_type}")
            
    #         pred_norm = output * emulator.cp_nn.features_std + emulator.cp_nn.features_mean
    #         if len(pred_norm.shape) == 2:
    #             pred_norm = pred_norm[0]
            
    #         pred_intermediate = pred_norm * y_std_tf + y_mean_tf
            
    #         if emulator.data_trafo == 'log_norm':
    #             predictions = tf.pow(10.0, pred_intermediate)
    #         elif emulator.data_trafo == 'norm':
    #             predictions = pred_intermediate
    #         elif emulator.data_trafo == 'PCA':
    #             pca_matrix_tf = tf.constant(emulator.pca_transform_matrix, dtype=DTYPE)
    #             features_std_tf = tf.constant(emulator.features_std, dtype=DTYPE)
    #             features_mean_tf = tf.constant(emulator.features_mean, dtype=DTYPE)
    #             pca_reconstructed = tf.matmul(tf.expand_dims(pred_intermediate, 0), pca_matrix_tf)
    #             predictions = (pca_reconstructed * features_std_tf + features_mean_tf)[0]
    #         else:
    #             predictions = pred_intermediate
            
    #         # Compute log_likelihood from predictions using TensorFlow operations
    #         # This is fully differentiable if we have data_vector_tf and inv_covariance_tf
    #         if data_vector_tf is not None and inv_covariance_tf is not None:
    #             # Gaussian likelihood: -0.5 * (x - mu)^T * C^-1 * (x - mu) - 0.5 * log_det
    #             diff = predictions - data_vector_tf
    #             # Compute chi-squared: diff^T * inv_cov * diff
    #             chi2 = tf.tensordot(tf.tensordot(diff, inv_covariance_tf, axes=1), diff, axes=1)
    #             log_like = -0.5 * chi2 #- 0.5 * log_det_cov_tf
    #         else:
    #             # Fallback: use py_function to get likelihood from pipeline
    #             # This breaks gradient flow for likelihood, but we still get emulator gradients
    #             def get_likelihood(params_np):
    #                 try:
    #                     r = self.emu_pipeline.run_results(params_np)
    #                     return float(r.like)
    #                 except:
    #                     return -np.inf
                
    #             log_like = tf.py_function(
    #                 func=get_likelihood,
    #                 inp=[params_var],
    #                 Tout=tf.float32
    #             )
    #             log_like.set_shape([])
            
    #         # Get log_prior (this typically doesn't depend on emulator, so gradient is 0)
    #         # But we still need it for the log probability value
    #         def get_prior(params_np):
    #             try:
    #                 unit_params = self.pipeline.normalize_vector_to_prior(params_np)
    #                 if np.any(unit_params < 0) or np.any(unit_params > 1):
    #                     return -np.inf
    #                 return self.pipeline.prior(unit_params)
    #             except:
    #                 return -np.inf
            
    #         log_prior = tf.py_function(
    #             func=get_prior,
    #             inp=[params_var],
    #             Tout=tf.float32
    #         )
    #         log_prior.set_shape([])
            
    #         # Log posterior with tempering
    #         log_post = (log_like + log_prior) * tempering
        
    #     # Compute gradient using TensorFlow autodiff
    #     # This will compute gradients through the emulator and likelihood computation
    #     grad = tape.gradient(log_post, params_var)
    #     if grad is None:
    #         grad = tf.zeros_like(params_var)
        
    #     return log_post, grad

    # def log_prob_tf_fn(physical_params_tf):
    #     """TensorFlow function to compute log probability with autodiff.
        
    #     This function uses TensorFlow's automatic differentiation similar to
    #     compute_gradients, computing log_posterior = log_likelihood + log_prior.
    #     """
    #     # Ensure 1D
    #     if len(physical_params_tf.shape) > 1:
    #         physical_params_tf = tf.reshape(physical_params_tf, [-1])
        
    #     # Check prior bounds by normalizing
    #     params_norm = (physical_params_tf - X_mean_tf) / X_std_tf
        
    #     # Normalize for neural network input
    #     params_nn_norm = (params_norm - emulator.cp_nn.parameters_mean) / emulator.cp_nn.parameters_std
        
    #     # Ensure batch dimension
    #     if len(params_nn_norm.shape) == 1:
    #         params_nn_norm = tf.expand_dims(params_nn_norm, 0)
        
    #     # Forward pass through neural network (same as compute_gradients)
    #     x = params_nn_norm
    #     layers = [x]
        
    #     if emulator.cp_nn.architecture_type == "MLP":
    #         for i in range(emulator.cp_nn.n_layers - 1):
    #             linear_out = tf.matmul(layers[-1], emulator.cp_nn.W[i]) + emulator.cp_nn.b[i]
    #             activated = emulator.cp_nn.activation(linear_out, emulator.cp_nn.alphas[i], emulator.cp_nn.betas[i])
    #             layers.append(activated)
            
    #         output = tf.matmul(layers[-1], emulator.cp_nn.W[-1]) + emulator.cp_nn.b[-1]
    #     else:
    #         raise NotImplementedError(f"NUTS not implemented for architecture: {emulator.cp_nn.architecture_type}")
        
    #     # Denormalize output
    #     pred_norm = output * emulator.cp_nn.features_std + emulator.cp_nn.features_mean
    #     if len(pred_norm.shape) == 2:
    #         pred_norm = pred_norm[0]
        
    #     # Transform from normalized to original space
    #     pred_intermediate = pred_norm * y_std_tf + y_mean_tf
        
    #     if emulator.data_trafo == 'log_norm':
    #         predictions = tf.pow(10.0, pred_intermediate)
    #     elif emulator.data_trafo == 'norm':
    #         predictions = pred_intermediate
    #     elif emulator.data_trafo == 'PCA':
    #         pca_matrix_tf = tf.constant(emulator.pca_transform_matrix, dtype=DTYPE)
    #         features_std_tf = tf.constant(emulator.features_std, dtype=DTYPE)
    #         features_mean_tf = tf.constant(emulator.features_mean, dtype=DTYPE)
    #         pca_reconstructed = tf.matmul(tf.expand_dims(pred_intermediate, 0), pca_matrix_tf)
    #         predictions = (pca_reconstructed * features_std_tf + features_mean_tf)[0]
    #     else:
    #         predictions = pred_intermediate
        

    #     def compute_log_prob_np(params_np):
    #         """Numpy function to compute full log probability."""
    #         try:
    #             unit_params = self.pipeline.normalize_vector_to_prior(params_np)
    #             if np.any(unit_params < 0) or np.any(unit_params > 1):
    #                 return -np.inf
    #             r = self.emu_pipeline.run_results(params_np)
    #             return float(r.post) * tempering
    #         except:
    #             return -np.inf
        
    #     # Get predictions as numpy for likelihood computation
    #     predictions_np = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
    #     print(f"predictions_np: {predictions_np}")

    #     # The key is that predictions are computed with TF autodiff
    #     log_prob_val = tf.py_function(
    #         func=compute_log_prob_np,
    #         inp=[physical_params_tf],
    #         Tout=tf.float32
    #     )
    #     log_prob_val.set_shape([])
        
    #     return log_prob_val      


    # def _verify_nuts_gradients(self, test_params: np.ndarray, tempering: float,
    #                            emulator, X_mean_tf, X_std_tf, y_mean_tf, y_std_tf,
    #                            data_vector_tf, inv_covariance_tf, DTYPE) -> None:
    #     """Verify gradient computation by comparing autodiff with finite differences."""
    #     import tensorflow as tf
        
    #     logger.info("Verifying NUTS gradient computation...")
        
    #     # Convert test params to TensorFlow
    #     test_params_tf = tf.constant(test_params, dtype=DTYPE)
        
    #     # Compute gradient using TensorFlow autodiff
    #     with tf.GradientTape() as tape:
    #         tape.watch(test_params_tf)
    #         log_prob_tf = self._log_prob_nuts_impl(
    #             test_params_tf, emulator, X_mean_tf, X_std_tf,
    #             y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
    #             tempering, DTYPE
    #         )
        
    #     grad_autodiff = tape.gradient(log_prob_tf, test_params_tf)
        
    #     if grad_autodiff is None:
    #         logger.error("Autodiff gradient is None! Gradient computation may be broken.")
    #         return
        
    #     grad_autodiff_np = grad_autodiff.numpy()
        
    #     # Compute gradient using finite differences
    #     eps = 1e-5
    #     grad_finite_diff = np.zeros_like(test_params)
        
    #     # Get baseline log probability
    #     log_prob_base = self._log_prob_nuts_impl(
    #         test_params_tf, emulator, X_mean_tf, X_std_tf,
    #         y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
    #         tempering, DTYPE
    #     ).numpy()
        
    #     for i in range(len(test_params)):
    #         # Forward difference
    #         params_forward = test_params.copy()
    #         params_forward[i] += eps
    #         params_forward_tf = tf.constant(params_forward, dtype=DTYPE)
            
    #         log_prob_forward = self._log_prob_nuts_impl(
    #             params_forward_tf, emulator, X_mean_tf, X_std_tf,
    #             y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
    #             tempering, DTYPE
    #         ).numpy()
            
    #         # Central difference (more accurate)
    #         params_backward = test_params.copy()
    #         params_backward[i] -= eps
    #         params_backward_tf = tf.constant(params_backward, dtype=DTYPE)
            
    #         log_prob_backward = self._log_prob_nuts_impl(
    #             params_backward_tf, emulator, X_mean_tf, X_std_tf,
    #             y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf,
    #             tempering, DTYPE
    #         ).numpy()
            
    #         # Central difference gradient
    #         grad_finite_diff[i] = (log_prob_forward - log_prob_backward) / (2 * eps)
        
    #     # Compare gradients
    #     logger.info("=" * 60)
    #     logger.info("Gradient Verification Results")
    #     logger.info("=" * 60)
    #     logger.info(f"Test point: {test_params}")
    #     logger.info(f"Log probability: {log_prob_base:.6f}")
    #     logger.info("")
    #     logger.info("Parameter | Autodiff Gradient | Finite Diff Gradient | Relative Error")
    #     logger.info("-" * 70)
        
    #     max_rel_error = 0.0
    #     for i, param_name in enumerate(self.pipeline.varied_params):
    #         autodiff_val = grad_autodiff_np[i]
    #         finite_diff_val = grad_finite_diff[i]
            
    #         # Compute relative error (handle zero gradients)
    #         if abs(autodiff_val) > 1e-10 or abs(finite_diff_val) > 1e-10:
    #             rel_error = abs(autodiff_val - finite_diff_val) / max(abs(autodiff_val), abs(finite_diff_val), 1e-10)
    #         else:
    #             rel_error = abs(autodiff_val - finite_diff_val)
            
    #         max_rel_error = max(max_rel_error, rel_error)
            
    #         logger.info(f"{param_name.name:20s} | {autodiff_val:15.6e} | {finite_diff_val:15.6e} | {rel_error:10.2e}")
        
    #     logger.info("-" * 70)
    #     logger.info(f"Maximum relative error: {max_rel_error:.2e}")
        
    #     if max_rel_error > 0.1:
    #         logger.warning(f"Large gradient discrepancy detected! Max relative error: {max_rel_error:.2e}")
    #         logger.warning("This may cause poor NUTS exploration. Check gradient computation.")
    #     elif max_rel_error > 0.01:
    #         logger.warning(f"Moderate gradient discrepancy. Max relative error: {max_rel_error:.2e}")
    #     else:
    #         logger.info("Gradients match well. Gradient computation appears correct.")
        
    #     # Analyze gradient magnitudes for NUTS diagnostics
    #     grad_magnitude = np.linalg.norm(grad_autodiff_np)
    #     max_grad = np.max(np.abs(grad_autodiff_np))
    #     logger.info("")
    #     logger.info("Gradient Magnitude Analysis:")
    #     logger.info(f"  Gradient norm: {grad_magnitude:.2e}")
    #     logger.info(f"  Max gradient component: {max_grad:.2e}")
        
    #     # Check if gradients are very large (could cause small step sizes)
    #     if grad_magnitude > 1e4:
    #         logger.warning(f"Very large gradient magnitude ({grad_magnitude:.2e}) detected!")
    #         logger.warning("This may cause NUTS to use very small step sizes.")
    #         logger.warning("Consider: 1) Parameter scaling/normalization, 2) Mass matrix, 3) Larger initial step size")
    #     elif grad_magnitude > 1e3:
    #         logger.warning(f"Large gradient magnitude ({grad_magnitude:.2e}). May affect NUTS exploration.")
        
    #     # Check parameter scales
    #     param_scales = np.array([p.width() for p in self.pipeline.varied_params])
    #     logger.info("")
    #     logger.info("Parameter Scale Analysis:")
    #     for i, param in enumerate(self.pipeline.varied_params):
    #         scale = param_scales[i]
    #         grad_scale = abs(grad_autodiff_np[i]) * scale
    #         logger.info(f"  {param.name:20s}: scale={scale:.4f}, |grad|*scale={grad_scale:.2e}")
        
    #     # Suggest step size based on gradient magnitude
    #     # Rough heuristic: step_size ~ 1 / max_gradient for reasonable acceptance
    #     suggested_step = 1.0 / max_grad if max_grad > 0 else 0.1
    #     logger.info("")
    #     logger.info(f"Suggested step size (heuristic): {suggested_step:.6f}")
    #     logger.info(f"Current step size: {self.nuts_step_size:.6f}")
    #     if abs(suggested_step - self.nuts_step_size) / max(suggested_step, self.nuts_step_size) > 0.5:
    #         logger.warning(f"Step size differs significantly from suggestion. Consider adjusting.")
        
    #     logger.info("=" * 60)

    # def _run_nuts_sampling(self, tempering: float) -> None:
    #     """Run NUTS sampling using TensorFlow Probability."""
    #     try:
    #         import tensorflow as tf
    #         import tensorflow_probability as tfp
    #     except ImportError as e:
    #         raise ImportError(
    #             "TensorFlow and TensorFlow Probability are required for NUTS sampling. "
    #             f"Install with: pip install tensorflow tensorflow-probability. Error: {e}"
    #         )
        
    #     if self.emu_pipeline is None:
    #         logger.warning("emu_pipeline is None, setting it up now (this should happen in execute())")
    #         self.compute_fiducial_setup_emu_pipeline()
    #         # Update global sampler reference after setup
    #         utils_module._sampler = self
        
    #     logger.info(f"Starting NUTS sampling with step_size={self.nuts_step_size}, "
    #                f"max_tree_depth={self.nuts_max_tree_depth}, num_results={self.nuts_num_results}")
        
    #     # Get initial state in unit hypercube
    #     if self.trained_before:
    #         # Use randomized start from pipeline
    #         initial_state_unit = self.pipeline.randomized_start()
    #     else:
    #         # Use last training sample
    #         if len(self.unit_sample) == 0:
    #             raise RuntimeError("No training samples available for NUTS initialization")
    #         initial_state_unit = self.unit_sample[-1]
        
    #     # Ensure initial state is in [0,1] (unit hypercube)
    #     initial_state_unit = np.clip(initial_state_unit, 0.0, 1.0)
        
    #     # Convert to physical space for initial state
    #     try:
    #         initial_state_physical = self.pipeline.denormalize_vector_from_prior(initial_state_unit)
    #     except ValueError:
    #         # If denormalization fails, try to get a valid starting point
    #         logger.warning("Initial state denormalization failed, using pipeline default")
    #         initial_state_physical = np.array([
    #             self.pipeline.prior_info[i].start for i in range(self.ndim)
    #         ])
    #         initial_state_unit = self.pipeline.normalize_vector_to_prior(initial_state_physical)
    #         initial_state_unit = np.clip(initial_state_unit, 0.0, 1.0)
        
    #     logger.info(f"Initial state (unit): {initial_state_unit}")
    #     logger.info(f"Initial state (physical): {initial_state_physical}")
        
    #     # Create TensorFlow-compatible log probability function with autodiff
    #     # Similar to compute_gradients, but for log posterior
    #     # Access emulator (stored in both self.emulator and self.emu_module.data.emulator)
    #     emulator = self.emulator
    #     if emulator is None:
    #         raise RuntimeError("Emulator not set. This should be set during training or loading.")
    #     DTYPE = tf.float32
        
    #     # Get normalization constants
    #     X_mean_tf = tf.constant([emulator.X_mean[key] for key in emulator.model_parameters], dtype=DTYPE)
    #     X_std_tf = tf.constant([emulator.X_std[key] for key in emulator.model_parameters], dtype=DTYPE)
    #     y_mean_tf = tf.constant(emulator.y_mean, dtype=DTYPE)
    #     y_std_tf = tf.constant(emulator.y_std, dtype=DTYPE)
        
    #     # Get data vector and inverse covariance from pipeline for likelihood computation
    #     # Extract these from a sample run to use as TensorFlow constants
    #     data_vector_tf = None
    #     inv_covariance_tf = None
    #     logger.info(f"Initial state for NUTS: {initial_state_physical}")
    #     try:
    #         like, data_vectors_theory, data_vectors, data_inv_covariance, error_vectors, block = utils_module.task(initial_state_physical, self, True)
    #         logger.info(f"Extracted {len(data_vectors)} data vectors and {len(data_inv_covariance)} covariance matrices")
    #         # For now let's test just one probe's data vector and inverse covariance
    #         # TODO: Handle multiple probes properly
    #         data_vector_tf = tf.constant(np.atleast_1d(data_vectors[0]), dtype=DTYPE)
    #         inv_covariance_tf = tf.constant(np.atleast_2d(data_inv_covariance[0]), dtype=DTYPE)
    #         logger.info(f"Data vector shape: {data_vector_tf.shape}, Inv covariance shape: {inv_covariance_tf.shape}")
    #     except Exception as e:
    #         logger.warning(f"Could not get sample run for NUTS data extraction: {e}")
    #         logger.warning("NUTS will use fallback likelihood computation (slower, no gradients)")
        
    #     # Verify gradient computation by comparing with finite differences
    #     # This helps diagnose if poor exploration is due to incorrect gradients
    #     self._verify_nuts_gradients(initial_state_physical, tempering, emulator, 
    #                                  X_mean_tf, X_std_tf, y_mean_tf, y_std_tf, 
    #                                  data_vector_tf, inv_covariance_tf, DTYPE)

    #     # CRITICAL FIX: Sample in unit hypercube space [0,1]^n instead of physical space
    #     # This normalizes all parameters to the same scale, which is essential for NUTS
    #     # to explore effectively when parameters have very different scales
        
    #     # Get parameter limits for transformation (pure TensorFlow operations for gradients)
    #     param_lowers = np.array([p.limits[0] for p in self.pipeline.varied_params], dtype=np.float32)
    #     param_uppers = np.array([p.limits[1] for p in self.pipeline.varied_params], dtype=np.float32)
    #     param_lowers_tf = tf.constant(param_lowers, dtype=DTYPE)
    #     param_uppers_tf = tf.constant(param_uppers, dtype=DTYPE)
    #     param_widths_tf = param_uppers_tf - param_lowers_tf
        
    #     # Convert initial state to unit hypercube
    #     initial_state_unit_tf = tf.constant(initial_state_unit, dtype=DTYPE)
        
    #     def log_prob_nuts_unit(unit_params_tf):
    #         """Log probability function in unit hypercube [0,1]^ndim.
            
    #         This function samples in normalized space, which helps NUTS explore
    #         when parameters have very different scales. Uses pure TF operations
    #         for gradient flow.
    #         """
    #         # Use soft boundary constraint - penalize points outside [0,1]
    #         # This allows gradients to flow through boundaries while keeping samples valid
    #         # Quadratic penalty for values outside [0,1]
    #         below_zero = tf.minimum(unit_params_tf, 0.0)
    #         above_one = tf.maximum(unit_params_tf - 1.0, 0.0)
    #         penalty = 1000.0 * (tf.reduce_sum(tf.square(below_zero)) + tf.reduce_sum(tf.square(above_one)))
            
    #         # Clip for transformation to ensure valid physical parameters
    #         unit_params_clipped = tf.clip_by_value(unit_params_tf, 0.0, 1.0)
            
    #         # Transform from unit cube to physical space using pure TF operations
    #         # physical = lower + (upper - lower) * unit
    #         # This is a linear transformation, so gradients flow through
    #         physical_params_tf = param_lowers_tf + param_widths_tf * unit_params_clipped
            
    #         # Compute log probability in physical space
    #         # Gradients will flow through the transformation automatically
    #         log_prob = self._log_prob_nuts_impl(physical_params_tf, emulator, X_mean_tf, X_std_tf,
    #                                              y_mean_tf, y_std_tf, data_vector_tf,
    #                                              inv_covariance_tf, tempering, DTYPE)
            
    #         # Apply boundary penalty (subtract penalty from log prob)
    #         log_prob = log_prob - penalty
            
    #         # For uniform priors with linear transformation, the Jacobian is constant
    #         # so we don't need to add it (it doesn't affect the sampling distribution)
    #         return log_prob
        
    #     # Use unit hypercube for sampling
    #     # Ensure initial state is in [0,1]
    #     initial_state_unit_clipped = np.clip(initial_state_unit, 0.0, 1.0)
    #     initial_state = tf.constant(initial_state_unit_clipped, dtype=DTYPE)
        
    #     logger.info(f"Sampling in unit hypercube space [0,1]^{self.ndim}")
    #     logger.info(f"Parameter limits (physical): lower={param_lowers}, upper={param_uppers}")
    #     logger.info(f"Initial state (unit, clipped): {initial_state_unit_clipped}")
        
    #     # DIAGNOSTIC: Check parameter scales
    #     param_scales = np.array([p.width() for p in self.pipeline.varied_params])
    #     logger.info("")
    #     logger.info("Parameter Scale Analysis for NUTS:")
    #     for i, param in enumerate(self.pipeline.varied_params):
    #         scale = param_scales[i]
    #         logger.info(f"  {param.name:20s}: scale={scale:.6f}, initial_value={initial_state_physical[i]:.6f}")
        
    #     # Create NUTS kernel - now sampling in unit hypercube space
    #     # Use a reasonable step size for unit hypercube (typically 0.1-0.3 works well)
    #     # The step size should be relative to the [0,1] scale
    #     # Since we're in [0,1] space, a step size of 0.1-0.3 is reasonable
    #     # For better exploration, use a larger step size
        
    #     # Check if user wants fixed step size
    #     if hasattr(self, 'nuts_use_fixed_step_size') and self.nuts_use_fixed_step_size:
    #         # Use fixed step size (no adaptation)
    #         unit_step_size = self.nuts_fixed_step_size if hasattr(self, 'nuts_fixed_step_size') else 0.3
    #         # Ensure step size is reasonable for unit hypercube [0,1]
    #         # Step size of 0.001 means only 0.1% movement per step - way too small!
    #         if unit_step_size < 0.05:
    #             logger.warning(f"Fixed step size {unit_step_size:.6f} is VERY SMALL for unit hypercube [0,1]!")
    #             logger.warning(f"This will severely limit exploration. Increasing to 0.2 for better exploration.")
    #             logger.warning(f"Consider setting nuts_fixed_step_size >= 0.2 (0.3 is recommended)")
    #             unit_step_size = 0.2
    #         elif unit_step_size > 0.5:
    #             logger.warning(f"Fixed step size {unit_step_size:.6f} is very large for unit hypercube [0,1]!")
    #             logger.warning(f"Capping at 0.5. Consider setting nuts_fixed_step_size <= 0.5")
    #             unit_step_size = 0.5
    #         logger.info(f"Using FIXED step_size={unit_step_size:.6f} for unit hypercube sampling (no adaptation)")
    #     else:
    #         # Use adaptive step size
    #         unit_step_size = 0.25  # Larger default for better exploration in unit hypercube
    #         if hasattr(self, 'nuts_step_size') and self.nuts_step_size > 0:
    #             # If user specified a step size, use it directly if it's reasonable for [0,1] space
    #             if 0.01 <= self.nuts_step_size <= 0.5:
    #                 unit_step_size = self.nuts_step_size
    #             elif self.nuts_step_size < 0.01:
    #                 unit_step_size = 0.25  # Use default for very small step sizes
    #             else:
    #                 unit_step_size = 0.4  # Allow larger maximum for exploration
    #         logger.info(f"Using ADAPTIVE step_size={unit_step_size:.6f} for unit hypercube sampling")
        
    #     logger.info(f"Note: This step size is relative to [0,1] unit hypercube space")
        
    #     # Increase max_tree_depth for better exploration in unit space
    #     # In unit hypercube, we can afford deeper trees for better exploration
    #     effective_max_tree_depth = max(self.nuts_max_tree_depth, 12)  # At least 12 for exploration
        
    #     nuts_kernel = tfp.mcmc.NoUTurnSampler(
    #         target_log_prob_fn=log_prob_nuts_unit,
    #         step_size=unit_step_size,
    #         max_tree_depth=effective_max_tree_depth,
    #         max_energy_diff=self.nuts_max_energy_diff,
    #         unrolled_leapfrog_steps=self.nuts_unrolled_leapfrog_steps,
    #         parallel_iterations=self.nuts_parallel_iterations,
    #         name='nuts_kernel'
    #     )
    #     logger.info(f"NUTS kernel initialized with step_size={unit_step_size:.6f} (unit hypercube), "
    #                f"max_tree_depth={effective_max_tree_depth}")
        
    #     # Use fixed or adaptive step size based on configuration
    #     if hasattr(self, 'nuts_use_fixed_step_size') and self.nuts_use_fixed_step_size:
    #         # Use fixed step size - no adaptation wrapper
    #         adaptive_kernel = nuts_kernel
    #         logger.info("Using FIXED step size (no adaptation)")
    #     else:
    #         # Adaptive step size
    #         # target_accept_prob is critical for proper adaptation!
    #         # Use a higher target (0.7) for better exploration - allows larger steps
    #         # In unit hypercube space, we want more aggressive exploration
    #         adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #             inner_kernel=nuts_kernel,
    #             num_adaptation_steps=self.nuts_num_adaptation_steps,
    #             target_accept_prob=0.7,  # Higher target for better exploration in unit space
    #         )
    #         logger.info(f"Adaptive kernel initialized with {self.nuts_num_adaptation_steps} adaptation steps")
    #         logger.info(f"Note: Step size will adapt during the first {self.nuts_num_adaptation_steps} steps")
        
    #     logger.info("")
        
    #     # Run sampling
    #     start_time = default_timer()
        
    #     # Run multiple chains if requested
    #     all_chains = []
    #     all_log_probs = []
    #     all_blobs = []
    #     logger.info(f"Running {self.nuts_num_chains} chain(s) with {self.nuts_num_results} samples each")

    #     for chain_idx in range(self.nuts_num_chains):
    #         if self.nuts_num_chains > 1:
    #             # Slightly perturb initial state for each chain
    #             # Use larger perturbation to ensure different starting points
    #             # In unit space [0,1], use a perturbation of ~0.05
    #             perturbation_scale = 0.05
    #             chain_initial = initial_state + tf.random.normal(
    #                 shape=initial_state.shape,
    #                 stddev=perturbation_scale,
    #                 dtype=tf.float32,
    #                 seed=self.seed + chain_idx if self.seed is not None else None
    #             )
    #             # Clip to [0,1] to ensure valid unit hypercube
    #             chain_initial = tf.clip_by_value(chain_initial, 0.0, 1.0)
    #             logger.info(f"Chain {chain_idx + 1}: perturbed initial state by ~{perturbation_scale:.4f}")
    #         else:
    #             chain_initial = initial_state
    #             logger.info(f"Chain {chain_idx + 1}: using initial state {chain_initial.numpy()}")    
            
    #         # Run MCMC with trace to monitor step size (if adaptive)
    #         # Capture step_size for use in trace function
    #         if hasattr(self, 'nuts_use_fixed_step_size') and self.nuts_use_fixed_step_size:
    #             initial_step_size = tf.constant(unit_step_size, dtype=DTYPE)
    #         else:
    #             initial_step_size = tf.constant(unit_step_size, dtype=DTYPE)
            
    #         # Trace step size from kernel (adaptive or fixed)
    #         def trace_step_size(_, kernel_results):
    #             # For fixed step size, kernel_results is from NUTS directly
    #             # For adaptive step size, step_size is in inner_results
    #             if hasattr(self, 'nuts_use_fixed_step_size') and self.nuts_use_fixed_step_size:
    #                 # Fixed step size - return the fixed value
    #                 return initial_step_size
    #             else:
    #                 # Adaptive step size - try to get from inner_results
    #                 if hasattr(kernel_results, 'inner_results'):
    #                     inner = kernel_results.inner_results
    #                     if hasattr(inner, 'step_size'):
    #                         return inner.step_size
    #                 # Fallback: try direct access
    #                 if hasattr(kernel_results, 'step_size'):
    #                     return kernel_results.step_size
    #                 # If all else fails, return initial step size
    #                 return initial_step_size
            
    #         # Trace both step size and acceptance rate for diagnostics
    #         def trace_diagnostics(_, kernel_results):
    #             """Trace step size and acceptance information."""
    #             step_size = trace_step_size(_, kernel_results)
                
    #             # Try to get acceptance information
    #             if hasattr(self, 'nuts_use_fixed_step_size') and self.nuts_use_fixed_step_size:
    #                 # For fixed step size, kernel_results is from NUTS directly
    #                 if hasattr(kernel_results, 'is_accepted'):
    #                     accepted = kernel_results.is_accepted
    #                 else:
    #                     accepted = tf.constant(True, dtype=tf.bool)  # Assume accepted if not available
    #             else:
    #                 # For adaptive, check inner_results
    #                 if hasattr(kernel_results, 'inner_results'):
    #                     inner = kernel_results.inner_results
    #                     if hasattr(inner, 'is_accepted'):
    #                         accepted = inner.is_accepted
    #                     else:
    #                         accepted = tf.constant(True, dtype=tf.bool)
    #                 else:
    #                     accepted = tf.constant(True, dtype=tf.bool)
                
    #             return {
    #                 'step_size': step_size,
    #                 'is_accepted': accepted
    #             }
            
    #         # CRITICAL: Test log probability function before running NUTS
    #         # Check if it returns different values for different inputs
    #         test_point_1 = chain_initial
    #         test_point_2 = chain_initial + tf.constant([0.001, 0.001], dtype=DTYPE)  # Perturb slightly
            
    #         try:
    #             log_prob_1 = log_prob_nuts_unit(test_point_1)
    #             log_prob_2 = log_prob_nuts_unit(test_point_2)
    #             logger.info(f"Chain {chain_idx + 1}: Log prob test - point 1: {log_prob_1.numpy():.6f}, point 2: {log_prob_2.numpy():.6f}")
    #             if abs(log_prob_1.numpy() - log_prob_2.numpy()) < 1e-6:
    #                 logger.warning(f"Chain {chain_idx + 1}: Log prob function returns same value for different inputs!")
    #                 logger.warning("This will cause NUTS to reject all proposals. Check log probability implementation.")
    #         except Exception as e:
    #             logger.error(f"Chain {chain_idx + 1}: Error testing log probability function: {e}")
            
    #         # Try without @tf.function first to see if that's the issue
    #         # @tf.function can sometimes cause issues with complex log prob functions
    #         logger.info(f"Chain {chain_idx + 1}: Running NUTS sampling...")
    #         try:
    #             samples, trace = tfp.mcmc.sample_chain(
    #                 num_results=self.nuts_num_results,
    #                 num_burnin_steps=self.nuts_num_burnin_steps,
    #                 current_state=chain_initial,
    #                 kernel=adaptive_kernel,
    #                 trace_fn=trace_diagnostics,
    #                 seed=self.seed + chain_idx if self.seed is not None else None,
    #             )
    #         except Exception as e:
    #             logger.error(f"Chain {chain_idx + 1}: Error during NUTS sampling: {e}")
    #             raise
            
    #         # Extract step sizes and acceptance info
    #         step_sizes = trace['step_size'] if isinstance(trace, dict) else None
    #         acceptance_info = trace.get('is_accepted', None) if isinstance(trace, dict) else None
            
    #         # Log acceptance rate if available
    #         if acceptance_info is not None:
    #             try:
    #                 acceptance_np = acceptance_info.numpy() if hasattr(acceptance_info, 'numpy') else acceptance_info
    #                 if len(acceptance_np) > 0:
    #                     acceptance_rate = np.mean(acceptance_np.astype(float))
    #                     logger.info(f"Chain {chain_idx + 1}: Acceptance rate: {acceptance_rate:.3f}")
    #                     if acceptance_rate < 0.1:
    #                         logger.warning(f"Chain {chain_idx + 1}: Very low acceptance rate! This explains poor exploration.")
    #                         logger.warning("Consider: 1) Reducing step size, 2) Checking log probability function")
    #                     elif acceptance_rate > 0.95:
    #                         logger.warning(f"Chain {chain_idx + 1}: Very high acceptance rate! Step size may be too small.")
    #             except Exception as e:
    #                 logger.warning(f"Could not extract acceptance rate: {e}")
            
    #         # Log step size information for diagnostics
    #         if hasattr(self, 'nuts_use_fixed_step_size') and self.nuts_use_fixed_step_size:
    #             logger.info(f"Chain {chain_idx + 1}: Using FIXED step size (unit space): {unit_step_size:.6f}")
    #         else:
    #             try:
    #                 if step_sizes is not None:
    #                     step_sizes_np = step_sizes.numpy() if hasattr(step_sizes, 'numpy') else step_sizes
    #                     if len(step_sizes_np) > 0:
    #                         final_step_size = step_sizes_np[-1] if len(step_sizes_np) > 0 else unit_step_size
    #                         logger.info(f"Chain {chain_idx + 1}: Final adapted step size (unit space): {final_step_size:.6f}")
    #                         if len(step_sizes_np) > 10:
    #                             logger.info(f"Chain {chain_idx + 1}: Step size range: [{np.min(step_sizes_np):.6f}, {np.max(step_sizes_np):.6f}]")
    #                             if final_step_size < 0.01:
    #                                 logger.warning(f"Chain {chain_idx + 1}: Step size adapted to very small value ({final_step_size:.6f})!")
    #                                 logger.warning("This severely limits exploration. Consider:")
    #                                 logger.warning("  1) Increasing initial step_size (e.g., 0.3-0.5)")
    #                                 logger.warning("  2) Reducing target_accept_prob (e.g., 0.7)")
    #                                 logger.warning("  3) Using fixed step size: nuts_use_fixed_step_size = T, nuts_fixed_step_size = 0.3")
    #                             elif final_step_size > 0.4:
    #                                 logger.info(f"Chain {chain_idx + 1}: Large step size ({final_step_size:.6f}) - good for exploration")
    #             except Exception as e:
    #                 logger.warning(f"Could not extract step size information: {e}")
                
    #         # Convert to numpy - samples are in unit hypercube space
    #         samples_unit_np = samples.numpy()
    #         if len(samples_unit_np.shape) == 1:
    #             samples_unit_np = samples_unit_np.reshape(-1, 1)
            
    #         # DIAGNOSTIC: Check if samples are actually different
    #         if len(samples_unit_np) > 1:
    #             sample_variance = np.var(samples_unit_np, axis=0)
    #             unique_samples = len(np.unique(samples_unit_np, axis=0))
    #             logger.info(f"Chain {chain_idx + 1}: Sample variance (unit space): {sample_variance}")
    #             logger.info(f"Chain {chain_idx + 1}: Unique samples: {unique_samples} out of {len(samples_unit_np)}")
    #             if unique_samples == 1:
    #                 logger.error(f"Chain {chain_idx + 1}: CRITICAL - All samples are identical! NUTS is not exploring.")
    #                 logger.error("Possible causes:")
    #                 logger.error("  1) Log probability function always returns same value")
    #                 logger.error("  2) All proposals rejected (check acceptance rate)")
    #                 logger.error("  3) TensorFlow graph execution issue")
    #                 logger.error("  4) Step size too large causing all rejections")
    #                 # Log first few samples for debugging
    #                 logger.error(f"First 5 samples (unit space): {samples_unit_np[:5]}")
            
    #         # Transform samples from unit hypercube to physical space
    #         samples_np = []
    #         for sample_unit in samples_unit_np:
    #             try:
    #                 # Clip to [0,1] to ensure valid transformation
    #                 sample_unit_clipped = np.clip(sample_unit, 0.0, 1.0)
    #                 sample_physical = self.pipeline.denormalize_vector_from_prior(sample_unit_clipped)
    #                 samples_np.append(sample_physical)
    #             except (ValueError, Exception):
    #                 # If transformation fails, use a dummy value (will be rejected by likelihood)
    #                 samples_np.append(np.full(self.ndim, np.nan))
    #         samples_np = np.array(samples_np)
            
    #         # Compute log probabilities and blobs for each sample
    #         chain_log_probs = []
    #         chain_blobs = []
    #         for sample in samples_np:
    #             try:
    #                 if np.any(np.isnan(sample)):
    #                     raise ValueError("Invalid sample (NaN)")
    #                 r = self.emu_pipeline.run_results(sample)
    #                 chain_log_probs.append(r.post * tempering)
    #                 chain_blobs.append((r.prior, r.extra))
    #             except Exception:
    #                 chain_log_probs.append(-np.inf)
    #                 chain_blobs.append((-np.inf, [np.nan] * self.pipeline.number_extra))
            
    #         all_chains.append(samples_np)
    #         all_log_probs.append(chain_log_probs)
    #         all_blobs.append(chain_blobs)
            
    #         # DIAGNOSTIC: Check sample diversity
    #         if len(samples_np) > 10:
    #             # Compute distance from initial state (in physical space)
    #             distances = np.linalg.norm(samples_np - initial_state_physical, axis=1)
    #             max_distance = np.max(distances)
    #             mean_distance = np.mean(distances)
    #             param_ranges = np.ptp(samples_np, axis=0)  # Peak-to-peak (range) for each parameter
                
    #             # Also check diversity in unit space
    #             distances_unit = np.linalg.norm(samples_unit_np - initial_state_unit, axis=1)
    #             max_distance_unit = np.max(distances_unit)
    #             mean_distance_unit = np.mean(distances_unit)
                
    #             logger.info(f"Chain {chain_idx + 1} Sample Diversity:")
    #             logger.info(f"  Physical space - Max distance: {max_distance:.6f}, Mean: {mean_distance:.6f}")
    #             logger.info(f"  Unit space - Max distance: {max_distance_unit:.6f}, Mean: {mean_distance_unit:.6f}")
    #             logger.info(f"  Parameter ranges sampled (physical space):")
    #             for i, param in enumerate(self.pipeline.varied_params):
    #                 range_val = param_ranges[i]
    #                 scale = param_scales[i]
    #                 range_frac = range_val / scale if scale > 0 else 0
    #                 logger.info(f"    {param.name:20s}: range={range_val:.6f} ({range_frac*100:.1f}% of prior width)")
                
    #             if max_distance < 0.01 * np.mean(param_scales):
    #                 logger.warning(f"Chain {chain_idx + 1}: Very limited exploration! Samples stay very close to initial state.")
    #                 logger.warning("Possible causes: 1) Step size too small, 2) Gradients too large, 3) Poor adaptation")
    #             elif max_distance_unit < 0.01:
    #                 logger.warning(f"Chain {chain_idx + 1}: Limited exploration in unit space. Step size may need adjustment.")
        
    #     # Combine chains
    #     if self.nuts_num_chains > 1:
    #         self.chain = np.vstack(all_chains)
    #         self.nuts_logp = np.concatenate(all_log_probs)
    #         self.blobs = [item for sublist in all_blobs for item in sublist]
    #     else:
    #         # For single chain, use the results from the loop
    #         self.chain = all_chains[0] if all_chains else samples_np
    #         self.nuts_logp = np.array(all_log_probs[0]) if all_log_probs else np.array([])
    #         self.blobs = all_blobs[0] if all_blobs else []
        
    #     # Convert to unit cube
    #     self.unit_chain = np.array([
    #         self.pipeline.normalize_vector_to_prior(p) for p in self.chain
    #     ])
        
    #     end_time = default_timer()
    #     logger.info(f"NUTS sampling took {end_time - start_time:.1f} seconds")
        
    #     # Process NUTS results
    #     self._process_nuts_results(tempering)
    
    # def _log_prob_nuts_impl(self, physical_params_tf, emulator, X_mean_tf, X_std_tf,
    #                         y_mean_tf, y_std_tf, data_vector_tf, inv_covariance_tf, 
    #                         tempering, DTYPE):
    #     """Implementation of log probability function for NUTS."""
    #     import tensorflow as tf
        
    #     # Ensure 1D
    #     if len(physical_params_tf.shape) > 1:
    #         physical_params_tf = tf.reshape(physical_params_tf, [-1])
        
    #     # Compute predictions through emulator (fully differentiable)
    #     # Use the input tensor directly - TensorFlow will track gradients automatically
    #     params_norm = (physical_params_tf - X_mean_tf) / X_std_tf
    #     params_nn_norm = (params_norm - emulator.cp_nn.parameters_mean) / emulator.cp_nn.parameters_std
        
    #     if len(params_nn_norm.shape) == 1:
    #         params_nn_norm = tf.expand_dims(params_nn_norm, 0)
        
    #     x = params_nn_norm
    #     layers = [x]
        
    #     if emulator.cp_nn.architecture_type == "MLP":
    #         for i in range(emulator.cp_nn.n_layers - 1):
    #             linear_out = tf.matmul(layers[-1], emulator.cp_nn.W[i]) + emulator.cp_nn.b[i]
    #             activated = emulator.cp_nn.activation(linear_out, emulator.cp_nn.alphas[i], emulator.cp_nn.betas[i])
    #             layers.append(activated)
    #         output = tf.matmul(layers[-1], emulator.cp_nn.W[-1]) + emulator.cp_nn.b[-1]
        
    #     pred_norm = output * emulator.cp_nn.features_std + emulator.cp_nn.features_mean
    #     if len(pred_norm.shape) == 2:
    #         pred_norm = pred_norm[0]
        
    #     pred_intermediate = pred_norm * y_std_tf + y_mean_tf
        
    #     if emulator.data_trafo == 'log_norm':
    #         predictions = tf.pow(10.0, pred_intermediate)
    #     elif emulator.data_trafo == 'norm':
    #         predictions = pred_intermediate
    #     elif emulator.data_trafo == 'PCA':
    #         pca_matrix_tf = tf.constant(emulator.pca_transform_matrix, dtype=DTYPE)
    #         features_std_tf = tf.constant(emulator.features_std, dtype=DTYPE)
    #         features_mean_tf = tf.constant(emulator.features_mean, dtype=DTYPE)
    #         pca_reconstructed = tf.matmul(tf.expand_dims(pred_intermediate, 0), pca_matrix_tf)
    #         predictions = (pca_reconstructed * features_std_tf + features_mean_tf)[0]
    #     else:
    #         predictions = pred_intermediate
        
    #     # Compute log_likelihood from predictions using TensorFlow operations
    #     if data_vector_tf is not None and inv_covariance_tf is not None:
    #         # Gaussian likelihood: -0.5 * (x - mu)^T * C^-1 * (x - mu)
    #         diff = predictions - data_vector_tf
    #         # Compute chi-squared: diff^T * inv_cov * diff
    #         # Use einsum with explicit scalar output ('->' means output is scalar)
    #         chi2 = tf.einsum('i,ij,j->', diff, inv_covariance_tf, diff)
    #         log_like = -0.5 * chi2
    #         # Ensure it's a scalar (explicit reshape to empty shape)
    #         log_like = tf.reshape(log_like, [])
    #     else:
    #         # Fallback: use py_function to get likelihood from pipeline
    #         # This breaks gradient flow, but is a fallback
    #         def get_likelihood(params_np):
    #             try:
    #                 r = self.emu_pipeline.run_results(params_np)
    #                 return float(r.like)
    #             except:
    #                 return -np.inf
            
    #         log_like = tf.py_function(
    #             func=get_likelihood,
    #             inp=[physical_params_tf],
    #             Tout=tf.float32
    #         )
    #         log_like.set_shape([])
        
    #     # Get log_prior using py_function and stop gradients
    #     # For uniform priors, gradients are typically 0 anyway
    #     def get_prior(params_np):
    #         try:
    #             unit_params = self.pipeline.normalize_vector_to_prior(params_np)
    #             if np.any(unit_params < 0) or np.any(unit_params > 1):
    #                 return -np.inf
    #             return self.pipeline.prior(unit_params)
    #         except:
    #             return -np.inf
        
    #     log_prior = tf.py_function(
    #         func=get_prior,
    #         inp=[physical_params_tf],
    #         Tout=tf.float32
    #     )
    #     log_prior.set_shape([])
    #     # Ensure it's a scalar
    #     log_prior = tf.reshape(log_prior, [])
    #     # Stop gradients through prior (for uniform priors, gradients are 0)
    #     log_prior = tf.stop_gradient(log_prior)
        
    #     # Log posterior with tempering
    #     # Both log_like and log_prior are now guaranteed to be scalars
    #     log_post = (log_like + log_prior) * tempering
    #     # Ensure final result is a scalar
    #     log_post = tf.reshape(log_post, [])
    #     return log_post
    
    # def _process_nuts_results(self, tempering: float) -> None:
    #     """Process NUTS results and update output chains."""
    #     # Calculate burn-in (use same logic as emcee)
    #     if hasattr(self, 'emcee_burn') and self.emcee_burn < 1:
    #         burn = int(self.emcee_burn * len(self.chain))
    #     elif hasattr(self, 'emcee_burn'):
    #         burn = int(self.emcee_burn)
    #     else:
    #         burn = 0
        
    #     # Apply burn-in and thinning
    #     if hasattr(self, 'emcee_thin'):
    #         thin = self.emcee_thin
    #     else:
    #         thin = 1
        
    #     self.chain = self.chain[burn::thin]
    #     self.unit_chain = self.unit_chain[burn::thin]
    #     logp = self.nuts_logp[burn::thin]
    #     self.blobs = self.blobs[burn::thin]
        
    #     # Handle output file management
    #     if self.save_outputs == SAVE_ALL and 0 < self.iterations < self.max_iterations:
    #         suffix = f'_tempering_{self.tempering[self.iterations-1]}_iteration_{self.iterations}'
    #         self.output.save_and_reset_to_chain_start(suffix)
    #     else:
    #         self.output.reset_to_chain_start()
        
    #     # Output chain points (similar to emcee format)
    #     for params, tempered_post, blob in zip(self.chain, logp, self.blobs):
    #         prior, extra = blob
    #         post = tempered_post / tempering
            
    #         # NUTS doesn't use log weights (unlike nautilus)
    #         self.output.parameters(params, extra, prior, tempered_post, post)
        
    #     logger.info(f"Generated {len(self.chain)} NUTS samples")