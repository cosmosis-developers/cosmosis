"""
CosmoSIS Neural Network Emulator Implementation

This module provides neural network emulators for cosmological calculations,
supporting Multi-Layer Perceptron (MLP) architecture.
The emulators can be trained on cosmological data vectors and used for fast
predictions during parameter estimation.

Key Features:
- Data preprocessing with normalization and PCA
- GPU/CPU support with automatic device detection
- Comprehensive training with cooling schedules and early stopping
- Model saving/loading with full state preservation
- Extensive logging and diagnostics

Authors: CosmoSIS Team
License: BSD 2-Clause
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

import numpy as np
import tensorflow as tf
from sklearn.decomposition import IncrementalPCA
import pickle
from tqdm import trange

# Configure logging
logger = logging.getLogger(__name__)

# Set TensorFlow data type
DTYPE = tf.float32

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)

# Device detection with better error handling
def get_device() -> str:
    """Detect and return the best available compute device.
    
    Returns:
        Device string ('gpu:0' or 'cpu')
    """
    try:
        if tf.test.is_gpu_available():
            # Check if GPU is actually usable
            with tf.device('/gpu:0'):
                test = tf.constant([1.0])
                _ = tf.square(test)
            logger.info("GPU detected and verified - using GPU acceleration")
            return 'gpu:0'
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
    
    logger.info("Using CPU for computations")
    return 'cpu'

DEVICE = get_device()



class CosmoPowerNN(tf.keras.Model):
    """Main neural network model for cosmological emulation.
    
    This class implements MLP architecture for emulating
    cosmological calculations. It handles data preprocessing, training,
    and prediction with comprehensive error handling and logging.
    
    Args:
        parameters: List of parameter names
        modes: List of mode/feature names
        parameters_mean: Mean values for parameter normalization
        parameters_std: Standard deviation for parameter normalization
        features_mean: Mean values for feature normalization
        features_std: Standard deviation for feature normalization
        n_hidden: Hidden layer sizes for MLP architecture
        restore: Whether to restore from saved model
        restore_filename: Filename for model restoration
        trainable: Whether model parameters are trainable
        optimizer: TensorFlow optimizer to use
        verbose: Whether to print detailed information
        architecture_type: Type of architecture ('MLP' or implement other architecture of your choice)
    """
    
    def __init__(self,
                 parameters: Optional[List[str]] = None,
                 modes: Optional[List[str]] = None,
                 parameters_mean: Optional[np.ndarray] = None,
                 parameters_std: Optional[np.ndarray] = None,
                 features_mean: Optional[np.ndarray] = None,
                 features_std: Optional[np.ndarray] = None,
                 n_hidden: List[int] = [512, 512, 512],
                 restore: bool = False,
                 restore_filename: Optional[str] = None,
                 trainable: bool = True,
                 optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                 verbose: bool = False,
                 architecture_type: str = "MLP",
                 loss_function: str = "standard",
                 **kwargs):
        
        super(CosmoPowerNN, self).__init__(**kwargs)
        
        self.architecture_type = architecture_type
        self.verbose = verbose
        if loss_function == "standard":
            self.compute_loss = self.compute_loss_standard
        elif loss_function == "weighted_w_cov":
            self.compute_loss = self.compute_loss_weighted_w_cov
        else:
            raise ValueError(f"Unknown loss_function type: {loss_function}")
        self.loss_function = loss_function
        # Handle model restoration
        if restore:
            if not restore_filename:
                raise ValueError("restore_filename must be provided when restore=True")
            self.restore(restore_filename)
        else:
            # Initialize from parameters
            self._initialize_from_parameters(
                parameters, modes, parameters_mean, parameters_std,
                features_mean, features_std, n_hidden
            )
        
        # Set up normalization constants
        self._setup_normalization_constants()
        
        # Build network architecture
        self._build_network(trainable)
        
        # If we restored from file, assign loaded weights to Variables
        if restore and hasattr(self, 'W_'):
            if self.architecture_type == "MLP":
                for i in range(len(self.W_)):
                    self.W[i].assign(self.W_[i])
                    self.b[i].assign(self.b_[i])
                for i in range(len(self.alphas_)):
                    self.alphas[i].assign(self.alphas_[i])
                    self.betas[i].assign(self.betas_[i])
        
        # Set up optimizer
        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        
        # Print initialization info
        if self.verbose:
            self._print_initialization_info()

    def _initialize_from_parameters(self,
                                  parameters: List[str],
                                  modes: List[str],
                                  parameters_mean: np.ndarray,
                                  parameters_std: np.ndarray,
                                  features_mean: np.ndarray,
                                  features_std: np.ndarray,
                                  n_hidden: List[int]) -> None:
        """Initialize model from provided parameters."""
        if not parameters:
            raise ValueError("parameters must be provided when not restoring")
        
        self.parameters = parameters
        self.n_parameters = len(self.parameters)
        self.modes = modes or list(range(len(features_mean))) if features_mean is not None else []
        self.n_modes = len(self.modes)
        self.n_hidden = n_hidden
        
        # Store normalization parameters
        self.parameters_mean_ = parameters_mean
        self.parameters_std_ = parameters_std
        self.features_mean_ = features_mean
        self.features_std_ = features_std

    def _setup_normalization_constants(self) -> None:
        """Set up TensorFlow constants for normalization."""
        self.parameters_mean = tf.constant(
            self.parameters_mean_, dtype=DTYPE, name='parameters_mean'
        )
        self.parameters_std = tf.constant(
            self.parameters_std_, dtype=DTYPE, name='parameters_std'
        )
        self.features_mean = tf.constant(
            self.features_mean_, dtype=DTYPE, name='features_mean'
        )
        self.features_std = tf.constant(
            self.features_std_, dtype=DTYPE, name='features_std'
        )

    def _build_network(self, trainable: bool) -> None:
        """Build the neural network architecture."""
        if self.architecture_type == "MLP":
            self._build_mlp_network(trainable)
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture_type}")

    def _build_mlp_network(self, trainable: bool) -> None:
        """Build MLP architecture with custom activation functions."""
        # Architecture definition
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_modes]
        self.n_layers = len(self.architecture) - 1
        
        # Initialize weights and biases
        self.W, self.b, self.alphas, self.betas = [], [], [], []
        
        for i in range(self.n_layers):
            self.W.append(tf.Variable(
                tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3),
                name=f"W_{i}",
                trainable=trainable
            ))
            self.b.append(tf.Variable(
                tf.zeros([self.architecture[i+1]]),
                name=f"b_{i}",
                trainable=trainable
            ))
        
        # Activation function parameters (for all layers except output)
        for i in range(self.n_layers - 1):
            self.alphas.append(tf.Variable(
                tf.random.normal([self.architecture[i+1]]),
                name=f"alphas_{i}",
                trainable=trainable
            ))
            self.betas.append(tf.Variable(
                tf.random.normal([self.architecture[i+1]]),
                name=f"betas_{i}",
                trainable=trainable
            ))


    def _print_initialization_info(self) -> None:
        """Print model initialization information."""
        info_str = (
            f"\\nInitialized {self.architecture_type} model\\n"
            f"Mapping {self.n_parameters} input parameters to {self.n_modes} output modes\\n"
            f"Using {len(self.n_hidden)} hidden layers with {self.n_hidden} nodes\\n"
        )
        logger.info(info_str)

    def activation(self, x: tf.Tensor, alpha: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
        """Custom activation function with learnable parameters.
        
        This implements a parameterized activation function that can adapt
        during training to better fit the data.
        
        Args:
            x: Input tensor
            alpha: Scale parameter
            beta: Shift parameter
            
        Returns:
            Activated tensor
        """
        sigmoid_part = tf.sigmoid(alpha * x)
        return (beta + (1.0 - beta) * sigmoid_part) * x

    @tf.function
    def predictions_tf(self, parameters_tensor: tf.Tensor) -> tf.Tensor:
        """TensorFlow forward pass for predictions.
        
        Args:
            parameters_tensor: Input parameters tensor
            
        Returns:
            Predicted features tensor
        """
        # Normalize inputs
        x = (parameters_tensor - self.parameters_mean) / self.parameters_std
        
        if self.architecture_type == "MLP":
            # MLP forward pass
            layers = [x]
            
            for i in range(self.n_layers - 1):
                # Linear transformation
                linear_out = tf.matmul(layers[-1], self.W[i]) + self.b[i]
                # Apply custom activation
                activated = self.activation(linear_out, self.alphas[i], self.betas[i])
                layers.append(activated)
            
            # Output layer (linear)
            output = tf.matmul(layers[-1], self.W[-1]) + self.b[-1]
            
        else:
            raise NotImplementedError(f"Prediction not implemented for architecture: {self.architecture_type}")
        
        # Denormalize output
        return output * self.features_std + self.features_mean

    def forward_pass_np(self, parameters_arr: np.ndarray) -> np.ndarray:
        """NumPy forward pass for CPU predictions.
        
        This method provides a NumPy-only forward pass for cases where
        TensorFlow operations are not needed or desired.
        
        Args:
            parameters_arr: Input parameters array
            
        Returns:
            Predicted features array
        """
        if self.architecture_type != "MLP":
            raise NotImplementedError("NumPy forward pass only implemented for MLP")
        
        # Normalize inputs
        layers = [(parameters_arr - self.parameters_mean_) / self.parameters_std_]
        
        # Forward pass through layers
        for i in range(self.n_layers - 1):
            # Linear transformation
            linear_out = np.dot(layers[-1], self.W_[i]) + self.b_[i]
            
            # Custom activation function
            sigmoid_part = 1.0 / (1.0 + np.exp(-self.alphas_[i] * linear_out))
            activated = (self.betas_[i] + (1.0 - self.betas_[i]) * sigmoid_part) * linear_out
            layers.append(activated)
        
        # Output layer
        output = np.dot(layers[-1], self.W_[-1]) + self.b_[-1]
        
        # Denormalize and return
        return output * self.features_std_ + self.features_mean_

    def predictions_np(self, parameters_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions using NumPy forward pass (MLP).
        
        Args:
            parameters_dict: Dictionary mapping parameter names to values
            
        Returns:
            Predicted features array
        """
        if self.architecture_type == "MLP":
            parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
            return self.forward_pass_np(parameters_arr)
        else:
            raise NotImplementedError(f"Prediction not implemented for architecture: {self.architecture_type}")

    def dict_to_ordered_arr_np(self, input_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to ordered array.
        
        Args:
            input_dict: Dictionary of parameter values
            
        Returns:
            Ordered parameter array
        """
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)

    def update_emulator_parameters(self) -> None:
        """Update emulator parameters for saving.
        
        This method extracts the current model parameters and stores them
        in a format suitable for saving and later restoration.
        """
        if self.architecture_type == "MLP":
            self.emulator_parameters = {
                "W": [w.numpy() for w in self.W],
                "b": [b.numpy() for b in self.b],
                "alphas": [a.numpy() for a in self.alphas],
                "betas": [b.numpy() for b in self.betas],
            }
            # Also store as individual arrays for NumPy forward pass
            self.W_ = [w.numpy() for w in self.W]
            self.b_ = [b.numpy() for b in self.b]
            self.alphas_ = [a.numpy() for a in self.alphas]
            self.betas_ = [b.numpy() for b in self.betas]
        else:
            raise NotImplementedError(f"Update emulator parameters not implemented for architecture: {self.architecture_type}")

    @tf.function
    def compute_loss_standard(self, training_parameters: tf.Tensor, training_features: tf.Tensor) -> tf.Tensor:
        """Compute training loss (RMSE).
        
        Args:
            training_parameters: Parameter tensor
            training_features: Target features tensor
            
        Returns:
            Root mean squared error loss
        """
        predictions = self.predictions_tf(training_parameters)
        return tf.sqrt(tf.reduce_mean(tf.square(predictions - training_features)))

    @tf.function
    def compute_loss_weighted_w_cov(self, training_parameters: tf.Tensor, training_features: tf.Tensor) -> tf.Tensor:
        """Compute weighted loss using inverse covariance matrix.
        
        Args:
            training_parameters: Parameter tensor
            training_features: Target features tensor
            
        Returns:
            Weighted loss using inverse covariance
        """
        predictions = self.predictions_tf(training_parameters)
        diff = predictions - training_features
        weighted_diff = tf.matmul(diff, self.data_inv_cov)
        return tf.sqrt(tf.reduce_mean(tf.reduce_sum(diff * weighted_diff, axis=1)))

    @tf.function
    def compute_loss_and_gradients(self, training_parameters: tf.Tensor, training_features: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Compute loss and gradients for training step.
        
        Args:
            training_parameters: Parameter tensor
            training_features: Target features tensor
            
        Returns:
            Tuple of (loss, gradients)
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(training_parameters, training_features)
        gradients = tape.gradient(loss, self.trainable_variables)
        return loss, gradients

    # def compute_prediction_gradients(self, parameters_tensor: tf.Tensor) -> tf.Tensor:
    #     """Compute gradients of predictions with respect to input parameters using autodifferentiation.
        
    #     This method uses TensorFlow's automatic differentiation to compute the Jacobian
    #     matrix: d(predictions)/d(parameters). This is useful for Fisher matrix calculations,
    #     optimization, and sensitivity analysis.
        
    #     Args:
    #         parameters_tensor: Input parameters tensor of shape (batch_size, n_parameters)
            
    #     Returns:
    #         Gradient tensor of shape (batch_size, n_modes, n_parameters)
    #         where gradient[i, j, k] = d(prediction[i, j])/d(parameter[i, k])
    #     """
    #     # Ensure we have a proper tensor
    #     if not isinstance(parameters_tensor, tf.Variable):
    #         parameters_tensor = tf.Variable(parameters_tensor, trainable=False)
        
    #     # Get batch size as Python int
    #     try:
    #         # Try to get static shape first
    #         batch_size = int(parameters_tensor.shape[0])
    #     except (TypeError, ValueError):
    #         # If static shape is None, get from tensor
    #         batch_size_tensor = tf.shape(parameters_tensor)[0]
    #         try:
    #             batch_size = int(batch_size_tensor.numpy())
    #         except (AttributeError, ValueError, tf.errors.InvalidArgumentError):
    #             # If we can't evaluate, default to 1 (will process first sample)
    #             batch_size = 1
        
    #     n_modes = self.n_modes
    #     n_params = self.n_parameters
        
    #     # Compute gradients for each sample in the batch
    #     # We need to compute gradients separately for each sample because
    #     # GradientTape computes gradients element-wise
    #     all_gradients = []
        
    #     for batch_idx in range(batch_size):
    #         # Extract single sample (keep batch dimension for compatibility)
    #         sample_params = parameters_tensor[batch_idx:batch_idx+1]
    #         # Use a regular tensor and watch it (not a Variable)
    #         sample_params_tensor = tf.identity(sample_params)
            
    #         with tf.GradientTape(persistent=True) as tape:
    #             tape.watch(sample_params_tensor)
    #             sample_predictions = self.predictions_tf(sample_params_tensor)
    #             # Remove batch dimension for jacobian computation: (1, n_modes) -> (n_modes,)
    #             sample_predictions_flat = sample_predictions[0]
            
    #         # Compute full Jacobian at once: d(predictions)/d(parameters)
    #         # This is more efficient and avoids indexing issues
    #         # jacobian shape: (n_modes, 1, n_params) -> squeeze to (n_modes, n_params)
    #         try:
    #             jacobian = tape.jacobian(sample_predictions_flat, sample_params_tensor)
    #             # jacobian shape is (n_modes, 1, n_params), squeeze middle dimension
    #             if len(jacobian.shape) == 3:
    #                 jacobian = tf.squeeze(jacobian, axis=1)  # (n_modes, n_params)
    #             sample_gradient_matrix = jacobian
    #         except (AttributeError, ValueError) as e:
    #             # Fallback: compute gradients element by element if jacobian fails
    #             sample_gradients = []
    #             for mode_idx in range(n_modes):
    #                 output_element = sample_predictions_flat[mode_idx]
    #                 grad = tape.gradient(output_element, sample_params_tensor)
    #                 if grad is not None:
    #                     sample_gradients.append(tf.reshape(grad, [n_params]))
    #                 else:
    #                     sample_gradients.append(tf.zeros([n_params], dtype=DTYPE))
    #             sample_gradient_matrix = tf.stack(sample_gradients, axis=0)
            
    #         del tape
    #         all_gradients.append(sample_gradient_matrix)
        
    #     # Stack all samples to get shape (batch_size, n_modes, n_parameters)
    #     gradient_tensor = tf.stack(all_gradients, axis=0)
        
    #     return gradient_tensor

    def training_step(self, training_parameters: tf.Tensor, training_features: tf.Tensor) -> tf.Tensor:
        """Perform one training step.
        
        Args:
            training_parameters: Parameter tensor
            training_features: Target features tensor
            
        Returns:
            Training loss for this step
        """
        loss, gradients = self.compute_loss_and_gradients(training_parameters, training_features)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self,
              training_parameters: Dict[str, np.ndarray],
              training_features: np.ndarray,
              filename_saved_model: str,
              validation_split: float = 0.1,
              learning_rates: List[float] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              batch_sizes: List[int] = [1024, 1024, 1024, 1024, 1024],
              gradient_accumulation_steps: List[int] = [1, 1, 1, 1, 1],
              patience_values: List[int] = [100, 100, 100, 100, 100],
              max_epochs: List[int] = [1000, 1000, 1000, 1000, 1000]) -> None:
        """Train the neural network with cooling schedule and early stopping.
        
        This method implements a comprehensive training procedure with:
        - Multiple learning rate stages
        - Early stopping based on validation loss
        - Automatic model saving
        - Detailed progress tracking
        
        Args:
            training_parameters: Dictionary of training parameters
            training_features: Training target features
            filename_saved_model: Base filename for saving model
            validation_split: Fraction of data to use for validation
            learning_rates: Learning rates for each training stage
            batch_sizes: Batch sizes for each training stage
            gradient_accumulation_steps: Gradient accumulation steps (not currently used)
            patience_values: Early stopping patience for each stage
            max_epochs: Maximum epochs for each training stage
        """
        # Validate input arguments
        arg_lengths = [len(learning_rates), len(batch_sizes), len(gradient_accumulation_steps),
                      len(patience_values), len(max_epochs)]
        if not all(length == arg_lengths[0] for length in arg_lengths):
            raise ValueError("All training parameter lists must have the same length")

        # Log training start information
        if self.verbose:
            logger.info(f"Starting training with {int(100*validation_split)}% validation split")
            logger.info(f"Training stages: {len(learning_rates)}")
            logger.info(f"Learning rates: {learning_rates}")
            logger.info(f"Batch sizes: {batch_sizes}")
            logger.info(f"Patience values: {patience_values}")
            logger.info(f"Max epochs: {max_epochs}")

        # Convert parameters dictionary to array and normalize
        training_parameters_arr = self.dict_to_ordered_arr_np(training_parameters)
        
        # Compute normalization statistics
        self.parameters_mean = np.mean(training_parameters_arr, axis=0)
        self.parameters_std = np.std(training_parameters_arr, axis=0)
        self.features_mean = np.mean(training_features, axis=0)
        self.features_std = np.std(training_features, axis=0)
        
        # Store as numpy arrays for later use
        self.parameters_mean_ = self.parameters_mean.copy()
        self.parameters_std_ = self.parameters_std.copy()
        self.features_mean_ = self.features_mean.copy()
        self.features_std_ = self.features_std.copy()
        
        # Update TensorFlow constants
        self._setup_normalization_constants()
        
        # Convert to TensorFlow tensors
        training_parameters_tf = tf.convert_to_tensor(training_parameters_arr, dtype=DTYPE)
        training_features_tf = tf.convert_to_tensor(training_features, dtype=DTYPE)
        
        # Training/validation split
        n_samples = training_parameters_tf.shape[0]
        n_validation = int(n_samples * validation_split)
        n_training = n_samples - n_validation
        
        diagnostics = {}
        
        # Training loop with cooling schedule
        with tf.device(DEVICE):
            for stage in range(len(learning_rates)):
                logger.info(f"Training stage {stage + 1}/{len(learning_rates)}: "
                           f"lr={learning_rates[stage]}, batch_size={batch_sizes[stage]}")
                
                # Set learning rate
                self.optimizer.learning_rate = learning_rates[stage]
                
                # Create random training/validation split
                indices = tf.random.shuffle(tf.range(n_samples))
                train_indices = indices[:n_training]
                val_indices = indices[n_training:]
                
                # Create training dataset
                train_params = tf.gather(training_parameters_tf, train_indices)
                train_features = tf.gather(training_features_tf, train_indices)
                train_dataset = tf.data.Dataset.from_tensor_slices((train_params, train_features))
                train_dataset = train_dataset.shuffle(n_training).batch(batch_sizes[stage])
                
                # Validation data
                val_params = tf.gather(training_parameters_tf, val_indices)
                val_features = tf.gather(training_features_tf, val_indices)
                
                # Initialize tracking variables
                best_val_loss = np.inf
                patience_counter = 0
                stage_diagnostics = {
                    'epochs': [],
                    'training_loss': [],
                    'validation_loss': []
                }
                
                # Training epochs
                for epoch in range(max_epochs[stage]):
                    # Training step
                    epoch_train_losses = []
                    for batch_params, batch_features in train_dataset:
                        batch_loss = self.training_step(batch_params, batch_features)
                        epoch_train_losses.append(batch_loss.numpy())
                    
                    avg_train_loss = np.mean(epoch_train_losses)
                    
                    # Validation step
                    val_loss = self.compute_loss(val_params, val_features).numpy()
                    
                    # Record diagnostics
                    stage_diagnostics['epochs'].append(epoch)
                    stage_diagnostics['training_loss'].append(avg_train_loss)
                    stage_diagnostics['validation_loss'].append(val_loss)
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.update_emulator_parameters()
                    # Custom stopping condition
                    #TODO: Make this a parameter
                    elif best_val_loss < 5e-4:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.update_emulator_parameters()   
                    else:
                        patience_counter += 1
                    
                    # Log progress every 10 epochs
                    if epoch % 10 == 0 or patience_counter >= patience_values[stage]:
                        logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
                                  f"val_loss={val_loss:.6f}, best_val={best_val_loss:.6f}")
                        print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
                                  f"val_loss={val_loss:.6f}, best_val={best_val_loss:.6f}")
                    
                    # Early stopping
                    if patience_counter >= patience_values[stage]:
                        logger.info(f"Early stopping at epoch {epoch} (patience={patience_values[stage]})")
                        break
                
                # Store stage diagnostics
                diagnostics[f'learning_cycle_{stage}'] = stage_diagnostics
                logger.info(f"Stage {stage + 1} completed. Best validation loss: {best_val_loss:.6f}")
        
        # Final model save
        self.save(filename_saved_model, diagnostics)
        logger.info(f"Training completed. Final model saved to {filename_saved_model}")
        
        # Print model summary
        if self.verbose:
            self.summary()

    def save(self, filename: str, diagnostics: Dict[str, Any]) -> None:
        """Save model parameters and diagnostics.
        
        Args:
            filename: Base filename for saving (without extension)
            diagnostics: Training diagnostics to save
        """

        # First, update emulator parameters to extract current weights
        self.update_emulator_parameters()
        
        save_dict = {
            "architecture_type": self.architecture_type,
            "diagnostics": diagnostics,
            "parameters_mean": self.parameters_mean_.tolist(),
            "parameters_std": self.parameters_std_.tolist(),
            "features_mean": self.features_mean_.tolist(),
            "features_std": self.features_std_.tolist(),
            "parameters": self.parameters,
            "modes": self.modes,
            "loss_function": self.loss_function,
            "n_hidden": getattr(self, 'n_hidden', [])
        }
        
        if self.architecture_type == "MLP":
            # Use the extracted numpy weights (W_, b_, etc.) instead of TF variables
            save_dict["weights"] = {
                "W": [w.tolist() for w in self.W_],
                "b": [b.tolist() for b in self.b_],
                "alphas": [a.tolist() for a in self.alphas_],
                "betas": [b.tolist() for b in self.betas_]
            }
        else:
            raise NotImplementedError(f"Save not implemented for architecture: {self.architecture_type}")

        # Save main model file
        np.savez_compressed(filename + ".npz", **save_dict)
        logger.info(f"Model saved to {filename}.npz")

    def restore(self, filename: str) -> None:
        """Load pre-trained model from file.
        
        Args:
            filename: Filename to load from (with or without .npz extension)
        """
        # Handle filename with or without extension
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
        # Load model data
        data = np.load(filename, allow_pickle=True)
        
        # Restore basic attributes
        self.architecture_type = str(data["architecture_type"])
        self.loss_function = str(data["loss_function"])
        self.parameters_mean_ = np.array(data["parameters_mean"])
        self.parameters_std_ = np.array(data["parameters_std"])
        self.features_mean_ = np.array(data["features_mean"])
        self.features_std_ = np.array(data["features_std"])
        
        # Restore model-specific attributes
        if "parameters" in data:
            self.parameters = list(data["parameters"])
            self.n_parameters = len(self.parameters)
        if "modes" in data:
            self.modes = list(data["modes"])
            self.n_modes = len(self.modes)
        if "n_hidden" in data:
            self.n_hidden = list(data["n_hidden"])
        
        # Restore weights for MLP architecture
        if self.architecture_type == "MLP" and "weights" in data:
            weights = data["weights"].item()
            self.W_ = [np.array(w) for w in weights["W"]]
            self.b_ = [np.array(b) for b in weights["b"]]
            self.alphas_ = [np.array(a) for a in weights["alphas"]]
            self.betas_ = [np.array(b) for b in weights["betas"]]
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture_type}")
        
        logger.info(f"Model restored from {filename} with architecture: {self.architecture_type}")
            

    def summary(self) -> None:
        """Print detailed model summary."""
        print("\\n" + "="*60)
        print(f" CosmoPower Model Summary ({self.architecture_type})")
        print("="*60)
        
        if self.architecture_type == "MLP":
            self._print_mlp_summary()
        else:
            raise NotImplementedError(f"Summary not implemented for architecture: {self.architecture_type}")
        
        print("="*60 + "\\n")

    def _print_mlp_summary(self) -> None:
        """Print MLP-specific summary."""
        total_params = 0
        
        print(f"{'Layer (type)':<25} {'Output Shape':<15} {'Param #':<10}")
        print("-" * 50)
        
        input_dim = self.n_parameters
        for i in range(self.n_layers):
            output_dim = self.architecture[i + 1]
            
            # Weight and bias parameters
            w_params = input_dim * output_dim
            b_params = output_dim
            layer_params = w_params + b_params
            total_params += layer_params
            
            print(f"Dense_{i:<20} ({output_dim:>3},){'':<10} {layer_params:>10}")
            
            # Activation parameters (for hidden layers)
            if i < len(self.alphas):
                activation_params = 2 * output_dim  # alphas + betas
                total_params += activation_params
                print(f"Activation_{i:<15} ({output_dim:>3},){'':<10} {activation_params:>10}")
            
            input_dim = output_dim
        
        print("-" * 50)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print("Non-trainable params: 0")



class NNEmulator:
    """CosmoSIS-compatible emulator wrapper for neural network models.
    
    This class provides a high-level interface for training and using
    neural network emulators in CosmoSIS. It handles data preprocessing,
    model training, and prediction with various transformation options.
    
    Args:
        model_parameters: List of parameter names
        modes: Output modes/features
        nn_model: Neural network architecture ('MLP' or other models of your choice -- to be implemented)
        iteration: Current training iteration (for naming)
        data_trafo: Data transformation type ('log_norm', 'norm', 'PCA')
        n_pca: Number of PCA components (if using PCA)
        datavector: Reference data vector for weighted training
        inv_cov: Inverse covariance matrix for weighted training
    """
    
    def __init__(self,
                 model_parameters: List[str],
                 modes: Union[List[str], np.ndarray],
                 nn_model: str = 'MLP',
                 loss_function: str = "standard",
                 iteration: int = 1,
                 data_trafo: str = 'log_norm',
                 n_pca: int = 64,
                 datavector: Optional[np.ndarray] = None,
                 inv_cov: Optional[np.ndarray] = None):
        
        self.trained = False
        self.model_parameters = model_parameters
        self.modes = modes
        self.data_trafo = data_trafo
        self.datavector = datavector
        self.data_inv_cov = inv_cov
        self.n_pca = n_pca
        self.nn_model = nn_model
        self.loss_function = loss_function
        self.iteration = iteration
        
        # Initialize transformation attributes
        self.pca_transform_matrix = None
        self.y_mean = None
        self.y_std = None
        self.features_mean = None
        self.features_std = None
        self.X_mean = None
        self.X_std = None
        
        logger.info(f"NNEmulator initialized with {data_trafo} transformation")

    def transform(self, model_datavector: np.ndarray) -> np.ndarray:
        """Transform data vector for neural network training.
        
        Args:
            model_datavector: Raw data vector to transform
            
        Returns:
            Transformed data vector ready for training
        """
        if self.data_trafo == 'log_norm':
            return self._log_norm_transform(model_datavector)
        elif self.data_trafo == 'norm':
            return self._norm_transform(model_datavector)
        elif self.data_trafo == 'PCA':
            return self._pca_transform(model_datavector)
        else:
            raise ValueError(f"Unknown data transformation: {self.data_trafo}")

    def _log_norm_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply log-normalization transformation."""
        # Handle potential zeros or negative values
        data_safe = np.maximum(data, 1e-30)
        y = np.log10(data_safe)
        
        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0)
        
        # Avoid division by zero
        self.y_std = np.maximum(self.y_std, 1e-10)
              
        return (y - self.y_mean) / self.y_std

    def _norm_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply standard normalization transformation."""
        self.y_mean = np.mean(data, axis=0)
        self.y_std = np.std(data, axis=0)
        
        # Avoid division by zero
        self.y_std = np.maximum(self.y_std, 1e-10)
               
        return (data - self.y_mean) / self.y_std

    def _pca_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply PCA transformation."""
        # Normalize first
        y_mean = np.mean(data, axis=0)
        y_std = np.std(data, axis=0)
        y_std = np.maximum(y_std, 1e-10)
        normalized_data = (data - y_mean) / y_std
        
        # Apply PCA
        pca = IncrementalPCA(n_components=self.n_pca)
        pca.fit(normalized_data)
        
        # Store transformation matrix and parameters
        self.pca_transform_matrix = pca.components_
        self.features_mean = y_mean
        self.features_std = y_std
        
        # Transform data
        pca_data = pca.transform(normalized_data)
        
        # Normalize PCA components
        #self.y_mean = np.mean(pca_data, axis=0)
        #self.y_std = np.std(pca_data, axis=0)
        #self.y_std = np.maximum(self.y_std, 1e-10)
        self.y_mean = np.zeros(pca_data.shape[1])
        self.y_std = np.ones(pca_data.shape[1])
        
        return (pca_data - self.y_mean) / self.y_std

    def backtransform(self, model_datavector: np.ndarray) -> np.ndarray:
        """Transform predictions back to original space.
        
        Args:
            model_datavector: Transformed predictions
            
        Returns:
            Predictions in original data space
        """
        if self.data_trafo == 'log_norm':
            return 10 ** model_datavector
        elif self.data_trafo == 'norm':
            return model_datavector
        elif self.data_trafo == 'PCA':
            # Reverse PCA transformation
            pca_reconstructed = np.dot(model_datavector, self.pca_transform_matrix)
            return pca_reconstructed * self.features_std + self.features_mean
        else:
            raise ValueError(f"Unknown data transformation: {self.data_trafo}")

    def train(self,
              X: Dict[str, np.ndarray],
              y: np.ndarray,
              model_filename: str,
              test_split: float = 0.1,
              batch_size: int = 32,
              n_cycles: int = 5) -> None:
        """Train the neural network emulator.
        
        Args:
            X: Dictionary of input parameters
            y: Target output data
            model_filename: Base filename for saving model
            test_split: Fraction of data for validation
            batch_size: Training batch size
            n_cycles: Number of training cycles
        """
        logger.info("Starting emulator training")
        

        # Validate inputs
        if not X or len(X) == 0:
            raise ValueError("Input parameters X cannot be empty")
        if y is None or len(y) == 0:
            raise ValueError("Target data y cannot be empty")
        
        # Check parameter consistency
        param_lengths = [len(X[key]) for key in X.keys()]
        if not all(length == param_lengths[0] for length in param_lengths):
            raise ValueError("All input parameters must have the same length")
        if len(y) != param_lengths[0]:
            raise ValueError("Target data length must match input parameter length")
        
        # Normalize input parameters
        logger.info("Normalizing input parameters")
        self.X_mean = {key: np.mean(X[key], axis=0) for key in X.keys()}
        self.X_std = {key: np.maximum(np.std(X[key], axis=0), 1e-10) for key in X.keys()}
        
        # Transform target data
        logger.info(f"Applying {self.data_trafo} transformation to target data")
        y_train = self.transform(y)
        
        # Prepare normalization arrays
        X_mean_arr = np.array([self.X_mean[key] for key in self.model_parameters])
        X_std_arr = np.array([self.X_std[key] for key in self.model_parameters])
        
        # Create neural network
        logger.info(f"Creating {self.nn_model} neural network")
        output_dim = self.n_pca if self.data_trafo == 'PCA' else len(self.modes)
        
        self.cp_nn = CosmoPowerNN(
            parameters=self.model_parameters,
            modes=list(range(output_dim)),
            parameters_mean=X_mean_arr,
            parameters_std=X_std_arr,
            features_mean=self.y_mean,
            features_std=self.y_std,
            n_hidden=[512, 512, 512, 512],
            verbose=True,
            architecture_type=self.nn_model,
            loss_function=self.loss_function
        )
        
        # Prepare training data
        X_train = {key: (X[key] - self.X_mean[key]) / self.X_std[key] for key in X.keys()}
        
        
        # Train the model
        with tf.device(DEVICE):
            self.cp_nn.train(
                training_parameters=X_train,
                training_features=y_train,
                filename_saved_model=model_filename,
                validation_split=test_split,
                learning_rates=[10**(-2-i) for i in range(n_cycles)],
                batch_sizes=[batch_size] * n_cycles,
                gradient_accumulation_steps=[1] * n_cycles,
                patience_values=[100] * n_cycles,
                max_epochs=[1000] * n_cycles
            )
        
        # Save additional attributes
        self._save_attributes(model_filename)
        
        self.trained = True
        logger.info("Emulator training completed successfully")
            


    def _save_attributes(self, model_filename: str) -> None:
        """Save additional emulator attributes."""
        save_dict = { "data_transformation": {
            "data_trafo": self.data_trafo,
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "features_mean": self.features_mean,
            "features_std": self.features_std,
            "pca_transform_matrix": self.pca_transform_matrix}
        }

        npz_filename = model_filename + ".npz"
        
        # Check if file exists and is not empty
        if os.path.exists(npz_filename) and os.path.getsize(npz_filename) > 0:
            # Load existing data
            existing_data = np.load(npz_filename, allow_pickle=True)
            # Convert to regular dict and merge with new data
            existing_dict = {key: existing_data[key] for key in existing_data.keys()}
            existing_data.close()
            
            # Merge dictionaries (new data takes precedence for overlapping keys)
            merged_dict = {**existing_dict, **save_dict}
            np.savez_compressed(npz_filename, **merged_dict)
            logger.info(f"Emulator attributes merged and saved to {npz_filename}")
        else:
            np.savez_compressed(npz_filename, **save_dict)
            logger.info(f"Emulator attributes saved to {npz_filename}")


    def load(self, filename: str) -> None:
        """Load pre-trained emulator from file.
        
        Args:
            filename: Base filename to load from
        """

        logger.info(f"Loading pre-trained emulator from {filename}")
        
        # Load neural network
        self.cp_nn = CosmoPowerNN(restore=True, restore_filename=filename)
        
        # Load additional attributes
        npz_filename = filename + ".npz"
        with np.load(npz_filename, allow_pickle=True) as data:
            data_transformation = data["data_transformation"].item()
            self.data_trafo = data_transformation["data_trafo"]
            self.X_mean = data_transformation["X_mean"]
            self.X_std = data_transformation["X_std"]
            self.y_mean = data_transformation["y_mean"]
            self.y_std = data_transformation["y_std"]
            self.features_mean = data_transformation["features_mean"]
            self.features_std = data_transformation["features_std"]
            self.pca_transform_matrix = data_transformation["pca_transform_matrix"]
        
        self.trained = True
        logger.info("Emulator loaded successfully")
            


    def predict(self, X: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
        """Make predictions using the trained emulator.
        
        Args:
            X: Dictionary of input parameters
            
        Returns:
            Predicted output in original data space
        """
        if not self.trained:
            raise RuntimeError("Emulator must be trained before making predictions")
        

        # Normalize input parameters
        X_norm_dict = {}
        for key in X.keys():
            if key not in self.X_mean:
                raise KeyError(f"Parameter {key} not found in training data")
            
            # Handle scalar inputs - ensure consistent array format
            x_val = X[key]
            if np.isscalar(x_val):
                x_val = np.array([x_val])
            elif isinstance(x_val, (list, tuple)):
                x_val = np.array(x_val)
            
            # Normalize and format for neural network
            x_norm = (x_val - self.X_mean[key]) / self.X_std[key]
            X_norm_dict[key] = x_norm
        
        # Get predictions from neural network
        y_pred = self.cp_nn.predictions_np(X_norm_dict)
        
        # Denormalize predictions
        y_pred = y_pred * self.y_std + self.y_mean
        
        # Apply inverse transformation
        y_pred = self.backtransform(y_pred)
        
        return y_pred

    def compute_gradients(self, X: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
        """Compute gradients of predictions with respect to input parameters using autodifferentiation.
        
        This method uses TensorFlow's automatic differentiation to compute the Jacobian
        matrix: d(predictions)/d(parameters). The gradients are computed in the normalized
        space and then transformed back to the original parameter space.
        
        Args:
            X: Dictionary of input parameters (same format as predict method)
            
        Returns:
            Gradient array of shape (batch_size, n_modes, n_parameters)
            where gradient[i, j, k] = d(prediction[i, j])/d(parameter[i, k])
            in the original (denormalized) parameter space
        """
        if not self.trained:
            raise RuntimeError("Emulator must be trained before computing gradients")
        
        # Normalize input parameters
        X_norm_dict = {}
        X_norm_arr_list = []
        for key in self.model_parameters:
            if key not in X:
                raise KeyError(f"Parameter {key} not found in input dictionary")
            if key not in self.X_mean:
                raise KeyError(f"Parameter {key} not found in training data")
            
            # Handle scalar inputs - ensure consistent array format
            x_val = X[key]
            if np.isscalar(x_val):
                x_val = np.array([x_val])
            elif isinstance(x_val, (list, tuple)):
                x_val = np.array(x_val)
            
            # Normalize and format for neural network
            x_norm = (x_val - self.X_mean[key]) / self.X_std[key]
            X_norm_dict[key] = x_norm
            X_norm_arr_list.append(x_norm)
        
        # Convert to array format for TensorFlow
        # Stack parameters: shape (batch_size, n_parameters)
        X_orig_arr = np.stack([X[key] if not np.isscalar(X[key]) else np.array([X[key]]) 
                               for key in self.model_parameters], axis=1)
        if len(X_orig_arr.shape) == 1:
            X_orig_arr = X_orig_arr.reshape(1, -1)
        batch_size = X_orig_arr.shape[0]
        n_params = X_orig_arr.shape[1]
        # Convert to TensorFlow tensor (original parameter space)
        X_orig_tf = tf.convert_to_tensor(X_orig_arr, dtype=DTYPE)
        # Create TensorFlow constants for normalization
        X_mean_tf = tf.constant([self.X_mean[key] for key in self.model_parameters], dtype=DTYPE)
        X_std_tf = tf.constant([self.X_std[key] for key in self.model_parameters], dtype=DTYPE)
        y_mean_tf = tf.constant(self.y_mean, dtype=DTYPE)
        y_std_tf = tf.constant(self.y_std, dtype=DTYPE)
        # Define a function that takes original parameters and returns original predictions
        # This allows TensorFlow to automatically handle the chain rule
        # We manually implement the forward pass to avoid @tf.function decorator issues with gradients
        def predict_original_space(params_orig):
            """Full prediction pipeline in original space."""
            # Normalize inputs
            params_norm = (params_orig - X_mean_tf) / X_std_tf
            
            # Ensure params_norm has batch dimension for neural network
            # The NN expects shape (batch_size, n_params)
            if len(params_norm.shape) == 1:
                params_norm_batch = tf.expand_dims(params_norm, 0)  # (1, n_params)
            else:
                params_norm_batch = params_norm
            
            # Manually do forward pass through NN to avoid @tf.function gradient issues
            # This is equivalent to self.cp_nn.predictions_tf but without the decorator
            x = (params_norm_batch - self.cp_nn.parameters_mean) / self.cp_nn.parameters_std
            
            if self.cp_nn.architecture_type == "MLP":
                # MLP forward pass
                layers = [x]
                
                for i in range(self.cp_nn.n_layers - 1):
                    # Linear transformation
                    linear_out = tf.matmul(layers[-1], self.cp_nn.W[i]) + self.cp_nn.b[i]
                    # Apply custom activation
                    activated = self.cp_nn.activation(linear_out, self.cp_nn.alphas[i], self.cp_nn.betas[i])
                    layers.append(activated)
                
                # Output layer (linear)
                output = tf.matmul(layers[-1], self.cp_nn.W[-1]) + self.cp_nn.b[-1]
            else:
                raise NotImplementedError(f"Gradient computation not implemented for architecture: {self.cp_nn.architecture_type}")
            
            # Denormalize output (this gives pred_norm in normalized output space)
            pred_norm_batch = output * self.cp_nn.features_std + self.cp_nn.features_mean
            
            # Remove batch dimension if it was added
            if len(params_norm.shape) == 1 and len(pred_norm_batch.shape) == 2:
                pred_norm = pred_norm_batch[0]  # (n_modes,)
            else:
                pred_norm = pred_norm_batch
            
            # Denormalize outputs (e.g., transform from NN normalization to log10 space)
            pred_intermediate = pred_norm * y_std_tf + y_mean_tf
            
            # Apply backtransform (e.g., from log10 space to original space)
            if self.data_trafo == 'log_norm':
                pred_original = tf.pow(10.0, pred_intermediate)
            elif self.data_trafo == 'norm':
                pred_original = pred_intermediate
            elif self.data_trafo == 'PCA':
                # For PCA: pred_original = (pred_intermediate @ pca_transform_matrix) * features_std + features_mean
                pca_matrix_tf = tf.constant(self.pca_transform_matrix, dtype=DTYPE)
                features_std_tf = tf.constant(self.features_std, dtype=DTYPE)
                features_mean_tf = tf.constant(self.features_mean, dtype=DTYPE)
                pca_reconstructed = tf.matmul(pred_intermediate, pca_matrix_tf)
                pred_original = pca_reconstructed * features_std_tf + features_mean_tf
            else:
                pred_original = pred_intermediate
            
            return pred_original
        
        # Compute gradients directly in original space using TensorFlow's autodiff
        # This automatically handles all the chain rule transformations!
        # Process each sample separately to ensure proper gradient tracking
        all_gradients = []
        
        for batch_idx in range(batch_size):
            # Extract single sample (remove batch dimension for Variable)
            sample_params_orig_1d = X_orig_tf[batch_idx]  # Shape: (n_params,)
            # Use tf.Variable instead of tf.identity() for proper gradient tracking
            # tf.Variable is more reliable for gradient computation than watched tensors
            sample_params_var = tf.Variable(sample_params_orig_1d, trainable=True)
            
            with tf.GradientTape(persistent=True) as tape:
                # Compute predictions for this single sample
                sample_predictions = predict_original_space(sample_params_var)
                # Ensure we have the right shape: should be (n_modes,)
                if len(sample_predictions.shape) > 1:
                    sample_predictions_flat = sample_predictions[0]
                else:
                    sample_predictions_flat = sample_predictions
            
            # Compute full Jacobian for this sample: d(pred_original)/d(param_original)
            n_modes = sample_predictions_flat.shape[0]
            try:
                # Use jacobian to compute all gradients at once
                jacobian = tape.jacobian(sample_predictions_flat, sample_params_var)
                # jacobian shape: (n_modes, n_params) - no batch dimension since we used 1D Variable
                if len(jacobian.shape) == 3:
                    # If somehow we got 3D, squeeze middle dimension
                    jacobian = tf.squeeze(jacobian, axis=1)
                all_gradients.append(jacobian)
            except (AttributeError, ValueError, tf.errors.InvalidArgumentError) as e:
                # Fallback: compute gradients element by element
                sample_gradients = []
                for mode_idx in range(n_modes):
                    grad = tape.gradient(sample_predictions_flat[mode_idx], sample_params_var)
                    if grad is not None:
                        # grad should already be (n_params,) since sample_params_var is 1D
                        if len(grad.shape) > 1:
                            grad = tf.reshape(grad, [n_params])
                        sample_gradients.append(grad)
                    else:
                        sample_gradients.append(tf.zeros([n_params], dtype=DTYPE))
                all_gradients.append(tf.stack(sample_gradients, axis=0))
            
            del tape
        
        # Stack all samples: (batch_size, n_modes, n_params)
        gradients_original = tf.stack(all_gradients, axis=0).numpy()
        
        return gradients_original


    def summary(self) -> None:
        """Print emulator summary."""
        print("\\n" + "="*60)
        print(" NNEmulator Summary")
        print("="*60)
        print(f"Parameters: {len(self.model_parameters)}")
        print(f"Parameter names: {self.model_parameters}")
        print(f"Output modes: {len(self.modes)}")
        print(f"Architecture: {self.nn_model}")
        print(f"Data transformation: {self.data_trafo}")
        if self.data_trafo == 'PCA':
            print(f"PCA components: {self.n_pca}")
        print(f"Trained: {self.trained}")
        
        if self.trained and hasattr(self, 'cp_nn'):
            print("\\nUnderlying Neural Network:")
            self.cp_nn.summary()
        
        print("="*60 + "\\n")
