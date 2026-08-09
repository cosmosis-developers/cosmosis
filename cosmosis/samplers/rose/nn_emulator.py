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
import scipy
import scipy.stats

from .amplitude_prefactor import BlockAmplitudePrefactor, parse_amplitude_prefactor
from .bin_embedding import build_bin_pair_spec
from .shared_trunk import build_shared_trunk_spec
from .vector_blocks import iter_bin_pairs
from .utils import SIGNED_LOG_NORM_TRANSFORM_SCALE
# Configure logging
logger = logging.getLogger(__name__)

# Architectures whose weights are one flat dense stack (params -> outputs).
DENSE_ARCHITECTURES = ("MLP", "EmbMLP")
# Architectures with a cosmology trunk shared across probe-specific heads.
SHARED_TRUNK_ARCHITECTURES = ("SharedTrunkMLP", "SharedTrunkEmbMLP")

# Set TensorFlow data type
DTYPE = tf.float32

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)

# Value for the signed log norm transform
# optimised for 3x2pt with C_ell ~1e-11-1e-9
s = SIGNED_LOG_NORM_TRANSFORM_SCALE

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
            print("GPU detected and verified - using GPU acceleration")
            return 'gpu:0'
    except Exception as e:
        print(f"GPU detection failed: {e}")
    
    print("Using CPU for computations")
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
        architecture_type: 'MLP', 'EmbMLP', 'SharedTrunkMLP' or 'SharedTrunkEmbMLP'
        shared_trunk_spec: Probe partition / parameter routing (shared-trunk only)
        head_n_hidden: Per-probe head widths (shared-trunk only); defaults to
            the last two trunk widths
        trunk_skip_connection: Feed the raw cosmological parameters to each head
            alongside the trunk latent (shared-trunk only)
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
                 loss_data_feat: Optional[np.ndarray] = None,
                 loss_inv_feat: Optional[np.ndarray] = None,
                 bin_pair_spec: Optional[Dict[str, Any]] = None,
                 embedding_dim: int = 4,
                 shared_trunk_spec: Optional[Dict[str, Any]] = None,
                 head_n_hidden: Optional[List[int]] = None,
                 trunk_skip_connection: bool = True,
                 **kwargs):
        
        super(CosmoPowerNN, self).__init__(**kwargs)
        
        self.architecture_type = architecture_type
        # EmbMLP only; restore() overwrites these from the saved model.
        self.bin_pair_spec = bin_pair_spec
        self.embedding_dim = int(embedding_dim)
        # SharedTrunkMLP only; restore() overwrites these from the saved model.
        self.shared_trunk_spec = shared_trunk_spec
        self.head_n_hidden = list(head_n_hidden) if head_n_hidden else []
        self.trunk_skip_connection = bool(trunk_skip_connection)
        self.verbose = verbose
        self.loss_data_feat = None
        self.loss_inv_feat = None
        self.n_loss_modes = None
        if loss_function == "standard":
            self.compute_loss = self.compute_loss_standard
        elif loss_function == "weighted_w_cov":
            # Match linna.util.Auxilleryfunc / Loss_fn (github.com/chto/linna):
            # compare NN vs M in *feature* space with C^{-1} transformed to that
            # space; floor χ²(M,d) at 0.5 * n_modes.
            if loss_data_feat is None or loss_inv_feat is None:
                raise ValueError(
                    "loss_function='weighted_w_cov' requires loss_data_feat and "
                    "loss_inv_feat (feature-space data and C^{-1}, as in LINNA)"
                )
            d_feat = np.atleast_1d(np.asarray(loss_data_feat, dtype=np.float64))
            ic_feat = np.atleast_2d(np.asarray(loss_inv_feat, dtype=np.float64))
            if ic_feat.shape != (d_feat.shape[0], d_feat.shape[0]):
                raise ValueError(
                    "loss_data_feat / loss_inv_feat shape mismatch: "
                    f"d={d_feat.shape}, inv={ic_feat.shape}"
                )
            self.n_loss_modes = int(d_feat.shape[0])
            self.loss_data_feat = tf.constant(d_feat, dtype=DTYPE)
            self.loss_inv_feat = tf.constant(ic_feat, dtype=DTYPE)
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
        if restore:
            self.restore_emulator_parameters()
        
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
        elif self.architecture_type == "EmbMLP":
            self._build_embmlp_network(trainable)
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            self._build_shared_trunk_network(trainable)
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture_type}")

    def _build_embmlp_network(self, trainable: bool) -> None:
        """Build the bin-pair-conditioned MLP.

        The trunk maps ``[params | probe emb | slot1 emb | slot2 emb]`` to one
        bin pair's ell nodes, so its output width is the shared ell grid rather
        than the full theory vector. Pair predictions are gathered back into
        flat-vector order in :meth:`_embmlp_core`.
        """
        if not self.bin_pair_spec:
            raise ValueError(
                "architecture_type='EmbMLP' requires bin_pair_spec "
                "(build it with bin_embedding.build_bin_pair_spec)"
            )
        spec = self.bin_pair_spec
        if int(spec["n_modes"]) != int(self.n_modes):
            raise ValueError(
                f"EmbMLP: bin_pair_spec covers {spec['n_modes']} modes but the "
                f"network declares {self.n_modes}"
            )

        self.n_pairs = int(spec["n_pairs"])
        self.n_ell_grid = int(spec["n_ell_grid"])
        self.slot_rows = tf.constant(
            np.asarray(spec["slot_rows"], dtype=np.int64), dtype=tf.int64
        )
        self.gather_index = tf.constant(
            np.asarray(spec["gather_index"], dtype=np.int64), dtype=tf.int64
        )
        self.n_slots = int(self.slot_rows.shape[1])

        trunk_input = self.n_parameters + self.n_slots * self.embedding_dim
        self.architecture = [trunk_input] + self.n_hidden + [self.n_ell_grid]
        self.n_layers = len(self.architecture) - 1

        self.E = tf.Variable(
            tf.random.normal([int(spec["table_size"]), self.embedding_dim], 0., 1e-1),
            name="bin_embedding",
            trainable=trainable,
        )
        self._create_dense_variables(trainable)

    def _build_mlp_network(self, trainable: bool) -> None:
        """Build MLP architecture with custom activation functions."""
        # Architecture definition
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_modes]
        self.n_layers = len(self.architecture) - 1
        self._create_dense_variables(trainable)

    @property
    def _uses_bin_embedding(self) -> bool:
        """True when each head predicts one bin pair's ell nodes at a time."""
        return self.architecture_type == "SharedTrunkEmbMLP"

    def _build_shared_trunk_network(self, trainable: bool) -> None:
        """Build one cosmology trunk plus one head per probe.

        The trunk maps the cosmological parameters to a latent ``z`` that every
        probe reuses; each head maps ``[z | cosmo params | that probe's nuisance
        params]`` to that probe's modes. Head outputs are concatenated and
        gathered back into likelihood order in :meth:`_shared_trunk_core_tf`.

        For ``SharedTrunkEmbMLP`` the head input also carries two bin-embedding
        rows and the head emits a single bin pair's ell nodes, so every pair of
        a probe shares that head's weights.

        Unlike :meth:`_create_dense_variables`, the trunk activates *every*
        layer: its output is a hidden representation, not a prediction.
        """
        if not self.shared_trunk_spec:
            raise ValueError(
                f"architecture_type='{self.architecture_type}' requires "
                "shared_trunk_spec (build it with "
                "shared_trunk.build_shared_trunk_spec)"
            )
        spec = self.shared_trunk_spec
        if int(spec["n_modes"]) != int(self.n_modes):
            raise ValueError(
                f"{self.architecture_type}: shared_trunk_spec covers "
                f"{spec['n_modes']} modes but the network declares "
                f"{self.n_modes}"
            )
        if self._uses_bin_embedding and "per_probe_pairs" not in spec:
            raise ValueError(
                "architecture_type='SharedTrunkEmbMLP' requires a "
                "shared_trunk_spec built with bin_embedding=True"
            )

        self.probe_names = [str(p) for p in spec["probes"]]
        self.trunk_param_index = tf.constant(
            np.asarray(spec["trunk_param_index"], dtype=np.int64), dtype=tf.int64
        )
        self.head_param_index = [
            tf.constant(np.asarray(idx, dtype=np.int64), dtype=tf.int64)
            for idx in spec["head_param_index"]
        ]
        self.mode_gather_index = tf.constant(
            np.asarray(spec["gather_index"], dtype=np.int64), dtype=tf.int64
        )

        if not self.head_n_hidden:
            # Heads default to the last two trunk widths (or the only one).
            self.head_n_hidden = list(self.n_hidden[-2:])

        n_trunk_in = int(np.asarray(spec["trunk_param_index"]).size)
        self.trunk_architecture = [n_trunk_in] + list(self.n_hidden)
        latent = self.trunk_architecture[-1]
        # The skip connection lets a head see the raw cosmology as well as the
        # latent, so the trunk width does not cap cosmological sensitivity.
        head_shared_in = latent + (n_trunk_in if self.trunk_skip_connection else 0)

        self.trunk_W, self.trunk_b, self.trunk_alphas, self.trunk_betas = (
            self._dense_block_variables(
                self.trunk_architecture, trainable, "trunk", activate_last=True
            )
        )

        if self._uses_bin_embedding:
            pairs = spec["per_probe_pairs"]
            self.head_slot_rows = [
                tf.constant(np.asarray(p["slot_rows"], dtype=np.int64), dtype=tf.int64)
                for p in pairs
            ]
            self.head_n_pairs = [int(p["n_pairs"]) for p in pairs]
            self.head_n_ell_grid = [int(p["n_ell_grid"]) for p in pairs]
            self.n_slots = int(np.asarray(pairs[0]["slot_rows"]).shape[1])
            self.E = tf.Variable(
                tf.random.normal(
                    [int(spec["table_size"]), self.embedding_dim], 0., 1e-1
                ),
                name="bin_embedding",
                trainable=trainable,
            )

        self.head_architectures = []
        self.head_W, self.head_b, self.head_alphas, self.head_betas = [], [], [], []
        for i, probe in enumerate(self.probe_names):
            n_head_params = int(np.asarray(spec["head_param_index"][i]).size)
            if self._uses_bin_embedding:
                head_in = (
                    head_shared_in
                    + n_head_params
                    + self.n_slots * self.embedding_dim
                )
                head_out = self.head_n_ell_grid[i]
            else:
                head_in = head_shared_in + n_head_params
                head_out = int(np.asarray(spec["probe_mode_index"][i]).size)
            widths = [head_in] + list(self.head_n_hidden) + [head_out]
            self.head_architectures.append(widths)
            W, b, alphas, betas = self._dense_block_variables(
                widths, trainable, f"head_{i}", activate_last=False
            )
            self.head_W.append(W)
            self.head_b.append(b)
            self.head_alphas.append(alphas)
            self.head_betas.append(betas)

        logger.info(
            "%s: trunk %s shared by %d head(s) %s",
            self.architecture_type,
            self.trunk_architecture,
            len(self.probe_names),
            {p: w for p, w in zip(self.probe_names, self.head_architectures)},
        )

    def _dense_block_variables(
        self,
        widths: List[int],
        trainable: bool,
        prefix: str,
        activate_last: bool = False,
    ) -> Tuple[List[tf.Variable], List[tf.Variable], List[tf.Variable], List[tf.Variable]]:
        """Allocate weights/biases/activation parameters for one dense block."""
        W, b, alphas, betas = [], [], [], []
        n_layers = len(widths) - 1
        for i in range(n_layers):
            W.append(tf.Variable(
                tf.random.normal([widths[i], widths[i + 1]], 0., 1e-3),
                name=f"{prefix}_W_{i}",
                trainable=trainable,
            ))
            b.append(tf.Variable(
                tf.zeros([widths[i + 1]]),
                name=f"{prefix}_b_{i}",
                trainable=trainable,
            ))
        for i in range(n_layers if activate_last else n_layers - 1):
            alphas.append(tf.Variable(
                tf.random.normal([widths[i + 1]]),
                name=f"{prefix}_alphas_{i}",
                trainable=trainable,
            ))
            betas.append(tf.Variable(
                tf.random.normal([widths[i + 1]]),
                name=f"{prefix}_betas_{i}",
                trainable=trainable,
            ))
        return W, b, alphas, betas

    def _create_dense_variables(self, trainable: bool) -> None:
        """Allocate dense weights/biases and activation parameters."""
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

    def _dense_stack_tf(self, h: tf.Tensor) -> tf.Tensor:
        """Run the shared dense trunk (hidden layers + linear output)."""
        for i in range(self.n_layers - 1):
            linear_out = tf.matmul(h, self.W[i]) + self.b[i]
            h = self.activation(linear_out, self.alphas[i], self.betas[i])
        return tf.matmul(h, self.W[-1]) + self.b[-1]

    def _embmlp_core_tf(self, x: tf.Tensor) -> tf.Tensor:
        """Evaluate every bin pair, then gather back into flat-vector order.

        Pairs are stacked along the batch axis, so all pairs share the trunk
        weights and differ only through their embedding rows.
        """
        batch = tf.shape(x)[0]
        # (n_pairs, n_slots * embedding_dim): fixed for a trained model.
        emb = tf.reshape(
            tf.gather(self.E, self.slot_rows),
            [self.n_pairs, self.n_slots * self.embedding_dim],
        )
        x_pairs = tf.tile(tf.expand_dims(x, 1), [1, self.n_pairs, 1])
        emb_pairs = tf.tile(tf.expand_dims(emb, 0), [batch, 1, 1])
        h = tf.reshape(
            tf.concat([x_pairs, emb_pairs], axis=-1),
            [-1, self.architecture[0]],
        )
        out = self._dense_stack_tf(h)
        out = tf.reshape(out, [batch, self.n_pairs * self.n_ell_grid])
        return tf.gather(out, self.gather_index, axis=1)

    def _run_dense_block_tf(
        self,
        h: tf.Tensor,
        W: List[tf.Variable],
        b: List[tf.Variable],
        alphas: List[tf.Variable],
        betas: List[tf.Variable],
    ) -> tf.Tensor:
        """Run one dense block; layers past ``len(alphas)`` stay linear."""
        n_activated = len(alphas)
        for i in range(len(W)):
            linear_out = tf.matmul(h, W[i]) + b[i]
            if i < n_activated:
                h = self.activation(linear_out, alphas[i], betas[i])
            else:
                h = linear_out
        return h

    def _shared_trunk_core_tf(self, x: tf.Tensor) -> tf.Tensor:
        """Run the shared cosmology trunk, then each probe head.

        Heads write disjoint slices of the theory vector, so their outputs are
        concatenated and gathered back into likelihood order. For
        ``SharedTrunkEmbMLP`` a head's bin pairs are stacked along the batch
        axis, so they all share that head's weights and differ only through
        their embedding rows.
        """
        trunk_in = tf.gather(x, self.trunk_param_index, axis=1)
        z = self._run_dense_block_tf(
            trunk_in, self.trunk_W, self.trunk_b, self.trunk_alphas, self.trunk_betas
        )
        if self.trunk_skip_connection:
            z = tf.concat([z, trunk_in], axis=1)

        batch = tf.shape(x)[0]
        outputs = []
        for i in range(len(self.probe_names)):
            head_in = z
            if int(self.head_param_index[i].shape[0]) > 0:
                head_in = tf.concat(
                    [z, tf.gather(x, self.head_param_index[i], axis=1)], axis=1
                )
            if self._uses_bin_embedding:
                n_pairs = self.head_n_pairs[i]
                n_ell = self.head_n_ell_grid[i]
                # (n_pairs, n_slots * embedding_dim): fixed for a trained model.
                emb = tf.reshape(
                    tf.gather(self.E, self.head_slot_rows[i]),
                    [n_pairs, self.n_slots * self.embedding_dim],
                )
                ctx_pairs = tf.tile(tf.expand_dims(head_in, 1), [1, n_pairs, 1])
                emb_pairs = tf.tile(tf.expand_dims(emb, 0), [batch, 1, 1])
                head_in = tf.reshape(
                    tf.concat([ctx_pairs, emb_pairs], axis=-1),
                    [-1, self.head_architectures[i][0]],
                )
            out = self._run_dense_block_tf(
                head_in,
                self.head_W[i],
                self.head_b[i],
                self.head_alphas[i],
                self.head_betas[i],
            )
            if self._uses_bin_embedding:
                out = tf.reshape(out, [batch, n_pairs * n_ell])
            outputs.append(out)
        out = tf.concat(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return tf.gather(out, self.mode_gather_index, axis=1)

    def predictions_normalized_tf(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass from already-normalized parameters to features.

        Split out from :meth:`predictions_tf` so the NUTS autodiff path can
        share one implementation per architecture.
        """
        if self.architecture_type == "MLP":
            output = self._dense_stack_tf(x)
        elif self.architecture_type == "EmbMLP":
            output = self._embmlp_core_tf(x)
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            output = self._shared_trunk_core_tf(x)
        else:
            raise NotImplementedError(f"Prediction not implemented for architecture: {self.architecture_type}")

        # Denormalize output
        return output * self.features_std + self.features_mean

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
        return self.predictions_normalized_tf(x)

    def forward_pass_np(self, parameters_arr: np.ndarray) -> np.ndarray:
        """NumPy forward pass for CPU predictions.
        
        This method provides a NumPy-only forward pass for cases where
        TensorFlow operations are not needed or desired.
        
        Args:
            parameters_arr: Input parameters array
            
        Returns:
            Predicted features array
        """
        # Normalize inputs
        x = (parameters_arr - self.parameters_mean_) / self.parameters_std_

        if self.architecture_type == "MLP":
            output = self._dense_stack_np(x)
        elif self.architecture_type == "EmbMLP":
            output = self._embmlp_core_np(x)
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            output = self._shared_trunk_core_np(x)
        else:
            raise NotImplementedError(
                f"NumPy forward pass not implemented for architecture: "
                f"{self.architecture_type}"
            )

        # Denormalize and return
        return output * self.features_std_ + self.features_mean_

    def _dense_stack_np(self, h: np.ndarray) -> np.ndarray:
        """NumPy equivalent of :meth:`_dense_stack_tf`."""
        for i in range(self.n_layers - 1):
            linear_out = np.dot(h, self.W_[i]) + self.b_[i]
            sigmoid_part = 1.0 / (1.0 + np.exp(-self.alphas_[i] * linear_out))
            h = (self.betas_[i] + (1.0 - self.betas_[i]) * sigmoid_part) * linear_out

        return np.dot(h, self.W_[-1]) + self.b_[-1]

    def _run_dense_block_np(
        self,
        h: np.ndarray,
        W: List[np.ndarray],
        b: List[np.ndarray],
        alphas: List[np.ndarray],
        betas: List[np.ndarray],
    ) -> np.ndarray:
        """NumPy equivalent of :meth:`_run_dense_block_tf`."""
        n_activated = len(alphas)
        for i in range(len(W)):
            linear_out = np.dot(h, W[i]) + b[i]
            if i < n_activated:
                sigmoid_part = 1.0 / (1.0 + np.exp(-alphas[i] * linear_out))
                h = (betas[i] + (1.0 - betas[i]) * sigmoid_part) * linear_out
            else:
                h = linear_out
        return h

    def _shared_trunk_core_np(self, x: np.ndarray) -> np.ndarray:
        """NumPy equivalent of :meth:`_shared_trunk_core_tf`."""
        spec = self.shared_trunk_spec
        trunk_idx = np.asarray(spec["trunk_param_index"], dtype=int)
        gather_index = np.asarray(spec["gather_index"], dtype=int)

        trunk_in = x[:, trunk_idx]
        z = self._run_dense_block_np(
            trunk_in,
            self.trunk_W_,
            self.trunk_b_,
            self.trunk_alphas_,
            self.trunk_betas_,
        )
        if self.trunk_skip_connection:
            z = np.concatenate([z, trunk_in], axis=1)

        batch = x.shape[0]
        outputs = []
        for i in range(len(spec["probes"])):
            head_idx = np.asarray(spec["head_param_index"][i], dtype=int)
            head_in = z if head_idx.size == 0 else np.concatenate(
                [z, x[:, head_idx]], axis=1
            )
            if self._uses_bin_embedding:
                pair = spec["per_probe_pairs"][i]
                slot_rows = np.asarray(pair["slot_rows"], dtype=int)
                n_pairs = int(pair["n_pairs"])
                n_ell = int(pair["n_ell_grid"])
                emb = self.E_[slot_rows].reshape(n_pairs, -1)
                ctx_pairs = np.repeat(head_in[:, None, :], n_pairs, axis=1)
                emb_pairs = np.broadcast_to(emb, (batch, *emb.shape))
                head_in = np.concatenate(
                    [ctx_pairs, emb_pairs], axis=-1
                ).reshape(batch * n_pairs, -1)
            out = self._run_dense_block_np(
                head_in,
                self.head_W_[i],
                self.head_b_[i],
                self.head_alphas_[i],
                self.head_betas_[i],
            )
            if self._uses_bin_embedding:
                out = out.reshape(batch, n_pairs * n_ell)
            outputs.append(out)
        out = np.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return out[:, gather_index]

    def _embmlp_core_np(self, x: np.ndarray) -> np.ndarray:
        """NumPy equivalent of :meth:`_embmlp_core_tf`."""
        spec = self.bin_pair_spec
        slot_rows = np.asarray(spec["slot_rows"], dtype=int)
        gather_index = np.asarray(spec["gather_index"], dtype=int)
        n_pairs = int(spec["n_pairs"])
        n_ell_grid = int(spec["n_ell_grid"])

        batch = x.shape[0]
        emb = self.E_[slot_rows].reshape(n_pairs, -1)
        x_pairs = np.repeat(x[:, None, :], n_pairs, axis=1)
        emb_pairs = np.broadcast_to(emb, (batch, *emb.shape))
        h = np.concatenate([x_pairs, emb_pairs], axis=-1).reshape(batch * n_pairs, -1)
        out = self._dense_stack_np(h).reshape(batch, n_pairs * n_ell_grid)
        return out[:, gather_index]

    def predictions_np(self, parameters_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Make predictions using NumPy forward pass.
        
        Args:
            parameters_dict: Dictionary mapping parameter names to values
            
        Returns:
            Predicted features array
        """
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)

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
        """Snapshot current TF Variables into NumPy arrays (W_, b_, …).

        Used as the early-stopping checkpoint: call when validation loss
        improves so :meth:`restore_emulator_parameters` can roll back later.
        """
        if self.architecture_type in DENSE_ARCHITECTURES:
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
            if self.architecture_type == "EmbMLP":
                self.emulator_parameters["E"] = self.E.numpy()
                self.E_ = self.E.numpy()
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            self.trunk_W_ = [w.numpy() for w in self.trunk_W]
            self.trunk_b_ = [b.numpy() for b in self.trunk_b]
            self.trunk_alphas_ = [a.numpy() for a in self.trunk_alphas]
            self.trunk_betas_ = [b.numpy() for b in self.trunk_betas]
            self.head_W_ = [[w.numpy() for w in head] for head in self.head_W]
            self.head_b_ = [[b.numpy() for b in head] for head in self.head_b]
            self.head_alphas_ = [[a.numpy() for a in head] for head in self.head_alphas]
            self.head_betas_ = [[b.numpy() for b in head] for head in self.head_betas]
            self.emulator_parameters = {
                "trunk_W": self.trunk_W_,
                "trunk_b": self.trunk_b_,
                "trunk_alphas": self.trunk_alphas_,
                "trunk_betas": self.trunk_betas_,
                "head_W": self.head_W_,
                "head_b": self.head_b_,
                "head_alphas": self.head_alphas_,
                "head_betas": self.head_betas_,
            }
            if self._uses_bin_embedding:
                self.E_ = self.E.numpy()
                self.emulator_parameters["E"] = self.E_
        else:
            raise NotImplementedError(f"Update emulator parameters not implemented for architecture: {self.architecture_type}")

    def restore_emulator_parameters(self) -> None:
        """Load the last ``update_emulator_parameters`` snapshot into TF Variables."""
        if self.architecture_type in DENSE_ARCHITECTURES:
            if not hasattr(self, "W_") or self.W_ is None:
                logger.warning("No emulator parameter snapshot to restore")
                return
            for i in range(len(self.W_)):
                self.W[i].assign(self.W_[i])
                self.b[i].assign(self.b_[i])
            for i in range(len(self.alphas_)):
                self.alphas[i].assign(self.alphas_[i])
                self.betas[i].assign(self.betas_[i])
            if self.architecture_type == "EmbMLP":
                self.E.assign(self.E_)
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            if not hasattr(self, "trunk_W_") or self.trunk_W_ is None:
                logger.warning("No emulator parameter snapshot to restore")
                return
            for i in range(len(self.trunk_W_)):
                self.trunk_W[i].assign(self.trunk_W_[i])
                self.trunk_b[i].assign(self.trunk_b_[i])
            for i in range(len(self.trunk_alphas_)):
                self.trunk_alphas[i].assign(self.trunk_alphas_[i])
                self.trunk_betas[i].assign(self.trunk_betas_[i])
            for h in range(len(self.head_W_)):
                for i in range(len(self.head_W_[h])):
                    self.head_W[h][i].assign(self.head_W_[h][i])
                    self.head_b[h][i].assign(self.head_b_[h][i])
                for i in range(len(self.head_alphas_[h])):
                    self.head_alphas[h][i].assign(self.head_alphas_[h][i])
                    self.head_betas[h][i].assign(self.head_betas_[h][i])
            if self._uses_bin_embedding:
                self.E.assign(self.E_)
        else:
            raise NotImplementedError(
                f"Restore not implemented for architecture: {self.architecture_type}"
            )

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
        """LINNA cov-weighted loss (To et al. 2022; ``linna.util.Auxilleryfunc``).

        Follows https://github.com/chto/linna exactly:

        - ``y_pred`` / ``y_target`` live in **NN feature space** (after
          ``y→y/σ`` and z-score / median–MAD), not physical space.
        - ``C^{-1}`` is the inverse of the covariance transformed into that
          same feature space.
        - ``Loss = mean[ χ²(M,NN) / χ²(M,d) ]`` with
          ``χ²(M,d) ← max(χ²(M,d), 0.5 * n_modes)``.

        Args:
            training_parameters: Parameter tensor (NNEmulator-normalized space)
            training_features: Target features already in feature space

        Returns:
            Batch-mean LINNA loss
        """
        # Network output is already in feature space (same as training_features).
        y_pred = self.predictions_tf(training_parameters)
        y_target = training_features
        # (M - NN) and (M - d) — same ordering as LINNA Auxilleryfunc.
        delta_m_nn = y_target - y_pred
        delta_m_d = y_target - self.loss_data_feat
        chisq_mnn = tf.einsum(
            "bi,ij,bj->b", delta_m_nn, self.loss_inv_feat, delta_m_nn
        )
        chisq_md = tf.einsum(
            "bi,ij,bj->b", delta_m_d, self.loss_inv_feat, delta_m_d
        )
        floor = tf.constant(0.5 * float(self.n_loss_modes), dtype=DTYPE)
        chisq_md = tf.maximum(chisq_md, floor)
        ratio = chisq_mnn / chisq_md
        # Cap per-sample ratio so a single outlier cannot blow up Adam steps
        # (common with lr=1e-2 on the LINNA χ²-ratio loss).
        ratio = tf.minimum(ratio, tf.constant(1.0e4, dtype=DTYPE))
        return tf.reduce_mean(ratio)

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


    def training_step(self, training_parameters: tf.Tensor, training_features: tf.Tensor) -> tf.Tensor:
        """Perform one training step.
        
        Args:
            training_parameters: Parameter tensor
            training_features: Target features tensor
            
        Returns:
            Training loss for this step
        """
        loss, gradients = self.compute_loss_and_gradients(training_parameters, training_features)
        # Clip grads: LINNA χ²-ratio loss can spike and destroy weights at high lr.
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
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
                    #elif best_val_loss < 5e-4:
                    #    best_val_loss = val_loss
                    #    patience_counter = 0
                    #    # Save best model
                    #    self.update_emulator_parameters()   
                    else:
                        patience_counter += 1
                    
                    # Log progress every 10 epochs
                    if epoch % 10 == 0 or patience_counter >= patience_values[stage]:
                        logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
                                  f"val_loss={val_loss:.6f}, best_val={best_val_loss:.6f}")

                    
                    # Early stopping
                    if patience_counter >= patience_values[stage]:
                        logger.info(f"Early stopping at epoch {epoch} (patience={patience_values[stage]})")
                        break
                
                # Roll back to best validation weights (critical if loss exploded).
                self.restore_emulator_parameters()
                restored_val = float(self.compute_loss(val_params, val_features).numpy())
                # Store stage diagnostics
                diagnostics[f'learning_cycle_{stage}'] = stage_diagnostics
                logger.info(
                    f"Stage {stage + 1} completed. Best validation loss: {best_val_loss:.6f} "
                    f"(restored weights → val_loss={restored_val:.6f})"
                )
        
        # Ensure TF Variables match the best snapshot before writing to disk.
        self.restore_emulator_parameters()
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
        
        if self.architecture_type in DENSE_ARCHITECTURES:
            # Use the extracted numpy weights (W_, b_, etc.) instead of TF variables
            save_dict["weights"] = {
                "W": [w.tolist() for w in self.W_],
                "b": [b.tolist() for b in self.b_],
                "alphas": [a.tolist() for a in self.alphas_],
                "betas": [b.tolist() for b in self.betas_]
            }
            if self.architecture_type == "EmbMLP":
                save_dict["weights"]["E"] = self.E_.tolist()
                save_dict["bin_pair_spec"] = self.bin_pair_spec
                save_dict["embedding_dim"] = self.embedding_dim
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            save_dict["weights"] = {
                "trunk_W": [w.tolist() for w in self.trunk_W_],
                "trunk_b": [b.tolist() for b in self.trunk_b_],
                "trunk_alphas": [a.tolist() for a in self.trunk_alphas_],
                "trunk_betas": [b.tolist() for b in self.trunk_betas_],
                "head_W": [[w.tolist() for w in head] for head in self.head_W_],
                "head_b": [[b.tolist() for b in head] for head in self.head_b_],
                "head_alphas": [
                    [a.tolist() for a in head] for head in self.head_alphas_
                ],
                "head_betas": [
                    [b.tolist() for b in head] for head in self.head_betas_
                ],
            }
            if self._uses_bin_embedding:
                save_dict["weights"]["E"] = self.E_.tolist()
                save_dict["embedding_dim"] = self.embedding_dim
            save_dict["shared_trunk_spec"] = self._shared_trunk_spec_state()
            save_dict["head_n_hidden"] = list(self.head_n_hidden)
            save_dict["trunk_skip_connection"] = self.trunk_skip_connection
        else:
            raise NotImplementedError(f"Save not implemented for architecture: {self.architecture_type}")

        # Save main model file
        np.savez_compressed(filename + ".npz", **save_dict)
        logger.info(f"Model saved to {filename}.npz")

    def _shared_trunk_spec_state(self) -> Dict[str, Any]:
        """Plain-container copy of the spec for ``.npz`` storage.

        Keras wraps list/dict attributes of a Model, and those wrappers would
        otherwise be pickled into the saved file.
        """
        spec = self.shared_trunk_spec
        state: Dict[str, Any] = {
            "probes": [str(p) for p in spec["probes"]],
            "parameters": [str(p) for p in spec["parameters"]],
            "probe_mode_index": [
                np.asarray(idx, dtype=np.int64) for idx in spec["probe_mode_index"]
            ],
            "gather_index": np.asarray(spec["gather_index"], dtype=np.int64),
            "n_modes": int(spec["n_modes"]),
            "trunk_parameters": [str(p) for p in spec["trunk_parameters"]],
            "trunk_param_index": np.asarray(
                spec["trunk_param_index"], dtype=np.int64
            ),
            "head_parameters": [
                [str(p) for p in head] for head in spec["head_parameters"]
            ],
            "head_param_index": [
                np.asarray(idx, dtype=np.int64) for idx in spec["head_param_index"]
            ],
        }
        if "per_probe_pairs" in spec:
            state["table_size"] = int(spec["table_size"])
            state["source_bins"] = [int(b) for b in spec["source_bins"]]
            state["lens_bins"] = [int(b) for b in spec["lens_bins"]]
            state["per_probe_pairs"] = [
                {
                    "name": str(p["name"]),
                    "slot_rows": np.asarray(p["slot_rows"], dtype=np.int64),
                    "n_pairs": int(p["n_pairs"]),
                    "n_ell_grid": int(p["n_ell_grid"]),
                    "output_offset": int(p["output_offset"]),
                    "pairs": [list(t) for t in p["pairs"]],
                }
                for p in spec["per_probe_pairs"]
            ]
        return state

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
        
        # Restore weights for the dense-trunk architectures
        if self.architecture_type in DENSE_ARCHITECTURES and "weights" in data:
            weights = data["weights"].item()
            self.W_ = [np.array(w) for w in weights["W"]]
            self.b_ = [np.array(b) for b in weights["b"]]
            self.alphas_ = [np.array(a) for a in weights["alphas"]]
            self.betas_ = [np.array(b) for b in weights["betas"]]
            if self.architecture_type == "EmbMLP":
                from .bin_embedding import spec_from_state

                self.E_ = np.array(weights["E"])
                self.bin_pair_spec = spec_from_state(data["bin_pair_spec"])
                self.embedding_dim = int(data["embedding_dim"])
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES and "weights" in data:
            from .shared_trunk import spec_from_state as shared_trunk_spec_from_state

            weights = data["weights"].item()
            self.trunk_W_ = [np.array(w) for w in weights["trunk_W"]]
            self.trunk_b_ = [np.array(b) for b in weights["trunk_b"]]
            self.trunk_alphas_ = [np.array(a) for a in weights["trunk_alphas"]]
            self.trunk_betas_ = [np.array(b) for b in weights["trunk_betas"]]
            self.head_W_ = [[np.array(w) for w in head] for head in weights["head_W"]]
            self.head_b_ = [[np.array(b) for b in head] for head in weights["head_b"]]
            self.head_alphas_ = [
                [np.array(a) for a in head] for head in weights["head_alphas"]
            ]
            self.head_betas_ = [
                [np.array(b) for b in head] for head in weights["head_betas"]
            ]
            self.shared_trunk_spec = shared_trunk_spec_from_state(
                data["shared_trunk_spec"]
            )
            self.head_n_hidden = [int(n) for n in np.atleast_1d(data["head_n_hidden"])]
            self.trunk_skip_connection = bool(data["trunk_skip_connection"])
            if self.architecture_type == "SharedTrunkEmbMLP":
                self.E_ = np.array(weights["E"])
                self.embedding_dim = int(data["embedding_dim"])
        else:
            raise ValueError(f"Unknown architecture type: {self.architecture_type}")
        
        logger.info(f"Model restored from {filename} with architecture: {self.architecture_type}")
            

    def summary(self) -> None:
        """Print detailed model summary."""
        print("\\n" + "="*60)
        print(f" CosmoPower Model Summary ({self.architecture_type})")
        print("="*60)
        
        if self.architecture_type in DENSE_ARCHITECTURES:
            self._print_mlp_summary()
        elif self.architecture_type in SHARED_TRUNK_ARCHITECTURES:
            self._print_shared_trunk_summary()
        else:
            raise NotImplementedError(f"Summary not implemented for architecture: {self.architecture_type}")
        
        print("="*60 + "\\n")

    def _print_shared_trunk_summary(self) -> None:
        """Print trunk / per-head parameter counts."""
        def block_params(widths: List[int], activate_last: bool) -> int:
            n_layers = len(widths) - 1
            total = sum(
                widths[i] * widths[i + 1] + widths[i + 1] for i in range(n_layers)
            )
            n_activated = n_layers if activate_last else n_layers - 1
            total += sum(2 * widths[i + 1] for i in range(n_activated))
            return total

        print(f"{'Block':<25} {'Shape':<28} {'Param #':<10}")
        print("-" * 65)

        total_embedding = 0
        if self._uses_bin_embedding:
            table_size = int(self.shared_trunk_spec["table_size"])
            total_embedding = table_size * self.embedding_dim
            print(
                f"{'BinEmbedding (shared)':<25} "
                f"{str([table_size, self.embedding_dim]):<28} "
                f"{total_embedding:>10}"
            )

        trunk_params = block_params(self.trunk_architecture, activate_last=True)
        print(
            f"{'Trunk (shared)':<25} {str(list(self.trunk_architecture)):<28} "
            f"{trunk_params:>10}"
        )
        total = trunk_params + total_embedding
        for i, (probe, widths) in enumerate(
            zip(self.probe_names, self.head_architectures)
        ):
            head_params = block_params(widths, activate_last=False)
            total += head_params
            label = "Head " + probe
            if self._uses_bin_embedding:
                pair = self.shared_trunk_spec["per_probe_pairs"][i]
                label += f" (x{int(pair['n_pairs'])} pairs)"
            print(f"{label:<25} {str(list(widths)):<28} {head_params:>10}")

        print("-" * 65)
        print(f"Trunk parameters: {list(self.shared_trunk_spec['trunk_parameters'])}")
        for probe, head in zip(
            self.probe_names, self.shared_trunk_spec["head_parameters"]
        ):
            print(f"Head parameters [{probe}]: {list(head) if head else '(none)'}")
        print(f"Skip connection into heads: {self.trunk_skip_connection}")
        print(f"Total params: {total:,}")

    def _print_mlp_summary(self) -> None:
        """Print MLP-specific summary."""
        total_params = 0
        
        print(f"{'Layer (type)':<25} {'Output Shape':<15} {'Param #':<10}")
        print("-" * 50)
        
        if self.architecture_type == "EmbMLP":
            emb_params = int(self.bin_pair_spec["table_size"]) * self.embedding_dim
            total_params += emb_params
            print(
                f"{'BinEmbedding':<25} "
                f"({int(self.bin_pair_spec['table_size']):>3},{self.embedding_dim:>3})"
                f"{'':<4} {emb_params:>10}"
            )

        input_dim = self.architecture[0]
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
        nn_model: Neural network architecture ('MLP', 'EmbMLP', 'SharedTrunkMLP'
            or 'SharedTrunkEmbMLP')
        iteration: Current training iteration (for naming)
        data_transformation: Data transformation type
            ('log_norm', 'norm', 'signed_log_norm', 'weighted_norm',
             'weighted_median_norm', 'PCA', 'PCA_per_bin')
        n_pca: Number of PCA components (global PCA, or per bin-pair for PCA_per_bin)
        datavector: Reference data vector for weighted training / weighted transforms
        inv_cov: Inverse covariance matrix (used to build ``data_cov`` / ``cov_sigma``
            for weighted transforms, and for cov-weighted loss)
    """
    
    def __init__(self,
                 model_parameters: List[str],
                 modes: Union[List[str], np.ndarray],
                 nn_model: str = 'MLP',
                 loss_function: str = "standard",
                 iteration: int = 1,
                 data_transformation: str = 'log_norm',
                 n_pca: int = 64,
                 datavector: Optional[np.ndarray] = None,
                 inv_cov: Optional[np.ndarray] = None,
                 amplitude_prefactor: Union[bool, str] = False,
                 embedding_dim: int = 4,
                 head_n_hidden: Optional[List[int]] = None,
                 trunk_skip_connection: bool = True,
                 trunk_params: Optional[List[str]] = None):
        
        self.trained = False
        # When True, parameters passed to predict()/compute_gradients() that are
        # not part of this emulator's training data are silently ignored instead
        # of raising. Used when a pre-trained emulator is combined into a larger
        # pipeline that varies extra parameters it does not depend on.
        self.ignore_extra_params = False
        self.model_parameters = model_parameters
        self.modes = modes
        self.data_transformation = data_transformation
        self.datavector = datavector
        self.data_inv_cov = inv_cov
        # Covariance and per-mode σ = sqrt(diag(C)) for weighted_* transforms.
        self.data_cov = None
        self.cov_sigma = None
        if inv_cov is not None:
            inv = np.atleast_2d(np.asarray(inv_cov, dtype=float))
            self.data_cov = np.linalg.inv(inv)
            self.cov_sigma = np.maximum(np.sqrt(np.diag(self.data_cov)), 1e-30)
        self.n_pca = n_pca
        self.nn_model = nn_model
        self.embedding_dim = int(embedding_dim)
        self.head_n_hidden = list(head_n_hidden) if head_n_hidden else []
        self.trunk_skip_connection = bool(trunk_skip_connection)
        self.trunk_params = list(trunk_params) if trunk_params else None
        self.loss_function = loss_function
        self.iteration = iteration
        self.amplitude_prefactor_enabled = parse_amplitude_prefactor(amplitude_prefactor)
        self.amplitude_prefactor: Optional[BlockAmplitudePrefactor] = None
        # Metadata / probe list for PCA_per_bin (and name inference ≤3.19)
        self.vector_metadata: Optional[Dict[str, Any]] = None
        self.amplitude_prefactor_spectra: Optional[Any] = None
        
        # Initialize transformation attributes
        self.pca_transform_matrix = None
        self.pca_blocks: Optional[List[Dict[str, Any]]] = None
        self.y_mean = None
        self.y_std = None
        self.features_mean = None
        self.features_std = None
        self.X_mean = None
        self.X_std = None
        
        logger.info(
            f"NNEmulator initialized with {data_transformation} transformation"
            + (", amplitude_prefactor=T" if self.amplitude_prefactor_enabled else "")
        )

    def configure_amplitude_prefactor(
        self,
        metadata: Optional[Dict[str, Any]],
        spectra: Optional[Any] = None,
    ) -> None:
        """Build the block-wise 3x2pt amplitude prefactor from vector metadata."""
        if metadata is not None:
            self.vector_metadata = metadata
        if spectra is not None:
            self.amplitude_prefactor_spectra = spectra
        if not self.amplitude_prefactor_enabled:
            self.amplitude_prefactor = None
            return
        if not metadata:
            raise ValueError(
                "amplitude_prefactor=T but no fiducial vector metadata is available."
            )
        self.amplitude_prefactor = BlockAmplitudePrefactor.from_metadata(
            metadata, self.model_parameters, spectra=spectra
        )

    def configure_bin_pair_pca(
        self,
        metadata: Optional[Dict[str, Any]],
        spectra: Optional[Any] = None,
    ) -> None:
        """Attach vector metadata needed by ``PCA_per_bin`` transforms."""
        if not metadata:
            raise ValueError(
                "data_transformation=PCA_per_bin requires fiducial vector metadata "
                "(name/bin1/bin2, or angle+bin1+bin2 for name inference)."
            )
        self.vector_metadata = metadata
        if spectra is not None:
            self.amplitude_prefactor_spectra = spectra

    @staticmethod
    def _is_pca_family(data_transformation: str) -> bool:
        return data_transformation in ("PCA", "PCA_per_bin")

    @staticmethod
    def _is_weighted_norm_family(data_transformation: str) -> bool:
        return data_transformation in ("weighted_norm", "weighted_median_norm")

    def transform(self, model_datavector: np.ndarray) -> np.ndarray:
        """Transform data vector for neural network training.
        
        Args:
            model_datavector: Raw data vector to transform
            
        Returns:
            Transformed data vector ready for training
        """
        if self.data_transformation == 'log_norm':
            return self._log_norm_transform(model_datavector)
        elif self.data_transformation == 'signed_log_norm':
            return self._signed_log_norm_transform(model_datavector)
        elif self.data_transformation == 'norm':
            return self._norm_transform(model_datavector)
        elif self.data_transformation == 'weighted_norm':
            return self._weighted_norm_transform(model_datavector)
        elif self.data_transformation == 'weighted_median_norm':
            return self._weighted_median_norm_transform(model_datavector)
        elif self.data_transformation == 'PCA':
            return self._pca_transform(model_datavector)
        elif self.data_transformation == 'PCA_per_bin':
            return self._pca_per_bin_transform(model_datavector)
        else:
            raise ValueError(f"Unknown data transformation: {self.data_transformation}")

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

    def _signed_log_norm_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply signed log-normalization transformation."""
        # Handle potential zeros or negative values
        y = np.sign(data) * np.log10(np.abs(data)/s + 1.)
        
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

    def _weighted_norm_transform(self, data: np.ndarray) -> np.ndarray:
        """Rescale by data σ, then z-score (mean/std).

        Uses ``cov_sigma = sqrt(diag(C))`` with ``C = inv(inv_cov)``.
        """
        if self.cov_sigma is None:
            raise ValueError(
                "data_transformation='weighted_norm' requires inv_cov "
                "(to build per-mode data σ)"
            )
        weighted_data = data / self.cov_sigma
        self.y_mean = np.mean(weighted_data, axis=0)
        self.y_std = np.maximum(np.std(weighted_data, axis=0), 1e-10)
        return (weighted_data - self.y_mean) / self.y_std

    def _weighted_median_norm_transform(self, data: np.ndarray) -> np.ndarray:
        """Rescale by data σ, then robust median / MAD normalization.

        Uses ``cov_sigma = sqrt(diag(C))`` with ``C = inv(inv_cov)``.
        """
        if self.cov_sigma is None:
            raise ValueError(
                "data_transformation='weighted_median_norm' requires inv_cov "
                "(to build per-mode data σ)"
            )
        weighted_data = data / self.cov_sigma
        self.y_mean = np.median(weighted_data, axis=0)
        self.y_std = np.maximum(
            scipy.stats.median_abs_deviation(weighted_data, axis=0), 1e-10
        )
        return (weighted_data - self.y_mean) / self.y_std

    def _pca_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply global PCA transformation."""
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
        self.pca_blocks = None
        self.features_mean = y_mean
        self.features_std = y_std
        
        # Transform data
        pca_data = pca.transform(normalized_data)
        
        # Standardize PCA coefficients (CosmoPower PCAplusNN convention).
        # Equalizes loss weight across components; leading PCs otherwise dominate.
        self.y_mean = np.mean(pca_data, axis=0)
        self.y_std = np.std(pca_data, axis=0)
        self.y_std = np.maximum(self.y_std, 1e-10)
        
        return (pca_data - self.y_mean) / self.y_std

    def _pca_per_bin_transform(self, data: np.ndarray) -> np.ndarray:
        """PCA independently on each redshift-bin pair, then concatenate coeffs.

        For each contiguous ``(name, bin1, bin2)`` block: z-score modes →
        ``IncrementalPCA(n_components=min(n_pca, n_modes))`` → append coeffs.
        Optional amplitude prefactor is applied *before* this (in ``train``).
        """
        if self.vector_metadata is None:
            raise ValueError(
                "PCA_per_bin requires vector metadata; call "
                "configure_bin_pair_pca() or configure_amplitude_prefactor() first."
            )
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_samples, n_modes = data.shape
        blocks = iter_bin_pairs(
            self.vector_metadata, spectra=self.amplitude_prefactor_spectra
        )
        if not blocks:
            raise ValueError("PCA_per_bin: no bin-pair blocks found in metadata")

        pca_blocks: List[Dict[str, Any]] = []
        coeff_parts: List[np.ndarray] = []
        for block in blocks:
            idx = np.asarray(block["indices"], dtype=int)
            if idx.size == 0:
                continue
            y = data[:, idx]
            n_feat = y.shape[1]
            n_comp = min(int(self.n_pca), n_feat, n_samples)
            if n_comp < 1:
                continue
            feat_mean = np.mean(y, axis=0)
            feat_std = np.maximum(np.std(y, axis=0), 1e-10)
            normalized = (y - feat_mean) / feat_std
            pca = IncrementalPCA(n_components=n_comp)
            pca.fit(normalized)
            coeffs = pca.transform(normalized)
            pca_blocks.append(
                {
                    "name": block["name"],
                    "bin1": int(block["bin1"]),
                    "bin2": int(block["bin2"]),
                    "indices": idx,
                    "components": np.asarray(pca.components_, dtype=float),
                    "features_mean": feat_mean,
                    "features_std": feat_std,
                    "n_comp": int(n_comp),
                }
            )
            coeff_parts.append(coeffs)

        if not coeff_parts:
            raise ValueError("PCA_per_bin: no usable bin-pair blocks after clamping n_pca")

        pca_data = np.concatenate(coeff_parts, axis=1)
        self.pca_blocks = pca_blocks
        self.pca_transform_matrix = None
        # Full-vector placeholders unused by per-bin backtransform
        self.features_mean = np.zeros(n_modes)
        self.features_std = np.ones(n_modes)

        self.y_mean = np.mean(pca_data, axis=0)
        self.y_std = np.maximum(np.std(pca_data, axis=0), 1e-10)
        logger.info(
            "PCA_per_bin: %d bin-pairs, n_pca_per_bin<=%d → %d concatenated coeffs "
            "(from %d modes)",
            len(pca_blocks),
            int(self.n_pca),
            pca_data.shape[1],
            n_modes,
        )
        return (pca_data - self.y_mean) / self.y_std

    def _pca_per_bin_backtransform(self, model_datavector: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_pca_per_bin_transform` (PCA-coeff space → data)."""
        if not self.pca_blocks:
            raise RuntimeError("PCA_per_bin backtransform: pca_blocks missing")
        coeffs = np.asarray(model_datavector, dtype=float)
        single = coeffs.ndim == 1
        if single:
            coeffs = coeffs.reshape(1, -1)
        n_modes = int(len(self.modes)) if self.modes is not None else 0
        if self.vector_metadata and self.vector_metadata.get("size"):
            n_modes = int(self.vector_metadata["size"])
        if n_modes <= 0:
            n_modes = int(max(int(np.max(b["indices"])) for b in self.pca_blocks) + 1)
        out = np.zeros((coeffs.shape[0], n_modes), dtype=float)
        offset = 0
        for block in self.pca_blocks:
            n_comp = int(block["n_comp"])
            c = coeffs[:, offset:offset + n_comp]
            recon = np.dot(c, block["components"])
            out[:, block["indices"]] = (
                recon * block["features_std"] + block["features_mean"]
            )
            offset += n_comp
        if offset != coeffs.shape[1]:
            raise ValueError(
                f"PCA_per_bin backtransform: expected {offset} coeffs, got {coeffs.shape[1]}"
            )
        return out[0] if single else out

    @tf.autograph.experimental.do_not_convert
    def _linna_feature_space_loss_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``d`` and ``C^{-1}`` in NN feature space (LINNA ``Auxilleryfunc``).

        LINNA transforms physical ``y`` by ``y/σ`` then z-score / median–MAD, and
        applies the same Jacobians to ``C`` before inverting. The training loss
        then compares NN outputs to targets in that feature space
        (see ``linna.util.Auxilleryfunc``).
        """
        if self.datavector is None or self.data_inv_cov is None:
            raise ValueError(
                "weighted_w_cov requires datavector and inv_cov on NNEmulator"
            )
        if self.y_mean is None or self.y_std is None:
            raise ValueError(
                "weighted_w_cov requires y_mean/y_std from data_transformation"
            )
        d = np.atleast_1d(np.asarray(self.datavector, dtype=np.float64))
        ic = np.atleast_2d(np.asarray(self.data_inv_cov, dtype=np.float64))
        y_mean = np.atleast_1d(np.asarray(self.y_mean, dtype=np.float64))
        y_std = np.maximum(
            np.atleast_1d(np.asarray(self.y_std, dtype=np.float64)), 1e-30
        )
        n = d.shape[0]
        if y_mean.shape[0] != n or y_std.shape[0] != n or ic.shape != (n, n):
            raise ValueError(
                "datavector / inv_cov / y_mean / y_std shape mismatch for "
                f"weighted_w_cov: d={d.shape}, inv={ic.shape}, "
                f"y_mean={y_mean.shape}, y_std={y_std.shape}"
            )

        if self._is_weighted_norm_family(self.data_transformation):
            # Feature: ((y/σ) - μ) / s ; C_f^{-1} = diag(s σ) C^{-1} diag(s σ)
            if self.cov_sigma is None:
                raise ValueError(
                    f"{self.data_transformation} + weighted_w_cov requires cov_sigma"
                )
            sigma = np.maximum(
                np.atleast_1d(np.asarray(self.cov_sigma, dtype=np.float64)), 1e-30
            )
            if sigma.shape[0] != n:
                raise ValueError(
                    f"cov_sigma length {sigma.shape[0]} != datavector {n}"
                )
            d_feat = (d / sigma - y_mean) / y_std
            scale = y_std * sigma
        elif self.data_transformation == "norm":
            # Feature: (y - μ) / s ; C_f^{-1} = diag(s) C^{-1} diag(s)
            d_feat = (d - y_mean) / y_std
            scale = y_std
        else:
            raise ValueError(
                "loss_function='weighted_w_cov' currently supports "
                "data_transformation in "
                "{weighted_norm, weighted_median_norm, norm}; "
                f"got {self.data_transformation!r}. "
                "PCA / PCA_per_bin and amplitude_prefactor=T are not supported "
                "with weighted_w_cov."
            )

        inv_feat = scale[:, None] * ic * scale[None, :]
        return d_feat.astype(np.float64), inv_feat.astype(np.float64)

    def backtransform(self, model_datavector: np.ndarray) -> np.ndarray:
        """Transform predictions back to original space.
        
        Args:
            model_datavector: Transformed predictions
            
        Returns:
            Predictions in original data space
        """
        if self.data_transformation == 'log_norm':
            return 10 ** model_datavector
        elif self.data_transformation == 'signed_log_norm':
            return np.sign(model_datavector) * s * (10.0 ** np.abs(model_datavector) - 1.)
        elif self.data_transformation == 'norm':
            return model_datavector
        elif self._is_weighted_norm_family(self.data_transformation):
            if self.cov_sigma is None:
                raise RuntimeError(
                    f"{self.data_transformation} backtransform requires cov_sigma"
                )
            return model_datavector * self.cov_sigma
        elif self.data_transformation == 'PCA':
            # Reverse PCA transformation
            pca_reconstructed = np.dot(model_datavector, self.pca_transform_matrix)
            return pca_reconstructed * self.features_std + self.features_mean
        elif self.data_transformation == 'PCA_per_bin':
            return self._pca_per_bin_backtransform(model_datavector)
        else:
            raise ValueError(f"Unknown data transformation: {self.data_transformation}")

    def backtransform_tf(self, pred_intermediate: "tf.Tensor", dtype=None) -> "tf.Tensor":
        """TensorFlow inverse of ``data_transformation`` (for autodiff / NUTS)."""
        if dtype is None:
            dtype = DTYPE
        if self.data_transformation == 'log_norm':
            return tf.pow(10.0, pred_intermediate)
        if self.data_transformation == 'signed_log_norm':
            return tf.sign(pred_intermediate) * tf.multiply(
                s, tf.subtract(tf.pow(10.0, tf.abs(pred_intermediate)), 1.0)
            )
        if self.data_transformation == 'norm':
            return pred_intermediate
        if self._is_weighted_norm_family(self.data_transformation):
            if self.cov_sigma is None:
                raise RuntimeError(
                    f"{self.data_transformation} TF backtransform requires cov_sigma"
                )
            return pred_intermediate * tf.constant(self.cov_sigma, dtype=dtype)
        if self.data_transformation == 'PCA':
            pca_matrix_tf = tf.constant(self.pca_transform_matrix, dtype=dtype)
            features_std_tf = tf.constant(self.features_std, dtype=dtype)
            features_mean_tf = tf.constant(self.features_mean, dtype=dtype)
            # Support both (n_comp,) and (batch, n_comp)
            if len(pred_intermediate.shape) == 1:
                pca_reconstructed = tf.matmul(
                    tf.expand_dims(pred_intermediate, 0), pca_matrix_tf
                )[0]
            else:
                pca_reconstructed = tf.matmul(pred_intermediate, pca_matrix_tf)
            return pca_reconstructed * features_std_tf + features_mean_tf
        if self.data_transformation == 'PCA_per_bin':
            return self._pca_per_bin_backtransform_tf(pred_intermediate, dtype=dtype)
        return pred_intermediate

    def _pca_per_bin_backtransform_tf(
        self, pred_intermediate: "tf.Tensor", dtype=None
    ) -> "tf.Tensor":
        """TF inverse of per-bin-pair PCA.

        Bin-pair blocks are a contiguous partition of the flat theory vector in
        metadata order, so reconstructed segments are concatenated.
        """
        if dtype is None:
            dtype = DTYPE
        if not self.pca_blocks:
            raise RuntimeError("PCA_per_bin TF backtransform: pca_blocks missing")

        single = len(pred_intermediate.shape) == 1
        coeffs = (
            tf.expand_dims(pred_intermediate, 0) if single else pred_intermediate
        )
        parts = []
        offset = 0
        for block in self.pca_blocks:
            n_comp = int(block["n_comp"])
            c = coeffs[:, offset:offset + n_comp]
            components = tf.constant(block["components"], dtype=dtype)
            feat_std = tf.constant(block["features_std"], dtype=dtype)
            feat_mean = tf.constant(block["features_mean"], dtype=dtype)
            parts.append(tf.matmul(c, components) * feat_std + feat_mean)
            offset += n_comp
        out = tf.concat(parts, axis=-1)
        return out[0] if single else out

    def train(self,
              X: Dict[str, np.ndarray],
              y: np.ndarray,
              model_filename: str,
              test_split: float = 0.1,
              n_cycles_per_training: int = 5,
              n_hidden: Optional[List[int]] = None,
              learning_rates: Optional[List[float]] = None,
              batch_sizes: Optional[List[int]] = None,
              gradient_accumulation_steps: Optional[List[int]] = None,
              patience_values: Optional[List[int]] = None,
              max_epochs: Optional[List[int]] = None) -> None:
        """Train the neural network emulator.
        
        Args:
            X: Dictionary of input parameters
            y: Target output data
            model_filename: Base filename for saving model
            test_split: Fraction of data for validation
            n_cycles_per_training: Number of training cycles / learning-rate stages
            n_hidden: Hidden-layer widths; defaults to ``[512, 512, 512, 512]``
            learning_rates: Per-stage learning rates; defaults to
                ``[1e-2, 1e-3, ...]`` over ``n_cycles_per_training``
            batch_sizes: Per-stage batch sizes; a single value is broadcast.
                Defaults to ``[32] * n_cycles_per_training``
            gradient_accumulation_steps: Per-stage accumulation steps
            patience_values: Early-stopping patience per stage
            max_epochs: Maximum epochs per stage
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

        if n_hidden is None:
            n_hidden = [512, 512, 512, 512]
        if learning_rates is None:
            learning_rates = [10 ** (-2 - i) for i in range(n_cycles_per_training)]
        if batch_sizes is None:
            batch_sizes = [32] * n_cycles_per_training
        elif len(batch_sizes) == 1:
            batch_sizes = list(batch_sizes) * n_cycles_per_training
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = [1] * n_cycles_per_training
        if patience_values is None:
            patience_values = [100] * n_cycles_per_training
        if max_epochs is None:
            max_epochs = [1000] * n_cycles_per_training

        schedule_lengths = [
            len(learning_rates), len(batch_sizes), len(gradient_accumulation_steps),
            len(patience_values), len(max_epochs),
        ]
        if not all(length == n_cycles_per_training for length in schedule_lengths):
            raise ValueError(
                f"Training schedule lists must all have length "
                f"n_cycles_per_training={n_cycles_per_training}; "
                f"got learning_rates={len(learning_rates)}, "
                f"batch_sizes={len(batch_sizes)}, "
                f"gradient_accumulation_steps={len(gradient_accumulation_steps)}, "
                f"patience_values={len(patience_values)}, "
                f"max_epochs={len(max_epochs)}"
            )
        
        # Normalize input parameters
        logger.info("Normalizing input parameters")
        self.X_mean = {key: np.mean(X[key], axis=0) for key in X.keys()}
        self.X_std = {key: np.maximum(np.std(X[key], axis=0), 1e-10) for key in X.keys()}
        
        # Optional block-wise amplitude prefactor, then data_transformation
        y_for_transform = y
        if self.amplitude_prefactor is not None:
            if self.loss_function == "weighted_w_cov":
                raise ValueError(
                    "loss_function=weighted_w_cov is incompatible with "
                    "amplitude_prefactor=T (θ-dependent amplitude divide vs "
                    "fixed physical d / C^{-1} in the LINNA feature-space loss). "
                    "Use loss_function=standard or amplitude_prefactor=F."
                )
            logger.info("Applying block-wise 3x2pt amplitude prefactor to training targets")
            y_for_transform = self.amplitude_prefactor.divide(X, y)

        # Transform target data
        logger.info(f"Applying {self.data_transformation} transformation to target data")
        y_train = self.transform(y_for_transform)
        
        # Prepare normalization arrays
        X_mean_arr = np.array([self.X_mean[key] for key in self.model_parameters])
        X_std_arr = np.array([self.X_std[key] for key in self.model_parameters])
        
        # Create neural network
        logger.info(f"Creating {self.nn_model} neural network with n_hidden={n_hidden}")
        if self._is_pca_family(self.data_transformation):
            output_dim = int(y_train.shape[1])
        else:
            output_dim = len(self.modes)

        bin_pair_spec = None
        if self.nn_model == "EmbMLP":
            if self._is_pca_family(self.data_transformation):
                raise ValueError(
                    "nn_model=EmbMLP is incompatible with "
                    f"data_transformation={self.data_transformation!r}: the "
                    "network predicts one bin pair's ell nodes at a time, but "
                    "PCA coefficients have no bin-pair structure. Use "
                    "weighted_norm / weighted_median_norm / norm / log_norm."
                )
            bin_pair_spec = build_bin_pair_spec(
                self.vector_metadata,
                output_dim,
                spectra=self.amplitude_prefactor_spectra,
            )

        shared_trunk_spec = None
        if self.nn_model in ("SharedTrunkMLP", "SharedTrunkEmbMLP"):
            if self._is_pca_family(self.data_transformation):
                raise ValueError(
                    f"nn_model={self.nn_model} is incompatible with "
                    f"data_transformation={self.data_transformation!r}: each "
                    "head predicts one probe's modes, but PCA coefficients "
                    "mix probes and have no per-probe structure. Use "
                    "weighted_norm / weighted_median_norm / norm / log_norm."
                )
            shared_trunk_spec = build_shared_trunk_spec(
                self.vector_metadata,
                output_dim,
                self.model_parameters,
                spectra=self.amplitude_prefactor_spectra,
                trunk_params=self.trunk_params,
                bin_embedding=self.nn_model == "SharedTrunkEmbMLP",
            )

        # LINNA-style cov-weighted loss: feature-space d and C^{-1}.
        loss_data_feat = None
        loss_inv_feat = None
        if self.loss_function == "weighted_w_cov":
            loss_data_feat, loss_inv_feat = self._linna_feature_space_loss_arrays()
            logger.info(
                "weighted_w_cov: LINNA feature-space loss "
                "(n_modes=%d, floor χ²(M,d)=%.1f)",
                loss_data_feat.shape[0],
                0.5 * loss_data_feat.shape[0],
            )

        self.cp_nn = CosmoPowerNN(
            parameters=self.model_parameters,
            modes=list(range(output_dim)),
            parameters_mean=X_mean_arr,
            parameters_std=X_std_arr,
            features_mean=self.y_mean,
            features_std=self.y_std,
            n_hidden=n_hidden,
            verbose=True,
            architecture_type=self.nn_model,
            loss_function=self.loss_function,
            loss_data_feat=loss_data_feat,
            loss_inv_feat=loss_inv_feat,
            bin_pair_spec=bin_pair_spec,
            embedding_dim=self.embedding_dim,
            shared_trunk_spec=shared_trunk_spec,
            head_n_hidden=self.head_n_hidden,
            trunk_skip_connection=self.trunk_skip_connection,
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
                learning_rates=learning_rates,
                batch_sizes=batch_sizes,
                gradient_accumulation_steps=gradient_accumulation_steps,
                patience_values=patience_values,
                max_epochs=max_epochs
            )
        
        # Save additional attributes
        self._save_attributes(model_filename)
        
        self.trained = True
        logger.info("Emulator training completed successfully")


    def _save_attributes(self, model_filename: str) -> None:
        """Save additional emulator attributes."""
        save_dict = { "data_transformation": {
            "data_transformation": self.data_transformation,
            "n_pca": self.n_pca,
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "features_mean": self.features_mean,
            "features_std": self.features_std,
            "cov_sigma": self.cov_sigma,
            "pca_transform_matrix": self.pca_transform_matrix,
            "pca_blocks": self.pca_blocks,
            "vector_metadata": self.vector_metadata,
            "amplitude_prefactor_spectra": self.amplitude_prefactor_spectra,
            "amplitude_prefactor_enabled": self.amplitude_prefactor_enabled,
            "amplitude_prefactor_state": (
                self.amplitude_prefactor.to_state()
                if self.amplitude_prefactor is not None else None
            ),
            }
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
        # The saved model, not the constructor default, defines the architecture.
        self.nn_model = self.cp_nn.architecture_type
        self.embedding_dim = int(getattr(self.cp_nn, "embedding_dim", self.embedding_dim))
        self.head_n_hidden = list(getattr(self.cp_nn, "head_n_hidden", []) or [])
        self.trunk_skip_connection = bool(
            getattr(self.cp_nn, "trunk_skip_connection", self.trunk_skip_connection)
        )
        
        # Load additional attributes
        npz_filename = filename + ".npz"
        with np.load(npz_filename, allow_pickle=True) as data:
            data_transformation = data["data_transformation"].item()
            self.data_transformation = data_transformation["data_transformation"]
            if "n_pca" in data_transformation and data_transformation["n_pca"] is not None:
                self.n_pca = int(data_transformation["n_pca"])
            self.X_mean = data_transformation["X_mean"]
            self.X_std = data_transformation["X_std"]
            self.y_mean = data_transformation["y_mean"]
            self.y_std = data_transformation["y_std"]
            self.features_mean = data_transformation["features_mean"]
            self.features_std = data_transformation["features_std"]
            self.cov_sigma = data_transformation.get("cov_sigma")
            self.pca_transform_matrix = data_transformation["pca_transform_matrix"]
            self.pca_blocks = data_transformation.get("pca_blocks")
            self.vector_metadata = data_transformation.get("vector_metadata")
            self.amplitude_prefactor_spectra = data_transformation.get(
                "amplitude_prefactor_spectra"
            )
            self.amplitude_prefactor_enabled = bool(
                data_transformation.get("amplitude_prefactor_enabled", False)
            )
            amp_state = data_transformation.get("amplitude_prefactor_state")
            if self.amplitude_prefactor_enabled and amp_state is not None:
                self.amplitude_prefactor = BlockAmplitudePrefactor.from_state(amp_state)
            else:
                self.amplitude_prefactor = None
            if (
                self._is_weighted_norm_family(str(self.data_transformation))
                and self.cov_sigma is None
            ):
                raise ValueError(
                    f"Loaded emulator uses data_transformation="
                    f"{self.data_transformation!r} but cov_sigma is missing. "
                    "Retrain with a current ROSE that saves cov_sigma."
                )
        
        self.trained = True
        logger.info("Emulator loaded successfully")
            

    def save_to(self, filename: str) -> None:
        """Save current emulator state to filename so workers can load it from disk.
        
        Used when running with a pool (e.g. Nautilus) and the model may not have been
        written to this path yet (e.g. save_output is not 'all'). Call this before
        passing the path to worker processes.
        
        Args:
            filename: Base filename to save to (no .npz extension).
        """
        if not self.trained or not hasattr(self, 'cp_nn'):
            raise RuntimeError("Emulator must be trained before saving")
        self.cp_nn.save(filename, {})
        self._save_attributes(filename)
        logger.info(f"Emulator saved to {filename} for worker processes")

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
                if self.ignore_extra_params:
                    continue
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

        # Undo block-wise amplitude prefactor
        if self.amplitude_prefactor is not None:
            y_pred = self.amplitude_prefactor.multiply(X, y_pred)
        
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
            pred_norm_batch = self.cp_nn.predictions_normalized_tf(x)
            
            # Remove batch dimension if it was added
            if len(params_norm.shape) == 1 and len(pred_norm_batch.shape) == 2:
                pred_norm = pred_norm_batch[0]  # (n_modes,)
            else:
                pred_norm = pred_norm_batch
            
            # Denormalize outputs (e.g., transform from NN normalization to log10 space)
            pred_intermediate = pred_norm * y_std_tf + y_mean_tf
            
            # Apply backtransform (e.g., from log10 / PCA-coeff space to original space)
            pred_original = self.backtransform_tf(pred_intermediate, dtype=DTYPE)

            if self.amplitude_prefactor is not None:
                param_index = {p: i for i, p in enumerate(self.model_parameters)}
                amp = self.amplitude_prefactor.factors_tf(params_orig, param_index, DTYPE)
                if len(pred_original.shape) == 2:
                    pred_original = pred_original * amp[None, :]
                else:
                    pred_original = pred_original * amp
            
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
        print(f"Data transformation: {self.data_transformation}")
        if self.data_transformation == 'PCA':
            print(f"PCA components: {self.n_pca}")
        elif self.data_transformation == 'PCA_per_bin':
            n_blocks = len(self.pca_blocks) if self.pca_blocks else 0
            n_coeff = int(self.y_mean.shape[0]) if self.y_mean is not None else 0
            print(f"PCA per bin-pair: n_pca_per_bin={self.n_pca}, "
                  f"blocks={n_blocks}, total_coeffs={n_coeff}")
        print(f"Trained: {self.trained}")
        
        if self.trained and hasattr(self, 'cp_nn'):
            print("\\nUnderlying Neural Network:")
            self.cp_nn.summary()
        
        print("="*60 + "\\n")
