import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfpl = tfp.layers
import keras
from keras import Model, Sequential 
from keras.layers import InputLayer, Flatten
from bayesflow import default_settings as defaults
from bayesflow.attention import (
    InducedSelfAttentionBlock,
    MultiHeadAttentionBlock,
    PoolingWithAttention,
    SelfAttentionBlock,
)
from bayesflow.helper_networks import EquivariantModule, InvariantModule, MultiConv1D


class DeepSet(tf.keras.Model):
    """Implements a deep permutation-invariant network according to [1] and [2].

    [1] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017).
    Deep sets. Advances in neural information processing systems, 30.

    [2] Bloem-Reddy, B., & Teh, Y. W. (2020).
    Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1.
    """


    def __init__(
        self,
        summary_dim=10,
        num_dense_s1=2,
        num_dense_s2=2,
        num_dense_s3=2,
        num_equiv=2,
        dense_s1_args=None,
        dense_s2_args=None,
        dense_s3_args=None,
        pooling_fun="mean",
        input_shape=None,
        **kwargs,
    ):
        """Creates a stack of 'num_equiv' equivariant layers followed by a final invariant layer.

        Parameters
        ----------
        summary_dim   : int, optional, default: 10
            The number of learned summary statistics.
        num_dense_s1  : int, optional, default: 2
            The number of dense layers in the inner function of a deep set.
        num_dense_s2  : int, optional, default: 2
            The number of dense layers in the outer function of a deep set.
        num_dense_s3  : int, optional, default: 2
            The number of dense layers in an equivariant layer.
        num_equiv     : int, optional, default: 2
            The number of equivariant layers in the network.
        dense_s1_args : dict or None, optional, default: None
            The arguments for the dense layers of s1 (inner, pre-pooling function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        dense_s2_args : dict or None, optional, default: None
            The arguments for the dense layers of s2 (outer, post-pooling function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        dense_s3_args : dict or None, optional, default: None
            The arguments for the dense layers of s3 (equivariant function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        pooling_fun   : str of callable, optional, default: 'mean'
            If string argument provided, should be one in ['mean', 'max']. In addition, ac actual
            neural network can be passed for learnable pooling.
        **kwargs      : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model.
        """

        super().__init__(**kwargs)

        # Prepare settings dictionary
        settings = dict(
            num_dense_s1=num_dense_s1,
            num_dense_s2=num_dense_s2,
            num_dense_s3=num_dense_s3,
            dense_s1_args=defaults.DEFAULT_SETTING_DENSE_DEEP_SET if dense_s1_args is None else dense_s1_args,
            dense_s2_args=defaults.DEFAULT_SETTING_DENSE_DEEP_SET if dense_s2_args is None else dense_s2_args,
            dense_s3_args=defaults.DEFAULT_SETTING_DENSE_DEEP_SET if dense_s3_args is None else dense_s3_args,
            pooling_fun=pooling_fun,
        )

        # Create equivariant layers and final invariant layer
        layers = [EquivariantModule(settings) for _ in range(num_equiv)]
        bnn_layers = [
            InputLayer(input_shape=input_shape),
            tfpl.DenseFlipout(units=128, activation='relu'),
            tf.keras.layers.Flatten(),
            tfpl.DenseFlipout(units=128, activation='relu'),
        ]
        layers = bnn_layers + layers
        self.equiv_layers = Sequential(layers)
        self.inv = InvariantModule(settings)

        # Output layer to output "summary_dim" learned summary statistics
        self.out_layer = Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim





    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """

        # Pass through series of augmented equivariant transforms
        out_equiv = self.equiv_layers(x, **kwargs)

        # Pass through final invariant layer
        out = self.out_layer(self.inv(out_equiv, **kwargs), **kwargs)

        return out

