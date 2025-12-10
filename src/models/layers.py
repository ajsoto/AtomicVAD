"""
Custom layers for AtomicVAD model.
"""

import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package='Custom', name='ELU')
class ELU(tf.keras.layers.Layer):
    """
    Exponential Linear Unit (ELU) activation function.
    
    Applies: x if x > 0, else alpha * (exp(x) - 1)
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.elu(inputs, alpha=1.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='GCU')
class GCU(tf.keras.layers.Layer):
    """
    Growing Cosine Unit (GCU) activation function.
    
    Applies: x * cos(x)
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(GCU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return inputs*tf.cos(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='GELU')
class GELU(tf.keras.layers.Layer):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    Applies: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))) (approximate) or x * P(X <= x) where X ~ N(0, 1) (exact)
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.gelu(inputs, approximate=False)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='LReLU')
class LReLU(tf.keras.layers.Layer):
    """
    Leaky Rectified Linear Unit (LReLU) activation function.
    
    Applies: x if x >= 0, else negative_slope * x
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(LReLU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.leaky_relu(inputs, negative_slope=0.2)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='Mish')
class Mish(tf.keras.layers.Layer):
    """
    Mish activation function.
    
    Applies: x * tanh(softplus(x))
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.mish(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='ReLU')
class ReLU(tf.keras.layers.Layer):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Applies: max(0, x)
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.relu(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='SELU')
class SELU(tf.keras.layers.Layer):
    """
    Scaled Exponential Linear Unit (SELU) activation function.
    
    Applies: scale * x if x > 0, else scale * alpha * (exp(x) - 1)
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(SELU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.selu(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='SILU')
class SILU(tf.keras.layers.Layer):
    """
    Sigmoid Linear Unit (SiLU), also known as Swish, activation function.
    
    Applies: x * sigmoid(x)
    """
    def __init__(self, seed: int = 42, **kwargs):
        super(SILU, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        return tf.keras.activations.silu(inputs)#Swish
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='GGCU')
class GGCU(tf.keras.layers.Layer):
    """
    Generalized Growing Cosine Unit (GGCU) activation function.
    
    Applies: (w1*x + b1) * cos(w2*x + b2)
    """
    
    def __init__(self, seed: int = 42, **kwargs):
        super(GGCU, self).__init__(**kwargs)
        self.seed = seed
        
    def build(self, input_shape):
        # Initialize learnable parameters
        self.w1 = self.add_weight(
            name='w1',
            shape=(1,),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )

        self.w2 = self.add_weight(
            name='w2',
            shape=(1,),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )
        
        self.b1 = self.add_weight(
            name='b1',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )

        self.b2 = self.add_weight(
            name='b2',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        return (self.w1 * inputs + self.b1) * tf.cos(self.w2 * inputs + self.b2)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='PGCU')
class PGCU(tf.keras.layers.Layer):
    """
    Phase Growing Cosine Unit (PGCU) activation function.
    
    Applies: x * cos(w*x + b)
    """
    
    def __init__(self, seed: int = 42, **kwargs):
        super(PGCU, self).__init__(**kwargs)
        self.seed = seed
        
    def build(self, input_shape):
        # Initialize learnable parameters
        self.w1 = self.add_weight(
            name='w1',
            shape=(1,),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )
        
        self.b1 = self.add_weight(
            name='b1',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        return inputs * tf.cos(self.w1 * inputs + self.b1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config

@keras.saving.register_keras_serializable(package='Custom', name='AGCU')
class AGCU(tf.keras.layers.Layer):
    """
    Adaptive Gated Cosine Unit (AGCU) activation function.
    
    Applies: (w*x + b) * cos(x)
    """
    
    def __init__(self, seed: int = 42, **kwargs):
        super(AGCU, self).__init__(**kwargs)
        self.seed = seed
        
    def build(self, input_shape):
        # Initialize learnable parameters
        self.w1 = self.add_weight(
            name='w1',
            shape=(1,),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )
        
        self.b1 = self.add_weight(
            name='b1',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        return (self.w1 * inputs + self.b1) * tf.cos(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({'seed': self.seed})
        return config


@keras.saving.register_keras_serializable(package='Custom', name='Spectrogram')
class Spectrogram(tf.keras.layers.Layer):
    """
    Layer for computing spectrograms (MFCC or Mel) from audio waveforms.
    """
    
    def __init__(
        self,
        spec_type: str = 'mfcc',
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        n_mfcc: int = 64,
        normalize: bool = False,
        **kwargs
    ):
        super(Spectrogram, self).__init__(**kwargs)
        self.spec_type = spec_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        
    def call(self, inputs):
        # Compute mel spectrogram
        mel_spec = tf.keras.layers.MelSpectrogram(
            fft_length=self.n_fft,
            sequence_stride=self.hop_length,
            sequence_length=400,
            window='hamming',
            num_mel_bins=self.n_mels,
            sampling_rate=self.sample_rate
        )
        
        # Compute MFCC or Mel spectrogram
        if self.spec_type == 'mfcc':
            spectrogram = tf.transpose(
                tf.signal.mfccs_from_log_mel_spectrograms(
                    tf.transpose(mel_spec(inputs), perm=[0, 2, 1])
                ),
                perm=[0, 2, 1]
            )[..., :self.n_mfcc]
        else:
            spectrogram = mel_spec(inputs)
        
        # Normalize if required
        if self.normalize:
            min_val = tf.reduce_min(spectrogram)
            max_val = tf.reduce_max(spectrogram)
            spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-6)
        
        # Add channel dimension
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        return spectrogram
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'spec_type': self.spec_type,
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
            'normalize': self.normalize
        })
        return config


@keras.saving.register_keras_serializable(package='Custom', name='SpecCutout')
class SpecCutout(tf.keras.layers.Layer):
    """
    Cutout data augmentation for spectrograms.
    
    Based on: Improved Regularization of Convolutional Neural Networks with Cutout
    https://arxiv.org/abs/1708.04552
    """
    
    def __init__(
        self,
        masks_number: int = 2,
        time_mask_size: int = 5,
        frequency_mask_size: int = 2,
        seed: int = 42,
        **kwargs
    ):
        super(SpecCutout, self).__init__(**kwargs)
        self.masks_number = masks_number
        self.time_mask_size = time_mask_size
        self.frequency_mask_size = frequency_mask_size
        self.seed = seed
        
    def _random_cutout(
        self,
        inputs: tf.Tensor,
        mask_size: tuple,
        mask_value: float = 0,
        data_format: str = 'channels_last'
    ) -> tf.Tensor:
        """Apply random cutout mask to inputs."""
        mask_size = tf.convert_to_tensor(mask_size)
        
        if data_format == 'channels_last':
            time_size = tf.shape(inputs)[1]
            feature_size = tf.shape(inputs)[2]
        else:
            time_size = tf.shape(inputs)[2]
            feature_size = tf.shape(inputs)[3]
            
        batch_size = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[-1] if data_format == 'channels_last' else tf.shape(inputs)[1]
        
        # Generate random centers
        cutout_center_time = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=time_size,
            dtype=tf.int32,
            seed=self.seed
        )
        cutout_center_feature = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=feature_size,
            dtype=tf.int32,
            seed=self.seed
        )
        
        mask_size = mask_size // 2
        
        # Compute bounds
        time_lower = tf.maximum(0, cutout_center_time - mask_size[0])
        time_upper = tf.minimum(time_size, cutout_center_time + mask_size[0])
        feature_lower = tf.maximum(0, cutout_center_feature - mask_size[1])
        feature_upper = tf.minimum(feature_size, cutout_center_feature + mask_size[1])
        
        # Reshape for broadcasting
        time_lower = tf.reshape(time_lower, [-1, 1, 1])
        time_upper = tf.reshape(time_upper, [-1, 1, 1])
        feature_lower = tf.reshape(feature_lower, [-1, 1, 1])
        feature_upper = tf.reshape(feature_upper, [-1, 1, 1])
        
        # Create grids
        time_range = tf.reshape(tf.range(time_size), [1, time_size, 1])
        feature_range = tf.reshape(tf.range(feature_size), [1, 1, feature_size])
        
        # Create masks
        time_mask = tf.logical_and(time_range >= time_lower, time_range < time_upper)
        feature_mask = tf.logical_and(feature_range >= feature_lower, feature_range < feature_upper)
        mask = tf.logical_and(time_mask, feature_mask)
        
        # Expand mask for channels
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.tile(mask, [1, 1, 1, channels])
        
        # Apply mask
        masked_inputs = tf.where(
            mask,
            tf.ones_like(inputs, dtype=inputs.dtype) * mask_value,
            inputs
        )
        
        return masked_inputs
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
            
        # Apply multiple cutout masks
        net = inputs
        for _ in range(self.masks_number):
            net = self._random_cutout(
                net,
                (self.time_mask_size, self.frequency_mask_size)
            )
        return net
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'masks_number': self.masks_number,
            'time_mask_size': self.time_mask_size,
            'frequency_mask_size': self.frequency_mask_size,
            'seed': self.seed
        })
        return config