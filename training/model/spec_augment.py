"""
SpecAugment implementation for audio data augmentation.
Based on: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
https://arxiv.org/abs/1904.08779
"""

import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package='Custom', name='SpecAugment')
class SpecAugment(tf.keras.layers.Layer):
    """
    SpecAugment layer for spectrogram augmentation.
    
    Applies time and frequency masking to spectrograms during training.
    """
    
    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        n_freq_mask: int = 1,
        n_time_mask: int = 1,
        mask_value: float = 0.,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            freq_mask_param: Frequency mask parameter (F in the paper)
            time_mask_param: Time mask parameter (T in the paper)
            n_freq_mask: Number of frequency masks (mF in the paper)
            n_time_mask: Number of time masks (mT in the paper)
            mask_value: Imputation value for masked regions
            seed: Random seed for reproducibility
        """
        super(SpecAugment, self).__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_mask = n_freq_mask
        self.n_time_mask = n_time_mask
        self.mask_value = tf.cast(mask_value, tf.float32)
        self.seed = seed
        
    def _frequency_mask_single(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Generate a single frequency mask."""
        n_mels = tf.cast(tf.shape(input_mel_spectrogram)[1], tf.float32)
        freq_indices = tf.reshape(tf.cast(tf.range(n_mels), tf.int32), (1, -1, 1))
        
        # Generate random mask parameters
        f = tf.cast(
            tf.random.uniform(shape=(), maxval=self.freq_mask_param, seed=self.seed),
            tf.int32
        )
        f0 = tf.cast(
            tf.random.uniform(shape=(), maxval=n_mels - tf.cast(f, tf.float32), seed=self.seed),
            tf.int32
        )
        
        # Create mask condition
        condition = tf.logical_and(freq_indices >= f0, freq_indices <= f0 + f)
        return tf.cast(condition, tf.float32)
    
    def _frequency_masks(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Apply multiple frequency masks."""
        mel_repeated = tf.repeat(
            tf.expand_dims(input_mel_spectrogram, 0),
            self.n_freq_mask,
            axis=0
        )
        
        masks = tf.cast(
            tf.map_fn(elems=mel_repeated, fn=self._frequency_mask_single),
            tf.bool
        )
        mask = tf.math.reduce_any(masks, 0)
        
        return tf.where(mask, self.mask_value, input_mel_spectrogram)
    
    def _time_mask_single(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Generate a single time mask."""
        n_steps = tf.cast(tf.shape(input_mel_spectrogram)[0], tf.float32)
        time_indices = tf.reshape(tf.cast(tf.range(n_steps), tf.int32), (-1, 1, 1))
        
        # Generate random mask parameters
        t = tf.cast(
            tf.random.uniform(shape=(), maxval=self.time_mask_param, seed=self.seed),
            tf.int32
        )
        t0 = tf.cast(
            tf.random.uniform(shape=(), maxval=n_steps - tf.cast(t, tf.float32), seed=self.seed),
            tf.int32
        )
        
        # Create mask condition
        condition = tf.logical_and(time_indices >= t0, time_indices <= t0 + t)
        return tf.cast(condition, tf.float32)
    
    def _time_masks(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Apply multiple time masks."""
        mel_repeated = tf.repeat(
            tf.expand_dims(input_mel_spectrogram, 0),
            self.n_time_mask,
            axis=0
        )
        
        masks = tf.cast(
            tf.map_fn(elems=mel_repeated, fn=self._time_mask_single),
            tf.bool
        )
        mask = tf.math.reduce_any(masks, 0)
        
        return tf.where(mask, self.mask_value, input_mel_spectrogram)
    
    def _apply_spec_augment(self, input_mel_spectrogram: tf.Tensor) -> tf.Tensor:
        """Apply both frequency and time masking."""
        if self.n_freq_mask >= 1:
            input_mel_spectrogram = self._frequency_masks(input_mel_spectrogram)
        
        if self.n_time_mask >= 1:
            input_mel_spectrogram = self._time_masks(input_mel_spectrogram)
        
        return input_mel_spectrogram
    
    def call(self, inputs: tf.Tensor, training=None, **kwargs):
        """
        Apply SpecAugment to input spectrograms.
        
        Args:
            inputs: Input mel spectrogram tensor
            training: If True, augmentation is applied
            
        Returns:
            Augmented spectrogram (if training) or original spectrogram
        """
        if training:
            inputs_masked = tf.map_fn(elems=inputs, fn=self._apply_spec_augment)
            return inputs_masked
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "n_freq_mask": self.n_freq_mask,
            "n_time_mask": self.n_time_mask,
            "mask_value": self.mask_value.numpy(),
            "seed": self.seed
        })
        return config