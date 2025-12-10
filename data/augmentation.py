"""
Audio augmentation functions for training.
"""

import tensorflow as tf


@tf.function(jit_compile=True)
def shift_perturb(
    audio_input: tf.Tensor,
    probability: float = 0.8,
    duration: float = 0.63,
    sample_rate: int = 16000,
    min_shift_ms: int = -5,
    max_shift_ms: int = 5,
    seed: int = 42
) -> tf.Tensor:
    """
    Apply time-shift augmentation to audio signal.
    
    Args:
        audio_input: Input audio tensor
        probability: Probability of applying augmentation
        duration: Maximum duration for shift in seconds
        sample_rate: Audio sample rate
        min_shift_ms: Minimum shift in milliseconds (negative = shift left)
        max_shift_ms: Maximum shift in milliseconds (positive = shift right)
        seed: Random seed
        
    Returns:
        Augmented audio tensor
    """
    max_shift_ms_value = max(abs(min_shift_ms), abs(max_shift_ms))
    max_shift_samples = tf.cast(max_shift_ms_value * sample_rate // 1000, tf.int32)
    
    # Pad audio with zeros
    padded_audio = tf.pad(audio_input, [[max_shift_samples, max_shift_samples]])
    
    # Compute random shift
    shift_ms = tf.random.uniform(
        shape=(),
        minval=min_shift_ms,
        maxval=max_shift_ms,
        seed=seed
    )
    shift_samples = tf.cast(shift_ms * sample_rate // 1000, dtype=tf.int32)
    
    # Extract shifted audio
    audio_length = tf.shape(audio_input)[0]
    start_idx = max_shift_samples - shift_samples
    shifted_audio = padded_audio[start_idx : start_idx + audio_length]
    
    # Apply duration condition
    shifted_audio = tf.cond(
        tf.abs(shift_ms / 1000) <= duration,
        lambda: shifted_audio,
        lambda: audio_input
    )
    
    # Apply probability condition
    return tf.cond(
        tf.random.uniform([], 0, 1, seed=seed) <= probability,
        lambda: shifted_audio,
        lambda: audio_input
    )


@tf.function(jit_compile=True)
def white_noise_perturb(
    audio_input: tf.Tensor,
    noise_input: tf.Tensor,
    probability: float = 0.8,
    min_level: int = -90,
    max_level: int = -46,
    seed: int = 42
) -> tf.Tensor:
    """
    Apply white noise and background noise augmentation.
    
    Args:
        audio_input: Input audio tensor
        noise_input: Background noise tensor
        probability: Probability of applying augmentation
        min_level: Minimum noise level in dB
        max_level: Maximum noise level in dB
        seed: Random seed
        
    Returns:
        Augmented audio tensor
    """
    # Generate noise level
    noise_level_db = tf.random.uniform(
        shape=(),
        minval=min_level,
        maxval=max_level,
        dtype=tf.int32,
        seed=seed
    )
    
    # Generate white noise
    noise = tf.random.normal(shape=tf.shape(audio_input), seed=seed)
    noise_signal = noise * (10.0 ** (tf.cast(noise_level_db, dtype=tf.float32) / 20.0))
    
    # Create augmented versions
    audio_wnoise = audio_input + noise_signal
    
    # Random selection between augmentation strategies
    rand_val = tf.random.uniform([], 0, 1, seed=seed)
    
    return tf.cond(
        rand_val <= probability,
        lambda: audio_wnoise,
        lambda: audio_input
    )