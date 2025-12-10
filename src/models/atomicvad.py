"""
AtomicVAD model architecture definition.
"""
import os
import sys

import tensorflow as tf
from typing import Tuple

from layers import GGCU, Spectrogram, SpecCutout
from spec_augment import SpecAugment

# 1. Determine the absolute path to the directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP two levels and then DOWN into the folder
target_dir_training = os.path.join(current_file_dir, '..', '..', 'training')

# 3. Add this target directory to Python's search path
sys.path.append(target_dir_training)

# 4. Now you can import as if it were in the same folder
from config import TrainingConfig


def core_block(
    input_layer: tf.Tensor,
    depth_multiplier: int,
    kernel_size: int,
    pool_size: int,
    normalization: str,
    filter_conv1: int,
    activation_function: str,
    dropout_rate: float,
    filter_conv2: int,
    seed: int = 42,
    name_prefix: str = ""
) -> tf.Tensor:
    """
    Core building block of AtomicVAD.
    
    Architecture:
    DepthwiseConv2D -> MaxPooling2D -> Normalization -> Conv2D -> Activation -> Dropout -> Conv2D
    
    Args:
        input_layer: Input tensor
        depth_multiplier: Depth multiplier for depthwise convolution
        kernel_size: Kernel size for depthwise convolution
        pool_size: Pool size for max pooling
        normalization: Type of normalization ('layerNorm' or 'batchNorm')
        filter_conv1: Number of filters for first Conv2D
        activation_function: Activation function name
        dropout_rate: Dropout rate
        filter_conv2: Number of filters for second Conv2D
        seed: Random seed
        name_prefix: Prefix for layer names
        
    Returns:
        Output tensor from the core block
    """
    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=depth_multiplier,
        kernel_size=(kernel_size, kernel_size),
        padding='same',
        data_format='channels_last',
        activation=None,
        name=f'{name_prefix}_depthwise_conv'
    )(input_layer)
    
    # Max pooling
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(pool_size, pool_size),
        padding='same',
        data_format='channels_last',
        name=f'{name_prefix}_maxpool'
    )(x)
    
    # Normalization
    if normalization == 'batchNorm':
        x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_batchnorm')(x)
    elif normalization == 'layerNorm':
        x = tf.keras.layers.LayerNormalization(name=f'{name_prefix}_layernorm')(x)
    
    # First Conv2D
    x = tf.keras.layers.Conv2D(
        filters=filter_conv1,
        kernel_size=(1, 1),
        data_format='channels_last',
        name=f'{name_prefix}_conv1'
    )(x)
    
    # Activation
    if activation_function == 'GGCU':
        x = GGCU(seed=seed, name=f'{name_prefix}_ggcu')(x)
    else:
        x = tf.keras.layers.Activation(activation_function, name=f'{name_prefix}_activation')(x)
    
    # Dropout
    x = tf.keras.layers.Dropout(dropout_rate, seed=seed, name=f'{name_prefix}_dropout')(x)
    
    # Second Conv2D
    x = tf.keras.layers.Conv2D(
        filters=filter_conv2,
        kernel_size=(1, 1),
        data_format='channels_last',
        name=f'{name_prefix}_conv2'
    )(x)
    
    return x


def build_atomicvad_model(config: TrainingConfig, seed: int = 42) -> tf.keras.Model:
    """
    Build the complete AtomicVAD model.
    
    Args:
        config: Training configuration containing model hyperparameters
        seed: Random seed for reproducibility
        
    Returns:
        Compiled Keras model
    """
    # Input layer - raw audio waveform
    input_layer = tf.keras.layers.Input(
        shape=(int(config.sample_rate * config.duration),),
        name='audio_input'
    )
    
    # Spectrogram extraction
    x = Spectrogram(
        spec_type=config.spec_type,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        n_mfcc=config.n_mfcc,
        normalize=False,
        name='spectrogram'
    )(input_layer)
    
    # Data augmentation layers (only active during training)
    x = SpecAugment(
        freq_mask_param=config.spec_augment_freq_mask,
        time_mask_param=config.spec_augment_time_mask,
        n_freq_mask=config.spec_augment_n_freq_mask,
        n_time_mask=config.spec_augment_n_time_mask,
        mask_value=0,
        seed=seed,
        name='spec_augment'
    )(x)
    
    x = SpecCutout(
        masks_number=config.cutout_masks_number,
        time_mask_size=config.cutout_time_mask_size,
        frequency_mask_size=config.cutout_freq_mask_size,
        seed=seed,
        name='spec_cutout'
    )(x)
    
    # Block 1
    block1_out = core_block(
        input_layer=x,
        depth_multiplier=config.dm_block1,
        kernel_size=config.kernel_dc_block1,
        pool_size=config.maxpool_block1,
        normalization=config.normalization,
        filter_conv1=config.filter_c1_block1,
        activation_function=config.activation_function,
        dropout_rate=config.dropout,
        filter_conv2=config.filter_c2_block1,
        seed=seed,
        name_prefix='block1'
    )
    
    # Block 2
    block2_out = core_block(
        input_layer=block1_out,
        depth_multiplier=config.dm_block2,
        kernel_size=config.kernel_dc_block2,
        pool_size=config.maxpool_block2,
        normalization=config.normalization,
        filter_conv1=config.filter_c1_block2,
        activation_function=config.activation_function,
        dropout_rate=config.dropout,
        filter_conv2=config.filter_c2_block2,
        seed=seed,
        name_prefix='block2'
    )
    
    # Skip connection from Block 1 to Block 2
    # Pool the skip connection to match spatial dimensions
    skip_pooled = tf.keras.layers.MaxPooling2D(
        pool_size=(config.maxpool_block2, config.maxpool_block2),
        padding='same',
        data_format='channels_last',
        name='skip_connection_pool'
    )(block1_out)
    
    # Project skip connection if channel dimensions don't match
    if skip_pooled.shape[-1] != config.filter_c2_block2:
        skip_projected = tf.keras.layers.Conv2D(
            filters=config.filter_c2_block2,
            kernel_size=(1, 1),
            padding='same',
            data_format='channels_last',
            name='skip_connection_projection'
        )(skip_pooled)
    else:
        skip_projected = skip_pooled
    
    # Add skip connection
    x = tf.keras.layers.Add(name='skip_connection_add')([skip_projected, block2_out])
    
    # Reshape and flatten
    x = tf.keras.layers.Reshape(
        (x.shape[1] * x.shape[2], x.shape[3]),
        name='reshape'
    )(x)
    
    x = tf.keras.layers.Flatten(name='flatten')(x)
    
    # Output layer
    output = tf.keras.layers.Dense(
        config.num_classes,
        activation='softmax',
        name='output'
    )(x)
    
    # Create model
    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output,
        name='AtomicVAD-GGCU'
    )
    
    return model