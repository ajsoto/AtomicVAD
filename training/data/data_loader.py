"""
Data loading utilities for AtomicVAD training.
"""

import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional

from data.augmentation import shift_perturb, white_noise_perturb


def get_files(manifest_path: str) -> List[Dict]:
    """
    Load manifest file containing audio file information.
    
    Args:
        manifest_path: Path to JSON manifest file
        
    Returns:
        List of dictionaries containing file information
    """
    with open(manifest_path, 'r') as f:
        return [json.loads(line) for line in f]


def load_manifest(
    speech_train_json: str = None,
    background_train_json: str = None,
    speech_val_json: str = None,
    background_val_json: str = None,
    speech_test_json: str = None,
    background_test_json: str = None,
    return_noise_only: bool = False
) -> Tuple[List[Dict], ...]:
    """
    Load and combine train/val/test manifests.
    
    Args:
        speech_train_json: Path to speech training manifest
        background_train_json: Path to background training manifest
        speech_val_json: Path to speech validation manifest
        background_val_json: Path to background validation manifest
        speech_test_json: Path to speech testing manifest
        background_test_json: Path to background testing manifest
        return_noise_only: If True, only return noise samples
        
    Returns:
        Tuple of (train_json, val_json, test_json) or just noise_json if return_noise_only
    """
    if return_noise_only:
        return get_files(background_train_json)
    
    # Load and combine manifests
    train_json = get_files(speech_train_json) + get_files(background_train_json)
    val_json = get_files(speech_val_json) + get_files(background_val_json)
    test_json = get_files(speech_test_json) + get_files(background_test_json)
    
    # Shuffle the datasets
    np.random.shuffle(train_json)
    np.random.shuffle(val_json)
    np.random.shuffle(test_json)
    
    return train_json, val_json, test_json


def process_sample(
    audio_filepath: str,
    noise_filepath: str,
    duration: float,
    label: str,
    audio_offset: float,
    noise_offset: float,
    sample_rate: int = 16000,
    train: bool = True,
    seed: int = 42
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Process a single audio sample with optional augmentation.
    
    Args:
        audio_filepath: Path to audio file
        noise_filepath: Path to noise file for augmentation
        duration: Duration of audio segment in seconds
        label: Label ('speech' or 'background'/'NO_SPEECH')
        audio_offset: Start time offset in audio file
        noise_offset: Start time offset in noise file
        sample_rate: Audio sample rate
        train: Whether to apply training augmentations
        seed: Random seed for augmentation
        
    Returns:
        Tuple of (audio_waveform, label_tensor)
    """
    # Read and decode audio
    audio_binary = tf.io.read_file(audio_filepath)
    audio_waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio_waveform = tf.squeeze(audio_waveform, axis=-1)
    
    # Extract segment
    total_samples = tf.cast(sample_rate * duration, tf.int32)
    start_sample = tf.cast(sample_rate * audio_offset, tf.int32)
    end_sample = start_sample + total_samples
    audio_waveform = audio_waveform[start_sample:end_sample]
    
    # Apply augmentation during training
    if train:
        # Load noise
        noise_binary = tf.io.read_file(noise_filepath)
        noise_waveform, _ = tf.audio.decode_wav(noise_binary, desired_channels=1)
        noise_waveform = tf.squeeze(noise_waveform, axis=-1)
        
        # Extract noise segment
        start_noise = tf.cast(sample_rate * noise_offset, tf.int32)
        end_noise = start_noise + total_samples
        noise_waveform = noise_waveform[start_noise:end_noise]
        
        # Apply augmentations
        audio_waveform = white_noise_perturb(audio_waveform, noise_waveform, seed=seed)
        audio_waveform = shift_perturb(audio_waveform, seed=seed)
    
    # Create label tensor
    label_tensor = tf.cond(
        tf.logical_or(tf.equal(label, 'background'), tf.equal(label, 'NO_SPEECH')),
        lambda: tf.constant([1, 0], dtype=tf.int32),
        lambda: tf.constant([0, 1], dtype=tf.int32)
    )
    
    return audio_waveform, label_tensor


def create_dataset(
    audio_info_list: List[Dict],
    noise_info_list: List[Dict],
    batch_size: int,
    sample_rate: int,
    train: bool = True,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from manifest information.
    
    Args:
        audio_info_list: List of audio file information dictionaries
        noise_info_list: List of noise file information dictionaries
        batch_size: Batch size for training
        sample_rate: Audio sample rate
        train: Whether this is a training dataset (applies augmentation)
        seed: Random seed for shuffling and augmentation
        
    Returns:
        TensorFlow Dataset ready for training/evaluation
    """
    # Extract fields from dictionaries
    audio_filepaths = [info['audio_filepath'] for info in audio_info_list]
    noise_filepaths = [info['audio_filepath'] for info in noise_info_list]
    durations = [info['duration'] for info in audio_info_list]
    labels = [info['label'] for info in audio_info_list]
    audio_offsets = [info['offset'] for info in audio_info_list]
    noise_offsets = [info['offset'] for info in noise_info_list]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        audio_filepaths,
        noise_filepaths,
        durations,
        labels,
        audio_offsets,
        noise_offsets
    ))
    
    # Shuffle training data
    if train:
        dataset = dataset.shuffle(buffer_size=len(audio_info_list), seed=seed)
    
    # Set deterministic options
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    
    # Process samples
    dataset = dataset.map(
        lambda audio_filepath, noise_filepath, duration, label, audio_offset, noise_offset: process_sample(
            audio_filepath, noise_filepath, duration, label, audio_offset, noise_offset, sample_rate, train, seed
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset