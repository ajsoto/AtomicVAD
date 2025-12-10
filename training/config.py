"""
Training configuration for AtomicVAD model.
"""

import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Data paths
    data_dir: str = 'data/manifest'
    speech_train_json: str = os.path.join(data_dir, 'balanced_speech_training_manifest.json')
    background_train_json: str = os.path.join(data_dir, 'balanced_background_training_manifest.json')
    speech_val_json: str = os.path.join(data_dir, 'balanced_speech_validation_manifest.json')
    background_val_json: str = os.path.join(data_dir, 'balanced_background_validation_manifest.json')
    speech_test_json: str = os.path.join(data_dir, 'balanced_speech_testing_manifest.json')
    background_test_json: str = os.path.join(data_dir, 'balanced_background_testing_manifest.json')
    
    # Output paths
    checkpoint_dir: str = 'experiments/atomicvad-ggcu/checkpoints'
    backup_dir: str = 'experiments/atomicvad-ggcu/backups'
    
    # Audio parameters
    sample_rate: int = 16000
    duration: float = 0.63  # seconds
    
    # Spectrogram parameters
    spec_type: str = 'mfcc'  # 'mfcc' or 'mel'
    n_fft: int = 512
    hop_length: int = 160  # 10ms frame shift
    n_mels: int = 64
    n_mfcc: int = 64
    
    # Data augmentation parameters
    spec_augment_freq_mask: int = 15
    spec_augment_time_mask: int = 25
    spec_augment_n_freq_mask: int = 2
    spec_augment_n_time_mask: int = 2
    
    cutout_masks_number: int = 5
    cutout_time_mask_size: int = 25
    cutout_freq_mask_size: int = 15
    
    shift_perturb_prob: float = 0.8
    shift_perturb_min_ms: int = -5
    shift_perturb_max_ms: int = 5
    
    white_noise_prob: float = 0.8
    white_noise_min_db: int = -90
    white_noise_max_db: int = -46
    
    # Model architecture - Block 1
    dm_block1: int = 3
    kernel_dc_block1: int = 7
    filter_c1_block1: int = 8
    filter_c2_block1: int = 2
    maxpool_block1: int = 5
    
    # Model architecture - Block 2
    dm_block2: int = 1
    kernel_dc_block2: int = 3
    filter_c1_block2: int = 8
    filter_c2_block2: int = 2
    maxpool_block2: int = 7
    
    # Model regularization
    dropout: float = 0.2
    normalization: str = 'layerNorm'  # 'layerNorm' or 'batchNorm'
    activation_function: str = 'GGCU'
    
    # Training parameters
    batch_size: int = 256
    epochs: int = 2#150
    initial_lr: float = 0.001
    early_stopping_patience: int = 20
    
    # Output classes
    num_classes: int = 2
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """Separate configuration for model architecture only."""
    
    # Spectrogram
    spec_type: str = 'mfcc'
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 64
    n_mfcc: int = 64
    
    # Block 1
    dm_block1: int = 3
    kernel_dc_block1: int = 7
    filter_c1_block1: int = 8
    filter_c2_block1: int = 2
    maxpool_block1: int = 5
    
    # Block 2
    dm_block2: int = 1
    kernel_dc_block2: int = 3
    filter_c1_block2: int = 8
    filter_c2_block2: int = 2
    maxpool_block2: int = 7
    
    # Regularization
    dropout: float = 0.2
    normalization: str = 'layerNorm' # 'layerNorm' or 'batchNorm'
    activation_function: str = 'GGCU' # 'GGCU', 'ReLU', etc.
    
    # Data augmentation (training only)
    spec_augment_freq_mask: int = 15
    spec_augment_time_mask: int = 25
    spec_augment_n_freq_mask: int = 2
    spec_augment_n_time_mask: int = 2
    
    cutout_masks_number: int = 5
    cutout_time_mask_size: int = 25
    cutout_freq_mask_size: int = 15
    
    num_classes: int = 2