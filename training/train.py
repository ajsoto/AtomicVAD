"""
AtomicVAD Training Script
Main training loop for the AtomicVAD model with GGCU activation.
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras import mixed_precision

from config import TrainingConfig
from utils.seed import set_seed
from utils.gpu import configure_gpu

# 1. Determine the absolute path to the directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP one level and then DOWN into the folder
target_dir_data = os.path.join(current_file_dir, '..', 'data')
target_dir_models = os.path.join(current_file_dir, '..', 'src/models')

# 3. Add this target directory to Python's search path
sys.path.append(target_dir_data)
sys.path.append(target_dir_models)

# 4. Now you can import as if it were in the same folder
from data_loader import load_manifest, create_dataset
from atomicvad import build_atomicvad_model


def train_model(config: TrainingConfig, seed: int, fold_idx: int):
    """
    Train AtomicVAD model for a single fold.
    
    Args:
        config: Training configuration object
        seed: Random seed for reproducibility
        fold_idx: Fold index for organizing checkpoints
    """
    # Setup environment
    tf.keras.backend.clear_session()
    set_seed(seed)
    configure_gpu()
    
    # Load data manifests
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1} with seed {seed}")
    print(f"{'='*60}\n")
    
    train_json, val_json, test_json = load_manifest(
        config.speech_train_json,
        config.background_train_json,
        config.speech_val_json,
        config.background_val_json,
        config.speech_test_json,
        config.background_test_json
    )
    
    noise_json = load_manifest(
        background_train_json=config.background_train_json,
        return_noise_only=True
    )

    train_json = train_json[:100]
    val_json = val_json[:50]
    test_json = test_json[:50]
    noise_json = noise_json[:50]
    
    print(f'Training samples: {len(train_json):,}')
    print(f'Validation samples: {len(val_json):,}')
    print(f'Test samples: {len(test_json):,}\n')
    
    # Create datasets
    train_ds = create_dataset(
        train_json, 
        noise_json * 2,  # Duplicate noise samples
        config.batch_size, 
        config.sample_rate,
        train=True,
        seed=seed
    )
    
    val_ds = create_dataset(
        val_json,
        noise_json[:len(val_json)],
        config.batch_size,
        config.sample_rate,
        train=False,
        seed=seed
    )
    
    test_ds = create_dataset(
        test_json,
        noise_json[:len(test_json)],
        config.batch_size,
        config.sample_rate,
        train=False,
        seed=seed
    )
    
    # Build model
    model = build_atomicvad_model(config, seed)
    model.summary()
    
    # Setup training
    number_of_samples = len(train_json)
    total_steps = int(config.epochs * (number_of_samples // config.batch_size))
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.initial_lr,
        first_decay_steps=1,#int(0.1 * total_steps),
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        name="CosineDecayRestarts"
    )
    
    # Optimizer and loss
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)
    
    loss_func = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,
        gamma=2.0,
        from_logits=False
    )
    
    metrics = [
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.F1Score(average='micro', threshold=0.5, name='f1_score'),
        tf.keras.metrics.FBetaScore(average='micro', beta=2.0, threshold=0.5, name='f2_score'),
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        jit_compile=True,
        metrics=metrics
    )
    
    # Callbacks
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f'fold_{fold_idx}',
        'epoch_{epoch:03d}_auc_{val_auc:.4f}.keras'
    )
    
    backup_dir = os.path.join(config.backup_dir, f'fold_{fold_idx}')
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=config.early_stopping_patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.checkpoint_dir, f'fold_{fold_idx}', 'training_log.csv')
        ),
    ]
    
    # Train model
    print(f"\nStarting training for {config.epochs} epochs...\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}\n")
    
    test_results = model.evaluate(test_ds, verbose=1)
    
    print(f"\nTest Results - Fold {fold_idx + 1}:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")
    
    return history, test_results


def main():
    """Main training function."""
    # Enable mixed precision training
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    # Load configuration
    config = TrainingConfig()
    
    # Seeds for 10-run
    seeds = [43185, 86648, 92813, 89816, 69375, 47845, 10357, 37555, 79511, 78417]
    
    all_test_results = []
    
    # Train on each seed
    for fold_idx, seed in enumerate(seeds):
        try:
            history, test_results = train_model(config, seed, fold_idx)
            all_test_results.append(test_results)
            
        except Exception as e:
            print(f"\nError training fold {fold_idx + 1}: {str(e)}")
            raise
    
    # Print aggregate results
    print(f"\n{'='*60}")
    print("Aggregate Test Results Across All Seeds:")
    print(f"{'='*60}\n")
    
    metric_names = ['loss', 'auc', 'binary_accuracy', 'f1_score', 'f2_score']
    results_array = np.array(all_test_results)
    
    for idx, metric_name in enumerate(metric_names):
        mean_val = np.mean(results_array[:, idx])
        std_val = np.std(results_array[:, idx])
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()