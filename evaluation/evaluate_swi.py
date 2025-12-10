"""
SWI (Sliding Window Inference) Evaluation Protocol

This protocol evaluates VAD models using overlapping windows (87.5% overlap),
providing fine-grained frame-level predictions. More computationally intensive
but provides better temporal resolution.

Useful for:
- Fine-grained temporal analysis
- Real-world streaming scenarios
- Detailed performance evaluation

Usage:
    python evaluate_swi.py \
        --model_path experiments/atomicvad/checkpoint0/best_model.keras \
        --manifest data/manifest/ava_test_manifest.json \
        --output_csv results/swi_metrics.csv \
        --overlap 0.875
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

# 1. Determine the absolute path to the directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP one level and then DOWN into the folder
target_dir_models = os.path.join(current_file_dir, '..', 'src/models')

# 3. Add this target directory to Python's search path
sys.path.append(target_dir_models)

# 4. Now you can import as if it were in the same folder
from atomicvad import GGCU, Spectrogram, SpecAugment, SpecCutout

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from tqdm import tqdm

from eval_utils import (
    overlapping_segments,
    calculate_overall_metrics,
    calculate_per_class_metrics,
    print_metrics
)

from typing import Dict

def process_audio_with_sliding_window(
    row: pd.Series,
    model: tf.keras.Model,
    percent_overlap: float = 0.875,
    segment_duration: float = 0.63,
    sr: int = 16000,
    batch_size: int = 256
) -> pd.Series:
    """
    Process audio using sliding window with overlap.
    
    Creates overlapping windows and aggregates predictions across all windows.
    
    Args:
        row: DataFrame row with audio metadata
        model: Trained VAD model
        percent_overlap: Overlap percentage (0.875 = 87.5%)
        segment_duration: Window duration in seconds
        sr: Sample rate
        batch_size: Batch size for prediction
        
    Returns:
        Series with predictions and ground truth
    """
    offset = row['offset']
    duration = row['duration']
    audio_path = row['audio_filepath']
    
    # Load audio segment
    audio, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=duration)
    
    # Create overlapping segments
    segments = overlapping_segments(
        audio,
        percent_overlap=percent_overlap,
        segment_duration=segment_duration,
        sr=sr
    )
    
    # Get predictions for all segments
    predictions = model.predict(segments, verbose=0, batch_size=batch_size)
    speech_probs = predictions[:, 1]  # Speech probability
    
    # Aggregate predictions
    prediction_median = 1 if np.median(speech_probs) >= 0.5 else 0
    prediction_mean = 1 if np.mean(speech_probs) >= 0.5 else 0
    
    # Ground truth
    ground_truth = 1 if row['label'] not in ["NO_SPEECH", "non-speech"] else 0
    
    return pd.Series({
        'predicted_median': prediction_median,
        'predicted_mean': prediction_mean,
        'label': row['label'],
        'ground_truth': ground_truth,
        'num_windows': len(segments)
    })


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load manifest file into DataFrame.
    
    Args:
        manifest_path: Path to JSON manifest file
        
    Returns:
        DataFrame with audio metadata
    """
    data = []
    with open(manifest_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # Filter segments that are long enough
    df = df[df['duration'] >= 0.63]
    df = df.reset_index(drop=True)
    
    return df


def evaluate_swi(
    model_path: str,
    manifest_path: str,
    output_csv: str = None,
    overlap: float = 0.875,
    batch_size: int = 256
) -> Dict:
    """
    Evaluate model using SWI protocol.
    
    Args:
        model_path: Path to trained model
        manifest_path: Path to evaluation manifest
        output_csv: Path to save results (optional)
        overlap: Overlap percentage for sliding windows
        batch_size: Batch size for prediction
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("SWI (Sliding Window Inference) Evaluation")
    print(f"Overlap: {overlap * 100:.1f}%")
    print("=" * 60)
    
    # Load manifest
    print(f"\nLoading manifest: {manifest_path}")
    df = load_manifest(manifest_path)
    print(f"Total segments: {len(df):,}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path,
                                    custom_objects={'GGCU': GGCU,
                                        'Spectrogram': Spectrogram,
                                        'SpecAugment': SpecAugment,
                                        'SpecCutout': SpecCutout}, compile=False)
    
    # Process all segments with sliding windows
    print("\nProcessing audio with sliding windows...")
    print(f"This may take longer than SPI due to overlap...")
    
    tqdm.pandas()
    results = df.progress_apply(
        lambda x: process_audio_with_sliding_window(
            x,
            model=model,
            percent_overlap=overlap,
            batch_size=batch_size
        ),
        axis=1
    )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Print statistics about window counts
    avg_windows = results['num_windows'].mean()
    print(f"\nAverage windows per segment: {avg_windows:.1f}")
    
    # Overall binary metrics
    overall_metrics = calculate_overall_metrics(results)
    print_metrics(overall_metrics, "Overall Binary Metrics (Speech vs. Non-Speech)")
    
    # Per-class metrics (if multiple speech classes exist)
    unique_labels = results['label'].unique()
    if len(unique_labels) > 2:
        per_class_metrics = calculate_per_class_metrics(results)
        print_metrics(per_class_metrics, "Per-Class Metrics")
    else:
        per_class_metrics = None
    
    # Save results if requested
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten metrics for CSV
        flat_metrics = {
            'model': str(model_path),
            'overlap': overlap,
            'avg_windows_per_segment': float(avg_windows),
            **{f'overall_{k}': v for k, v in overall_metrics.items()}
        }
        
        if per_class_metrics:
            for cls, cls_metrics in per_class_metrics.items():
                for agg_method, metrics in cls_metrics.items():
                    for metric_name, value in metrics.items():
                        flat_metrics[f'{cls}_{agg_method}_{metric_name}'] = value
        
        df_results = pd.DataFrame([flat_metrics])
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    return {
        'overall': overall_metrics,
        'per_class': per_class_metrics,
        'avg_windows': float(avg_windows)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VAD model using SWI protocol'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model (.keras file)'
    )
    parser.add_argument(
        '--manifest_path',
        type=str,
        required=True,
        help='Path to evaluation manifest (.json file)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Path to save results CSV'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.875,
        help='Overlap percentage (default: 0.875 = 87.5%%)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for prediction (default: 256)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    tf.keras.utils.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Run evaluation
    results = evaluate_swi(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        output_csv=args.output_csv,
        overlap=args.overlap,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()