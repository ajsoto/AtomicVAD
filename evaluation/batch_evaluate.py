"""
Batch Evaluation Script for Multiple Models

Evaluates multiple trained models (e.g., from cross-validation folds)
on one or more datasets using either SPI or SWI protocol.

Usage:
    # Evaluate all models in a directory
    python batch_evaluate.py \
        --model_dir experiments/atomicvad/checkpoints/ \
        --manifest_path data/manifest/ava_test_manifest.json \
        --protocol swi \
        --output_csv results/batch_results.csv

    # Evaluate specific models
    python batch_evaluate.py \
        --model_paths model1.keras model2.keras model3.keras \
        --manifest_path data/manifest/ava_test_manifest.json \
        --protocol spi \
        --output_csv results/specific_models.csv
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
from pathlib import Path
from typing import List, Dict
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from evaluate_spi import evaluate_spi
from evaluate_swi import evaluate_swi
from eval_utils import aggregate_fold_results, print_metrics


def find_model_files(model_dir: str, pattern: str = "*.keras") -> List[str]:
    """
    Find all model files in a directory.
    
    Args:
        model_dir: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of model file paths
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")
    
    # Search recursively for model files
    model_files = list(model_dir.rglob(pattern))
    
    if not model_files:
        raise ValueError(f"No model files found in {model_dir} matching {pattern}")
    
    return [str(f) for f in sorted(model_files)]


def batch_evaluate(
    model_paths: List[str],
    manifest_path: str,
    protocol: str = 'swi',
    output_csv: str = None,
    overlap: float = 0.875,
    batch_size: int = 256
) -> pd.DataFrame:
    """
    Evaluate multiple models and aggregate results.
    
    Args:
        model_paths: List of model file paths
        manifest_path: Path to evaluation manifest
        protocol: 'spi' or 'swi'
        output_csv: Path to save results
        overlap: Overlap for SWI protocol
        batch_size: Batch size for predictions
        
    Returns:
        DataFrame with all results
    """
    print("\n" + "=" * 60)
    print(f"Batch Evaluation - {protocol.upper()} Protocol")
    print("=" * 60)
    print(f"Number of models: {len(model_paths)}")
    print(f"Evaluation dataset: {manifest_path}")
    print(f"Protocol: {protocol.upper()}")
    if protocol == 'swi':
        print(f"Overlap: {overlap * 100:.1f}%")
    print("=" * 60 + "\n")
    
    all_results = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n{'='*60}")
        print(f"Evaluating Model {i}/{len(model_paths)}")
        print(f"Model: {Path(model_path).name}")
        print(f"{'='*60}")
        
        try:
            # Choose evaluation protocol
            if protocol.lower() == 'spi':
                results = evaluate_spi(
                    model_path=model_path,
                    manifest_path=manifest_path,
                    output_csv=None  # Don't save individual results
                )
            elif protocol.lower() == 'swi':
                results = evaluate_swi(
                    model_path=model_path,
                    manifest_path=manifest_path,
                    output_csv=None,
                    overlap=overlap,
                    batch_size=batch_size
                )
            else:
                raise ValueError(f"Unknown protocol: {protocol}")
            
            # Flatten results for DataFrame
            flat_result = {
                'model': Path(model_path).name,
                'model_path': str(model_path),
                'fold': i - 1  # Assume sequential folds
            }
            
            # Add overall metrics
            for agg_method in ['Median', 'Mean']:
                if agg_method in results['overall']:
                    for metric, value in results['overall'][agg_method].items():
                        # Skip confusion matrix (too complex for CSV)
                        if 'Confusion' not in metric:
                            flat_result[f'{agg_method}_{metric}'] = value
            
            # Add per-class metrics if available
            if results.get('per_class'):
                for cls, cls_metrics in results['per_class'].items():
                    for agg_method, metrics in cls_metrics.items():
                        for metric_name, value in metrics.items():
                            if 'Confusion' not in metric_name:
                                flat_result[f'{cls}_{agg_method}_{metric_name}'] = value
            
            all_results.append(flat_result)
            
        except Exception as e:
            print(f"\n✗ Error evaluating {model_path}: {e}")
            continue
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    if df_results.empty:
        print("\n✗ No results collected!")
        return df_results
    
    # Calculate aggregate statistics
    print("\n" + "=" * 60)
    print("Aggregate Statistics Across All Models")
    print("=" * 60)
    
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['fold']]
    
    stats = {
        'metric': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    
    for col in numeric_cols:
        stats['metric'].append(col)
        stats['mean'].append(df_results[col].mean())
        stats['std'].append(df_results[col].std())
        stats['min'].append(df_results[col].min())
        stats['max'].append(df_results[col].max())
    
    df_stats = pd.DataFrame(stats)
    
    # Print key metrics
    key_metrics = [
        'Median_AUROC',
        'Mean_AUROC',
        'Median_F1',
        'Mean_F1',
        'Median_F2',
        'Mean_F2'
    ]
    
    print("\nKey Metrics:")
    print("-" * 60)
    for metric in key_metrics:
        matching = [m for m in stats['metric'] if metric in m]
        for m in matching:
            idx = stats['metric'].index(m)
            print(f"{m:40s}: {stats['mean'][idx]:.4f} ± {stats['std'][idx]:.4f}")
    print("-" * 60)
    
    # Save results
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df_results.to_csv(output_path, index=False)
        print(f"\n✓ Detailed results saved to: {output_path}")
        
        # Save aggregate statistics
        stats_path = output_path.parent / f"{output_path.stem}_stats.csv"
        df_stats.to_csv(stats_path, index=False)
        print(f"✓ Aggregate statistics saved to: {stats_path}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate multiple VAD models'
    )
    
    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model_dir',
        type=str,
        help='Directory containing model files'
    )
    model_group.add_argument(
        '--model_paths',
        type=str,
        nargs='+',
        help='List of specific model paths'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--manifest_path',
        type=str,
        required=True,
        help='Path to evaluation manifest'
    )
    parser.add_argument(
        '--protocol',
        type=str,
        choices=['spi', 'swi'],
        default='swi',
        help='Evaluation protocol (default: swi)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to save results CSV'
    )
    
    # SWI-specific parameters
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.875,
        help='Overlap for SWI protocol (default: 0.875)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for predictions (default: 256)'
    )
    
    # Model selection
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.keras',
        help='File pattern for finding models (default: *.keras)'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
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
    
    # Get model paths
    if args.model_dir:
        model_paths = find_model_files(args.model_dir, args.pattern)
    else:
        model_paths = args.model_paths
    
    # Run batch evaluation
    results = batch_evaluate(
        model_paths=model_paths,
        manifest_path=args.manifest_path,
        protocol=args.protocol,
        output_csv=args.output_csv,
        overlap=args.overlap,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 60)
    print("✓ Batch evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()