"""
Voice Activity Detection (VAD) Data Processing Script

This script prepares training data for VAD by:
1. Downloading and extracting Google Speech Commands V2
2. Splitting speech and background data into train/val/test sets
3. Segmenting audio files into fixed-length windows
4. Creating balanced manifest files for training

Usage:
    python process_vad_data.py \
        --out_dir=data/manifest/ \
        --speech_data_root=/path/to/speech_data \
        --background_data_root=/path/to/background_data \
        --rebalance_method='fixed' \
        --log

Original source: NVIDIA NeMo Toolkit
Modified for AtomicVAD project
"""

import argparse
import glob
import json
import logging
import os
import tarfile
import urllib.request
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# Constants
SAMPLE_RATE = 16000
GOOGLE_SPEECH_COMMANDS_V2_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def maybe_download_file(destination: str, source: str) -> str:
    """
    Download source to destination if it doesn't exist.
    
    Args:
        destination: Local filepath
        source: URL of resource
        
    Returns:
        Path to downloaded file
    """
    if not os.path.exists(destination):
        logging.info(f"{destination} does not exist. Downloading...")
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info(f"Downloaded {destination}")
    else:
        logging.info(f"Destination {destination} exists. Skipping download.")
    return destination


def extract_file(filepath: str, data_dir: str):
    """Extract tar archive to specified directory."""
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
        logging.info(f"Extracted {filepath} to {data_dir}")
    except Exception as e:
        logging.warning(f'Extraction failed or already extracted: {e}')


def extract_all_files(filepath: str, data_root: str, data_dir: str):
    """Extract archive if destination doesn't exist."""
    if not os.path.exists(data_dir):
        extract_file(filepath, data_root)
    else:
        logging.info(f'Skipping extraction. Data already exists at {data_dir}')


def split_train_val_test(
    data_dir: str,
    file_type: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    demo: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        data_dir: Directory containing audio files
        file_type: 'speech' or 'background'
        test_size: Proportion of test set
        val_size: Proportion of validation set
        demo: If True, use small subset for demonstration
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    X = []
    
    if file_type == "speech":
        # Get all speech files except background noise
        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path) and subdir != "_background_noise_":
                X.extend(glob.glob(os.path.join(subdir_path, '*.wav')))
        
        if demo:
            logging.info(
                f"DEMO MODE: Using {int(len(X)/100)}/{len(X)} speech samples. "
                "Remove --demo flag for full training!"
            )
            X = np.random.choice(X, int(len(X) / 100), replace=False)
    
    else:  # background
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                X.extend(glob.glob(os.path.join(item_path, '*.wav')))
            elif item.endswith(".wav"):
                # Include background noise from Google Speech Commands
                X.append(item_path)
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val = train_test_split(X_train, test_size=val_size_adjusted, random_state=42)
    
    # Save file lists
    list_files = {
        'training': X_train,
        'validation': X_val,
        'testing': X_test
    }
    
    for split_name, files in list_files.items():
        output_path = os.path.join(data_dir, f"{file_type}_{split_name}_list.txt")
        with open(output_path, "w") as f:
            f.write("\n".join(files))
    
    logging.info(
        f'{file_type.capitalize()}: Total={len(X)}, '
        f'Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}'
    )
    
    return X_train, X_val, X_test


def write_manifest(
    out_dir: str,
    files: List[str],
    prefix: str,
    manifest_name: str,
    start: float = 0.0,
    end: float = None,
    duration_stride: float = 1.0,
    duration_max: float = None,
    duration_limit: float = 100.0,
    filter_long: bool = True
) -> Tuple[int, int, str]:
    """
    Segment audio files and write manifest.
    
    Args:
        out_dir: Output directory for manifest
        files: List of audio files to process
        prefix: Label prefix ('speech' or 'background')
        manifest_name: Name of output manifest
        start: Start time for segmentation
        end: End time for segmentation
        duration_stride: Stride between segments
        duration_max: Maximum duration per segment
        duration_limit: Filter out files longer than this
        filter_long: Whether to filter long files
        
    Returns:
        Tuple of (skipped_count, segment_count, output_path)
    """
    seg_count = 0
    skip_count = 0
    
    if duration_max is None:
        duration_max = 1e9
    
    os.makedirs(out_dir, exist_ok=True)
    
    output_path = os.path.join(out_dir, manifest_name + '.json')
    
    with open(output_path, 'w') as fout:
        for file_path in files:
            label = prefix
            
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                duration = librosa.get_duration(y=audio, sr=sr)
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
                continue
            
            # Filter out long files
            if filter_long and duration > duration_limit:
                skip_count += 1
                continue
            
            # Generate segments
            offsets = []
            durations = []
            
            if duration > duration_max:
                current_offset = start
                
                while current_offset < duration:
                    if end is not None and current_offset > end:
                        break
                    
                    remaining = duration - current_offset
                    if remaining < duration_max:
                        break
                    
                    offsets.append(current_offset)
                    durations.append(duration_max)
                    current_offset += duration_stride
            else:
                # Duration too short, skip
                skip_count += 1
                continue
            
            # Write segments to manifest
            for dur, offset in zip(durations, offsets):
                metadata = {
                    'audio_filepath': file_path,
                    'duration': dur,
                    'label': label,
                    'text': '_',  # Compatibility with ASR
                    'offset': offset,
                }
                json.dump(metadata, fout)
                fout.write('\n')
                seg_count += 1
    
    return skip_count, seg_count, output_path


def load_list_write_manifest(
    data_dir: str,
    out_dir: str,
    filename: str,
    prefix: str,
    start: float,
    end: float,
    duration_stride: float = 1.0,
    duration_max: float = 1.0,
    duration_limit: float = 100.0,
    filter_long: bool = True
) -> Tuple[int, int, str]:
    """
    Load file list and create manifest.
    
    Args:
        data_dir: Directory containing file list
        out_dir: Output directory for manifest
        filename: Name of file list (e.g., 'training_list.txt')
        prefix: Label prefix
        start: Start time for segmentation
        end: End time for segmentation
        duration_stride: Stride between segments
        duration_max: Maximum segment duration
        duration_limit: Filter threshold
        filter_long: Whether to filter long files
        
    Returns:
        Tuple of (skip_count, segment_count, output_path)
    """
    list_filename = f"{prefix}_{filename}"
    list_path = os.path.join(data_dir, list_filename)
    
    with open(list_path, 'r') as f:
        files = f.read().splitlines()
    
    manifest_name = list_filename.replace('_list.txt', '_manifest')
    
    return write_manifest(
        out_dir,
        files,
        prefix,
        manifest_name,
        start,
        end,
        duration_stride,
        duration_max,
        duration_limit,
        filter_long
    )


def rebalance_manifest(
    data_dir: str,
    manifest_path: str,
    num_samples: int,
    prefix: str
) -> str:
    """
    Rebalance manifest to specified number of samples.
    
    Args:
        data_dir: Output directory
        manifest_path: Path to input manifest
        num_samples: Target number of samples
        prefix: Prefix for output filename
        
    Returns:
        Path to rebalanced manifest
    """
    # Load manifest
    data = []
    with open(manifest_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Sample with or without replacement
    if len(data) >= num_samples:
        selected = np.random.choice(data, num_samples, replace=False)
    else:
        selected = np.random.choice(data, num_samples, replace=True)
    
    # Write rebalanced manifest
    filename = os.path.basename(manifest_path)
    output_path = os.path.join(data_dir, f"{prefix}_{filename}")
    
    with open(output_path, 'w') as fout:
        for sample in selected:
            json.dump(sample, fout)
            fout.write('\n')
    
    logging.info(f'Rebalanced: {len(selected)}/{num_samples} samples -> {output_path}')
    return output_path


def generate_background_variations(
    data_dir: str,
    filename: str,
    prefix: str
) -> int:
    """
    Generate additional background noise variations.
    
    Creates multiple segments with random magnitude variations
    to increase background data diversity.
    
    Args:
        data_dir: Directory containing background files
        filename: File list name
        prefix: Label prefix
        
    Returns:
        Number of generated files
    """
    base_dir = data_dir.rstrip("_background_noise_")
    output_dir = os.path.join(base_dir, "_background_noise_more")
    os.makedirs(output_dir, exist_ok=True)
    
    stride_samples = 1000  # 1/16 seconds
    rng = np.random.RandomState(0)
    
    list_path = os.path.join(data_dir, f"{prefix}_{filename}")
    
    with open(list_path, 'r') as f:
        files = f.read().splitlines()
    
    generated_files = []
    
    for file_path in files:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Generate segments with stride
        for i in range(0, len(audio) - SAMPLE_RATE, stride_samples * 100):
            segment = audio[i:i + SAMPLE_RATE]
            
            # Apply random magnitude
            magnitude = rng.uniform(0.0, 1.0)
            segment *= magnitude
            
            # Save segment
            filename_out = f"{os.path.basename(file_path)}_{i}.wav"
            output_path = os.path.join(output_dir, filename_out)
            sf.write(output_path, segment, sr)
            generated_files.append(output_path)
    
    # Write new file list
    new_list_path = os.path.join(output_dir, f"{prefix}_{filename}")
    with open(new_list_path, "w") as f:
        f.write("\n".join(generated_files))
    
    logging.info(
        f"Generated {len(generated_files)} background variations "
        f"from {len(files)} files -> {new_list_path}"
    )
    
    return len(generated_files)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare VAD training data from speech and background sources'
    )
    
    # Required arguments
    parser.add_argument(
        '--speech_data_root',
        required=True,
        type=str,
        help='Path to speech data directory'
    )
    parser.add_argument(
        '--background_data_root',
        required=True,
        type=str,
        help='Path to background data directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--out_dir',
        default='./manifest/',
        type=str,
        help='Output directory for manifest files'
    )
    parser.add_argument(
        '--test_size',
        default=0.1,
        type=float,
        help='Proportion of test set (0.0-1.0)'
    )
    parser.add_argument(
        '--val_size',
        default=0.1,
        type=float,
        help='Proportion of validation set (0.0-1.0)'
    )
    parser.add_argument(
        '--window_length_in_sec',
        default=0.63,
        type=float,
        help='Segment window length in seconds'
    )
    parser.add_argument(
        '--rebalance_method',
        choices=['over', 'under', 'fixed'],
        default=None,
        type=str,
        help='Rebalancing method: over/under/fixed'
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Use small subset for demonstration'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    logging.info("=" * 60)
    logging.info("VAD Data Preparation Pipeline")
    logging.info("=" * 60)
    
    # Setup paths
    speech_data_root = args.speech_data_root
    dataset_name = "google_speech_recognition_v2"
    speech_data_folder = os.path.join(speech_data_root, dataset_name)
    background_data_folder = args.background_data_root
    
    # Download and extract speech data
    if not os.path.exists(speech_data_folder):
        archive_path = os.path.join(speech_data_root, dataset_name + ".tar.bz2")
        logging.info("Downloading Google Speech Commands V2...")
        maybe_download_file(archive_path, GOOGLE_SPEECH_COMMANDS_V2_URL)
        logging.info("Extracting Google Speech Commands V2...")
        extract_all_files(archive_path, speech_data_root, speech_data_folder)
    
    # Split datasets
    logging.info("\nSplitting speech data...")
    split_train_val_test(
        speech_data_folder,
        "speech",
        args.test_size,
        args.val_size,
        args.demo
    )
    
    logging.info("\nSplitting background data...")
    split_train_val_test(
        background_data_folder,
        "background",
        args.test_size,
        args.val_size
    )
    
    # Process speech manifests
    logging.info("\n" + "=" * 60)
    logging.info("Processing Speech Manifests")
    logging.info("=" * 60)
    
    window_length = args.window_length_in_sec
    
    speech_results = {}
    
    for split, (start, end, stride) in [
        ('validation', (0.2, 0.8, window_length)),
        ('testing', (0.2, 0.8, 0.01)),
        ('training', (0.2, 0.8, window_length))
    ]:
        skip, segments, path = load_list_write_manifest(
            speech_data_folder,
            args.out_dir,
            f'{split}_list.txt',
            'speech',
            start,
            end,
            stride,
            window_length
        )
        speech_results[split] = (skip, segments, path)
        logging.info(f'{split.capitalize()}: Skipped={skip}, Segments={segments}')
    
    # Process background manifests
    if args.demo:
        logging.info("\nGenerating background variations...")
        for split in ['validation', 'training', 'testing']:
            generate_background_variations(
                background_data_folder,
                f'{split}_list.txt',
                'background'
            )
        background_data_folder = os.path.join(
            background_data_folder.rstrip("_background_noise_"),
            "_background_noise_more"
        )
    
    logging.info("\n" + "=" * 60)
    logging.info("Processing Background Manifests")
    logging.info("=" * 60)
    
    background_results = {}
    
    for split, (start, end, stride) in [
        ('validation', (0, None, 0.15)),
        ('testing', (0, None, 0.01)),
        ('training', (0, None, 0.15))
    ]:
        skip, segments, path = load_list_write_manifest(
            background_data_folder,
            args.out_dir,
            f'{split}_list.txt',
            'background',
            start,
            end,
            stride,
            window_length
        )
        background_results[split] = (skip, segments, path)
        logging.info(f'{split.capitalize()}: Skipped={skip}, Segments={segments}')
    
    # Rebalance if requested
    if args.rebalance_method:
        logging.info("\n" + "=" * 60)
        logging.info(f"Rebalancing using '{args.rebalance_method}' method")
        logging.info("=" * 60)
        
        # Calculate target sizes
        if args.rebalance_method == 'fixed':
            targets = {'training': 80500, 'validation': 10500, 'testing': 30500}
        elif args.rebalance_method == 'under':
            targets = {
                split: min(speech_results[split][1], background_results[split][1])
                for split in ['training', 'validation', 'testing']
            }
        else:  # over
            targets = {
                split: max(speech_results[split][1], background_results[split][1])
                for split in ['training', 'validation', 'testing']
            }
        
        logging.info(f"Target sizes: {targets}")
        
        # Rebalance all manifests
        for split in ['training', 'validation', 'testing']:
            target = targets[split]
            
            # Rebalance speech
            rebalance_manifest(
                args.out_dir,
                speech_results[split][2],
                target,
                'balanced'
            )
            
            # Rebalance background
            rebalance_manifest(
                args.out_dir,
                background_results[split][2],
                target,
                'balanced'
            )
    
    logging.info("\n" + "=" * 60)
    logging.info("Data Preparation Complete!")
    logging.info("=" * 60)
    logging.info(f"Manifests saved to: {args.out_dir}")


if __name__ == '__main__':
    main()