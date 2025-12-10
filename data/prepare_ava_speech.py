"""
AVA-Speech Dataset Manifest Generator

Converts AVA-Speech CSV labels to NeMo-compatible manifest format.

AVA-Speech CSV Format:
- Column 0: video_id (e.g., '5BDj0ow5hnA')
- Column 1: start_time (e.g., 900.0)
- Column 2: end_time (e.g., 905.67)
- Column 3: label (e.g., 'SPEECH_WITH_NOISE', 'CLEAN_SPEECH', 'NO_SPEECH')

Output Manifest Format:
{
  "audio_filepath": "path/to/audio/video_id.wav",
  "offset": offset_from_start,
  "duration": segment_duration,
  "label": "SPEECH_WITH_NOISE",
  "text": "_"
}

Usage:
    python data/prepare_ava_speech.py \
        --csv_file data/ava_speech_labels_v1.csv \
        --audio_dir data/ava-speech/audio \
        --output_manifest data/manifest/ava_test_manifest.json \
        --min_duration 0.63
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import librosa
from tqdm import tqdm


def load_ava_csv(csv_file: str) -> pd.DataFrame:
    """
    Load AVA-Speech CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        DataFrame with columns: video_id, start_time, end_time, label
    """
    df = pd.read_csv(
        csv_file,
        names=['video_id', 'start_time', 'end_time', 'label'],
        dtype={
            'video_id': str,
            'start_time': float,
            'end_time': float,
            'label': str
        }
    )
    
    return df


def validate_audio_files(df: pd.DataFrame, audio_dir: str) -> pd.DataFrame:
    """
    Validate that audio files exist and filter out missing ones.
    
    Args:
        df: DataFrame with video_id column
        audio_dir: Directory containing audio files
        
    Returns:
        Filtered DataFrame with only valid entries
    """
    audio_dir = Path(audio_dir)
    
    print("\nValidating audio files...")
    valid_indices = []
    missing_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        audio_path = audio_dir / f"{row['video_id']}.wav"
        
        if audio_path.exists():
            valid_indices.append(idx)
        else:
            missing_files.append(str(audio_path))
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} audio files not found")
        if len(missing_files) <= 10:
            print("Missing files:")
            for f in missing_files:
                print(f"  - {f}")
        else:
            print(f"First 10 missing files:")
            for f in missing_files[:10]:
                print(f"  - {f}")
    
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    print(f"\n✓ Valid entries: {len(df_valid):,} / {len(df):,}")
    
    return df_valid


def create_manifest_entry(
    video_id: str,
    start_time: float,
    end_time: float,
    label: str,
    audio_dir: str
) -> Dict:
    """
    Create a single manifest entry.
    
    Args:
        video_id: Video identifier
        start_time: Segment start time (absolute)
        end_time: Segment end time (absolute)
        label: Speech label
        audio_dir: Directory containing audio files
        
    Returns:
        Manifest entry dictionary
    """
    # AVA-Speech uses absolute timestamps starting from 900s
    # We need to calculate offset from the actual audio file start
    offset = start_time - 900.0
    duration = end_time - start_time
    
    audio_path = os.path.join(audio_dir, f"{video_id}.wav")
    
    return {
        "audio_filepath": audio_path,
        "offset": offset,
        "duration": duration,
        "label": label,
        "text": "_"  # Compatibility with ASR systems
    }


def generate_ava_manifest(
    csv_file: str,
    audio_dir: str,
    output_manifest: str,
    min_duration: float = 0.63,
    validate_files: bool = True,
    filter_corrupted: bool = True
) -> int:
    """
    Generate AVA-Speech manifest from CSV labels.
    
    Args:
        csv_file: Path to AVA-Speech CSV file
        audio_dir: Directory containing audio files
        output_manifest: Output manifest path
        min_duration: Minimum segment duration
        validate_files: Check if audio files exist
        filter_corrupted: Remove entries that can't be loaded
        
    Returns:
        Number of entries written
    """
    print("=" * 60)
    print("AVA-Speech Manifest Generation")
    print("=" * 60)
    
    # Load CSV
    print(f"\nLoading CSV: {csv_file}")
    df = load_ava_csv(csv_file)
    print(f"Total entries in CSV: {len(df):,}")
    
    # Calculate durations
    df['duration'] = df['end_time'] - df['start_time']
    
    # Filter by minimum duration
    df_filtered = df[df['duration'] >= min_duration].copy()
    print(f"After duration filter (>= {min_duration}s): {len(df_filtered):,}")
    
    # Show label distribution
    print("\nLabel distribution:")
    print(df_filtered['label'].value_counts())
    
    # Validate audio files if requested
    if validate_files:
        df_filtered = validate_audio_files(df_filtered, audio_dir)
    
    # Filter corrupted entries if requested
    if filter_corrupted:
        print("\nFiltering potentially corrupted entries...")
        # Known corrupted entry in AVA-Speech
        corrupted_indices = []
        for idx, row in df_filtered.iterrows():
            # Check for known issues
            if idx == 26399:  # Known corrupted entry
                corrupted_indices.append(idx)
        
        if corrupted_indices:
            df_filtered = df_filtered.drop(corrupted_indices).reset_index(drop=True)
            print(f"Removed {len(corrupted_indices)} corrupted entries")
    
    # Create output directory
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate manifest entries
    print(f"\nGenerating manifest entries...")
    entries_written = 0
    
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Writing manifest"):
            entry = create_manifest_entry(
                video_id=row['video_id'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                label=row['label'],
                audio_dir=audio_dir
            )
            
            f.write(json.dumps(entry) + '\n')
            entries_written += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Manifest generated: {output_manifest}")
    print(f"  Total entries: {entries_written:,}")
    print("=" * 60)
    
    return entries_written


def analyze_manifest(manifest_path: str):
    """
    Analyze and display statistics about the generated manifest.
    
    Args:
        manifest_path: Path to manifest file
    """
    print("\n" + "=" * 60)
    print("Manifest Analysis")
    print("=" * 60)
    
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
    
    df = pd.DataFrame(entries)
    
    print(f"\nTotal entries: {len(df):,}")
    print(f"\nDuration statistics:")
    print(f"  Mean: {df['duration'].mean():.2f}s")
    print(f"  Median: {df['duration'].median():.2f}s")
    print(f"  Min: {df['duration'].min():.2f}s")
    print(f"  Max: {df['duration'].max():.2f}s")
    
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nUnique videos: {df['audio_filepath'].nunique():,}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate AVA-Speech manifest from CSV labels'
    )
    
    parser.add_argument(
        '--csv_file',
        type=str,
        required=True,
        help='Path to AVA-Speech CSV file (e.g., ava_speech_labels_v1.csv)'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help='Directory containing AVA-Speech audio files (.wav)'
    )
    parser.add_argument(
        '--output_manifest',
        type=str,
        default='data/manifest/ava_test_manifest.json',
        help='Output manifest file path'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=0.63,
        help='Minimum segment duration in seconds (default: 0.63)'
    )
    parser.add_argument(
        '--no_validate',
        action='store_true',
        help='Skip audio file validation'
    )
    parser.add_argument(
        '--no_filter_corrupted',
        action='store_true',
        help='Keep potentially corrupted entries'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze manifest after generation'
    )
    
    args = parser.parse_args()
    
    # Generate manifest
    num_entries = generate_ava_manifest(
        csv_file=args.csv_file,
        audio_dir=args.audio_dir,
        output_manifest=args.output_manifest,
        min_duration=args.min_duration,
        validate_files=not args.no_validate,
        filter_corrupted=not args.no_filter_corrupted
    )
    
    # Analyze if requested
    if args.analyze:
        analyze_manifest(args.output_manifest)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()