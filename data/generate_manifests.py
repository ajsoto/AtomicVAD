"""
NeMo VAD Manifest Generation Toolkit

Supports multiple datasets for cross-dataset and multilingual evaluation:
- LibriSpeech (English)
- Multilingual LibriSpeech (MLS) - Multiple languages
- Common Voice - Multiple languages
- VoxCeleb (Speaker identification)
- MUSAN (Music, Speech, Noise)
- AMI Corpus
- Speech Activity Detection (SAD) Kaggle
- Custom datasets

Usage:
    python generate_manifests.py --dataset librispeech \
        --data_root /path/to/LibriSpeech \
        --output_dir ./manifests \
        --splits test-clean dev-clean
"""

import json
import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_textgrid(textgrid_path: str) -> List[Dict]:
    """
    Parse Praat TextGrid file to extract speech/non-speech intervals.
    
    Args:
        textgrid_path: Path to TextGrid file
        
    Returns:
        List of intervals with xmin, xmax, duration, and label
    """
    intervals = []
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all intervals in the file
    pattern = r'intervals \[(\d+)\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([01])"'
    matches = re.findall(pattern, content)
    
    for match in matches:
        interval_num, xmin, xmax, text = match
        intervals.append({
            'xmin': float(xmin),
            'xmax': float(xmax),
            'duration': float(xmax) - float(xmin),
            'label': 'speech' if text == '1' else 'non-speech'
        })
    
    return intervals


class ManifestGenerator:
    """Base class for generating NeMo-compatible manifest files."""
    
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_manifest_entry(
        self,
        audio_path: str,
        duration: float,
        label: str = "infer",
        text: str = "",
        offset: float = 0.0
    ) -> Dict:
        """Create a single manifest entry in NeMo format."""
        return {
            "audio_filepath": str(audio_path),
            "offset": offset,
            "duration": duration,
            "label": label,
            "text": text
        }
    
    def write_manifest(self, entries: List[Dict], output_file: str):
        """Write manifest entries to JSON file."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        logging.info(f"✓ Manifest saved: {output_path}")
        logging.info(f"  Total entries: {len(entries):,}")


class LibriSpeechManifest(ManifestGenerator):
    """Generate manifests for LibriSpeech dataset."""
    
    def generate(self, splits: List[str] = ['dev-clean', 'test-clean']):
        """
        LibriSpeech structure:
        LibriSpeech/
        ├── dev-clean/
        ├── test-clean/
        └── train-clean-100/
        """
        for split in splits:
            entries = []
            split_path = self.data_root / split
            
            if not split_path.exists():
                logging.warning(f"Path not found: {split_path}, skipping...")
                continue
            
            # LibriSpeech structure: split/speaker_id/chapter_id/*.flac
            audio_files = list(split_path.rglob("*.flac"))
            
            for audio_path in tqdm(audio_files, desc=f"Processing LibriSpeech {split}"):
                try:
                    duration = librosa.get_duration(path=str(audio_path))
                    
                    # Get corresponding transcript
                    trans_file = audio_path.parent / f"{audio_path.parent.name}.trans.txt"
                    text = ""
                    
                    if trans_file.exists():
                        with open(trans_file, 'r') as f:
                            for line in f:
                                if line.startswith(audio_path.stem):
                                    text = line.split(' ', 1)[1].strip()
                                    break
                    
                    entry = self.create_manifest_entry(
                        audio_path=str(audio_path),
                        duration=duration,
                        label="speech",
                        text=text
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
            
            self.write_manifest(entries, f"librispeech_{split}_manifest.json")


class MultilingualLibriSpeechManifest(ManifestGenerator):
    """Generate manifests for Multilingual LibriSpeech (MLS)."""
    
    def generate(self, language: str, splits: List[str] = ['dev', 'test']):
        """
        MLS structure:
        mls_{language}/
        ├── dev/
        ├── test/
        └── train/
        """
        for split in splits:
            entries = []
            split_path = self.data_root / f"mls_{language}" / split
            
            if not split_path.exists():
                logging.warning(f"Path not found: {split_path}, skipping...")
                continue
            
            audio_files = list(split_path.rglob("*.flac"))
            
            for audio_path in tqdm(audio_files, desc=f"Processing MLS {language} {split}"):
                try:
                    duration = librosa.get_duration(path=str(audio_path))
                    
                    # Try to find transcript
                    trans_file = audio_path.parent / "transcripts.txt"
                    text = ""
                    
                    if trans_file.exists():
                        with open(trans_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split('\t')
                                if len(parts) >= 2 and parts[0] == audio_path.stem:
                                    text = parts[1]
                                    break
                    
                    entry = self.create_manifest_entry(
                        audio_path=str(audio_path),
                        duration=duration,
                        label="speech",
                        text=text
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
            
            self.write_manifest(entries, f"mls_{language}_{split}_manifest.json")


class MUSANManifest(ManifestGenerator):
    """Generate manifests for MUSAN corpus (Music, Speech, and Noise)."""
    
    def generate(self, categories: List[str] = ['speech', 'music', 'noise']):
        """
        MUSAN structure:
        musan/
        ├── music/
        ├── speech/
        │   ├── librivox/
        │   └── us-gov/
        └── noise/
            ├── free-sound/
            └── sound-bible/
        """
        all_entries = []
        
        for category in categories:
            category_path = self.data_root / category
            
            if not category_path.exists():
                logging.warning(f"Path not found: {category_path}, skipping...")
                continue
            
            entries = []
            audio_files = list(category_path.rglob("*.wav"))
            
            # Determine label based on category
            label = 'speech' if category == 'speech' else 'non-speech'
            
            for audio_path in tqdm(audio_files, desc=f"Processing MUSAN {category}"):
                try:
                    duration = librosa.get_duration(path=str(audio_path))
                    
                    entry = self.create_manifest_entry(
                        audio_path=str(audio_path),
                        duration=duration,
                        label=label
                    )
                    entries.append(entry)
                    all_entries.append(entry)
                    
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
            
            # Save per-category manifest
            self.write_manifest(entries, f"musan_{category}_manifest.json")
        
        # Generate combined manifest for VAD evaluation
        if all_entries:
            logging.info("\nGenerating combined MUSAN manifest for VAD evaluation...")
            self.write_manifest(all_entries, "musan_combined_manifest.json")


class VoxCelebManifest(ManifestGenerator):
    """Generate manifests for VoxCeleb dataset."""
    
    def generate(self, dataset_version: str = "vox1"):
        """
        VoxCeleb structure:
        vox1_dev_wav/wav/ or vox2_dev_aac/aac/
        └── id*/
            └── video_id/
                └── *.wav or *.m4a
        """
        entries = []
        
        # Determine file extension
        if dataset_version == "vox1":
            search_pattern = "**/*.wav"
        elif dataset_version == "vox2":
            search_pattern = "**/*.m4a"
        else:
            search_pattern = "**/*.*"
        
        audio_files = list(self.data_root.rglob(search_pattern))
        
        for audio_path in tqdm(audio_files, desc=f"Processing VoxCeleb {dataset_version}"):
            try:
                duration = librosa.get_duration(path=str(audio_path))
                
                entry = self.create_manifest_entry(
                    audio_path=str(audio_path),
                    duration=duration,
                    label="speech"
                )
                entries.append(entry)
                
            except Exception as e:
                logging.error(f"Error processing {audio_path}: {e}")
        
        self.write_manifest(entries, f"voxceleb_{dataset_version}_manifest.json")


class CommonVoiceManifest(ManifestGenerator):
    """Generate manifests for Mozilla Common Voice dataset."""
    
    def generate(self, language_code: str, version: str = "cv-corpus-12.0-2022-12-07"):
        """
        Common Voice structure:
        cv-corpus-{version}/
        └── {language_code}/
            ├── clips/
            ├── train.tsv
            ├── dev.tsv
            └── test.tsv
        """
        lang_path = self.data_root / version / language_code
        
        if not lang_path.exists():
            logging.error(f"Path not found: {lang_path}")
            return
        
        clips_dir = lang_path / "clips"
        
        for split in ['train', 'dev', 'test']:
            tsv_file = lang_path / f"{split}.tsv"
            
            if not tsv_file.exists():
                logging.warning(f"File not found: {tsv_file}, skipping...")
                continue
            
            entries = []
            df = pd.read_csv(tsv_file, sep='\t')
            
            for _, row in tqdm(df.iterrows(), total=len(df), 
                             desc=f"Processing Common Voice {language_code} {split}"):
                audio_file = clips_dir / row['path']
                
                if not audio_file.exists():
                    continue
                
                try:
                    duration = librosa.get_duration(path=str(audio_file))
                    
                    entry = self.create_manifest_entry(
                        audio_path=str(audio_file),
                        duration=duration,
                        label="speech",
                        text=row.get('sentence', '')
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    logging.error(f"Error processing {audio_file}: {e}")
            
            self.write_manifest(
                entries,
                f"common_voice_{language_code}_{split}_manifest.json"
            )


class SpeechActivityDetectionManifest(ManifestGenerator):
    """
    Generate manifests for Speech Activity Detection (SAD) Kaggle dataset.
    
    Expected structure:
        KaggleDataset/
        ├── Annotation/
        │   ├── Male/
        │   ├── Female/
        │   └── Noizeus/
        └── Audio/
            (mirrors the same structure)
    """
    
    def generate(self):
        entries_by_category = {}
        annotations_root = self.data_root / "Annotation"
        audio_root = self.data_root / "Audio"
        
        # Process all categories
        for category_dir in annotations_root.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            entries_by_category[category_name] = []
            
            # Find all TextGrid files
            for textgrid_path in category_dir.rglob("*.TextGrid"):
                # Derive corresponding audio path
                rel_path = textgrid_path.relative_to(annotations_root)
                audio_path = audio_root / rel_path.with_suffix(".wav")
                
                if not audio_path.exists():
                    logging.warning(f"Missing audio: {audio_path}")
                    continue
                
                try:
                    intervals = parse_textgrid(str(textgrid_path))
                    
                    for interval in intervals:
                        entry = self.create_manifest_entry(
                            audio_path=str(audio_path),
                            duration=interval["duration"],
                            label=interval["label"],
                            offset=interval["xmin"]
                        )
                        entries_by_category[category_name].append(entry)
                        
                except Exception as e:
                    logging.error(f"Error processing {textgrid_path}: {e}")
        
        # Write per-category manifests
        for category, entries in entries_by_category.items():
            if entries:
                self.write_manifest(entries, f"sad_{category.lower()}_manifest.json")
        
        # Generate combined manifest
        all_entries = [e for entries in entries_by_category.values() for e in entries]
        if all_entries:
            self.write_manifest(all_entries, "sad_combined_manifest.json")


class CustomVADManifest(ManifestGenerator):
    """Generate manifests for any custom audio dataset."""
    
    def generate(self, audio_extensions: List[str] = ['.wav', '.flac', '.mp3', '.m4a']):
        """Process any directory structure with audio files."""
        entries = []
        
        for ext in audio_extensions:
            audio_files = list(self.data_root.rglob(f"*{ext}"))
            
            for audio_path in tqdm(audio_files, desc=f"Processing {ext} files"):
                try:
                    duration = librosa.get_duration(path=str(audio_path))
                    
                    entry = self.create_manifest_entry(
                        audio_path=str(audio_path),
                        duration=duration,
                        label="infer"
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    logging.error(f"Error processing {audio_path}: {e}")
        
        self.write_manifest(entries, "custom_manifest.json")


def main():
    parser = argparse.ArgumentParser(
        description='Generate NeMo VAD manifests for various datasets'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['librispeech', 'common_voice', 'voxceleb', 'mls', 
                'musan', 'sad_kaggle', 'custom'],
        help='Dataset type'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory of the dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./manifests',
        help='Output directory for manifest files'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code (for Common Voice and MLS)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['dev', 'test'],
        help='Dataset splits to process'
    )
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info(f"Generating manifests for: {args.dataset}")
    logging.info("=" * 60)
    
    # Generate manifests based on dataset type
    if args.dataset == 'librispeech':
        generator = LibriSpeechManifest(args.data_root, args.output_dir)
        generator.generate(splits=args.splits)
    
    elif args.dataset == 'common_voice':
        generator = CommonVoiceManifest(args.data_root, args.output_dir)
        generator.generate(language_code=args.language)
    
    elif args.dataset == 'voxceleb':
        generator = VoxCelebManifest(args.data_root, args.output_dir)
        generator.generate(dataset_version='vox1')
    
    elif args.dataset == 'mls':
        generator = MultilingualLibriSpeechManifest(args.data_root, args.output_dir)
        generator.generate(language=args.language, splits=args.splits)
    
    elif args.dataset == 'musan':
        generator = MUSANManifest(args.data_root, args.output_dir)
        generator.generate()
    
    elif args.dataset == 'sad_kaggle':
        generator = SpeechActivityDetectionManifest(args.data_root, args.output_dir)
        generator.generate()
    
    elif args.dataset == 'custom':
        generator = CustomVADManifest(args.data_root, args.output_dir)
        generator.generate()
    
    logging.info("\n" + "=" * 60)
    logging.info("✓ Manifest generation complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()