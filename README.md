# Audio Feature Extraction Pipeline for Pitt Corpus

A complete Python pipeline for extracting **clinician-interpretable audio features** from the Pitt Corpus dataset for Alzheimer's disease detection.

Based on the cross-linguistic study's proven implementation (`Timing_VAD.py`).

## Features

This pipeline extracts **36+ acoustic features** organized into three categories:

### 1. VAD-Based Timing Features (17 features)
Speech timing and pause characteristics extracted using energy-based Voice Activity Detection:

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `pause_ratio` | Total pause time / recording duration | Higher in AD (word retrieval difficulty) |
| `speech_ratio` | Total speech time / recording duration | Lower in AD |
| `pause_speech_ratio` | Speech segments / pause count | Lower in AD |
| `num_pauses_per_sec` | Pause frequency | Higher in AD |
| `pause_dur_mean/std/skew/kurt/min/max` | Pause duration statistics | Longer pauses in AD |
| `speech_dur_mean/std/skew/kurt/min/max` | Speech segment statistics | Shorter segments in AD |

### 2. Prosodic Features (11 features)
Fundamental frequency (pitch) and intensity (loudness):

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `f0_mean/std/min/max/range` | Pitch statistics (Hz) | Reduced range in AD (monotonous speech) |
| `intensity_mean/std/min/max/range` | Loudness statistics (dB) | Flatter intensity in AD |
| `voice_rate` | Voiced segments per second | Lower in AD |

### 3. Voice Quality Features (8 features)
Jitter, shimmer, and harmonics-to-noise ratio:

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `jitter_mean/std` | F0 perturbation (pitch irregularity) | Higher in AD/aging |
| `shimmer_mean/std` | Amplitude perturbation | Higher in AD/aging |
| `hnr_mean/std/min/max` | Harmonics-to-noise ratio (dB) | Lower in AD (noisier voice) |

## Installation

```bash
# Clone or navigate to the project
cd audio_feature_extraction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **librosa** ≥ 0.10.0 — Audio processing, F0 extraction
- **scipy** ≥ 1.7.0 — Signal processing, statistics
- **numpy** ≥ 1.20.0 — Numerical operations
- **parselmouth** ≥ 0.4.0 — Praat interface for robust voice quality analysis
- **soundfile** ≥ 0.12.0 — Audio I/O
- **pandas** ≥ 1.3.0 — Data organization, CSV output
- **pyyaml** ≥ 5.4.0 — Configuration files
- **tqdm** ≥ 4.60.0 — Progress bars

## Quick Start

### Single File Extraction

```python
from audio_feature_extraction import extract_audio_features

# Extract features from a single audio file
features = extract_audio_features(
    audio_path='path/to/audio.wav',
    participant_id='001',
    group='AD',  # or 'HC' for healthy control
    age=72,
    mmse=18
)

# Features is a dict with ~36 audio features
print(f"Pause ratio: {features['pause_ratio']:.3f}")
print(f"F0 mean: {features['f0_mean']:.1f} Hz")
print(f"Jitter: {features['jitter_mean']:.4f}")
```

### Batch Processing (Full Corpus)

```python
from audio_feature_extraction import process_pitt_corpus

# Process entire Pitt Corpus
df = process_pitt_corpus(
    corpus_root='../Pitt Corpus/Pitt Corpus',
    output_csv='../output/pitt_audio_features.csv',
    task='cookie'  # Cookie Theft picture description
)

print(f"Processed {len(df)} recordings")
print(df.groupby('group').mean()[['pause_ratio', 'f0_mean', 'hnr_mean']])
```

### Command Line

```bash
# Process corpus
python -m audio_feature_extraction.main_extractor \
    --corpus-root "../Pitt Corpus/Pitt Corpus" \
    --output "../output/features.csv" \
    --task cookie

# List all features with descriptions
python -m audio_feature_extraction.main_extractor --list-features
```

## Directory Structure

```
audio_feature_extraction/
├── __init__.py                # Package exports
├── main_extractor.py          # Main pipeline: load → process → save
├── vad_features.py            # VAD implementation (from Timing_VAD.py)
│   ├── eVAD()                 # Energy-based VAD
│   └── extract_vad_features() # Extract timing features
├── prosody_features.py        # F0 and intensity extraction
│   ├── extract_f0()           # Multiple F0 methods (pYIN, YIN, Praat)
│   └── extract_intensity()    # Energy contour
├── voice_quality_features.py  # Jitter, shimmer, HNR
│   ├── compute_jitter()       # F0 perturbation
│   ├── compute_shimmer()      # Amplitude perturbation
│   └── compute_hnr()          # Harmonics-to-noise ratio
├── utils/
│   ├── audio_loader.py        # Load/preprocess Pitt Corpus audio
│   ├── pitt_metadata.py       # Parse CHAT transcripts for demographics
│   └── validation.py          # QC checks on features
├── config.yaml                # Parameters (sample rate, VAD settings, etc.)
├── tests/
│   └── test_features.py       # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration

Edit `config.yaml` to customize extraction parameters:

```yaml
# Audio preprocessing
audio:
  target_sample_rate: 16000  # Resample all audio to 16 kHz
  normalize: true

# VAD parameters (from Timing_VAD.py)
vad:
  window_size: 0.025  # 25ms frames
  hop_size: 0.01      # 10ms hop

# Prosody parameters  
prosody:
  f0_min: 50          # Hz - minimum F0
  f0_max: 500         # Hz - maximum F0
  f0_method: pyin     # pyin, yin, or parselmouth

# Voice quality
voice_quality:
  hnr_min_pitch: 75   # Hz
```

## Pitt Corpus Structure

The pipeline expects the Pitt Corpus in this structure:

```
Pitt Corpus/
├── Media/
│   ├── Dementia/
│   │   └── Cookie/
│   │       ├── WAV/
│   │       │   ├── 001-0.wav
│   │       │   ├── 001-2.wav
│   │       │   └── ...
│   │       └── MP3/
│   └── Control/
│       └── Cookie/
│           └── WAV/
└── Transcripts/
    └── Pitt/
        ├── Dementia/
        │   └── cookie/
        │       ├── 001-0.cha
        │       └── ...
        └── Control/
            └── cookie/
```

## Output Format

The output CSV contains one row per audio file:

```csv
participant_id,session,group,age,mmse,gender,duration_seconds,pause_ratio,speech_ratio,...
001,0,AD,57,18,M,54.76,0.32,0.68,...
002,0,HC,58,30,F,61.49,0.15,0.85,...
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Clinical Interpretation

### Key Features for AD Detection

1. **Pause Characteristics** — Most discriminative
   - AD patients show longer and more frequent pauses
   - Reflects word-finding difficulties and semantic memory decline

2. **Prosodic Features**
   - Reduced F0 range → monotonous, less expressive speech
   - Lower intensity variation → flat affect

3. **Voice Quality**
   - Increased jitter/shimmer → vocal instability
   - Lower HNR → noisier voice quality
   - Note: Overlaps with normal aging effects

### Typical Feature Ranges

| Feature | AD (typical) | HC (typical) |
|---------|--------------|--------------|
| pause_ratio | 0.25-0.45 | 0.10-0.25 |
| pause_dur_mean | 0.5-2.0s | 0.2-0.8s |
| f0_range | 30-80 Hz | 60-150 Hz |
| hnr_mean | 5-15 dB | 12-25 dB |

## References

- Based on: Cross-linguistic AD detection study (Timing_VAD.py by Tomas & Paula, 2021)
- Pitt Corpus: DementiaBank, DOI: 10.21415/CQCW-1F92
- Cookie Theft picture description task

## License

For research use with proper attribution to the Pitt Corpus and original authors.

## Troubleshooting

### Common Issues

1. **"Parselmouth not available"**
   ```bash
   pip install praat-parselmouth
   ```
   Falls back to librosa if unavailable.

2. **"Audio file not found"**
   - Check corpus path structure matches expected layout
   - Ensure WAV files exist (not just MP3)

3. **"No voiced frames detected"**
   - Audio may be too quiet or noisy
   - Check `f0_min`/`f0_max` settings for speaker's pitch range

4. **Memory issues with large corpus**
   - Process in batches using `max_files` parameter
   - Enable multiprocessing in config

## Contributing

Contributions welcome! Please ensure tests pass:
```bash
pytest tests/ -v
```

