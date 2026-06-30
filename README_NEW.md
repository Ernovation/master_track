# Audio Mastering Tool - Refactored Architecture

## Overview

This is a comprehensive audio mastering tool that uses an **analysis-driven architecture**. Instead of applying independent heuristics, it performs a coherent spectral analysis and makes intelligent mastering decisions based on the audio's characteristics.

## Key Design Principles

1. **Conservative Processing**: Well-mastered tracks remain nearly unchanged
2. **Analysis Before Processing**: A single analysis stage drives all processing decisions
3. **Modular Architecture**: Separate analysis and processing modules
4. **Profile-Based**: External JSON profiles for different genres
5. **Comprehensive Reporting**: Detailed mastering report showing all decisions

## New Processing Chain

```
Load Audio
    ↓
Enhanced Start Noise Removal (detects 5 noise patterns)
    ↓
Adaptive Highpass Filter (based on sub-bass analysis)
    ↓
Spectral Analysis (loudest 35% of frames)
    ↓
Tonal Balance Analysis (replaces mud + air)
    ↓
Static Mastering EQ (up to 4 broad filters)
    ↓
Harshness Analysis (2.5-9 kHz resonances)
    ↓
Dynamic EQ (frequency-selective compression)
    ↓
Three-Band Stereo Analysis (Low/Mid/High)
    ↓
Stereo Width Adjustment
    ↓
Dynamics Analysis (Crest Factor + LRA)
    ↓
Adaptive Compression (based on both metrics)
    ↓
LUFS Normalization
    ↓
True Peak Limiter
    ↓
Fade Repair (if needed)
    ↓
Final Clip Protection
```

### Enhanced Start Noise Detection

The improved noise detection catches multiple patterns and **selects the longest trim**:

1. **Initial burst → silence → music**: Loud burst followed by near-silence, then music starts
2. **Large level jumps**: Sudden increase indicating transition from noise to content  
3. **Sustained low-level noise**: Hiss, hum, or static before music starts
4. **Very quiet start**: Below -60dB noise followed by normal level music
5. **Clicks/pops**: High peak-to-RMS ratio in first window indicating transients
6. **Short Suno artifacts (40ms)**: Compares first 40ms against following section - detects quieter, noisier, or louder anomalies
7. **Very short anomalies (20ms)**: Aggressive check for any oddity in first 20ms
8. **Noise before silence**: Detects moderate-level noise (e.g., -33dB) followed by silence (< -60dB)
9. **Multi-segment noise removal** ⭐: Scans forward to find the actual song fade-in and removes ALL noise before it - handles multiple consecutive noise segments!

**Pattern 9 is the most powerful** - it handles your case where there's -33dB noise for 22ms, then -55dB noise for another 23ms, then the actual song fade-in at -90dB. It automatically finds where the song starts and removes everything before it.

**All patterns are evaluated** and the one trimming the most is selected, ensuring comprehensive noise removal even with multiple consecutive artifacts.

## Architecture

### Analysis Modules (`analysis/`)

- **`frame_selection.py`**: Selects loudest frames for analysis
- **`spectral_analysis.py`**: FFT analysis and smoothing utilities
- **`tonal_balance.py`**: Unified tonal balance analyzer (replaces mud + air)
- **`harshness.py`**: Detects harsh resonances (dynamic problem)
- **`stereo_analysis.py`**: Three-band stereo width analysis
- **`dynamics.py`**: Crest factor + Loudness Range analysis

### Processing Modules (`processing/`)

- **`eq.py`**: Static EQ filters (shelf, bell, highpass)
- **`dynamic_eq.py`**: Dynamic EQ for harshness correction
- **`compressor.py`**: Adaptive compressor with soft knee
- **`limiter.py`**: True peak limiter with lookahead

### Profiles (`profiles/`)

Tonal balance profiles stored as JSON:
- **`edm.json`**: Enhanced bass and high-end for electronic music
- **`pop.json`**: Balanced with vocal presence boost
- **`rock.json`**: Warm low-mids with controlled highs
- **`neutral.json`**: Flat reference for well-balanced material

## Major Changes from Previous Version

### 1. Unified Tonal Balance Analysis

**Before**: Separate adaptive mud detector + air enhancer

**After**: Single tonal balance analyzer that:
- Analyzes loudest 35% of FFT frames
- Computes median spectrum using overlapping 10-second windows
- Smooths to 1/3 octave resolution
- Compares against configurable target curve
- Generates up to 4 broad EQ filters (1 low shelf, 2 bells, 1 high shelf)
- Maximum correction: ±2 dB
- Ignores corrections below 0.75 dB

### 2. Improved Harshness Detection

**Unchanged principle** but better implementation:
- Analyzes only loudest frames
- Detects resonances between 2.5-9 kHz
- Detects only peaks above local spectral trend
- Rejects narrow FFT spikes
- Uses **dynamic EQ only** (no static EQ)
- Maximum reduction: 2.5 dB

### 3. Three-Band Stereo Analysis

**Before**: Single global side/mid ratio

**After**: Frequency-dependent stereo analysis:
- **Low (20-150 Hz)**: Keep nearly mono
- **Mid (150-2000 Hz)**: Leave mostly unchanged
- **High (2000-20000 Hz)**: Allow wider stereo image

### 4. Smarter Compression Decisions

**Before**: Based only on crest factor

**After**: Uses both crest factor and LRA (Loudness Range):

| Crest Factor | LRA  | Mode     | Action              |
|--------------|------|----------|---------------------|
| High         | High | Normal   | Full compression    |
| High         | Low  | Light    | Gentle compression  |
| Low          | Low  | None     | Skip compression    |
| Low          | High | Minimal  | Very light          |

### 5. External Profile Support

**Before**: Hard-coded single target spectrum

**After**: JSON-based profiles with:
```json
{
  "name": "EDM",
  "description": "...",
  "target_curve": [
    {"frequency": 20, "db": 2.5},
    {"frequency": 60, "db": 1.5},
    ...
  ],
  "max_boost_db": 2.0,
  "max_cut_db": 2.0
}
```

## Usage

### Basic Usage

```bash
python master_track_new.py input.wav output.wav
```

### With Profile

```bash
python master_track_new.py input.wav output.wav --profile edm
```

### Custom LUFS Target

```bash
python master_track_new.py input.wav output.wav --profile pop --lufs -14.0
```

### All Options

```bash
python master_track_new.py input.wav output.wav \
  --profile rock \
  --lufs -12.0 \
  --max-gain 3.0 \
  --fade-seconds 5.0 \
  --no-start-trim \
  --no-harshness-fix
```

## Command-Line Arguments

- `input_file`: Input audio file
- `output_file`: Output audio file
- `--profile`: Tonal balance profile (edm, pop, rock, neutral) [default: neutral]
- `--lufs`: Target LUFS level [default: -12.0]
- `--max-gain`: Maximum loudness increase in dB [default: 3.0]
- `--fade-seconds`: Maximum fade tail duration [default: 5.0]
- `--no-start-trim`: Disable start noise removal
- `--no-harshness-fix`: Disable dynamic harshness correction

## Mastering Report

At the end of processing, a comprehensive report is printed:

```
==================================================
MASTERING REPORT
==================================================

Profile: EDM

Spectral Balance:
  Sub Bass: +1.0 dB
  Bass: OK
  Low Mids: -1.5 dB
  Presence: OK
  Air: +0.8 dB

Generated EQ:
  Low Shelf @ 55 Hz: +0.8 dB
  Bell @ 400 Hz (Q=0.8): -1.5 dB
  High Shelf @ 10000 Hz: +0.8 dB

Harshness:
  Detected: Yes
  Frequency: 3700 Hz (peak: +3.2 dB, bandwidth: 850 Hz, max reduction: -2.1 dB)

Stereo:
  Low Band: side/mid=0.08, adjustment=1.00x
  Mid Band: side/mid=0.43, adjustment=1.00x
  High Band: side/mid=0.72, adjustment=1.00x
  Overall: No adjustment needed

Compression:
  Crest Factor: 11.4 dB
  LRA: 6.2 LU
  Mode: Normal
  Settings: threshold=-18 dB, ratio=2.0:1

Output:
  Integrated LUFS: -14.0
  True Peak: -1.0 dBTP
==================================================
```

## Creating Custom Profiles

Create a new JSON file in `profiles/`:

```json
{
  "name": "My Profile",
  "description": "Custom tonal balance",
  "target_curve": [
    {"frequency": 20, "db": 0.0},
    {"frequency": 60, "db": 1.0},
    {"frequency": 200, "db": 0.0},
    {"frequency": 1000, "db": 0.0},
    {"frequency": 4000, "db": 1.0},
    {"frequency": 10000, "db": 1.5},
    {"frequency": 20000, "db": 1.0}
  ],
  "max_boost_db": 2.0,
  "max_cut_db": 2.0
}
```

Use it with:
```bash
python master_track_new.py input.wav output.wav --profile my_profile
```

## Preserved Features

All existing features are preserved:

✓ Start noise removal (Suno artifacts)  
✓ Fade detection and repair  
✓ LUFS normalization  
✓ True peak limiting  
✓ Batch processing support (via CLI)  
✓ Command-line interface  

## Technical Details

### Start Noise Detection

Enhanced detection with 9 pattern types, **all evaluated and the longest trim selected**:

1. **Burst-Silence-Music**: Detects loud burst → near-silence (< -80dB) → music rising above -70dB
2. **Level Jumps**: Finds sustained jumps > 15dB indicating noise-to-music transition
3. **Low-Level Noise**: Compares first 0.6s to music section median; trims if first portion is > 6dB quieter
4. **Very Quiet Start**: Detects < -60dB start followed by normal levels (> -50dB)
5. **Transients/Clicks**: High crest factor (peak-RMS > 20dB) in first 10ms window
6. **Short Suno Artifacts (40ms)**: Compares first 4 windows against windows 5-10; detects if first 40ms is 3dB+ quieter, more variable (high std dev), or has anomalous peaks
7. **Very Short Anomalies (20ms)**: Aggressive check for any 6dB+ difference in first 2 windows vs. next 6 windows
8. **Noise Before Silence**: Detects moderate-level noise (e.g., -33dB) in first 20-40ms followed by silence (< -60dB); typical of Suno pre-roll artifacts
9. **Multi-Segment Noise Removal** ⭐: Scans forward up to 1 second to find where the actual song fade-in begins (sustained quiet section < -80dB); removes ALL consecutive noise segments before that point, up to 200ms total

**Key improvement**: All 9 patterns are evaluated, and the one that would trim the most is selected. This ensures Pattern 9 (which scans forward for the fade-in) wins over Pattern 8 (which only checks first 20-40ms) when there are multiple consecutive noise segments.

**Logging**: Prints which pattern was selected and the total trim duration, e.g.:
```
[Noise Detection] Pattern 9: Multi-segment noise before fade-in (45.0ms, window 5)
[Noise Detection] Selected longest trim: 45.0ms
Start noise detected, trimming 45.0ms
```

Analysis uses 10ms windows. **Pattern 9 is the most comprehensive** - it handles multiple consecutive noise segments by finding the actual fade-in start point and trimming everything before it.

### Frame Selection

- Analyzes loudest 35% of frames (percentile 65)
- Window size: 4096 samples
- Hop size: 2048 samples
- Reduces influence of silence and quiet sections

### Spectral Smoothing

- Approximately 1/3 octave resolution
- Frequency-dependent smoothing
- Median-based (robust to outliers)

### Filter Design

- Frequency-domain implementation for accuracy
- Smooth transitions (no artifacts)
- Bell filters use Gaussian curves
- Shelves use sigmoid transitions

### Dynamic EQ

- Overlap-add processing (4096 window, 2048 hop)
- 2:1 compression ratio
- Threshold: band RMS + 2 dB
- Maximum reduction scales with detected peak

### Stereo Processing

- Mid-side based
- Frequency-dependent (via bandpass filtering)
- Preserves mono compatibility
- Never widens bass frequencies

## Migration from Old Version

The old version is preserved as `master_track.py`. The new version is `master_track_new.py`.

Key differences:

1. **Profile selection**: Add `--profile edm|pop|rock|neutral`
2. **No manual stereo width**: Now automatic (3-band analysis)
3. **No `--no-air` flag**: Air is part of tonal balance (use neutral profile)
4. **Better logging**: Comprehensive mastering report

To switch to the new version:
```bash
mv master_track.py master_track_legacy.py
mv master_track_new.py master_track.py
```

## Dependencies

Same as before:
- numpy
- scipy
- soundfile
- pyloudnorm

Install with:
```bash
pip install numpy scipy soundfile pyloudnorm
```

## Testing

Test with a well-mastered track (should apply minimal processing):
```bash
python master_track_new.py well_mastered.wav test_output.wav --profile neutral
```

Test with an EDM track:
```bash
python master_track_new.py edm_track.wav mastered_edm.wav --profile edm --lufs -14
```

## Philosophy

The goal is to create a **mastering engine that thinks like a mastering engineer**:

1. **Analyze first**: Understand the audio before processing
2. **Be conservative**: Don't fix what isn't broken
3. **Be coherent**: Make decisions based on overall spectral balance
4. **Be transparent**: Report every decision clearly
5. **Be flexible**: Support different genres and targets

Instead of asking:
> "Does this track have mud? Does it need air?"

Ask:
> "What broad EQ corrections are needed to match the target tonal balance?"

This fundamental shift from independent heuristics to coherent analysis is the core of this refactoring.
