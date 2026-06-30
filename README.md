# Audio Mastering Tool

A professional audio mastering tool specifically optimized for AI-generated music (Suno, etc.). Provides intelligent fade completion, loudness normalization, dynamic compression, and spectral enhancement.

## Features

### Core Mastering Chain
- **Start Noise Removal** - Automatically detects and removes noise artifacts at track beginning
- **Adaptive Fade Completion** - Automatically detects and extends incomplete fades to silence
- **Intelligent Loudness Normalization** - LUFS-based normalization with configurable targets
- **Adaptive Dynamics Processing** - Measures crest factor, applies compression only when needed (many Suno tracks are pre-compressed)
- **Transparent Limiting** - Lookahead limiter with smooth release
- **Adaptive Highpass Filtering** - Analyzes sub-bass content, uses 23Hz cutoff to preserve EDM sub-bass or 32Hz to remove rumble
- **Harshness Correction** - Dynamic EQ reduces harsh resonances (2.5-9kHz)

### Intelligent Processing
- **Dynamic Fade Detection** - Finds where fades actually start (not fixed duration)
- **Adaptive Fade Extension** - Continues existing fade rate, extends only as needed
- **Adaptive Sub-Bass Analysis** - Detects intentional sub-bass (28-35Hz EDM content) vs rumble, adjusts HPF accordingly
- **Adaptive Dynamics (Compression)** - Measures crest factor (peak-to-RMS), skips compression on pre-compressed tracks, applies light or full compression based on dynamic range
- **Adaptive Muddiness Detection & Cleanup** - Finds actual problem frequency (180-500Hz) and applies dynamic cut
- **Harshness Detection & Dynamic EQ** - Analyzes loudest 30% of frames for harsh resonances (2.5-9kHz), applies frame-by-frame correction
- **Adaptive Stereo Width Control** - Analyzes side/mid ratio, widens narrow tracks, narrows excessively wide tracks for mono compatibility
- **Adaptive High-Frequency Air Enhancement** - Analyzes 8-16kHz vs 1-4kHz balance, boosts only when needed (0-1.5dB)

### Smart Fade Repair
The tool analyzes the last several seconds of audio to:
1. Detect where the fade actually begins
2. Calculate the fade rate (dB/second)
3. Determine current ending level
4. Extend the fade only as long as needed to reach silence
5. Generate natural reverb tail with smooth crossfade

## Installation

### Prerequisites
- Python 3.8 or higher
- libsndfile (for soundfile)

### Linux/Ubuntu
```bash
# Install system dependencies
sudo apt-get install libsndfile1

# Create virtual environment
python3 -m venv master_track_venv
source master_track_venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### macOS
```bash
# Install system dependencies
brew install libsndfile

# Create virtual environment
python3 -m venv master_track_venv
source master_track_venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Windows
```bash
# libsndfile is bundled with soundfile on Windows

# Create virtual environment
python -m venv master_track_venv
master_track_venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python master_track.py input.wav output.wav
```

### With Custom Settings
```bash
# Custom LUFS target
python master_track.py input.wav output.wav --lufs -14

# Adjust stereo width (wider)
python master_track.py input.wav output.wav --stereo-width 1.15

# Custom fade extension limit
python master_track.py input.wav output.wav --fade-seconds 6.0

# Maximum loudness boost
python master_track.py input.wav output.wav --max-gain 5.0

# Disable high-frequency air enhancement
python master_track.py input.wav output.wav --no-air
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lufs` | -12.0 | Target loudness in LUFS |
| `--max-gain` | 3.0 | Maximum loudness increase in dB |
| `--fade-seconds` | 5.0 | Maximum fade extension duration |
| `--stereo-width` | 1.0 | Stereo width override (0.5-1.5: <1.0=narrow, >1.0=wide). Default 1.0 uses adaptive analysis |
| `--no-air` | false | Disable high-frequency enhancement |
| `--no-start-trim` | false | Disable automatic start noise removal |

## How It Works

### Processing Chain

1. **Load & Analyze** - Read audio file, measure original LUFS and peak
2. **Start Noise Detection** - Check for and remove initial artifacts (common in AI-generated audio)
3. **Adaptive Highpass Filter** - Analyzes sub-bass (20-35Hz), uses 23Hz cutoff if intentional content detected (preserves EDM sub-bass), 32Hz if just rumble
4. **Adaptive Muddiness Detection** - Find strongest resonance between 180-500Hz
5. **Adaptive Mud Cleanup** (if needed) - Apply dynamic EQ cut at detected frequency
6. **Adaptive Stereo Width Adjustment** - Analyzes side/mid ratio, adjusts width automatically (manual override available)
7. **Adaptive Compression** - Measures crest factor: >14dB→full compression, 10-14dB→light compression, <10dB→skip (let limiter handle)
8. **Harshness Detection** - Analyze loudest 30% of frames for harsh resonances (2.5-9kHz)
9. **Dynamic EQ Correction** (if needed) - Frame-by-frame dynamic EQ on harsh frequencies
10. **Loudness Normalization** - Target LUFS with safety limit
11. **High-Frequency Air** - Adaptive 8kHz+ shelf boost (0-1.5dB based on HF/Mid ratio)
12. **Limiting** - Transparent brick-wall limiter with lookahead
13. **Fade Completion** (if needed) - Intelligent reverb tail generation
14. **Output** - Write 24-bit PCM WAV file

### Fade Detection Algorithm

The tool uses a sophisticated multi-stage approach:

1. **Window Analysis** - Divide last 8 seconds into 200ms windows
2. **RMS Calculation** - Measure loudness of each window
3. **Trend Detection** - Work backwards to find where decline begins (>1.5dB drop)
4. **Completion Check** - Fade needs extension if:
   - Ending is above silence threshold (-35dB peak OR -40dB RMS), **OR**
   - Fade exists but is too short (< 2.0 seconds) for smooth listening
5. **Rate Calculation** - Measure dB/second from detected fade start
6. **Extension Calculation** - Determine duration needed to reach -70dB silence
7. **Limiter** - Cap at user-specified maximum duration

This ensures both incomplete fades and abrupt short fades are properly extended for professional-sounding endings.

### Adaptive Mud Detection Algorithm

Suno-generated audio often has muddy resonances, but not always at the same frequency. The adaptive algorithm finds and corrects the actual problem:

**Detection Process:**
1. **Segment Analysis** - Analyze 5 segments spread across the track
2. **Spectral Analysis** - Calculate FFT spectrum for each segment
3. **Peak Finding** - Find strongest resonance between 180-500Hz
4. **Excess Measurement** - Compare peak energy to neighboring frequency bands (±80Hz)
5. **Threshold Check** - Excess ratio > 1.5x triggers mud detection
6. **Reduction Scaling** - Dynamic cut based on severity:
   - 1.5-1.7x excess → -0.5dB reduction
   - 1.7-2.0x excess → -1.0dB reduction
   - 2.0-2.5x excess → -1.5dB reduction
   - 2.5x+ excess → -2.5dB reduction

**EQ Application:**
- Bell-shaped cut at detected frequency
- Q factor of 0.8 (gentle, musical)
- Preserves frequencies outside problem area

**Example:**
- Detects 213Hz resonance with 2.6x excess energy
- Applies -2.5dB bell cut at 213Hz
- Much more effective than fixed 320Hz approach

### Adaptive Stereo Width Analysis

Many AI-generated tracks have inconsistent or problematic stereo imaging. The adaptive stereo width system analyzes and corrects these issues automatically:

**Analysis Process:**
1. **Mid-Side Conversion** - Convert L/R to mid (mono) and side (stereo difference) signals
2. **RMS Measurement** - Calculate RMS energy of both mid and side components
3. **Width Ratio** - Compute `side_rms / mid_rms` to quantify stereo spread
4. **Intelligent Adjustment** - Apply correction based on measured ratio:
   - **< 0.2** (Very narrow/mono) → Widen to 1.4x (add stereo presence)
   - **0.2-0.4** (Narrow) → Widen to 1.2x (moderate enhancement)
   - **0.4-0.9** (Normal) → Leave unchanged (optimal range)
   - **0.9-1.2** (Quite wide) → Narrow to 0.85x (slight correction)
   - **> 1.2** (Excessively wide) → Narrow to 0.7x (protect mono compatibility)

**Benefits:**
- Prevents mono compatibility issues (excessive width damages playback on mono systems)
- Adds stereo presence to narrow/mono-ish tracks
- Avoids fixed multipliers that can create phase problems
- User can override with `--stereo-width` for manual control

**Example:**
- Detects side/mid ratio of 0.32 (narrow)
- Applies 1.20x width adjustment
- Results in more spacious stereo image while maintaining mono compatibility

### Harshness Detection and Correction

AI-generated audio often contains harsh resonances in the upper-mid range, especially in vocals, synths, and cymbals. The dynamic EQ system detects and corrects these issues:

**Detection Process:**
1. **Frame Analysis** - Divide audio into 4096-sample frames with Hann windowing
2. **Loudest Frames Selection** - Analyze only the loudest 30% of frames (choruses, drops)
3. **Spectral Analysis** - Compute FFT and average spectrum from loud frames
4. **Range Restriction** - Focus on 2.5-9 kHz (harshness range)
5. **Trend Estimation** - Apply median filtering to estimate local spectral baseline
6. **Resonance Detection** - Find where actual spectrum exceeds smoothed trend by >2.5dB
7. **Region Identification** - Identify contiguous harsh regions with 100-3000Hz bandwidth
8. **Scoring** - Rank by: `peak_height_dB × log(bandwidth)`
9. **Selection** - Keep top 1-3 harsh regions

**Correction Process (Dynamic EQ):**
- **Per-Frame Processing** - Each frame analyzed and corrected independently
- **Frequency-Specific** - Only affects detected harsh frequencies
- **Threshold-Based** - Compresses only when band exceeds: `band_RMS + 2dB`
- **Ratio** - 2:1 compression ratio
- **Attack/Release** - 5ms attack, 50ms release (frame-by-frame)
- **Maximum Reduction** - -2.5dB cap (scaled with peak severity)
- **Q Factor** - 0.8 to 6.0 (calculated from bandwidth)
- **Bell Curve** - Smooth Gaussian-shaped gain reduction

**Safety Checks:**
- No processing if max resonance < 2.5dB (balanced tracks)
- No processing if LUFS > -10dB AND crest factor < 8dB (dense mixes)
- Preserves natural tonality while removing harshness

**Example Output:**
```
Applying dynamic EQ for harshness: 3400Hz (4.2dB), 6200Hz (3.1dB)
```

**Why This Works:**
- Analyzes loudest sections (not quiet intros/outros that dilute results)
- Dynamic processing (doesn't affect clean sections)
- Adaptive frequency targeting (finds actual problem, not fixed frequency)
- Conservative thresholds (balanced tracks pass through unchanged)

### Start Noise Detection Algorithm

AI-generated audio (especially from Suno) sometimes begins with artifacts:

**Common Pattern:**
1. Brief noise burst (5-20ms) at moderate level (-30 to -20dB)
2. Near-silence period (100-400ms below -80dB)
3. Music gradually fades in

**Detection Process:**
1. **Window Analysis** - Divide first 600ms into 10ms windows
2. **Level Measurement** - Calculate RMS for each window
3. **Pattern Matching** - Detect burst → silence → rise pattern
4. **Trim Point** - Find where sustained music begins (above -70dB)
5. **Fade-In** - Apply 10ms fade-in to trimmed audio for smooth start

The algorithm is conservative and only trims when clear artifacts are detected.

### Reverb Tail Generation

For smooth fade completion:
- Generates impulse response with natural decay
- Convolves with last 2 seconds of audio
- Matches level precisely to track ending (last 50ms)
- Applies exponential fade curve
- Uses 500ms equal-power crossfade for seamless transition

## Technical Details

### Audio Specifications
- **Input Formats**: WAV, FLAC, OGG, MP3 (via soundfile)
- **Output Format**: 24-bit PCM WAV
- **Sample Rates**: Any (preserves input rate)
- **Channels**: Mono (converted to stereo) or Stereo

### Processing Parameters
- **Highpass**: 4th-order Butterworth @ 35Hz
- **Compression**: Adaptive based on crest factor
  - Crest > 14 dB: -18dB threshold, 2:1 ratio (full compression)
  - Crest 10-14 dB: -15dB threshold, 1.5:1 ratio (light compression)  
  - Crest < 10 dB: Skipped (already compressed)
  - All modes: 6dB knee, 20ms attack, 100ms release, 5ms lookahead
- **Harshness Correction**: 2.5-9kHz range, dynamic EQ with 2:1 ratio, 5ms attack, 50ms release, -2.5dB max reduction
- **Limiter**: -1dBFS ceiling, 5ms lookahead, 100ms release
- **Adaptive Mud Cleanup**: 180-500Hz detection range, Q=0.8, -0.5 to -2.5dB dynamic reduction
- **Air Enhancement**: 8kHz shelf, +1.2dB boost

## Examples

### Suno Track (typical use case)
```bash
# Standard mastering for streaming
python master_track.py suno_track.wav mastered.wav --lufs -14

# More aggressive (louder)
python master_track.py suno_track.wav mastered.wav --lufs -12 --max-gain 4

# Subtle enhancement with wider stereo
python master_track.py suno_track.wav mastered.wav --stereo-width 1.1

# Keep original start (disable noise removal)
python master_track.py suno_track.wav mastered.wav --no-start-trim
```

### YouTube Content
```bash
# Optimize for YouTube's -13 LUFS normalization
python master_track.py video_audio.wav youtube.wav --lufs -13
```

### Podcast/Voice
```bash
# Narrower stereo, skip air enhancement
python master_track.py podcast.wav mastered.wav --stereo-width 0.8 --no-air
```

## Troubleshooting

### "No module named 'soundfile'"
Install libsndfile system library (see Installation section)

### "Possible clipped samples in output"
This warning is suppressed internally - the limiter prevents actual clipping

### Fade not detected
- Check that the track actually has a fade (>1.5dB drop at end)
- Fade must be at least 0.2 seconds long
- Fade will be extended if:
  - Ending is above -35dB peak or -40dB RMS, OR
  - Fade is shorter than 2.0 seconds (prevents abrupt endings)
- The tool shows fade analysis for all tracks for transparency

### Output too quiet/loud
Adjust `--lufs` target (typical: -14 for streaming, -12 for energetic, -16 for ambient)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional EQ curves for different genres
- Multiband compression option
- Batch processing support
- GUI interface
- Additional output formats

## License

MIT License - feel free to use and modify for your projects.

## Acknowledgments

Built specifically to enhance AI-generated music from platforms like Suno, addressing common issues like:
- Initial noise artifacts and glitches
- Incomplete fades to silence
- Variable muddy resonances in the low-mid frequency range (180-500Hz)
- Harsh resonances in vocals, synths, and cymbals (2.5-9kHz)
- Lack of high-frequency detail
- Inconsistent stereo imaging
- Variable loudness levels
