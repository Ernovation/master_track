# Audio Mastering Tool

A professional audio mastering tool specifically optimized for AI-generated music (Suno, etc.). Provides intelligent fade completion, loudness normalization, dynamic compression, and spectral enhancement.

## Features

### Core Mastering Chain
- **Start Noise Removal** - Automatically detects and removes noise artifacts at track beginning
- **Adaptive Fade Completion** - Automatically detects and extends incomplete fades to silence
- **Intelligent Loudness Normalization** - LUFS-based normalization with configurable targets
- **Professional Dynamics Processing** - Soft-knee compression with lookahead
- **Transparent Limiting** - Lookahead limiter with smooth release
- **Highpass Filtering** - Removes sub-bass rumble (35Hz)

### Intelligent Processing
- **Dynamic Fade Detection** - Finds where fades actually start (not fixed duration)
- **Adaptive Fade Extension** - Continues existing fade rate, extends only as needed
- **Muddiness Detection & Cleanup** - Segment-based analysis with automatic correction
- **Stereo Width Control** - Mid-side processing for precise imaging adjustment
- **High-Frequency Air Enhancement** - Adds subtle sparkle and presence

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
| `--stereo-width` | 1.0 | Stereo width (0.5-1.5: <1.0=narrow, >1.0=wide) |
| `--no-air` | false | Disable high-frequency enhancement |
| `--no-start-trim` | false | Disable automatic start noise removal |

## How It Works

### Processing Chain

1. **Load & Analyze** - Read audio file, measure original LUFS and peak
2. **Start Noise Detection** - Check for and remove initial artifacts (common in AI-generated audio)
3. **Highpass Filter** - Remove sub-bass rumble below 35Hz
3. **Muddiness Detection** - Analyze 250-400Hz range across multiple segments
4. **Mud Cleanup** (if needed) - Apply gentle dip around 320Hz
5. **Stereo Width Adjustment** - Mid-side processing (if configured)
6. **Compression** - Soft-knee dynamics with 5ms lookahead
7. **Loudness Normalization** - Target LUFS with safety limit
8. **High-Frequency Air** - Subtle 8kHz+ shelf boost (~1.2dB)
9. **Limiting** - Transparent brick-wall limiter with lookahead
10. **Fade Completion** (if needed) - Intelligent reverb tail generation
11. **Output** - Write 24-bit PCM WAV file

### Fade Detection Algorithm

The tool uses a sophisticated multi-stage approach:

1. **Window Analysis** - Divide last 8 seconds into 200ms windows
2. **RMS Calculation** - Measure loudness of each window
3. **Trend Detection** - Work backwards to find where decline begins
4. **Rate Calculation** - Measure dB/second from detected fade start
5. **Extension Calculation** - Determine duration needed to reach -70dB silence
6. **Limiter** - Cap at user-specified maximum duration

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
- **Compression**: -18dB threshold, 2:1 ratio, 6dB knee, 20ms attack, 100ms release
- **Limiter**: -1dBFS ceiling, 5ms lookahead, 100ms release
- **Mud Cleanup**: 320Hz center, 120Hz width, -1.3dB max cut
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
- Check that the track actually has a fade (>3dB drop at end)
- Fade must be at least 0.5 seconds long
- Ending must be above -40dB peak or -45dB RMS

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
- Muddy low-mids
- Lack of high-frequency detail
- Inconsistent stereo imaging
- Variable loudness levels
