#!/usr/bin/env python3
"""
Audio Mastering Tool - Refactored Architecture

Analysis-driven mastering with coherent spectral analysis engine.
"""

import argparse
import warnings
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from scipy.signal import fftconvolve
from scipy.signal.windows import hann

# Import analysis modules
from analysis.tonal_balance import TonalBalanceAnalyzer
from analysis.harshness import HarshnessAnalyzer
from analysis.stereo_analysis import StereoAnalyzer
from analysis.dynamics import DynamicsAnalyzer
from analysis.spectral_analysis import linear_to_db, db_to_linear

# Import processing modules
from processing.eq import apply_eq_filters, apply_highpass_filter
from processing.dynamic_eq import apply_dynamic_eq
from processing.compressor import apply_compressor, apply_stereo_width_adjustment
from processing.limiter import apply_limiter, hard_clip_protection


# Default constants
TARGET_LUFS = -12.0
MAX_GAIN_DB = 3.0
LIMITER_CEILING_DB = -1.0
HPF_FREQ = 35.0


def load_profile(profile_name):
    """
    Load a tonal balance profile from JSON.
    
    Args:
        profile_name: Profile name (e.g., 'edm', 'pop', 'rock')
    
    Returns:
        Profile dict
    """
    # Get script directory
    script_dir = Path(__file__).parent
    profile_path = script_dir / 'profiles' / f'{profile_name}.json'
    
    if not profile_path.exists():
        print(f"Warning: Profile '{profile_name}' not found, using neutral profile")
        profile_path = script_dir / 'profiles' / 'neutral.json'
    
    with open(profile_path, 'r') as f:
        return json.load(f)


def peak_db(audio):
    """Calculate peak level in dBFS."""
    return linear_to_db(np.max(np.abs(audio)))


# ============================================================================
# PRESERVED FUNCTIONS (Start noise removal, fade detection/repair)
# ============================================================================


def detect_start_noise(audio, sr, max_check_duration=0.6):
    """
    Detect noise artifacts at the start of tracks.
    
    Detects various noise patterns:
    - Initial burst followed by silence then music
    - Sustained low-level noise before music starts
    - Pops, clicks, or static at the beginning
    - Level jumps indicating transition from noise to music
    
    Returns: sample index where actual content begins (0 if no noise detected)
    """
    mono = np.mean(audio, axis=1)
    
    check_samples = min(int(sr * max_check_duration), len(mono))
    analysis_region = mono[:check_samples]
    
    window_size = int(sr * 0.01)  # 10ms windows
    num_windows = check_samples // window_size
    
    if num_windows < 10:
        return 0
    
    rms_values = []
    peak_values = []
    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(analysis_region))
        window = analysis_region[start:end]
        rms = np.sqrt(np.mean(window ** 2))
        peak = np.max(np.abs(window))
        rms_values.append(rms)
        peak_values.append(peak)
    
    rms_db = [linear_to_db(rms) if rms > 0 else -120.0 for rms in rms_values]
    peak_db = [linear_to_db(peak) if peak > 0 else -120.0 for peak in peak_values]
    
    # Collect all pattern matches: list of (trim_samples, pattern_name, description)
    candidates = []
    
    # Pattern 1: Initial burst followed by silence then music
    first_window_db = rms_db[0]
    
    if first_window_db > -40.0:
        next_windows = rms_db[1:min(5, len(rms_db))]
        if next_windows:
            min_next = min(next_windows)
            avg_next = np.mean(next_windows)
            
            if avg_next < -80.0 and min_next < -90.0:
                for i in range(5, min(60, len(rms_db))):
                    if i < len(rms_db) - 3:
                        window_avg = np.mean(rms_db[i:min(i+3, len(rms_db))])
                        if window_avg > -70.0:
                            trim_samples = i * window_size
                            candidates.append((trim_samples, 'Pattern 1a', f'Initial burst > -70dB followed by silence, music at window {i}'))
                            break
                
                if not any(c[1] == 'Pattern 1a' for c in candidates):
                    for i in range(1, min(50, len(rms_db))):
                        if rms_db[i] > -90.0:
                            trim_samples = max(0, (i - 1) * window_size)
                            candidates.append((trim_samples, 'Pattern 1b', f'Initial burst > -90dB followed by silence, music at window {i}'))
                            break
    
    # Pattern 2: Large level jump indicating transition to music
    for i in range(1, min(10, len(rms_db) - 2)):
        level_jump = rms_db[i] - rms_db[i-1]
        
        if level_jump > 15.0:
            if i < len(rms_db) - 2:
                avg_after = np.mean(rms_db[i:min(i+5, len(rms_db))])
                avg_before = np.mean(rms_db[max(0, i-3):i])
                
                if avg_after - avg_before > 12.0:
                    trim_samples = i * window_size
                    candidates.append((trim_samples, 'Pattern 2', f'Level jump at window {i}'))
                    break
    
    # Pattern 3: Sustained low-level noise before music
    # Find where sustained musical content actually starts
    if len(rms_db) >= 15:
        # Calculate median level of later portion (assumed to be music)
        music_section_start = min(10, len(rms_db) - 5)
        music_median = np.median(rms_db[music_section_start:])
        
        # Look for first window where level approaches music level
        threshold_db = music_median - 10.0  # Within 10dB of music level
        
        for i in range(min(15, len(rms_db))):
            # Check if we've reached sustained music level
            if i < len(rms_db) - 2:
                # Need at least 3 consecutive windows near music level
                window_avg = np.mean(rms_db[i:min(i+3, len(rms_db))])
                if window_avg >= threshold_db:
                    # Check that preceding windows are significantly quieter
                    if i >= 2:
                        prev_avg = np.mean(rms_db[max(0, i-2):i])
                        if prev_avg < threshold_db - 6.0:  # At least 6dB quieter
                            # Trim back to the start of the quiet section
                            trim_samples = max(0, (i - 1) * window_size)
                            candidates.append((trim_samples, 'Pattern 3', f'Sustained music start at window {i}'))
                            break
    
    # Pattern 4: Very quiet start (below -60dB) followed by normal level
    # This catches noise that's just low-level hiss or hum
    if len(rms_db) >= 10:
        # Check if first 1-3 windows are unusually quiet
        first_few = rms_db[:min(3, len(rms_db))]
        avg_first = np.mean(first_few)
        
        if avg_first < -60.0:  # Very quiet start
            # Find where level rises to normal
            for i in range(1, min(10, len(rms_db))):
                if rms_db[i] > -50.0:  # Normal level reached
                    # Check if it stays there
                    if i < len(rms_db) - 2:
                        sustained = np.mean(rms_db[i:min(i+3, len(rms_db))])
                        if sustained > -50.0:
                            trim_samples = i * window_size
                            candidates.append((trim_samples, 'Pattern 4', f'Quiet start < -60dB followed by normal level at window {i}'))
                            break
    
    # Pattern 5: High peak but low RMS in first window (click/pop)
    if len(peak_db) > 0 and len(rms_db) > 0:
        first_crest = peak_db[0] - rms_db[0]
        if first_crest > 20.0 and peak_db[0] > -30.0:  # High crest factor = transient
            # Check if following windows are more normal
            if len(rms_db) >= 3:
                following_avg = np.mean(rms_db[1:3])
                if following_avg > rms_db[0] + 10.0:  # Following is much louder
                    trim_samples = window_size  # Trim first window
                    candidates.append((trim_samples, 'Pattern 5', 'Initial transient (click/pop) at window 0'))
    
    # Pattern 6: Very short noise in first 40-50ms (common Suno artifact)
    # Compare first 4-5 windows (40-50ms) against windows 5-10
    if len(rms_db) >= 10:
        # First section: first 4 windows (40ms)
        first_section = rms_db[:4]
        first_avg = np.mean(first_section)
        first_max = np.max(first_section)
        first_std = np.std(first_section)
        
        # Following section: windows 5-10 (50-100ms after start)
        following_section = rms_db[5:10]
        following_avg = np.mean(following_section)
        following_std = np.std(following_section)
        
        # Music section: windows 10+ 
        if len(rms_db) > 15:
            music_section = rms_db[10:15]
            music_avg = np.mean(music_section)
        else:
            music_avg = following_avg
        
        # Detection criteria for very short start noise:
        # 1. First section is notably different (quieter OR spectrally different)
        # 2. Following section is more consistent and closer to music level
        
        # Check if first 40ms is significantly quieter than following
        if first_avg < following_avg - 3.0:  # At least 3dB quieter
            # And following is closer to music level
            if abs(following_avg - music_avg) < abs(first_avg - music_avg):
                # Trim up to the start of the following section
                trim_samples = 4 * window_size  # Trim first 40ms
                candidates.append((trim_samples, 'Pattern 6a', 'Short noise (quieter) in first 40ms'))
        
        # Check if first 40ms has different characteristics (more variable)
        # while following is more stable
        if first_std > 3.0 and following_std < first_std / 2:
            # First part is noisy/inconsistent, following is stable
            if following_avg > first_avg - 5.0:  # Following isn't much quieter
                trim_samples = 4 * window_size  # Trim first 40ms
                if not any(c[1] == 'Pattern 6a' for c in candidates):
                    candidates.append((trim_samples, 'Pattern 6b', 'Inconsistent noise in first 40ms'))
        
        # Check if first 40ms is notably louder but brief (burst noise)
        if first_max > following_avg + 6.0:  # First has peaks 6dB+ above following
            # But music section is similar to following (so first is anomalous)
            if abs(music_avg - following_avg) < 4.0:
                trim_samples = 4 * window_size  # Trim first 40ms
                if not any(c[1].startswith('Pattern 6') for c in candidates):
                    candidates.append((trim_samples, 'Pattern 6c', 'Brief burst noise in first 40ms'))
    
    # Pattern 7: Any oddity in first 20ms compared to rest
    # More aggressive check specifically for first 2 windows (20ms)
    if len(rms_db) >= 8:
        first_20ms = np.mean(rms_db[:2])
        next_60ms = np.mean(rms_db[2:8])
        
        # If first 20ms is weird compared to next 60ms
        diff = abs(first_20ms - next_60ms)
        
        if diff > 6.0:  # Significant difference
            # Check consistency of just the next section (don't include music start)
            next_section_std = np.std(rms_db[2:8])
            if next_section_std < 5.0:  # Next section is relatively stable
                trim_samples = 2 * window_size  # Trim first 20ms
                candidates.append((trim_samples, 'Pattern 7', 'Anomaly in first 20ms vs next 60ms'))
    
    # Pattern 8: Noise before silence (first section LOUDER than following)
    # Specifically for Suno artifacts: moderate noise followed by silence
    if len(rms_db) >= 6:
        first_20ms = np.mean(rms_db[:2])
        next_40ms = np.mean(rms_db[2:6])
        
        # If first part is much louder than what follows
        if first_20ms > next_40ms + 10.0:  # First 20ms is 10dB+ louder
            # And the following part is very quiet (silence-like)
            if next_40ms < -60.0:  # Following is essentially silence
                trim_samples = 2 * window_size  # Trim first 20ms
                candidates.append((trim_samples, 'Pattern 8a', 'Noise before silence in first 20ms'))
        
        # Also check for 40ms version (more generous)
        if len(rms_db) >= 8:
            first_40ms = np.mean(rms_db[:4])
            next_60ms = np.mean(rms_db[4:10])
            
            if first_40ms > next_60ms + 10.0:  # First 40ms is 10dB+ louder
                if next_60ms < -60.0:  # Following is essentially silence
                    trim_samples = 4 * window_size  # Trim first 40ms
                    if not any(c[1] == 'Pattern 8a' for c in candidates):
                        candidates.append((trim_samples, 'Pattern 8b', 'Noise before silence in first 40ms'))
    
    # Pattern 9: Comprehensive Suno multi-segment noise removal
    # Scan forward and find where the actual song fade-in begins
    # Remove ALL consecutive noise segments before that point
    
    # Define thresholds
    FADE_IN_THRESHOLD = -80.0  # Song fade-ins typically start below this
    NOISE_MIN_LEVEL = -80.0    # Noise is typically louder than this
    STABILITY_WINDOW = 5       # Number of windows to check for stability
    
    # Find the first point where we have sustained quiet audio (fade-in start)
    fade_in_start = None
    
    for i in range(len(rms_db) - STABILITY_WINDOW):
        # Check if this window and several following are below fade-in threshold
        window_section = rms_db[i:i + STABILITY_WINDOW]
        
        # All windows in this section should be reasonably quiet
        if all(level < FADE_IN_THRESHOLD for level in window_section):
            # Check that this isn't just a momentary dip
            # The section should be relatively stable (not wildly varying)
            section_std = np.std(window_section)
            
            # If this is a stable quiet section, it's likely the fade-in
            if section_std < 8.0:  # Relatively consistent
                fade_in_start = i
                break
    
    # If we found a fade-in point, check if there's noise before it
    if fade_in_start is not None and fade_in_start > 0:
        # Check if the section before fade-in contains noise
        # (any windows louder than the fade-in threshold)
        pre_fade_section = rms_db[:fade_in_start]
        
        # Count how many windows are louder than fade-in threshold
        noisy_windows = sum(1 for level in pre_fade_section if level > NOISE_MIN_LEVEL)
        
        # If more than 30% of pre-fade windows are louder than fade-in level
        if noisy_windows > len(pre_fade_section) * 0.3:
            # This is noise that should be removed
            trim_samples = fade_in_start * window_size
            
            # Only trim if it's a reasonable amount (< 200ms)
            trim_ms = (trim_samples / sr) * 1000
            if trim_ms < 200:
                candidates.append((trim_samples, 'Pattern 9', f'Multi-segment noise before fade-in ({trim_ms:.1f}ms, window {fade_in_start})'))
    
    # Select the pattern that trims the most (longest noise detection)
    if not candidates:
        return 0
    
    # Find the maximum trim amount
    best_trim, best_pattern, best_desc = max(candidates, key=lambda x: x[0])
    
    # Log which pattern was selected
    print(f"[Noise Detection] {best_pattern}: {best_desc}")
    print(f"[Noise Detection] Selected longest trim: {(best_trim / sr) * 1000:.1f}ms")
    
    return best_trim


def remove_start_noise(audio, sr, trim_samples):
    """
    Remove noise from the start and add a short fade-in.
    """
    if trim_samples <= 0:
        return audio
    
    trimmed = audio[trim_samples:]
    
    fade_samples = int(sr * 0.01)  # 10ms fade
    fade_samples = min(fade_samples, len(trimmed))
    
    fade_curve = np.linspace(0.0, 1.0, fade_samples)
    fade_curve = fade_curve ** 0.5
    
    trimmed[:fade_samples] *= fade_curve[:, np.newaxis]
    
    return trimmed


def detect_fade_start(audio, sr, max_lookback=8.0):
    """
    Detect where the fade actually starts.
    
    Returns:
        - fade_start_sample: Index where fade begins
        - fade_duration: Duration of detected fade in seconds
        - rms_windows: RMS values for analysis
    """
    mono = np.mean(audio, axis=1)
    
    lookback_samples = min(int(max_lookback * sr), len(mono))
    analysis_region = mono[-lookback_samples:]
    
    window_size = int(sr * 0.2)  # 200ms windows
    num_windows = max(1, lookback_samples // window_size)
    
    rms_db = []
    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(analysis_region))
        window = analysis_region[start:end]
        rms_db.append(linear_to_db(np.sqrt(np.mean(window ** 2))))
    
    fade_start_idx = len(rms_db) - 1
    
    for i in range(len(rms_db) - 2, -1, -1):
        is_declining = True
        for j in range(i, len(rms_db) - 1):
            if rms_db[j] - rms_db[j + 1] < -1.0:
                is_declining = False
                break
        
        total_drop = rms_db[i] - rms_db[-1]
        
        if is_declining and total_drop >= 1.5:
            fade_start_idx = i
            break
    
    fade_start_sample = len(mono) - lookback_samples + (fade_start_idx * window_size)
    fade_duration = (len(mono) - fade_start_sample) / sr
    
    return fade_start_sample, fade_duration, rms_db


def needs_fade_completion(audio, sr):
    """
    Detect if the track is fading but hasn't reached silence.
    
    Returns: bool
    """
    mono = np.mean(audio, axis=1)
    
    fade_start_sample, fade_duration, rms_windows = detect_fade_start(audio, sr)
    
    if fade_duration < 0.2:
        return False
    
    fade_drop = rms_windows[0] - rms_windows[-1] if len(rms_windows) > 1 else 0
    
    end_section = mono[-int(sr * 0.2):]
    end_peak = linear_to_db(np.max(np.abs(end_section)))
    end_rms = linear_to_db(np.sqrt(np.mean(end_section ** 2)))
    
    fading = fade_drop > 1.5
    unfinished = end_peak > -35.0 or end_rms > -40.0
    too_short = fade_duration < 2.0
    
    needs_extension = fading and (unfinished or too_short)
    
    print(f"  Fade analysis:")
    print(f"    Duration: {fade_duration:.2f}s, drop: {fade_drop:.1f} dB")
    print(f"    End: peak={end_peak:.1f} dB, RMS={end_rms:.1f} dB")
    print(f"    Needs completion: {needs_extension}")
    
    return needs_extension


def generate_impulse_response(sr, tail_seconds=5.0, decay_db=80.0):
    """Generate impulse response for reverb tail."""
    n = int(sr * tail_seconds)
    ir = np.zeros(n)
    ir[0] = 1.0
    
    rng = np.random.default_rng()
    reflections = 500
    positions = rng.integers(1, n, size=reflections)
    
    for pos in positions:
        decay = np.exp(-pos / n * 8)
        ir[pos] += rng.uniform(-1, 1) * decay
    
    ir /= np.max(np.abs(ir))
    return ir


def generate_reverb_tail(audio, sr, source_seconds=2.0, tail_seconds=5.0, decay_db=80.0):
    """Generate a realistic reverb tail using convolution."""
    source_len = int(source_seconds * sr)
    tail_len = int(tail_seconds * sr)
    
    source = audio[-source_len:]
    ir = generate_impulse_response(sr, tail_seconds, decay_db)
    
    channels = []
    for ch in range(source.shape[1]):
        wet = fftconvolve(source[:, ch], ir, mode='full')
        wet = wet[source_len:source_len + tail_len]
        channels.append(wet)
    
    tail = np.column_stack(channels)
    
    # Match level
    end_sample_len = min(int(sr * 0.05), len(source))
    end_rms = np.sqrt(np.mean(source[-end_sample_len:] ** 2))
    
    start_sample_len = min(int(sr * 0.05), len(tail))
    start_rms = np.sqrt(np.mean(tail[:start_sample_len] ** 2))
    
    if start_rms > 1e-9:
        tail *= (end_rms / start_rms)
    
    # Apply fade
    fade_curve = np.linspace(1.0, 0.0, len(tail))
    fade_curve = fade_curve ** 1.5
    tail *= fade_curve[:, np.newaxis]
    
    # Safety
    tail_peak = np.max(np.abs(tail))
    source_peak = np.max(np.abs(source))
    if tail_peak > source_peak:
        tail *= (source_peak / tail_peak)
    
    return tail


def repair_fade_with_reverb(audio, sr, tail_seconds=5.0):
    """Append a generated reverb tail with smooth crossfade."""
    tail = generate_reverb_tail(
        audio, sr,
        source_seconds=2.0,
        tail_seconds=tail_seconds,
        decay_db=80.0
    )
    
    crossfade_len = int(sr * 0.5)  # 500ms
    
    if len(audio) < crossfade_len:
        return np.vstack([audio, tail])
    
    fade_out = np.sqrt(np.linspace(1.0, 0.0, crossfade_len))
    fade_in = np.sqrt(np.linspace(0.0, 1.0, crossfade_len))
    
    output = np.copy(audio)
    output[-crossfade_len:] = (
        output[-crossfade_len:] * fade_out[:, np.newaxis]
        + tail[:crossfade_len] * fade_in[:, np.newaxis]
    )
    
    remaining_tail = tail[crossfade_len:]
    return np.vstack([output, remaining_tail])


def analyze_sub_bass(audio, sr):
    """
    Analyze sub-bass content to determine optimal highpass frequency.
    
    Returns:
        (has_sub_bass, cutoff_freq, sub_ratio)
    """
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    sub_mask = (freqs >= 20) & (freqs <= 35)
    sub_energy = np.mean(np.abs(fft[sub_mask]) ** 2)
    
    ref_mask = (freqs >= 100) & (freqs <= 200)
    ref_energy = np.mean(np.abs(fft[ref_mask]) ** 2)
    
    if ref_energy > 0:
        sub_to_ref_ratio = sub_energy / ref_energy
    else:
        sub_to_ref_ratio = 0.0
    
    has_sub_bass = sub_to_ref_ratio > 0.15
    cutoff_freq = 23.0 if has_sub_bass else 32.0
    
    return has_sub_bass, cutoff_freq, sub_to_ref_ratio


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================


def process_file(
    input_file,
    output_file,
    profile_name='neutral',
    target_lufs=TARGET_LUFS,
    max_gain_db=MAX_GAIN_DB,
    fade_seconds=4.0,
    trim_start=True,
    fix_harshness=True
):
    """
    Process audio file with new analysis-driven architecture.
    """
    print(f"\nProcessing {input_file}")
    print(f"Profile: {profile_name}")
    
    # Load audio
    audio, sr = sf.read(input_file)
    
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    
    # Load profile
    profile = load_profile(profile_name)
    
    # Original metrics
    meter = pyln.Meter(sr)
    original_lufs = meter.integrated_loudness(audio)
    print(f"Original LUFS: {original_lufs:.2f}")
    print(f"Original Peak: {peak_db(audio):.2f} dBFS")
    
    # ========================================================================
    # STAGE 1: Start Noise Removal
    # ========================================================================
    
    if trim_start:
        trim_samples = detect_start_noise(audio, sr)
        if trim_samples > 0:
            trim_ms = (trim_samples / sr) * 1000
            print(f"\nStart noise detected, trimming {trim_ms:.1f}ms")
            audio = remove_start_noise(audio, sr, trim_samples)
    
    # ========================================================================
    # STAGE 2: Highpass Filter (Adaptive)
    # ========================================================================
    
    print("\n--- Spectral Analysis ---")
    has_sub_bass, hpf_cutoff, sub_ratio = analyze_sub_bass(audio, sr)
    print(f"Sub-bass: ratio={sub_ratio:.3f} ({linear_to_db(sub_ratio):.1f} dB)")
    print(f"Highpass: {hpf_cutoff:.0f} Hz")
    
    audio = apply_highpass_filter(audio, sr, hpf_cutoff)
    
    # ========================================================================
    # STAGE 3: Tonal Balance Analysis
    # ========================================================================
    
    print("\n--- Tonal Balance Analysis ---")
    tonal_analyzer = TonalBalanceAnalyzer(profile)
    tonal_analysis = tonal_analyzer.analyze(audio, sr)
    
    print("\nSpectral Balance:")
    print(tonal_analyzer.format_band_report(tonal_analysis['bands']))
    
    print("\nGenerated EQ:")
    print(tonal_analyzer.format_filter_report(tonal_analysis['filters']))
    
    # Apply static EQ
    if tonal_analysis['filters']:
        audio = apply_eq_filters(audio, sr, tonal_analysis['filters'])
    
    # ========================================================================
    # STAGE 4: Harshness Analysis
    # ========================================================================
    
    print("\n--- Harshness Analysis ---")
    harshness_analyzer = HarshnessAnalyzer()
    harshness_analysis = harshness_analyzer.analyze(audio, sr)
    
    print(harshness_analyzer.format_report(harshness_analysis))
    
    # Apply dynamic EQ if needed
    if fix_harshness and harshness_analysis['should_correct']:
        audio = apply_dynamic_eq(
            audio, sr,
            harshness_analysis['regions'],
            harshness_analyzer
        )
    
    # ========================================================================
    # STAGE 5: Stereo Analysis
    # ========================================================================
    
    print("\n--- Stereo Analysis ---")
    stereo_analyzer = StereoAnalyzer()
    stereo_analysis = stereo_analyzer.analyze(audio, sr)
    
    print(stereo_analyzer.format_report(stereo_analysis))
    
    # Apply stereo width adjustment
    if audio.shape[1] == 2:
        audio = apply_stereo_width_adjustment(
            audio,
            stereo_analysis['global_adjustment']
        )
    
    # ========================================================================
    # STAGE 6: Dynamics Analysis and Compression
    # ========================================================================
    
    print("\n--- Dynamics Analysis ---")
    dynamics_analyzer = DynamicsAnalyzer()
    dynamics_analysis = dynamics_analyzer.analyze(audio, sr)
    
    print(dynamics_analyzer.format_report(dynamics_analysis))
    
    # Apply compression
    audio = apply_compressor(
        audio, sr,
        dynamics_analysis['compression_settings']
    )
    
    # ========================================================================
    # STAGE 7: LUFS Normalization
    # ========================================================================
    
    print("\n--- Loudness Normalization ---")
    current_lufs = meter.integrated_loudness(audio)
    desired_gain_db = target_lufs - current_lufs
    actual_gain_db = min(desired_gain_db, max_gain_db)
    actual_target = current_lufs + actual_gain_db
    
    print(f"Target LUFS: {target_lufs:.2f}")
    print(f"Actual LUFS: {actual_target:.2f}")
    print(f"Gain: {actual_gain_db:+.2f} dB")
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Possible clipped samples')
        audio = pyln.normalize.loudness(audio, current_lufs, actual_target)
    
    # ========================================================================
    # STAGE 8: True Peak Limiter
    # ========================================================================
    
    print("\n--- Limiting ---")
    audio, limiting_db = apply_limiter(audio, sr, LIMITER_CEILING_DB)
    print(f"Limiting: {limiting_db:.2f} dB")
    
    # ========================================================================
    # STAGE 9: Fade Repair
    # ========================================================================
    
    print("\n--- Fade Detection ---")
    if needs_fade_completion(audio, sr):
        print("Extending fade...")
        audio = repair_fade_with_reverb(audio, sr, fade_seconds)
    
    # ========================================================================
    # STAGE 10: Final Protection
    # ========================================================================
    
    audio = hard_clip_protection(audio)
    
    # ========================================================================
    # Final Metrics
    # ========================================================================
    
    print("\n--- Output ---")
    final_lufs = meter.integrated_loudness(audio)
    final_peak = peak_db(audio)
    
    print(f"Final LUFS: {final_lufs:.2f}")
    print(f"Final Peak: {final_peak:.2f} dBFS")
    print(f"True Peak: {LIMITER_CEILING_DB:.2f} dBTP")
    
    # Write output
    sf.write(output_file, audio, sr, subtype='PCM_24')
    print(f"\nWritten: {output_file}")
    
    # ========================================================================
    # Mastering Report
    # ========================================================================
    
    print("\n" + "=" * 50)
    print("MASTERING REPORT")
    print("=" * 50)
    print(f"\nProfile: {profile['name']}")
    print(f"\nSpectral Balance:")
    print(tonal_analyzer.format_band_report(tonal_analysis['bands']))
    print(f"\nGenerated EQ:")
    print(tonal_analyzer.format_filter_report(tonal_analysis['filters']))
    print(f"\nHarshness:")
    print(harshness_analyzer.format_report(harshness_analysis))
    print(f"\nStereo:")
    print(stereo_analyzer.format_report(stereo_analysis))
    print(f"\nCompression:")
    print(dynamics_analyzer.format_report(dynamics_analysis))
    print(f"\nOutput:")
    print(f"  Integrated LUFS: {final_lufs:.2f}")
    print(f"  True Peak: {LIMITER_CEILING_DB:.2f} dBTP")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Audio mastering tool with analysis-driven processing'
    )
    
    parser.add_argument('input_file', help='Input audio file')
    parser.add_argument('output_file', help='Output audio file')
    
    parser.add_argument(
        '--profile',
        type=str,
        default='neutral',
        help='Tonal balance profile (edm, pop, rock, neutral)'
    )
    
    parser.add_argument(
        '--lufs',
        type=float,
        default=TARGET_LUFS,
        help='Target LUFS level'
    )
    
    parser.add_argument(
        '--max-gain',
        type=float,
        default=MAX_GAIN_DB,
        help='Maximum loudness increase in dB'
    )
    
    parser.add_argument(
        '--fade-seconds',
        type=float,
        default=5.0,
        help='Maximum duration of fade tail in seconds'
    )
    
    parser.add_argument(
        '--no-start-trim',
        action='store_true',
        help='Disable automatic removal of start noise'
    )
    
    parser.add_argument(
        '--no-harshness-fix',
        action='store_true',
        help='Disable dynamic harshness correction'
    )
    
    args = parser.parse_args()
    
    process_file(
        args.input_file,
        args.output_file,
        profile_name=args.profile,
        target_lufs=args.lufs,
        max_gain_db=args.max_gain,
        fade_seconds=args.fade_seconds,
        trim_start=not args.no_start_trim,
        fix_harshness=not args.no_harshness_fix
    )


if __name__ == '__main__':
    main()
