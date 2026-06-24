#!/usr/bin/env python3

import argparse
import warnings

import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from scipy.signal import butter, sosfilt
from scipy.signal import fftconvolve
from scipy.signal import get_window, medfilt
from scipy.ndimage import median_filter
from scipy.signal import get_window, medfilt
from scipy.ndimage import median_filter
from scipy.signal.windows import hann
from scipy.ndimage import median_filter


TARGET_LUFS = -12.0
MAX_GAIN_DB = 3.0
LIMITER_CEILING_DB = -1.0
HPF_FREQ = 35.0


def db_to_linear(db):
    return 10 ** (db / 20.0)


def linear_to_db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def peak_db(audio):
    return linear_to_db(np.max(np.abs(audio)))


def calculate_crest_factor(audio):
    """Calculate crest factor (peak to RMS ratio) in dB.
    
    Higher values indicate more dynamic range (less compressed).
    Lower values indicate already compressed audio.
    
    Returns:
        crest_factor_db: Peak - RMS in dB
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    
    peak_db = linear_to_db(peak)
    rms_db = linear_to_db(rms)
    
    crest_factor_db = peak_db - rms_db
    
    return crest_factor_db


def determine_compression_settings(crest_factor_db):
    """Determine compression settings based on crest factor.
    
    Many modern AI-generated tracks are already heavily compressed.
    Measure dynamics and apply compression only when needed.
    
    Args:
        crest_factor_db: Peak - RMS in dB
        
    Returns:
        (apply_compression, threshold_db, ratio)
    """
    if crest_factor_db > 14.0:
        # High dynamic range - needs full compression
        return True, -18.0, 2.0
    elif crest_factor_db > 10.0:
        # Moderate dynamic range - light compression
        return True, -15.0, 1.5
    else:
        # Already compressed - skip compression, let limiter handle peaks
        return False, None, None


def highpass_filter(audio, sr, cutoff=35):
    sos = butter(
        4,
        cutoff,
        btype="highpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, audio, axis=0)


def analyze_sub_bass(audio, sr):
    """Analyze sub-bass content to determine optimal highpass frequency.
    
    Some EDM tracks have intentional sub-bass content in the 28-35 Hz range.
    This function detects whether significant sub-bass is present.
    
    Returns:
        has_sub_bass: True if significant sub-bass content exists
        cutoff_freq: Recommended highpass cutoff (23 Hz or 32 Hz)
    """
    # Analyze frequency content
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    # Calculate energy in sub-bass range (20-35 Hz)
    sub_mask = (freqs >= 20) & (freqs <= 35)
    sub_energy = np.mean(np.abs(fft[sub_mask]) ** 2)
    
    # Calculate energy in low-mid reference range (100-200 Hz)
    ref_mask = (freqs >= 100) & (freqs <= 200)
    ref_energy = np.mean(np.abs(fft[ref_mask]) ** 2)
    
    # Calculate ratio (with safety for zero division)
    if ref_energy > 0:
        sub_to_ref_ratio = sub_energy / ref_energy
    else:
        sub_to_ref_ratio = 0.0
    
    # Threshold: if sub-bass is more than 15% of the reference energy,
    # consider it intentional content rather than rumble
    has_sub_bass = sub_to_ref_ratio > 0.15
    
    # Choose cutoff frequency
    if has_sub_bass:
        # Lower cutoff to preserve intentional sub-bass content
        cutoff_freq = 23.0
    else:
        # Higher cutoff to more aggressively remove rumble
        cutoff_freq = 32.0
    
    return has_sub_bass, cutoff_freq, sub_to_ref_ratio


def detect_muddiness_adaptive(audio, sr):
    """
    Adaptive mud detection that finds the actual problem frequency.
    
    Returns:
        tuple: (has_mud: bool, peak_freq: float, reduction_db: float)
    """
    mono = np.mean(audio, axis=1)

    # Analyze multiple segments instead of entire track
    segment_duration = 5.0  # seconds
    segment_samples = int(segment_duration * sr)
    
    # Take up to 5 segments spread across the track
    num_segments = min(5, len(mono) // segment_samples)
    if num_segments == 0:
        # Track too short, analyze whole thing
        segment_samples = len(mono)
        num_segments = 1
    
    all_peak_freqs = []
    all_excess_ratios = []
    
    for i in range(num_segments):
        start = i * (len(mono) - segment_samples) // max(1, num_segments - 1)
        if num_segments == 1:
            start = 0
        segment = mono[start:start + segment_samples]
        
        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1 / sr)

        # Focus on mud range: 180-500 Hz
        mud_mask = (freqs >= 180) & (freqs <= 500)
        mud_spectrum = spectrum[mud_mask]
        mud_freqs = freqs[mud_mask]
        
        if len(mud_spectrum) == 0:
            continue
        
        # Find peak in mud range
        peak_idx = np.argmax(mud_spectrum)
        peak_freq = mud_freqs[peak_idx]
        peak_energy = mud_spectrum[peak_idx]
        
        # Compare to neighboring bands (80Hz below and above)
        lower_mask = (freqs >= max(20, peak_freq - 80)) & (freqs < peak_freq - 20)
        upper_mask = (freqs > peak_freq + 20) & (freqs <= min(1000, peak_freq + 80))
        
        neighbor_energy = 0
        neighbor_count = 0
        
        if np.any(lower_mask):
            neighbor_energy += np.mean(spectrum[lower_mask])
            neighbor_count += 1
        if np.any(upper_mask):
            neighbor_energy += np.mean(spectrum[upper_mask])
            neighbor_count += 1
        
        if neighbor_count > 0:
            avg_neighbor = neighbor_energy / neighbor_count
            if avg_neighbor > 0:
                excess_ratio = peak_energy / avg_neighbor
                all_peak_freqs.append(peak_freq)
                all_excess_ratios.append(excess_ratio)
    
    if not all_excess_ratios:
        return False, 0, 0
    
    # Use median frequency and ratio
    median_peak_freq = np.median(all_peak_freqs)
    median_excess_ratio = np.median(all_excess_ratios)
    
    # Determine if there's mud and how much reduction needed
    # Excess ratio > 1.5 means peak is 50% louder than neighbors
    has_mud = median_excess_ratio > 1.5
    
    if has_mud:
        # Scale reduction based on excess
        # 1.5x excess = -0.5dB, 2.0x = -1.5dB, 2.5x+ = -2.5dB
        if median_excess_ratio >= 2.5:
            reduction_db = -2.5
        elif median_excess_ratio >= 2.0:
            reduction_db = -1.5
        elif median_excess_ratio >= 1.7:
            reduction_db = -1.0
        else:
            reduction_db = -0.5
    else:
        reduction_db = 0
    
    return has_mud, median_peak_freq, reduction_db


def adaptive_mud_cleanup(audio, sr, peak_freq, reduction_db, q=0.8):
    """
    Apply adaptive EQ cut at detected mud frequency.
    
    Args:
        audio: Audio array
        sr: Sample rate
        peak_freq: Center frequency for cut (Hz)
        reduction_db: Amount of reduction (negative dB)
        q: Q factor (bandwidth control, ~0.8 for gentle cut)
    """
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)

    # Calculate bandwidth from Q factor
    # Q = center_freq / bandwidth
    bandwidth = peak_freq / q
    
    # Create bell curve for EQ
    # Using Gaussian-like shape
    width_factor = bandwidth / 2.355  # Convert to sigma
    dip = np.exp(
        -0.5 * ((freqs - peak_freq) / width_factor) ** 2
    )

    # Convert reduction_db to linear gain
    # reduction_db is negative, so gain will be < 1.0
    min_gain = 10 ** (reduction_db / 20.0)
    
    # Apply the cut
    gain = 1.0 - (1.0 - min_gain) * dip

    fft *= gain[:, np.newaxis]

    return np.fft.irfft(
        fft,
        n=audio.shape[0],
        axis=0
    )


def detect_harshness(audio, sr, min_freq=2500, max_freq=9000):
    """
    Detect harsh resonances in the frequency range 2.5-9 kHz.
    
    Analyzes only the loudest 30% of frames for better detection.
    
    Returns:
        list of dicts: [{"freq": Hz, "peak": dB, "bandwidth": Hz}, ...]
    """
    mono = np.mean(audio, axis=1)
    
    # Calculate RMS of overlapping frames to find loudest sections
    window_size = 4096
    hop_size = 2048
    
    if len(mono) < window_size:
        return []
    
    num_frames = (len(mono) - window_size) // hop_size + 1
    frame_rms = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        frame = mono[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        frame_rms.append((i, rms))
    
    # Keep only loudest 30% of frames
    frame_rms.sort(key=lambda x: x[1], reverse=True)
    num_loud_frames = max(1, int(len(frame_rms) * 0.3))
    loud_frame_indices = sorted([idx for idx, _ in frame_rms[:num_loud_frames]])
    
    # Compute FFT for loud frames
    window = hann(window_size)
    spectra = []
    
    for frame_idx in loud_frame_indices:
        start = frame_idx * hop_size
        end = start + window_size
        if end > len(mono):
            continue
        frame = mono[start:end] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectra.append(spectrum)
    
    if len(spectra) == 0:
        return []
    
    # Average spectrum
    avg_spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.rfftfreq(window_size, 1 / sr)
    
    # Convert to dB
    spectrum_db = 20 * np.log10(np.maximum(avg_spectrum, 1e-12))
    
    # Restrict to harshness range
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    analysis_freqs = freqs[mask]
    analysis_spectrum = spectrum_db[mask]
    
    if len(analysis_spectrum) < 20:
        return []
    
    # Estimate local trend using median filter
    # 1/3 octave smoothing: approximate with frequency-dependent window
    # At 4kHz, 1/3 octave ≈ 1000Hz, which is ~256 bins at our resolution
    freq_resolution = sr / window_size  # Hz per bin
    octave_third_bins = int(1000 / freq_resolution)  # Approximate 1/3 octave
    octave_third_bins = max(11, min(octave_third_bins, len(analysis_spectrum) // 4))
    
    # Make odd for median filter
    if octave_third_bins % 2 == 0:
        octave_third_bins += 1
    
    smoothed_spectrum = median_filter(analysis_spectrum, size=octave_third_bins)
    
    # Compute resonance strength
    resonance_db = analysis_spectrum - smoothed_spectrum
    
    # Find candidate harsh regions (resonance > 2.5 dB)
    harsh_mask = resonance_db > 2.5
    
    if not np.any(harsh_mask):
        return []
    
    # Find contiguous regions
    regions = []
    in_region = False
    region_start = 0
    
    for i in range(len(harsh_mask)):
        if harsh_mask[i] and not in_region:
            region_start = i
            in_region = True
        elif not harsh_mask[i] and in_region:
            region_end = i
            regions.append((region_start, region_end))
            in_region = False
    
    if in_region:
        regions.append((region_start, len(harsh_mask)))
    
    # Analyze each region
    harsh_regions = []
    
    for start_idx, end_idx in regions:
        if end_idx - start_idx < 2:
            continue
        
        region_freqs = analysis_freqs[start_idx:end_idx]
        region_resonance = resonance_db[start_idx:end_idx]
        
        # Find peak
        peak_idx = np.argmax(region_resonance)
        peak_height = region_resonance[peak_idx]
        center_freq = region_freqs[peak_idx]
        
        # Estimate bandwidth
        bandwidth = region_freqs[-1] - region_freqs[0]
        
        # Reject narrow (<100Hz) and very wide (>3000Hz) regions
        if bandwidth < 100 or bandwidth > 3000:
            continue
        
        # Score = peak_height * log(bandwidth)
        score = peak_height * np.log10(bandwidth)
        
        harsh_regions.append({
            "freq": float(center_freq),
            "peak": float(peak_height),
            "bandwidth": float(bandwidth),
            "score": float(score)
        })
    
    # Sort by score and keep top 3
    harsh_regions.sort(key=lambda x: x["score"], reverse=True)
    return harsh_regions[:3]


def apply_dynamic_eq_band(audio, sr, center_freq, q, threshold_db, ratio=2.0, 
                          attack_ms=5.0, release_ms=50.0, max_reduction_db=-2.5):
    """
    Apply dynamic EQ to a specific frequency band.
    
    This implements a dynamic cut that only engages when the band exceeds threshold.
    """
    # Create bandpass filter to isolate the problem frequency
    bandwidth = center_freq / q
    low_freq = max(20, center_freq - bandwidth / 2)
    high_freq = min(sr / 2 - 100, center_freq + bandwidth / 2)
    
    # Design bandpass
    sos_bp = butter(4, [low_freq, high_freq], btype='bandpass', fs=sr, output='sos')
    
    # Extract band
    band_signal = sosfilt(sos_bp, audio, axis=0)
    
    # Compute envelope of band
    envelope_signal = np.abs(band_signal)
    
    # Smooth envelope (RMS-like)
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    
    threshold_linear = db_to_linear(threshold_db)
    max_gain = db_to_linear(max_reduction_db)  # This is < 1.0
    
    gain_reduction = np.ones(audio.shape[0])
    envelope = 0.0
    
    for i in range(audio.shape[0]):
        sample = np.max(np.abs(envelope_signal[i]))
        
        # Envelope follower
        if sample > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * sample
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * sample
        
        # Apply compression-style gain reduction
        if envelope > threshold_linear:
            # Amount over threshold
            over_db = linear_to_db(envelope) - threshold_db
            
            # Compressed amount
            compressed_over = over_db / ratio
            
            # Gain reduction in dB
            gr_db = -(over_db - compressed_over)
            
            # Clamp to max reduction
            gr_db = max(gr_db, max_reduction_db)
            
            gain_reduction[i] = db_to_linear(gr_db)
    
    # Apply gain reduction with the bandpass filter in frequency domain for precision
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    # Create bell-shaped gain curve
    bandwidth_factor = bandwidth / 2.355  # Gaussian width
    bell = np.exp(-0.5 * ((freqs - center_freq) / bandwidth_factor) ** 2)
    
    # Apply time-varying gain reduction
    for ch in range(audio.shape[1]):
        for freq_idx in range(len(freqs)):
            # Blend between full signal and reduced signal based on bell curve
            reduction_curve = 1.0 - (1.0 - gain_reduction) * bell[freq_idx]
            fft[freq_idx, ch] *= np.mean(reduction_curve)  # Simplified: use average
    
    return np.fft.irfft(fft, n=audio.shape[0], axis=0)


def apply_harshness_correction(audio, sr, harsh_regions):
    """
    Apply dynamic EQ to all detected harsh regions.
    """
    if not harsh_regions:
        return audio
    
    corrected = np.copy(audio)
    
    for region in harsh_regions:
        center_freq = region["freq"]
        bandwidth = region["bandwidth"]
        peak_height = region["peak"]
        
        # Calculate Q from bandwidth
        q = center_freq / bandwidth
        q = np.clip(q, 0.8, 4.0)
        
        # Estimate band RMS for threshold
        # Use a narrow analysis window around the frequency
        mono = np.mean(audio, axis=1)
        sos = butter(4, [max(20, center_freq - bandwidth), 
                         min(sr/2 - 100, center_freq + bandwidth)], 
                     btype='bandpass', fs=sr, output='sos')
        band = sosfilt(sos, mono)
        band_rms_db = linear_to_db(np.sqrt(np.mean(band ** 2)))
        
        # Threshold = band RMS + 2dB
        threshold_db = band_rms_db + 2.0
        
        # Max reduction scales with peak height
        if peak_height >= 4.5:
            max_reduction = -2.5
        elif peak_height >= 3.5:
            max_reduction = -2.0
        else:
            max_reduction = -1.5
        
        print(f"    Dynamic EQ at {center_freq:.0f}Hz (Q={q:.1f}, max reduction={max_reduction:.1f}dB)")
        
        corrected = apply_dynamic_eq_band(
            corrected, sr, center_freq, q, threshold_db,
            ratio=2.0, attack_ms=5.0, release_ms=50.0,
            max_reduction_db=max_reduction
        )
    
    return corrected


def compressor(
    audio,
    sr,
    threshold_db=-18,
    ratio=2.0,
    attack_ms=20,
    release_ms=100,
    knee_db=6.0,
    lookahead_ms=5.0
):
    """Improved compressor with soft knee and lookahead for smoother compression."""
    threshold = db_to_linear(threshold_db)
    knee = db_to_linear(knee_db)

    attack_coeff = np.exp(
        -1.0 / (sr * attack_ms / 1000)
    )

    release_coeff = np.exp(
        -1.0 / (sr * release_ms / 1000)
    )

    # Lookahead buffer
    lookahead_samples = int(sr * lookahead_ms / 1000)

    envelope = 0.0
    gain_curve = np.ones(audio.shape[0])

    # Stereo-linked detector (max of both channels)
    detector = np.max(
        np.abs(audio),
        axis=1
    )

    # Pad detector for lookahead
    detector_padded = np.pad(
        detector,
        (lookahead_samples, 0),
        mode='edge'
    )

    for i in range(len(detector)):
        # Look ahead
        sample = detector_padded[i + lookahead_samples]

        # Envelope follower
        if sample > envelope:
            envelope = (
                attack_coeff * envelope
                + (1 - attack_coeff) * sample
            )
        else:
            envelope = (
                release_coeff * envelope
                + (1 - release_coeff) * sample
            )

        if envelope > threshold:
            # Soft knee compression
            over = envelope - threshold
            
            if over < knee:
                # Soft knee region
                gain_reduction = over * over / (2 * knee * ratio)
                compressed = envelope - gain_reduction
            else:
                # Hard compression above knee
                compressed = (
                    threshold + knee / 2
                    + (over - knee) / ratio
                )

            gain_curve[i] = compressed / envelope

    return audio * gain_curve[:, np.newaxis]


def loudness_normalize(
    audio,
    sr,
    target_lufs,
    max_gain_db
):
    meter = pyln.Meter(sr)

    current_lufs = meter.integrated_loudness(audio)

    desired_gain_db = target_lufs - current_lufs

    actual_gain_db = min(
        desired_gain_db,
        max_gain_db
    )

    actual_target = (
        current_lufs + actual_gain_db
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Possible clipped samples in output."
        )

        normalized = pyln.normalize.loudness(
            audio,
            current_lufs,
            actual_target
        )

    return (
        normalized,
        current_lufs,
        actual_target,
        actual_gain_db
    )


def limiter(audio, sr, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=100.0):
    """Improved limiter with lookahead and smooth release."""
    ceiling = db_to_linear(ceiling_db)
    
    lookahead_samples = int(sr * lookahead_ms / 1000)
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    
    # Stereo-linked peak detection
    detector = np.max(np.abs(audio), axis=1)
    
    # Pad for lookahead
    detector_padded = np.pad(
        detector,
        (lookahead_samples, 0),
        mode='edge'
    )
    
    envelope = 0.0
    gain_curve = np.ones(audio.shape[0])
    
    for i in range(len(detector)):
        # Look ahead
        sample = detector_padded[i + lookahead_samples]
        
        # Instant attack, smooth release
        if sample > envelope:
            envelope = sample
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * sample
        
        # Calculate gain reduction
        if envelope > ceiling:
            gain_curve[i] = ceiling / envelope
    
    # Apply gain reduction
    audio_limited = audio * gain_curve[:, np.newaxis]
    
    # Calculate total limiting
    limiting_db = linear_to_db(np.min(gain_curve))
    
    return audio_limited, -limiting_db


def hard_clip_protection(audio):
    return np.clip(
        audio,
        -0.999,
        0.999
    )


def enhance_high_frequency_air(audio, sr):
    """Add adaptive high-frequency enhancement for 'air' and presence.
    
    Analyzes the balance between high frequencies (8-16 kHz) and
    mid frequencies (1-4 kHz). Boosts only when highs are deficient,
    avoiding the amplification of AI artifacts.
    """
    # Analyze frequency content
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    # Calculate energy in different bands
    # Mid range: 1-4 kHz (reference)
    mid_mask = (freqs >= 1000) & (freqs <= 4000)
    mid_energy = np.mean(np.abs(fft[mid_mask]) ** 2)
    
    # High range: 8-16 kHz (air band)
    high_mask = (freqs >= 8000) & (freqs <= 16000)
    high_energy = np.mean(np.abs(fft[high_mask]) ** 2)
    
    # Calculate ratio (with safety for zero division)
    if mid_energy > 0:
        hf_to_mid_ratio = high_energy / mid_energy
    else:
        hf_to_mid_ratio = 1.0
    
    # Target ratio: highs should be about 40-60% of mids
    # (in energy terms, which is about -4 to -2 dB)
    target_ratio = 0.5  # -3 dB
    
    # Calculate needed boost
    if hf_to_mid_ratio >= target_ratio:
        # Already has enough high frequency content
        boost_db = 0.0
    else:
        # Calculate boost needed (max 1.5 dB)
        ratio_deficit = target_ratio / max(hf_to_mid_ratio, 0.01)
        boost_db = min(1.5, linear_to_db(ratio_deficit) * 0.3)
    
    print(f"  High-frequency analysis:")
    print(f"    HF/Mid ratio: {hf_to_mid_ratio:.3f} ({linear_to_db(hf_to_mid_ratio):.1f} dB)")
    print(f"    Air boost: {boost_db:.1f} dB")
    
    if boost_db < 0.1:
        # No boost needed
        return audio
    
    # Apply adaptive shelf boost above 8kHz
    shelf_freq = 8000.0
    shelf_gain = db_to_linear(boost_db)
    transition_width = 2000.0
    
    # Smooth transition using sigmoid-like curve
    boost = np.ones_like(freqs)
    mask = freqs > (shelf_freq - transition_width)
    boost[mask] = 1.0 + (shelf_gain - 1.0) * (
        1.0 / (1.0 + np.exp(-5 * (freqs[mask] - shelf_freq) / transition_width))
    )
    
    fft *= boost[:, np.newaxis]
    
    return np.fft.irfft(fft, n=audio.shape[0], axis=0)


def analyze_stereo_width(audio):
    """Analyze stereo width by measuring mid/side energy ratio.
    
    Returns:
        width_ratio: Side RMS / Mid RMS
        recommended_width: Suggested width adjustment
    """
    if audio.shape[1] < 2:
        return 0.0, 1.0
    
    # Convert to mid-side
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0
    
    # Calculate RMS
    mid_rms = np.sqrt(np.mean(mid ** 2))
    side_rms = np.sqrt(np.mean(side ** 2))
    
    # Calculate width ratio (with safety for zero division)
    if mid_rms > 1e-12:
        width_ratio = side_rms / mid_rms
    else:
        width_ratio = 0.0
    
    # Determine recommended width adjustment
    # Typical values:
    # - Mono or very narrow: < 0.3
    # - Normal stereo: 0.3 - 0.8
    # - Wide stereo: 0.8 - 1.2
    # - Excessively wide: > 1.2 (can cause mono compatibility issues)
    
    if width_ratio < 0.2:
        # Very narrow - widen significantly
        recommended_width = 1.4
    elif width_ratio < 0.4:
        # Narrow - widen moderately
        recommended_width = 1.2
    elif width_ratio > 1.2:
        # Excessively wide - narrow to protect mono compatibility
        recommended_width = 0.7
    elif width_ratio > 0.9:
        # Quite wide - narrow slightly
        recommended_width = 0.85
    else:
        # Normal range (0.4 - 0.9) - leave alone
        recommended_width = 1.0
    
    return width_ratio, recommended_width


def adjust_stereo_width(audio, width=1.0):
    """Adjust stereo width using mid-side processing.
    
    width < 1.0: narrow
    width = 1.0: unchanged
    width > 1.0: wider
    
    Suno tracks sometimes have inconsistent stereo imaging.
    """
    if audio.shape[1] < 2 or width == 1.0:
        return audio
    
    # Convert to mid-side
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0
    
    # Adjust side signal
    side *= width
    
    # Convert back to left-right
    left = mid + side
    right = mid - side
    
    return np.column_stack([left, right])


def detect_harshness(audio, sr):
    """
    Detect harsh resonances in the 2.5-9 kHz range.
    
    Analyzes the loudest 30% of frames to focus on actual content.
    
    Returns:
        list: Detected harsh regions with freq, peak_db, bandwidth
    """
    mono = np.mean(audio, axis=1)
    
    # FFT parameters
    window_size = 4096
    hop_size = 2048
    window = get_window('hann', window_size)
    
    # Calculate number of frames
    num_frames = (len(mono) - window_size) // hop_size + 1
    
    if num_frames < 10:
        return []  # Too short to analyze
    
    # Compute spectra for all frames
    spectra = []
    frame_rms = []
    
    for i in range(num_frames):
        start = i * hop_size
        frame = mono[start:start + window_size]
        
        if len(frame) < window_size:
            break
        
        # Apply window
        windowed = frame * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        
        spectra.append(magnitude)
        frame_rms.append(np.sqrt(np.mean(frame ** 2)))
    
    if not spectra:
        return []
    
    spectra = np.array(spectra)
    frame_rms = np.array(frame_rms)
    
    # Analyze only the loudest 30% of frames
    threshold_idx = int(len(frame_rms) * 0.7)
    rms_threshold = np.sort(frame_rms)[threshold_idx]
    loud_frames = frame_rms >= rms_threshold
    
    if np.sum(loud_frames) < 5:
        # Not enough loud frames
        return []
    
    # Average spectrum from loud frames only
    avg_spectrum = np.mean(spectra[loud_frames], axis=0)
    
    # Convert to dB
    spectrum_db = linear_to_db(avg_spectrum)
    
    # Get frequency bins
    freqs = np.fft.rfftfreq(window_size, 1 / sr)
    
    # Restrict to 2.5-9 kHz range
    mask = (freqs >= 2500) & (freqs <= 9000)
    analysis_freqs = freqs[mask]
    analysis_spectrum = spectrum_db[mask]
    
    if len(analysis_spectrum) < 20:
        return []
    
    # Estimate local spectral trend using median filter
    # Use 1/3 octave smoothing width
    smooth_width = max(5, int(len(analysis_spectrum) / 20))  # Adaptive smoothing
    if smooth_width % 2 == 0:
        smooth_width += 1  # Must be odd for median filter
    
    smoothed_spectrum = medfilt(analysis_spectrum, kernel_size=smooth_width)
    
    # Compute resonance strength
    resonance_db = analysis_spectrum - smoothed_spectrum
    
    # Find harsh regions where resonance > 2.5 dB
    harsh_threshold = 2.5
    is_harsh = resonance_db > harsh_threshold
    
    if not np.any(is_harsh):
        return []  # No harshness detected
    
    # Find contiguous regions
    harsh_regions = []
    in_region = False
    region_start = 0
    
    for i, harsh in enumerate(is_harsh):
        if harsh and not in_region:
            region_start = i
            in_region = True
        elif not harsh and in_region:
            # End of region
            region_end = i
            
            # Calculate region properties
            region_slice = slice(region_start, region_end)
            region_freqs = analysis_freqs[region_slice]
            region_resonance = resonance_db[region_slice]
            
            center_freq = float(np.mean(region_freqs))
            peak_height = float(np.max(region_resonance))
            bandwidth = float(region_freqs[-1] - region_freqs[0])
            
            # Filter by bandwidth (reject too narrow or too wide)
            if 100 < bandwidth < 3000:
                # Score by peak height and bandwidth
                score = peak_height * np.log10(bandwidth)
                
                harsh_regions.append({
                    'freq': center_freq,
                    'peak': peak_height,
                    'bandwidth': bandwidth,
                    'score': score
                })
            
            in_region = False
    
    # Handle case where region extends to end
    if in_region:
        region_slice = slice(region_start, len(is_harsh))
        region_freqs = analysis_freqs[region_slice]
        region_resonance = resonance_db[region_slice]
        
        center_freq = float(np.mean(region_freqs))
        peak_height = float(np.max(region_resonance))
        bandwidth = float(region_freqs[-1] - region_freqs[0])
        
        if 100 < bandwidth < 3000:
            score = peak_height * np.log10(bandwidth)
            harsh_regions.append({
                'freq': center_freq,
                'peak': peak_height,
                'bandwidth': bandwidth,
                'score': score
            })
    
    # Sort by score and keep top 3
    harsh_regions.sort(key=lambda x: x['score'], reverse=True)
    return harsh_regions[:3]


def apply_dynamic_eq_multiband(audio, sr, harsh_regions):
    """
    Apply dynamic EQ to reduce harsh resonances.
    
    Uses frame-by-frame processing to compress only when harshness exceeds threshold.
    """
    if not harsh_regions:
        return audio
    
    # Process in frequency domain with overlap-add
    window_size = 4096
    hop_size = 2048
    window = get_window('hann', window_size)
    
    output = np.zeros_like(audio)
    window_sum = np.zeros(len(audio))
    
    num_frames = (len(audio) - window_size) // hop_size + 1
    
    for ch in range(audio.shape[1]):
        for i in range(num_frames):
            start = i * hop_size
            end = start + window_size
            
            if end > len(audio):
                break
            
            frame = audio[start:end, ch].copy()
            windowed = frame * window
            
            # FFT
            fft_result = np.fft.rfft(windowed)
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)
            
            freqs = np.fft.rfftfreq(window_size, 1 / sr)
            magnitude_db = linear_to_db(magnitude + 1e-12)
            
            # Apply dynamic reduction for each harsh region
            for region in harsh_regions:
                center = region['freq']
                bandwidth = region['bandwidth']
                peak = region['peak']
                
                # Calculate Q
                q = np.clip(center / bandwidth, 0.8, 6.0)
                
                # Create frequency mask for this band
                half_bandwidth = bandwidth / 2
                band_mask = (freqs >= center - half_bandwidth) & (freqs <= center + half_bandwidth)
                
                if not np.any(band_mask):
                    continue
                
                # Calculate band RMS
                band_energy = magnitude[band_mask]
                if len(band_energy) == 0:
                    continue
                
                band_rms_db = linear_to_db(np.sqrt(np.mean(band_energy ** 2)) + 1e-12)
                
                # Threshold: band_rms + 2dB
                threshold_db = band_rms_db + 2.0
                
                # Check if this frame exceeds threshold
                band_peak_db = np.max(magnitude_db[band_mask])
                
                if band_peak_db > threshold_db:
                    # Apply compression with 2:1 ratio
                    excess_db = band_peak_db - threshold_db
                    reduction_db = excess_db / 2.0  # 2:1 ratio
                    
                    # Clamp to maximum reduction
                    max_reduction = min(2.5, peak * 0.6)  # Scale with detected peak
                    reduction_db = min(reduction_db, max_reduction)
                    
                    # Apply bell-shaped reduction
                    reduction_linear = 10 ** (-reduction_db / 20.0)
                    
                    # Create smooth bell curve
                    bell = np.exp(-0.5 * ((freqs - center) / (bandwidth / 2.355)) ** 2)
                    gain = 1.0 - (1.0 - reduction_linear) * bell
                    
                    magnitude *= gain
            
            # Reconstruct signal
            fft_modified = magnitude * np.exp(1j * phase)
            reconstructed = np.fft.irfft(fft_modified, n=window_size)
            
            # Overlap-add
            output[start:end, ch] += reconstructed * window
            if ch == 0:  # Only compute window sum once
                window_sum[start:end] += window ** 2
    
    # Normalize by window sum
    window_sum[window_sum < 1e-8] = 1.0
    output /= window_sum[:, np.newaxis]
    
    return output


def should_apply_harshness_correction(audio, sr, harsh_regions):
    """
    Safety checks to avoid processing balanced tracks.
    
    Returns: True if harshness correction should be applied
    """
    if not harsh_regions:
        return False
    
    # Check max resonance
    max_resonance = max(r['peak'] for r in harsh_regions)
    if max_resonance < 2.5:
        return False
    
    # Check LUFS and crest factor for extremely dense mixes
    try:
        meter = pyln.Meter(sr)
        track_lufs = meter.integrated_loudness(audio)
        
        # Calculate crest factor
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        crest_factor_db = linear_to_db(peak / (rms + 1e-12))
        
        # Skip extremely dense mixes (false positive risk)
        if track_lufs > -10 and crest_factor_db < 8:
            return False
    except:
        pass  # If measurement fails, proceed anyway
    
    return True


def detect_start_noise(audio, sr, max_check_duration=0.6):
    """
    Detect noise artifacts at the start of Suno-generated tracks.
    
    Common pattern:
    - Brief noise burst (5-10ms) at moderate/high level
    - Followed by very low level or near-silence
    - Then music gradually starts
    
    Returns: sample index where actual content begins (0 if no noise detected)
    """
    mono = np.mean(audio, axis=1)
    
    # Analyze first max_check_duration seconds
    check_samples = min(int(sr * max_check_duration), len(mono))
    analysis_region = mono[:check_samples]
    
    # Split into small windows (10ms)
    window_size = int(sr * 0.01)
    num_windows = check_samples // window_size
    
    if num_windows < 10:
        return 0  # Too short to analyze
    
    # Calculate RMS for each window
    rms_values = []
    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(analysis_region))
        window = analysis_region[start:end]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
    
    # Convert to dB
    rms_db = [linear_to_db(rms) if rms > 0 else -120.0 for rms in rms_values]
    
    # Pattern detection: burst -> silence -> gradual rise
    # Check if first window is much louder than what follows
    first_window_db = rms_db[0]
    
    # Look for near-silence period after first window
    if first_window_db > -40.0:  # First window has content
        # Check if next few windows drop significantly
        next_windows = rms_db[1:min(5, len(rms_db))]
        if next_windows:
            min_next = min(next_windows)
            avg_next = np.mean(next_windows)
            
            # If drops to near-silence (below -80dB)
            if avg_next < -80.0 and min_next < -90.0:
                # Find where music actually starts (sustained rise)
                for i in range(5, min(60, len(rms_db))):
                    # Look for sustained level above -70dB
                    if i < len(rms_db) - 3:
                        window_avg = np.mean(rms_db[i:min(i+3, len(rms_db))])
                        if window_avg > -70.0:
                            # Music starts here, trim everything before
                            trim_point = i * window_size
                            return trim_point
                
                # If we found silence but no clear music start,
                # trim at least past the silence
                for i in range(1, min(50, len(rms_db))):
                    if rms_db[i] > -90.0:
                        trim_point = max(0, (i - 1) * window_size)
                        return trim_point
    
    # Alternative pattern: initial burst followed by drop then immediate rise
    for i in range(1, min(10, len(rms_db) - 2)):
        level_jump = rms_db[i] - rms_db[i-1]
        
        # Significant upward jump
        if level_jump > 15.0:
            # Verify sustained
            if i < len(rms_db) - 2:
                avg_after = np.mean(rms_db[i:min(i+5, len(rms_db))])
                avg_before = np.mean(rms_db[max(0, i-3):i])
                
                if avg_after - avg_before > 12.0:
                    trim_point = i * window_size
                    return trim_point
    
    return 0  # No noise detected


def remove_start_noise(audio, sr, trim_samples):
    """
    Remove noise from the start and add a short fade-in.
    
    Args:
        audio: Audio array
        sr: Sample rate
        trim_samples: Number of samples to trim from start
    
    Returns: Trimmed audio with fade-in
    """
    if trim_samples <= 0:
        return audio
    
    # Trim the noise
    trimmed = audio[trim_samples:]
    
    # Add short fade-in (10ms) to avoid click
    fade_samples = int(sr * 0.01)
    fade_samples = min(fade_samples, len(trimmed))
    
    fade_curve = np.linspace(0.0, 1.0, fade_samples)
    fade_curve = fade_curve ** 0.5  # Slight curve
    
    trimmed[:fade_samples] *= fade_curve[:, np.newaxis]
    
    return trimmed


def process_file(
    input_file,
    output_file,
    target_lufs=TARGET_LUFS,
    max_gain_db=MAX_GAIN_DB,
    fade_seconds=4.0,
    stereo_width=1.0,
    add_air=True,
    trim_start=True,
    fix_harshness=True
):
    print(f"\nProcessing {input_file}")

    audio, sr = sf.read(input_file)

    if audio.ndim == 1:
        audio = np.column_stack(
            [audio, audio]
        )

    # Check for and remove start noise (common in Suno tracks)
    if trim_start:
        trim_samples = detect_start_noise(audio, sr)
        if trim_samples > 0:
            trim_ms = (trim_samples / sr) * 1000
            print(
                f"Detected start noise, trimming {trim_ms:.1f}ms"
            )
            audio = remove_start_noise(audio, sr, trim_samples)

    meter = pyln.Meter(sr)

    original_lufs = (
        meter.integrated_loudness(audio)
    )

    print(
        f"Original LUFS: "
        f"{original_lufs:.2f}"
    )

    print(
        f"Original Peak: "
        f"{peak_db(audio):.2f} dBFS"
    )

    # Phase 1: Corrective EQ
    # Analyze sub-bass content to determine optimal HPF frequency
    has_sub_bass, hpf_cutoff, sub_ratio = analyze_sub_bass(audio, sr)
    
    print(f"  Sub-bass analysis:")
    print(f"    Sub/Ref ratio: {sub_ratio:.3f} ({linear_to_db(sub_ratio):.1f} dB)")
    print(f"    Intentional sub-bass detected: {has_sub_bass}")
    print(f"    Highpass cutoff: {hpf_cutoff:.0f} Hz")
    
    audio = highpass_filter(
        audio,
        sr,
        hpf_cutoff
    )

    # Adaptive mud detection and cleanup
    has_mud, mud_freq, reduction_db = detect_muddiness_adaptive(audio, sr)
    
    if has_mud:
        print(
            f"Applying adaptive mud cleanup at {mud_freq:.0f}Hz "
            f"({reduction_db:.1f}dB reduction)..."
        )
        audio = adaptive_mud_cleanup(
            audio,
            sr,
            mud_freq,
            reduction_db,
            q=0.8
        )

    # Harshness detection and correction (2.5-9 kHz)
    if fix_harshness:
        # Safety check: measure current loudness and dynamics
        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(audio)
        
        # Calculate crest factor
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / rms if rms > 0 else 999
        crest_factor_db = linear_to_db(crest_factor)
        
        # Apply safety rules
        safe_to_process = True
        
        # Skip extremely loud and dense mixes (likely already mastered)
        if current_lufs > -10 and crest_factor_db < 8:
            safe_to_process = False
            print("  Skipping harshness correction (track appears heavily limited)")
        
        if safe_to_process:
            harsh_regions = detect_harshness(audio, sr)
            
            # Check if any resonance is strong enough
            if harsh_regions:
                max_resonance = max(r["peak"] for r in harsh_regions)
                
                if max_resonance >= 2.5:
                    print(f"  Detected {len(harsh_regions)} harsh resonance(s):")
                    for r in harsh_regions:
                        print(f"    {r['freq']:.0f}Hz: +{r['peak']:.1f}dB (bandwidth: {r['bandwidth']:.0f}Hz)")
                    
                    audio = apply_harshness_correction(audio, sr, harsh_regions)

    # Phase 2: Stereo width adjustment (before compression)
    if audio.shape[1] == 2:
        # Analyze current stereo width
        width_ratio, recommended_width = analyze_stereo_width(audio)
        
        # Use manual override if specified, otherwise use adaptive width
        if stereo_width != 1.0:
            # User specified a manual width
            final_width = stereo_width
            print(f"  Stereo width analysis:")
            print(f"    Current side/mid ratio: {width_ratio:.2f}")
            print(f"    Manual width override: {final_width:.2f}")
        else:
            # Use adaptive width based on analysis
            final_width = recommended_width
            print(f"  Stereo width analysis:")
            print(f"    Current side/mid ratio: {width_ratio:.2f}")
            if final_width != 1.0:
                print(f"    Recommended adjustment: {final_width:.2f}")
            else:
                print(f"    Width is optimal, no adjustment needed")
        
        if final_width != 1.0:
            audio = adjust_stereo_width(audio, final_width)

    # Phase 3: Dynamics (conditional compression)
    # Measure crest factor to determine if compression is needed
    crest_factor = calculate_crest_factor(audio)
    apply_comp, threshold, ratio = determine_compression_settings(crest_factor)
    
    print(f"  Dynamic range analysis:")
    print(f"    Crest factor: {crest_factor:.1f} dB")
    
    if apply_comp:
        print(f"    Compression: threshold={threshold:.0f}dB, ratio={ratio:.1f}:1")
        audio = compressor(
            audio,
            sr,
            threshold_db=threshold,
            ratio=ratio,
            knee_db=6.0,
            lookahead_ms=5.0
        )
    else:
        print(f"    Compression: skipped (already compressed)")

    # Phase 4: Harshness detection and correction
    harsh_regions = detect_harshness(audio, sr)
    
    if should_apply_harshness_correction(audio, sr, harsh_regions):
        harsh_desc = ", ".join(
            f"{r['freq']:.0f}Hz ({r['peak']:.1f}dB)"
            for r in harsh_regions
        )
        print(
            f"Applying dynamic EQ for harshness: {harsh_desc}"
        )
        audio = apply_dynamic_eq_multiband(audio, sr, harsh_regions)

    # Phase 5: Loudness normalization
    (
        audio,
        measured_lufs,
        actual_target,
        gain_added
    ) = loudness_normalize(
        audio,
        sr,
        target_lufs,
        max_gain_db
    )

    print(
        f"Requested LUFS: {target_lufs:.2f}"
    )

    print(
        f"Actual LUFS target: "
        f"{actual_target:.2f}"
    )

    print(
        f"Gain added: "
        f"{gain_added:.2f} dB"
    )

    peak_after_norm = peak_db(audio)

    print(
        f"Peak after normalization: "
        f"{peak_after_norm:.2f} dBFS"
    )

    # Phase 6: Enhancement (after normalization, before limiting)
    if add_air:
        print("Analyzing high-frequency content...")
        audio = enhance_high_frequency_air(audio, sr)

    # Phase 7: Limiting
    audio, limiting_db = limiter(
        audio,
        sr,
        LIMITER_CEILING_DB,
        lookahead_ms=5.0,
        release_ms=100.0
    )

    audio = hard_clip_protection(audio)

    if needs_fade_completion(audio, sr):
        print(
            "Fade appears unfinished, "
            "extending fade..."
        )

        audio = repair_fade_with_reverb(
            audio,
            sr,
            tail_seconds=fade_seconds
        )

    final_lufs = (
        meter.integrated_loudness(audio)
    )

    final_peak = peak_db(audio)

    print(
        f"Limiter reduction: "
        f"{limiting_db:.2f} dB"
    )

    print(
        f"Final LUFS: "
        f"{final_lufs:.2f}"
    )

    print(
        f"Final Peak: "
        f"{final_peak:.2f} dBFS"
    )

    print(
        f"LUFS loss from limiting: "
        f"{actual_target - final_lufs:.2f}"
    )

    sf.write(
        output_file,
        audio,
        sr,
        subtype="PCM_24"
    )

    print(
        f"Written: {output_file}"
    )

def needs_fade_completion(audio, sr):
    """
    Detect if the track is fading but hasn't reached silence.
    
    Uses dynamic fade detection to find where the fade actually starts.
    
    Checks:
    1. Consistent downward trend exists
    2. Total fade amount is significant (>1.5dB)
    3. Ending hasn't reached silence OR fade is too short (< 2.0s)
    """
    mono = np.mean(audio, axis=1)

    # Use dynamic fade detection
    fade_start_sample, fade_duration, rms_windows = detect_fade_start(audio, sr)
    
    # If a fade was detected, check if it's incomplete
    if fade_duration < 0.2:  # Less than 0.2 seconds - probably not a fade
        return False
    
    # Get fade drop amount
    fade_drop = rms_windows[0] - rms_windows[-1] if len(rms_windows) > 1 else 0
    
    # Check both peak and RMS of the ending
    end_section = mono[-int(sr * 0.2):]
    
    end_peak = linear_to_db(
        np.max(np.abs(end_section))
    )
    
    end_rms = linear_to_db(
        np.sqrt(np.mean(end_section ** 2))
    )

    # Criteria for needing completion:
    # 1. Fade detected with at least 1.5 dB drop (from detect_fade_start)
    # 2. EITHER:
    #    a) Ending must be above silence threshold (-35 dB peak OR -40 dB RMS)
    #    b) Fade exists but is too short to sound smooth (< 2.0 seconds)
    
    fading = fade_drop > 1.5
    unfinished = end_peak > -35.0 or end_rms > -40.0
    too_short = fade_duration < 2.0
    
    needs_extension = fading and (unfinished or too_short)

    # Always show debug output for transparency
    print(f"  Fade analysis:")
    print(f"    Detected fade duration: {fade_duration:.2f}s")
    print(f"    Fade drop: {fade_drop:.1f} dB")
    print(f"    End peak: {end_peak:.1f} dB")
    print(f"    End RMS: {end_rms:.1f} dB")
    print(f"    Fading detected: {fading}")
    print(f"    Unfinished: {unfinished}")
    print(f"    Too short: {too_short}")
    print(f"    Needs completion: {needs_extension}")

    return needs_extension

def generate_impulse_response(
    sr,
    tail_seconds=5.0,
    decay_db=80.0
):

    n = int(sr * tail_seconds)

    ir = np.zeros(n)

    # direct impulse
    ir[0] = 1.0

    rng = np.random.default_rng()

    reflections = 500

    positions = rng.integers(
        1,
        n,
        size=reflections
    )

    for pos in positions:

        decay = np.exp(
            -pos / n * 8
        )

        ir[pos] += (
            rng.uniform(
                -1,
                1
            )
            * decay
        )

    ir /= np.max(
        np.abs(ir)
    )

    return ir

def generate_reverb_tail(
    audio,
    sr,
    source_seconds=2.0,
    tail_seconds=5.0,
    decay_db=80.0
):
    """
    Generate a realistic reverb tail
    using convolution.

    Returns only the generated tail.
    """

    source_len = int(
        source_seconds * sr
    )

    tail_len = int(
        tail_seconds * sr
    )

    source = audio[-source_len:]

    ir = generate_impulse_response(
        sr,
        tail_seconds,
        decay_db
    )

    channels = []

    for ch in range(
        source.shape[1]
    ):

        wet = fftconvolve(
            source[:, ch],
            ir,
            mode="full"
        )

        wet = wet[
            source_len:
            source_len + tail_len
        ]

        channels.append(wet)

    tail = np.column_stack(
        channels
    )

    # Match level to actual ending more precisely
    # Use the very last portion (50ms) for better continuity
    
    end_sample_len = min(
        int(sr * 0.05),  # 50ms
        len(source)
    )
    
    end_rms = np.sqrt(
        np.mean(
            source[-end_sample_len:] ** 2
        )
    )

    # Match to the start of the tail
    start_sample_len = min(
        int(sr * 0.05),
        len(tail)
    )
    
    start_rms = np.sqrt(
        np.mean(
            tail[:start_sample_len] ** 2
        )
    )

    if start_rms > 1e-9:
        tail *= (
            end_rms
            / start_rms
        )
    
    # Apply additional fade to ensure smooth decay to silence
    fade_curve = np.linspace(1.0, 0.0, len(tail))
    fade_curve = fade_curve ** 1.5  # Gentle exponential curve
    tail *= fade_curve[:, np.newaxis]

    return tail

def detect_fade_start(audio, sr, max_lookback=8.0):
    """
    Detect where the fade actually starts by analyzing RMS in small windows.
    
    Returns:
        - fade_start_sample: Index where fade begins
        - fade_duration: Duration of detected fade in seconds
    """
    mono = np.mean(audio, axis=1)
    
    # Look back up to max_lookback seconds
    lookback_samples = min(int(max_lookback * sr), len(mono))
    analysis_region = mono[-lookback_samples:]
    
    # Analyze in small windows (200ms)
    window_size = int(sr * 0.2)
    num_windows = max(1, lookback_samples // window_size)
    
    # Calculate RMS for each window
    rms_db = []
    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(analysis_region))
        window = analysis_region[start:end]
        rms_db.append(linear_to_db(np.sqrt(np.mean(window ** 2))))
    
    # Find where consistent decline begins
    # Start from the end and work backwards
    fade_start_idx = len(rms_db) - 1  # Default: assume entire region is fading
    
    # Look for the point where the decline becomes consistent
    # We want at least 2dB drop over at least 2 windows
    for i in range(len(rms_db) - 2, -1, -1):
        # Check if there's a consistent downward trend from this point forward
        is_declining = True
        for j in range(i, len(rms_db) - 1):
            # Allow small increases (1dB tolerance)
            if rms_db[j] - rms_db[j + 1] < -1.0:
                is_declining = False
                break
        
        # Check if total drop from this point is significant
        total_drop = rms_db[i] - rms_db[-1]
        
        # Lower threshold to 1.5 dB to catch gentler fades
        if is_declining and total_drop >= 1.5:
            fade_start_idx = i
            break
    
    # Convert to sample position and duration
    fade_start_sample = len(mono) - lookback_samples + (fade_start_idx * window_size)
    fade_duration = (len(mono) - fade_start_sample) / sr
    
    return fade_start_sample, fade_duration, rms_db

def calculate_fade_duration(audio, sr, max_duration=5.0, silence_threshold_db=-70.0):
    """
    Calculate how long the fade extension should be based on:
    - The existing fade rate (dB per second) - detected dynamically
    - Current ending level
    - Target silence threshold
    
    Returns the needed duration in seconds, capped at max_duration.
    """
    mono = np.mean(audio, axis=1)
    
    # Dynamically detect where the fade starts
    fade_start_sample, fade_duration, rms_windows = detect_fade_start(audio, sr)
    
    # Analyze only the fading portion
    fade_region = mono[fade_start_sample:]
    
    # Get start and end RMS of the fade
    fade_start_rms_db = linear_to_db(
        np.sqrt(np.mean(fade_region[:int(sr * 0.2)] ** 2))
    )
    
    end_samples = int(sr * 0.1)  # Last 100ms
    end_rms_db = linear_to_db(
        np.sqrt(np.mean(mono[-end_samples:] ** 2))
    )
    
    # Calculate fade rate from the actual fading portion
    fade_drop_total = fade_start_rms_db - end_rms_db
    fade_rate_db_per_sec = fade_drop_total / fade_duration if fade_duration > 0 else 0
    
    # Calculate needed drop to reach silence
    needed_drop = end_rms_db - silence_threshold_db
    
    # Calculate duration needed
    if fade_rate_db_per_sec > 0.5:  # Only if there's meaningful fade
        needed_duration = needed_drop / fade_rate_db_per_sec
        # Cap at maximum duration
        actual_duration = min(needed_duration, max_duration)
    else:
        # Not fading significantly, use max duration
        actual_duration = max_duration
    
    # Ensure minimum duration of 1 second
    actual_duration = max(1.0, actual_duration)
    
    print(f"  Adaptive fade calculation:")
    print(f"    Detected fade duration: {fade_duration:.2f}s")
    print(f"    Fade start level: {fade_start_rms_db:.1f} dB")
    print(f"    Current end level: {end_rms_db:.1f} dB")
    print(f"    Fade rate: {fade_rate_db_per_sec:.1f} dB/sec")
    print(f"    Needed drop to silence: {needed_drop:.1f} dB")
    print(f"    Calculated extension: {actual_duration:.2f}s (max: {max_duration:.1f}s)")
    
    return actual_duration

def repair_fade_with_reverb(
    audio,
    sr,
    tail_seconds=5.0
):
    """
    Append a generated reverb tail with smooth crossfade.
    Tail duration is adaptive based on existing fade rate.
    """
    
    # Calculate adaptive duration based on fade rate
    adaptive_duration = calculate_fade_duration(
        audio, sr, max_duration=tail_seconds
    )

    tail = generate_reverb_tail(
        audio,
        sr,
        source_seconds=2.0,
        tail_seconds=adaptive_duration,
        decay_db=80.0
    )

    # Add crossfade to ensure smooth transition
    # Use a longer crossfade (500ms) for more natural blending
    crossfade_len = int(sr * 0.5)
    
    if len(audio) < crossfade_len:
        # If audio is shorter than crossfade, just append
        return np.vstack([audio, tail])
    
    # Create smooth crossfade curves
    fade_out = np.linspace(1.0, 0.0, crossfade_len)
    fade_in = np.linspace(0.0, 1.0, crossfade_len)
    
    # Apply power curve for more natural fade (equal power crossfade)
    fade_out = np.sqrt(fade_out)
    fade_in = np.sqrt(fade_in)
    
    # Make a copy to avoid modifying original
    output = np.copy(audio)
    
    # Apply crossfade to the overlapping region
    output[-crossfade_len:] = (
        output[-crossfade_len:] * fade_out[:, np.newaxis]
        + tail[:crossfade_len] * fade_in[:, np.newaxis]
    )
    
    # Append the rest of the tail after the crossfade region
    remaining_tail = tail[crossfade_len:]
    
    return np.vstack([output, remaining_tail])


def repair_fade(audio, sr, fade_seconds=4.0):
    """
    Extend an unfinished fade naturally.

    Assumes the track is already fading but
    has not yet reached silence.
    """

    source_len = int(sr * 1.0)
    fade_len = int(sr * fade_seconds)

    source = audio[-source_len:]

    # Repeat enough times to create the tail
    repeats = int(
        np.ceil(fade_len / source_len)
    )

    tail = np.tile(
        source,
        (repeats, 1)
    )

    tail = tail[:fade_len]

    # Estimate current level
    end_rms = np.sqrt(
        np.mean(source ** 2)
    )

    # Fade by another 60 dB
    fade_curve_db = np.linspace(
        0,
        -60,
        fade_len
    )

    fade_curve = (
        10 ** (fade_curve_db / 20.0)
    )

    tail *= fade_curve[:, np.newaxis]

    # Crossfade first 100 ms
    crossfade_len = int(sr * 0.1)

    fade_out = np.linspace(
        1.0,
        0.0,
        crossfade_len
    )

    fade_in = np.linspace(
        0.0,
        1.0,
        crossfade_len
    )

    output = np.copy(audio)

    output[-crossfade_len:] = (
        output[-crossfade_len:] *
        fade_out[:, np.newaxis]
        +
        tail[:crossfade_len] *
        fade_in[:, np.newaxis]
    )

    tail = tail[crossfade_len:]

    return np.vstack([
        output,
        tail
    ])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file"
    )

    parser.add_argument(
        "output_file"
    )

    parser.add_argument(
        "--lufs",
        type=float,
        default=TARGET_LUFS
    )

    parser.add_argument(
        "--max-gain",
        type=float,
        default=MAX_GAIN_DB,
        help="Maximum loudness increase in dB"
    )

    parser.add_argument(
        "--fade-seconds",
        type=float,
        default=5.0,
        help="Maximum duration of the artificial fade tail in seconds"
    )

    parser.add_argument(
        "--stereo-width",
        type=float,
        default=1.0,
        help="Stereo width adjustment (0.5-1.5, where 1.0=unchanged, <1.0=narrower, >1.0=wider)"
    )

    parser.add_argument(
        "--no-air",
        action="store_true",
        help="Disable high-frequency air enhancement"
    )

    parser.add_argument(
        "--no-start-trim",
        action="store_true",
        help="Disable automatic removal of start noise artifacts"
    )

    parser.add_argument(
        "--no-harshness-fix",
        action="store_true",
        help="Disable dynamic harshness correction (2.5-9kHz)"
    )

    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        args.lufs,
        args.max_gain,
        args.fade_seconds,
        args.stereo_width,
        not args.no_air,
        not args.no_start_trim,
        not args.no_harshness_fix
    )


if __name__ == "__main__":
    main()

