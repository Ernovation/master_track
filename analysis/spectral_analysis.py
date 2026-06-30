"""
Spectral analysis utilities.

Provides smoothing and spectral processing functions.
"""

import numpy as np
from scipy.ndimage import median_filter


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 20.0)


def linear_to_db(x):
    """Convert linear to dB scale."""
    return 20 * np.log10(np.maximum(x, 1e-12))


def smooth_spectrum_octave_bands(freqs, spectrum_db, octave_fraction=3):
    """
    Smooth spectrum to approximate fractional octave resolution.
    
    Args:
        freqs: Frequency array
        spectrum_db: Spectrum in dB
        octave_fraction: Fraction of octave (3 = 1/3 octave)
    
    Returns:
        Smoothed spectrum in dB
    """
    if len(spectrum_db) < 10:
        return spectrum_db
    
    smoothed = np.copy(spectrum_db)
    
    # Frequency-dependent smoothing
    for i, freq in enumerate(freqs):
        if freq < 20:
            continue
        
        # Calculate bandwidth for this center frequency
        bandwidth = freq / octave_fraction
        
        # Find indices within bandwidth
        mask = (freqs >= freq - bandwidth / 2) & (freqs <= freq + bandwidth / 2)
        
        if np.sum(mask) > 0:
            smoothed[i] = np.median(spectrum_db[mask])
    
    return smoothed


def compute_median_spectrum(spectra):
    """
    Compute median spectrum across multiple frames.
    
    Args:
        spectra: Array of spectra (num_frames, num_bins)
    
    Returns:
        Median spectrum
    """
    return np.median(spectra, axis=0)


def compute_windowed_median_spectra(audio, sr, window_duration=10.0, overlap=0.5):
    """
    Compute median spectra using overlapping time windows.
    
    Args:
        audio: Audio array
        sr: Sample rate
        window_duration: Duration of each window in seconds
        overlap: Overlap fraction (0.0 to 1.0)
    
    Returns:
        List of median spectra
    """
    from .frame_selection import select_loud_frames, analyze_with_selected_frames
    
    window_samples = int(window_duration * sr)
    hop_samples = int(window_samples * (1.0 - overlap))
    
    if len(audio) < window_samples:
        # Single window
        frame_info = select_loud_frames(audio, sr)
        result = analyze_with_selected_frames(audio, sr, frame_info['indices'])
        return [compute_median_spectrum(result['spectra'])], result['freqs']
    
    num_windows = (len(audio) - window_samples) // hop_samples + 1
    median_spectra = []
    freqs = None
    
    for i in range(num_windows):
        start = i * hop_samples
        end = start + window_samples
        
        if end > len(audio):
            end = len(audio)
        
        window_audio = audio[start:end]
        
        # Select loud frames in this window
        frame_info = select_loud_frames(window_audio, sr)
        result = analyze_with_selected_frames(window_audio, sr, frame_info['indices'])
        
        median_spectrum = compute_median_spectrum(result['spectra'])
        median_spectra.append(median_spectrum)
        
        if freqs is None:
            freqs = result['freqs']
    
    return median_spectra, freqs


def interpolate_curve(frequency_points, target_freqs):
    """
    Interpolate a curve defined by frequency/dB points.
    
    Args:
        frequency_points: List of dicts [{"frequency": Hz, "db": dB}, ...]
        target_freqs: Frequency array to interpolate onto
    
    Returns:
        Interpolated curve in dB
    """
    if not frequency_points:
        return np.zeros_like(target_freqs)
    
    # Sort by frequency
    points = sorted(frequency_points, key=lambda p: p['frequency'])
    
    freqs = np.array([p['frequency'] for p in points])
    dbs = np.array([p['db'] for p in points])
    
    # Interpolate in log-frequency space for audio
    interpolated = np.interp(
        np.log10(np.maximum(target_freqs, 1e-6)),
        np.log10(freqs),
        dbs
    )
    
    return interpolated
