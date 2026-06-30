"""
Static EQ processing.

Implements shelf and bell filters for tonal balance correction.
"""

import numpy as np


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 20.0)


def apply_eq_filters(audio, sr, filters):
    """
    Apply a list of EQ filters to audio.
    
    Args:
        audio: Audio array (samples, channels)
        sr: Sample rate
        filters: List of filter dicts from tonal balance analyzer
    
    Returns:
        Processed audio
    """
    if not filters:
        return audio
    
    processed = np.copy(audio)
    
    for f in filters:
        if f['type'] == 'low_shelf':
            processed = apply_low_shelf(
                processed, sr, f['frequency'], f['gain_db']
            )
        elif f['type'] == 'high_shelf':
            processed = apply_high_shelf(
                processed, sr, f['frequency'], f['gain_db']
            )
        elif f['type'] == 'bell':
            processed = apply_bell_filter(
                processed, sr, f['frequency'], f['gain_db'], f['q']
            )
    
    return processed


def apply_low_shelf(audio, sr, frequency, gain_db):
    """
    Apply a low shelf filter.
    
    Uses frequency-domain implementation for accuracy.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frequency: Shelf frequency (Hz)
        gain_db: Gain in dB
    
    Returns:
        Filtered audio
    """
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    # Gain factor
    gain = db_to_linear(gain_db)
    
    # Smooth transition using sigmoid
    # Transition width: 1 octave
    transition_width = frequency / 2.0
    
    # Create shelf curve
    shelf = np.ones_like(freqs)
    for i, f in enumerate(freqs):
        if f < frequency - transition_width:
            shelf[i] = gain
        elif f < frequency + transition_width:
            # Smooth transition
            t = (f - (frequency - transition_width)) / (2 * transition_width)
            # Sigmoid-like curve
            shelf[i] = gain + (1.0 - gain) * (1.0 / (1.0 + np.exp(-10 * (t - 0.5))))
        else:
            shelf[i] = 1.0
    
    # Apply filter
    fft *= shelf[:, np.newaxis]
    
    return np.fft.irfft(fft, n=audio.shape[0], axis=0)


def apply_high_shelf(audio, sr, frequency, gain_db):
    """
    Apply a high shelf filter.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frequency: Shelf frequency (Hz)
        gain_db: Gain in dB
    
    Returns:
        Filtered audio
    """
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    gain = db_to_linear(gain_db)
    
    # Transition width: 1 octave
    transition_width = frequency / 2.0
    
    # Create shelf curve
    shelf = np.ones_like(freqs)
    for i, f in enumerate(freqs):
        if f > frequency + transition_width:
            shelf[i] = gain
        elif f > frequency - transition_width:
            # Smooth transition
            t = (f - (frequency - transition_width)) / (2 * transition_width)
            shelf[i] = 1.0 + (gain - 1.0) * (1.0 / (1.0 + np.exp(-10 * (t - 0.5))))
        else:
            shelf[i] = 1.0
    
    # Apply filter
    fft *= shelf[:, np.newaxis]
    
    return np.fft.irfft(fft, n=audio.shape[0], axis=0)


def apply_bell_filter(audio, sr, frequency, gain_db, q):
    """
    Apply a bell (peaking) filter.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frequency: Center frequency (Hz)
        gain_db: Gain in dB
        q: Q factor (bandwidth control)
    
    Returns:
        Filtered audio
    """
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    gain = db_to_linear(gain_db)
    
    # Calculate bandwidth from Q
    bandwidth = frequency / q
    
    # Create bell curve (Gaussian-like)
    sigma = bandwidth / 2.355  # Convert to Gaussian width
    bell = np.exp(-0.5 * ((freqs - frequency) / sigma) ** 2)
    
    # Create gain curve
    # At center: full gain
    # Away from center: unity gain
    gain_curve = 1.0 + (gain - 1.0) * bell
    
    # Apply filter
    fft *= gain_curve[:, np.newaxis]
    
    return np.fft.irfft(fft, n=audio.shape[0], axis=0)


def apply_highpass_filter(audio, sr, cutoff):
    """
    Apply a highpass filter for subsonic rumble removal.
    
    Args:
        audio: Audio array
        sr: Sample rate
        cutoff: Cutoff frequency (Hz)
    
    Returns:
        Filtered audio
    """
    from scipy.signal import butter, sosfilt
    
    sos = butter(4, cutoff, btype='highpass', fs=sr, output='sos')
    return sosfilt(sos, audio, axis=0)
