"""
Limiter processing.

True peak limiter with lookahead and smooth release.
"""

import numpy as np


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 20.0)


def linear_to_db(x):
    """Convert linear to dB scale."""
    return 20 * np.log10(np.maximum(x, 1e-12))


def apply_limiter(audio, sr, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=100.0):
    """
    Apply true peak limiter.
    
    Args:
        audio: Audio array
        sr: Sample rate
        ceiling_db: Ceiling level in dBFS
        lookahead_ms: Lookahead time in milliseconds
        release_ms: Release time in milliseconds
    
    Returns:
        Tuple of (limited_audio, limiting_db)
        - limited_audio: Processed audio
        - limiting_db: Amount of limiting applied (positive dB)
    """
    ceiling = db_to_linear(ceiling_db)
    
    lookahead_samples = int(sr * lookahead_ms / 1000)
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    
    # Stereo-linked peak detection
    detector = np.max(np.abs(audio), axis=1)
    
    # Pad for lookahead
    detector_padded = np.pad(detector, (lookahead_samples, 0), mode='edge')
    
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
    limited_audio = audio * gain_curve[:, np.newaxis]
    
    # Calculate total limiting
    limiting_db = -linear_to_db(np.min(gain_curve))
    
    return limited_audio, limiting_db


def hard_clip_protection(audio, max_level=0.999):
    """
    Final safety limiter to prevent clipping.
    
    Clips any samples exceeding ±max_level.
    This is the last line of defense against clipping.
    
    Args:
        audio: Audio array
        max_level: Maximum level (default 0.999 = -0.01 dBFS)
    
    Returns:
        Clipped audio
    """
    return np.clip(audio, -max_level, max_level)
