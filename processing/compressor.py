"""
Compressor processing.

Adaptive compressor with soft knee and lookahead.
"""

import numpy as np


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 20.0)


def linear_to_db(x):
    """Convert linear to dB scale."""
    return 20 * np.log10(np.maximum(x, 1e-12))


def apply_compressor(audio, sr, settings, attack_ms=20, release_ms=100, 
                    knee_db=6.0, lookahead_ms=5.0):
    """
    Apply compression with soft knee and lookahead.
    
    Args:
        audio: Audio array
        sr: Sample rate
        settings: Dict with 'threshold_db', 'ratio', 'apply'
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        knee_db: Knee width in dB
        lookahead_ms: Lookahead time in milliseconds
    
    Returns:
        Compressed audio
    """
    if not settings['apply']:
        return audio
    
    threshold_db = settings['threshold_db']
    ratio = settings['ratio']
    
    threshold = db_to_linear(threshold_db)
    knee = db_to_linear(knee_db)
    
    # Calculate coefficients
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    
    # Lookahead buffer
    lookahead_samples = int(sr * lookahead_ms / 1000)
    
    envelope = 0.0
    gain_curve = np.ones(audio.shape[0])
    
    # Stereo-linked detector (max of both channels)
    detector = np.max(np.abs(audio), axis=1)
    
    # Pad detector for lookahead
    detector_padded = np.pad(detector, (lookahead_samples, 0), mode='edge')
    
    for i in range(len(detector)):
        # Look ahead
        sample = detector_padded[i + lookahead_samples]
        
        # Envelope follower
        if sample > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * sample
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * sample
        
        if envelope > threshold:
            # Soft knee compression
            over = envelope - threshold
            
            if over < knee:
                # Soft knee region
                gain_reduction = over * over / (2 * knee * ratio)
                compressed = envelope - gain_reduction
            else:
                # Hard compression above knee
                compressed = threshold + knee / 2 + (over - knee) / ratio
            
            gain_curve[i] = compressed / envelope
    
    return audio * gain_curve[:, np.newaxis]


def apply_stereo_width_adjustment(audio, adjustment):
    """
    Adjust stereo width using mid-side processing.
    
    Args:
        audio: Stereo audio array (samples, 2)
        adjustment: Width multiplier (1.0 = no change)
    
    Returns:
        Adjusted audio
    """
    if audio.shape[1] < 2 or abs(adjustment - 1.0) < 0.01:
        return audio
    
    # Convert to mid-side
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0
    
    # Adjust side signal
    side *= adjustment
    
    # Convert back to left-right
    left = mid + side
    right = mid - side
    
    return np.column_stack([left, right])
