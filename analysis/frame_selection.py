"""
Frame selection for spectral analysis.

Selects the loudest frames for accurate mastering analysis.
"""

import numpy as np


def select_loud_frames(audio, sr, percentile=65, window_size=4096, hop_size=2048):
    """
    Select the loudest frames from audio for analysis.
    
    Args:
        audio: Audio array (samples, channels)
        sr: Sample rate
        percentile: Keep frames above this percentile (0-100)
                   Default 65 keeps the loudest 35% of frames
        window_size: FFT window size
        hop_size: Hop size for frame advancement
    
    Returns:
        dict: {
            'indices': List of loud frame indices,
            'count': Number of selected frames,
            'threshold_rms': RMS threshold used
        }
    """
    mono = np.mean(audio, axis=1)
    
    if len(mono) < window_size:
        return {
            'indices': [0],
            'count': 1,
            'threshold_rms': 0.0
        }
    
    num_frames = (len(mono) - window_size) // hop_size + 1
    frame_rms = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        if end > len(mono):
            break
        frame = mono[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        frame_rms.append((i, rms))
    
    if not frame_rms:
        return {
            'indices': [0],
            'count': 1,
            'threshold_rms': 0.0
        }
    
    # Calculate RMS threshold
    rms_values = np.array([rms for _, rms in frame_rms])
    threshold_rms = np.percentile(rms_values, percentile)
    
    # Select frames above threshold
    loud_indices = [idx for idx, rms in frame_rms if rms >= threshold_rms]
    
    return {
        'indices': loud_indices,
        'count': len(loud_indices),
        'threshold_rms': float(threshold_rms)
    }


def analyze_with_selected_frames(audio, sr, frame_indices, window_size=4096, hop_size=2048):
    """
    Compute FFT spectra for selected frames.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frame_indices: List of frame indices to analyze
        window_size: FFT window size
        hop_size: Hop size
    
    Returns:
        dict: {
            'spectra': Array of magnitude spectra (num_frames, num_bins),
            'freqs': Frequency array,
            'window_size': Window size used
        }
    """
    from scipy.signal.windows import hann
    
    mono = np.mean(audio, axis=1)
    window = hann(window_size)
    
    spectra = []
    
    for frame_idx in frame_indices:
        start = frame_idx * hop_size
        end = start + window_size
        
        if end > len(mono):
            continue
        
        frame = mono[start:end] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectra.append(spectrum)
    
    if not spectra:
        # Fallback: analyze entire signal
        padded = np.pad(mono, (0, max(0, window_size - len(mono))), mode='constant')
        frame = padded[:window_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectra = [spectrum]
    
    freqs = np.fft.rfftfreq(window_size, 1 / sr)
    
    return {
        'spectra': np.array(spectra),
        'freqs': freqs,
        'window_size': window_size
    }
