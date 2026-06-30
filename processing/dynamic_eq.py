"""
Dynamic EQ processing.

Applies frequency-selective compression for harshness correction.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.signal.windows import hann


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 20.0)


def linear_to_db(x):
    """Convert linear to dB scale."""
    return 20 * np.log10(np.maximum(x, 1e-12))


def apply_dynamic_eq(audio, sr, harsh_regions, analyzer):
    """
    Apply dynamic EQ to correct harsh resonances.
    
    Args:
        audio: Audio array
        sr: Sample rate
        harsh_regions: List of harsh region dicts from harshness analyzer
        analyzer: HarshnessAnalyzer instance (for reduction calculation)
    
    Returns:
        Processed audio
    """
    if not harsh_regions:
        return audio
    
    # Process using overlap-add for better quality
    window_size = 4096
    hop_size = 2048
    window = hann(window_size)
    
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
            magnitude_db = linear_to_db(magnitude)
            
            # Apply reduction for each harsh region
            for region in harsh_regions:
                center = region['freq']
                bandwidth = region['bandwidth']
                
                # Calculate Q
                q = np.clip(center / bandwidth, 0.8, 6.0)
                
                # Create frequency mask
                half_bandwidth = bandwidth / 2
                band_mask = (freqs >= center - half_bandwidth) & (freqs <= center + half_bandwidth)
                
                if not np.any(band_mask):
                    continue
                
                # Calculate band RMS
                band_energy = magnitude[band_mask]
                if len(band_energy) == 0:
                    continue
                
                band_rms_db = linear_to_db(np.sqrt(np.mean(band_energy ** 2)))
                
                # Threshold: band RMS + 2 dB
                threshold_db = band_rms_db + 2.0
                
                # Check if this frame exceeds threshold
                band_peak_db = np.max(magnitude_db[band_mask])
                
                if band_peak_db > threshold_db:
                    # Apply compression with 2:1 ratio
                    excess_db = band_peak_db - threshold_db
                    reduction_db = excess_db / 2.0
                    
                    # Clamp to maximum reduction
                    max_reduction_db = -analyzer.calculate_reduction_amount(region)
                    reduction_db = min(reduction_db, max_reduction_db)
                    
                    # Apply bell-shaped reduction
                    reduction_linear = 10 ** (-reduction_db / 20.0)
                    
                    # Create smooth bell curve
                    bell = np.exp(-0.5 * ((freqs - center) / (bandwidth / 2.355)) ** 2)
                    gain = 1.0 - (1.0 - reduction_linear) * bell
                    
                    magnitude *= gain
            
            # Reconstruct
            fft_modified = magnitude * np.exp(1j * phase)
            reconstructed = np.fft.irfft(fft_modified, n=window_size)
            
            # Overlap-add
            output[start:end, ch] += reconstructed * window
            if ch == 0:
                window_sum[start:end] += window ** 2
    
    # Normalize
    window_sum[window_sum < 1e-8] = 1.0
    output /= window_sum[:, np.newaxis]
    
    return output
