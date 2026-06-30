"""
Dynamics analysis.

Analyzes both crest factor and Loudness Range (LRA) to make
intelligent compression decisions.
"""

import numpy as np
import pyloudnorm as pyln
from .spectral_analysis import linear_to_db


class DynamicsAnalyzer:
    """
    Analyzes audio dynamics using both crest factor and LRA.
    
    Combines two metrics:
    - Crest factor: Peak-to-RMS ratio (dB)
    - LRA: Loudness Range (LU)
    
    This allows more intelligent compression decisions.
    """
    
    def __init__(self):
        """Initialize dynamics analyzer."""
        pass
    
    def analyze(self, audio, sr):
        """
        Analyze dynamics.
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            dict: {
                'crest_factor_db': Crest factor in dB,
                'lra': Loudness Range in LU,
                'compression_mode': Recommended compression mode,
                'compression_settings': Dict with threshold and ratio
            }
        """
        # Calculate crest factor
        crest_factor_db = self._calculate_crest_factor(audio)
        
        # Calculate LRA
        try:
            lra = self._calculate_lra(audio, sr)
        except:
            # Fallback if LRA calculation fails
            lra = 0.0
        
        # Determine compression strategy
        mode, settings = self._determine_compression(crest_factor_db, lra)
        
        return {
            'crest_factor_db': float(crest_factor_db),
            'lra': float(lra),
            'compression_mode': mode,
            'compression_settings': settings
        }
    
    def _calculate_crest_factor(self, audio):
        """
        Calculate crest factor (peak-to-RMS ratio in dB).
        
        Higher values = more dynamic range.
        Lower values = already compressed.
        """
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-12:
            return 999.0  # Effectively silent
        
        return linear_to_db(peak / rms)
    
    def _calculate_lra(self, audio, sr):
        """
        Calculate Loudness Range (LRA) in LU.
        
        LRA measures the variation in loudness over time.
        Lower values = more consistent loudness.
        Higher values = more dynamic variation.
        """
        meter = pyln.Meter(sr)
        
        # pyloudnorm doesn't have built-in LRA, so we'll approximate it
        # by analyzing loudness variation in windows
        
        window_duration = 3.0  # seconds
        hop_duration = 1.0  # seconds
        
        window_samples = int(window_duration * sr)
        hop_samples = int(hop_duration * sr)
        
        if len(audio) < window_samples:
            # Too short for windowed analysis
            return 0.0
        
        num_windows = (len(audio) - window_samples) // hop_samples + 1
        loudness_values = []
        
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            
            if end > len(audio):
                break
            
            window = audio[start:end]
            
            try:
                loudness = meter.integrated_loudness(window)
                if not np.isnan(loudness) and not np.isinf(loudness):
                    loudness_values.append(loudness)
            except:
                continue
        
        if len(loudness_values) < 2:
            return 0.0
        
        # LRA approximation: difference between 95th and 10th percentile
        loudness_array = np.array(loudness_values)
        lra = np.percentile(loudness_array, 95) - np.percentile(loudness_array, 10)
        
        return max(0.0, lra)
    
    def _determine_compression(self, crest_factor_db, lra):
        """
        Determine compression strategy based on crest factor and LRA.
        
        Decision matrix:
        - High crest + High LRA: Normal mastering compression
        - High crest + Low LRA: Light compression
        - Low crest + Low LRA: Skip compression entirely
        - Low crest + High LRA: Minimal compression (unusual case)
        
        Args:
            crest_factor_db: Crest factor in dB
            lra: Loudness Range in LU
        
        Returns:
            (mode: str, settings: dict)
        """
        high_crest = crest_factor_db > 10.0
        high_lra = lra > 6.0
        
        if high_crest and high_lra:
            # Dynamic material with loudness variation - needs compression
            return 'normal', {
                'threshold_db': -18.0,
                'ratio': 2.0,
                'apply': True
            }
        
        elif high_crest and not high_lra:
            # Dynamic peaks but consistent loudness - light compression
            return 'light', {
                'threshold_db': -15.0,
                'ratio': 1.5,
                'apply': True
            }
        
        elif not high_crest and not high_lra:
            # Already compressed and consistent - skip compression
            return 'none', {
                'threshold_db': None,
                'ratio': None,
                'apply': False
            }
        
        else:  # not high_crest and high_lra
            # Unusual: low peaks but high loudness variation
            # Apply minimal compression to even out loudness
            return 'minimal', {
                'threshold_db': -12.0,
                'ratio': 1.3,
                'apply': True
            }
    
    def format_report(self, analysis):
        """
        Format dynamics analysis for logging.
        
        Returns:
            String with formatted report
        """
        lines = [
            f"  Crest Factor: {analysis['crest_factor_db']:.1f} dB",
            f"  LRA: {analysis['lra']:.1f} LU",
            f"  Mode: {analysis['compression_mode'].title()}"
        ]
        
        settings = analysis['compression_settings']
        if settings['apply']:
            lines.append(
                f"  Settings: threshold={settings['threshold_db']:.0f} dB, "
                f"ratio={settings['ratio']:.1f}:1"
            )
        else:
            lines.append("  Compression skipped (already dense)")
        
        return '\n'.join(lines)
