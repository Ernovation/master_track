"""
Stereo analysis.

Three-band stereo analysis replacing global side/mid ratio analysis.
"""

import numpy as np
from scipy.signal import butter, sosfilt


class StereoAnalyzer:
    """
    Analyzes stereo width in three frequency bands.
    
    Low (20-150 Hz): Should be nearly mono
    Mid (150-2000 Hz): Generally leave unchanged
    High (2000-20000 Hz): Allow wider stereo image
    """
    
    def __init__(self):
        """Initialize stereo analyzer with default band definitions."""
        self.bands = {
            'Low': {'low': 20, 'high': 150},
            'Mid': {'low': 150, 'high': 2000},
            'High': {'low': 2000, 'high': 20000}
        }
    
    def analyze(self, audio, sr):
        """
        Analyze stereo width in three frequency bands.
        
        Args:
            audio: Stereo audio array (samples, 2)
            sr: Sample rate
        
        Returns:
            dict: {
                'bands': Dict of band analysis,
                'adjustments': Dict of recommended width adjustments per band,
                'global_adjustment': Overall recommended adjustment
            }
        """
        if audio.shape[1] < 2:
            # Mono audio
            return {
                'bands': {},
                'adjustments': {},
                'global_adjustment': 1.0
            }
        
        # Convert to mid-side
        mid = (audio[:, 0] + audio[:, 1]) / 2.0
        side = (audio[:, 0] - audio[:, 1]) / 2.0
        
        band_analysis = {}
        adjustments = {}
        
        for band_name, band_def in self.bands.items():
            # Filter mid and side to this band
            mid_band = self._bandpass_filter(
                mid, sr, band_def['low'], band_def['high']
            )
            side_band = self._bandpass_filter(
                side, sr, band_def['low'], band_def['high']
            )
            
            # Calculate RMS for each
            mid_rms = np.sqrt(np.mean(mid_band ** 2))
            side_rms = np.sqrt(np.mean(side_band ** 2))
            
            # Calculate width ratio
            if mid_rms > 1e-12:
                width_ratio = side_rms / mid_rms
            else:
                width_ratio = 0.0
            
            # Determine recommended adjustment for this band
            adjustment = self._calculate_band_adjustment(band_name, width_ratio)
            
            band_analysis[band_name] = {
                'width_ratio': float(width_ratio),
                'mid_rms': float(mid_rms),
                'side_rms': float(side_rms)
            }
            
            adjustments[band_name] = adjustment
        
        # Calculate global adjustment (weighted average)
        # Prioritize low band (bass mono compatibility is critical)
        weights = {'Low': 0.5, 'Mid': 0.3, 'High': 0.2}
        global_adjustment = sum(
            adjustments[band] * weights[band] 
            for band in ['Low', 'Mid', 'High']
        )
        
        return {
            'bands': band_analysis,
            'adjustments': adjustments,
            'global_adjustment': float(global_adjustment)
        }
    
    def _bandpass_filter(self, audio, sr, low_freq, high_freq):
        """
        Apply bandpass filter to audio.
        
        Args:
            audio: Audio array (mono)
            sr: Sample rate
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
        
        Returns:
            Filtered audio
        """
        # Handle edge cases
        nyquist = sr / 2.0
        
        if high_freq >= nyquist:
            # Highpass only
            sos = butter(4, low_freq, btype='highpass', fs=sr, output='sos')
        elif low_freq <= 20:
            # Lowpass only
            sos = butter(4, high_freq, btype='lowpass', fs=sr, output='sos')
        else:
            # Bandpass
            sos = butter(4, [low_freq, high_freq], btype='bandpass', fs=sr, output='sos')
        
        return sosfilt(sos, audio)
    
    def _calculate_band_adjustment(self, band_name, width_ratio):
        """
        Calculate recommended width adjustment for a band.
        
        Args:
            band_name: 'Low', 'Mid', or 'High'
            width_ratio: Current side/mid ratio
        
        Returns:
            Recommended width multiplier (1.0 = no change)
        """
        if band_name == 'Low':
            # Low frequencies should be nearly mono
            # Target: width_ratio < 0.15
            if width_ratio > 0.25:
                # Too wide - narrow significantly
                return 0.5
            elif width_ratio > 0.15:
                # Somewhat wide - narrow moderately
                return 0.7
            else:
                # Good - no change needed
                return 1.0
        
        elif band_name == 'Mid':
            # Mid frequencies - maintain balance
            # Target: 0.3 to 0.6
            if width_ratio > 0.8:
                # Too wide - narrow slightly
                return 0.85
            elif width_ratio < 0.2:
                # Too narrow - widen slightly
                return 1.15
            else:
                # Good range - no change
                return 1.0
        
        elif band_name == 'High':
            # High frequencies - allow wider stereo
            # Target: 0.5 to 1.0
            if width_ratio > 1.2:
                # Excessively wide - narrow for mono compatibility
                return 0.8
            elif width_ratio < 0.4:
                # Too narrow - widen
                return 1.2
            else:
                # Good range
                return 1.0
        
        return 1.0
    
    def format_report(self, analysis):
        """
        Format stereo analysis for logging.
        
        Returns:
            String with formatted report
        """
        if not analysis['bands']:
            return "  Mono audio - no stereo analysis"
        
        lines = []
        
        for band_name in ['Low', 'Mid', 'High']:
            band = analysis['bands'][band_name]
            adjustment = analysis['adjustments'][band_name]
            
            lines.append(
                f"  {band_name} Band: side/mid={band['width_ratio']:.2f}, "
                f"adjustment={adjustment:.2f}x"
            )
        
        global_adj = analysis['global_adjustment']
        if abs(global_adj - 1.0) > 0.05:
            change_pct = int((global_adj - 1.0) * 100)
            action = "widened" if change_pct > 0 else "narrowed"
            lines.append(f"  Overall: {action} {abs(change_pct)}%")
        else:
            lines.append("  Overall: No adjustment needed")
        
        return '\n'.join(lines)
