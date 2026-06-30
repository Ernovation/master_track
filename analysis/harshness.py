"""
Harshness detection and analysis.

Detects harsh resonances in the 2.5-9 kHz range.
Harshness is treated as a dynamic problem, not a tonal balance issue.
"""

import numpy as np
from scipy.signal import medfilt
from .spectral_analysis import linear_to_db
from .frame_selection import select_loud_frames, analyze_with_selected_frames


class HarshnessAnalyzer:
    """
    Detects harsh resonances that require dynamic EQ correction.
    
    Harshness is NOT a tonal balance issue - it's a dynamic problem
    that requires frequency-specific compression.
    """
    
    def __init__(self, min_freq=2500, max_freq=9000, threshold_db=2.5):
        """
        Initialize harshness analyzer.
        
        Args:
            min_freq: Minimum frequency to analyze (Hz)
            max_freq: Maximum frequency to analyze (Hz)
            threshold_db: Resonance threshold in dB above local trend
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.threshold_db = threshold_db
    
    def analyze(self, audio, sr):
        """
        Detect harsh resonances.
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            dict: {
                'regions': List of harsh region dicts,
                'max_resonance': Maximum resonance detected,
                'should_correct': Whether correction is recommended
            }
        """
        # Select loudest frames (35% = percentile 65)
        frame_info = select_loud_frames(audio, sr, percentile=65)
        
        if frame_info['count'] < 5:
            return {
                'regions': [],
                'max_resonance': 0.0,
                'should_correct': False
            }
        
        # Analyze selected frames
        result = analyze_with_selected_frames(audio, sr, frame_info['indices'])
        
        # Average spectrum from loud frames
        avg_spectrum = np.mean(result['spectra'], axis=0)
        freqs = result['freqs']
        
        # Convert to dB
        spectrum_db = linear_to_db(avg_spectrum)
        
        # Restrict to harshness range
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        analysis_freqs = freqs[mask]
        analysis_spectrum = spectrum_db[mask]
        
        if len(analysis_spectrum) < 20:
            return {
                'regions': [],
                'max_resonance': 0.0,
                'should_correct': False
            }
        
        # Estimate local spectral trend using median filter
        # Adaptive smoothing: approximately 1/3 octave
        smooth_width = max(5, int(len(analysis_spectrum) / 20))
        if smooth_width % 2 == 0:
            smooth_width += 1  # Must be odd
        
        smoothed_spectrum = medfilt(analysis_spectrum, kernel_size=smooth_width)
        
        # Compute resonance strength (peaks above trend)
        resonance_db = analysis_spectrum - smoothed_spectrum
        
        # Find harsh regions
        harsh_regions = self._find_harsh_regions(
            analysis_freqs,
            resonance_db
        )
        
        # Determine if correction should be applied
        max_resonance = max([r['peak'] for r in harsh_regions], default=0.0)
        should_correct = max_resonance >= self.threshold_db and len(harsh_regions) > 0
        
        return {
            'regions': harsh_regions,
            'max_resonance': float(max_resonance),
            'should_correct': should_correct
        }
    
    def _find_harsh_regions(self, freqs, resonance_db):
        """
        Find contiguous harsh regions.
        
        Args:
            freqs: Frequency array
            resonance_db: Resonance strength in dB
        
        Returns:
            List of harsh region dicts sorted by severity
        """
        # Find where resonance exceeds threshold
        is_harsh = resonance_db > self.threshold_db
        
        if not np.any(is_harsh):
            return []
        
        # Find contiguous regions
        regions = []
        in_region = False
        region_start = 0
        
        for i, harsh in enumerate(is_harsh):
            if harsh and not in_region:
                region_start = i
                in_region = True
            elif not harsh and in_region:
                # End of region
                regions.append(self._analyze_region(
                    region_start, i, freqs, resonance_db
                ))
                in_region = False
        
        # Handle region extending to end
        if in_region:
            regions.append(self._analyze_region(
                region_start, len(is_harsh), freqs, resonance_db
            ))
        
        # Filter by bandwidth and sort by severity
        filtered_regions = []
        
        for region in regions:
            # Reject narrow FFT spikes (<100 Hz) and very wide regions (>3000 Hz)
            if 100 < region['bandwidth'] < 3000:
                # Score by peak height and bandwidth (broad resonances are worse)
                region['score'] = region['peak'] * np.log10(region['bandwidth'])
                filtered_regions.append(region)
        
        # Sort by score and return top 3
        filtered_regions.sort(key=lambda x: x['score'], reverse=True)
        return filtered_regions[:3]
    
    def _analyze_region(self, start_idx, end_idx, freqs, resonance_db):
        """
        Analyze a single harsh region.
        
        Returns:
            Dict with region properties
        """
        region_freqs = freqs[start_idx:end_idx]
        region_resonance = resonance_db[start_idx:end_idx]
        
        # Find peak
        peak_idx = np.argmax(region_resonance)
        peak_height = region_resonance[peak_idx]
        center_freq = region_freqs[peak_idx]
        
        # Calculate bandwidth
        bandwidth = region_freqs[-1] - region_freqs[0]
        
        return {
            'freq': float(center_freq),
            'peak': float(peak_height),
            'bandwidth': float(bandwidth),
            'score': 0.0  # Will be calculated later
        }
    
    def calculate_reduction_amount(self, region):
        """
        Calculate dynamic EQ reduction amount for a harsh region.
        
        Args:
            region: Harsh region dict
        
        Returns:
            Maximum reduction in dB (negative)
        """
        peak_height = region['peak']
        
        # Scale reduction with peak height
        if peak_height >= 4.5:
            return -2.5
        elif peak_height >= 3.5:
            return -2.0
        elif peak_height >= 2.5:
            return -1.5
        else:
            return -1.0
    
    def format_report(self, analysis):
        """
        Format harshness analysis for logging.
        
        Returns:
            String with formatted report
        """
        if not analysis['should_correct']:
            return "  Detected: No\n  No harsh resonances found"
        
        lines = [f"  Detected: Yes"]
        
        for region in analysis['regions']:
            max_reduction = self.calculate_reduction_amount(region)
            lines.append(
                f"  Frequency: {region['freq']:.0f} Hz "
                f"(peak: +{region['peak']:.1f} dB, "
                f"bandwidth: {region['bandwidth']:.0f} Hz, "
                f"max reduction: {max_reduction:.1f} dB)"
            )
        
        return '\n'.join(lines)
