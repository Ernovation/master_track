"""
Tonal balance analysis.

Replaces separate mud detection and air enhancement with a unified
spectral balance analyzer that generates broad EQ corrections.
"""

import numpy as np
from .spectral_analysis import (
    linear_to_db,
    db_to_linear,
    smooth_spectrum_octave_bands,
    compute_windowed_median_spectra,
    interpolate_curve
)


class TonalBalanceAnalyzer:
    """
    Analyzes tonal balance and generates static mastering EQ.
    
    Replaces individual mud detection and air enhancement with
    a coherent spectral analysis approach.
    """
    
    def __init__(self, profile):
        """
        Initialize with a tonal balance profile.
        
        Args:
            profile: Dict with:
                - target_curve: List of {"frequency": Hz, "db": dB}
                - max_boost_db: Maximum boost allowed
                - max_cut_db: Maximum cut allowed (negative or positive)
        """
        self.profile = profile
        self.max_boost = profile.get('max_boost_db', 2.0)
        self.max_cut = abs(profile.get('max_cut_db', 2.0))
        self.min_correction = 0.75  # Ignore corrections below this threshold
    
    def analyze(self, audio, sr):
        """
        Analyze tonal balance and generate EQ corrections.
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            dict: {
                'bands': Analysis of frequency bands,
                'filters': List of EQ filter specs,
                'spectrum': The analyzed spectrum,
                'target': The target curve,
                'freqs': Frequency array
            }
        """
        # Compute median spectra using overlapping 10-second windows
        median_spectra, freqs = compute_windowed_median_spectra(
            audio, sr, window_duration=10.0, overlap=0.5
        )
        
        # Take overall median across all windows
        overall_median = np.median(median_spectra, axis=0)
        
        # Convert to dB
        spectrum_db = linear_to_db(overall_median)
        
        # Smooth to approximately 1/3 octave resolution
        smoothed_spectrum = smooth_spectrum_octave_bands(freqs, spectrum_db, octave_fraction=3)
        
        # Interpolate target curve
        target_curve = interpolate_curve(self.profile['target_curve'], freqs)
        
        # Calculate difference
        difference = target_curve - smoothed_spectrum
        
        # Analyze frequency bands
        bands = self._analyze_bands(freqs, difference)
        
        # Generate filters
        filters = self._generate_filters(bands)
        
        return {
            'bands': bands,
            'filters': filters,
            'spectrum': smoothed_spectrum,
            'target': target_curve,
            'freqs': freqs
        }
    
    def _analyze_bands(self, freqs, difference):
        """
        Analyze frequency bands to determine needed corrections.
        
        Band definitions:
        - Sub Bass: 20-60 Hz
        - Bass: 60-200 Hz
        - Low Mids: 200-800 Hz
        - Mids: 800-2000 Hz
        - Presence: 2000-6000 Hz
        - Air: 6000-20000 Hz
        """
        bands = {}
        
        band_definitions = [
            ('Sub Bass', 20, 60),
            ('Bass', 60, 200),
            ('Low Mids', 200, 800),
            ('Mids', 800, 2000),
            ('Presence', 2000, 6000),
            ('Air', 6000, 20000)
        ]
        
        for name, low, high in band_definitions:
            mask = (freqs >= low) & (freqs <= high)
            
            if np.sum(mask) == 0:
                continue
            
            band_diff = difference[mask]
            median_correction = np.median(band_diff)
            
            # Clamp to limits
            clamped = np.clip(median_correction, -self.max_cut, self.max_boost)
            
            # Check if correction is significant
            needs_correction = abs(clamped) >= self.min_correction
            
            bands[name] = {
                'low': low,
                'high': high,
                'correction_db': float(clamped) if needs_correction else 0.0,
                'raw_correction': float(median_correction),
                'needs_correction': needs_correction
            }
        
        return bands
    
    def _generate_filters(self, bands):
        """
        Generate EQ filter specifications based on band analysis.
        
        Generates at most:
        - 1 low shelf (for sub bass or bass)
        - 2 bell filters (for mids)
        - 1 high shelf (for air)
        
        Returns:
            List of filter dicts: [{
                'type': 'low_shelf' | 'bell' | 'high_shelf',
                'frequency': Hz,
                'gain_db': dB,
                'q': Q factor (for bells)
            }]
        """
        filters = []
        
        # Low shelf: Prioritize sub bass, then bass
        low_shelf_band = None
        low_shelf_freq = None
        
        if bands['Sub Bass']['needs_correction']:
            low_shelf_band = bands['Sub Bass']
            low_shelf_freq = 55  # Centered around sub bass
        elif bands['Bass']['needs_correction']:
            low_shelf_band = bands['Bass']
            low_shelf_freq = 100  # Centered around bass
        
        if low_shelf_band:
            filters.append({
                'type': 'low_shelf',
                'frequency': low_shelf_freq,
                'gain_db': low_shelf_band['correction_db']
            })
        
        # Bell filters: Low mids and mids/presence
        bell_candidates = []
        
        if bands['Low Mids']['needs_correction']:
            bell_candidates.append({
                'type': 'bell',
                'frequency': 400,  # Center of low mids
                'gain_db': bands['Low Mids']['correction_db'],
                'q': 0.8,
                'priority': abs(bands['Low Mids']['correction_db'])
            })
        
        if bands['Mids']['needs_correction']:
            bell_candidates.append({
                'type': 'bell',
                'frequency': 1200,  # Center of mids
                'gain_db': bands['Mids']['correction_db'],
                'q': 0.8,
                'priority': abs(bands['Mids']['correction_db'])
            })
        
        if bands['Presence']['needs_correction']:
            bell_candidates.append({
                'type': 'bell',
                'frequency': 3500,  # Center of presence
                'gain_db': bands['Presence']['correction_db'],
                'q': 0.8,
                'priority': abs(bands['Presence']['correction_db'])
            })
        
        # Sort by priority and take top 2
        bell_candidates.sort(key=lambda x: x['priority'], reverse=True)
        filters.extend([{k: v for k, v in f.items() if k != 'priority'} 
                       for f in bell_candidates[:2]])
        
        # High shelf: Air
        if bands['Air']['needs_correction']:
            filters.append({
                'type': 'high_shelf',
                'frequency': 10000,  # High shelf frequency
                'gain_db': bands['Air']['correction_db']
            })
        
        return filters
    
    def format_band_report(self, bands):
        """
        Format band analysis for logging.
        
        Returns:
            String with formatted band report
        """
        lines = []
        
        for name, info in bands.items():
            if info['needs_correction']:
                sign = '+' if info['correction_db'] > 0 else ''
                lines.append(f"  {name}: {sign}{info['correction_db']:.1f} dB")
            else:
                lines.append(f"  {name}: OK")
        
        return '\n'.join(lines)
    
    def format_filter_report(self, filters):
        """
        Format filter specifications for logging.
        
        Returns:
            String with formatted filter report
        """
        if not filters:
            return "  No EQ corrections needed"
        
        lines = []
        
        for f in filters:
            if f['type'] == 'low_shelf':
                lines.append(f"  Low Shelf @ {f['frequency']} Hz: {f['gain_db']:+.1f} dB")
            elif f['type'] == 'high_shelf':
                lines.append(f"  High Shelf @ {f['frequency']} Hz: {f['gain_db']:+.1f} dB")
            elif f['type'] == 'bell':
                lines.append(f"  Bell @ {f['frequency']} Hz (Q={f['q']:.1f}): {f['gain_db']:+.1f} dB")
        
        return '\n'.join(lines)
