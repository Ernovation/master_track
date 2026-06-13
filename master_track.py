#!/usr/bin/env python3

import argparse
import warnings

import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from scipy.signal import butter, sosfilt
from scipy.signal import fftconvolve


TARGET_LUFS = -12.0
MAX_GAIN_DB = 3.0
LIMITER_CEILING_DB = -1.0
HPF_FREQ = 35.0


def db_to_linear(db):
    return 10 ** (db / 20.0)


def linear_to_db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def peak_db(audio):
    return linear_to_db(np.max(np.abs(audio)))


def highpass_filter(audio, sr, cutoff=35):
    sos = butter(
        4,
        cutoff,
        btype="highpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, audio, axis=0)


def detect_muddiness(audio, sr):
    """Improved segment-based muddiness detection."""
    mono = np.mean(audio, axis=1)

    # Analyze multiple segments instead of entire track
    segment_duration = 5.0  # seconds
    segment_samples = int(segment_duration * sr)
    
    # Take up to 5 segments spread across the track
    num_segments = min(5, len(mono) // segment_samples)
    if num_segments == 0:
        # Track too short, analyze whole thing
        segment_samples = len(mono)
        num_segments = 1
    
    muddy_ratios = []
    
    for i in range(num_segments):
        start = i * (len(mono) - segment_samples) // max(1, num_segments - 1)
        if num_segments == 1:
            start = 0
        segment = mono[start:start + segment_samples]
        
        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1 / sr)

        total_energy = np.sum(spectrum)

        mask = (freqs >= 250) & (freqs <= 400)
        muddy_energy = np.sum(spectrum[mask])

        if total_energy > 0:
            muddy_ratios.append(muddy_energy / total_energy)
    
    # Use median to avoid outliers
    return np.median(muddy_ratios) > 0.08 if muddy_ratios else False


def low_mid_cleanup(audio, sr):
    fft = np.fft.rfft(audio, axis=0)

    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)

    center = 320.0
    width = 120.0

    dip = np.exp(
        -0.5 * ((freqs - center) / width) ** 2
    )

    gain = 1.0 - 0.15 * dip

    fft *= gain[:, np.newaxis]

    return np.fft.irfft(
        fft,
        n=audio.shape[0],
        axis=0
    )


def compressor(
    audio,
    sr,
    threshold_db=-18,
    ratio=2.0,
    attack_ms=20,
    release_ms=100,
    knee_db=6.0,
    lookahead_ms=5.0
):
    """Improved compressor with soft knee and lookahead for smoother compression."""
    threshold = db_to_linear(threshold_db)
    knee = db_to_linear(knee_db)

    attack_coeff = np.exp(
        -1.0 / (sr * attack_ms / 1000)
    )

    release_coeff = np.exp(
        -1.0 / (sr * release_ms / 1000)
    )

    # Lookahead buffer
    lookahead_samples = int(sr * lookahead_ms / 1000)

    envelope = 0.0
    gain_curve = np.ones(audio.shape[0])

    # Stereo-linked detector (max of both channels)
    detector = np.max(
        np.abs(audio),
        axis=1
    )

    # Pad detector for lookahead
    detector_padded = np.pad(
        detector,
        (lookahead_samples, 0),
        mode='edge'
    )

    for i in range(len(detector)):
        # Look ahead
        sample = detector_padded[i + lookahead_samples]

        # Envelope follower
        if sample > envelope:
            envelope = (
                attack_coeff * envelope
                + (1 - attack_coeff) * sample
            )
        else:
            envelope = (
                release_coeff * envelope
                + (1 - release_coeff) * sample
            )

        if envelope > threshold:
            # Soft knee compression
            over = envelope - threshold
            
            if over < knee:
                # Soft knee region
                gain_reduction = over * over / (2 * knee * ratio)
                compressed = envelope - gain_reduction
            else:
                # Hard compression above knee
                compressed = (
                    threshold + knee / 2
                    + (over - knee) / ratio
                )

            gain_curve[i] = compressed / envelope

    return audio * gain_curve[:, np.newaxis]


def loudness_normalize(
    audio,
    sr,
    target_lufs,
    max_gain_db
):
    meter = pyln.Meter(sr)

    current_lufs = meter.integrated_loudness(audio)

    desired_gain_db = target_lufs - current_lufs

    actual_gain_db = min(
        desired_gain_db,
        max_gain_db
    )

    actual_target = (
        current_lufs + actual_gain_db
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Possible clipped samples in output."
        )

        normalized = pyln.normalize.loudness(
            audio,
            current_lufs,
            actual_target
        )

    return (
        normalized,
        current_lufs,
        actual_target,
        actual_gain_db
    )


def limiter(audio, sr, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=100.0):
    """Improved limiter with lookahead and smooth release."""
    ceiling = db_to_linear(ceiling_db)
    
    lookahead_samples = int(sr * lookahead_ms / 1000)
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
    
    # Stereo-linked peak detection
    detector = np.max(np.abs(audio), axis=1)
    
    # Pad for lookahead
    detector_padded = np.pad(
        detector,
        (lookahead_samples, 0),
        mode='edge'
    )
    
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
    audio_limited = audio * gain_curve[:, np.newaxis]
    
    # Calculate total limiting
    limiting_db = linear_to_db(np.min(gain_curve))
    
    return audio_limited, -limiting_db


def hard_clip_protection(audio):
    return np.clip(
        audio,
        -0.999,
        0.999
    )


def enhance_high_frequency_air(audio, sr):
    """Add subtle high-frequency enhancement for 'air' and presence.
    
    Suno tracks sometimes lack sparkle in the high end.
    """
    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)
    
    # Gentle shelf boost above 8kHz
    shelf_freq = 8000.0
    shelf_gain = 1.15  # ~1.2dB boost
    transition_width = 2000.0
    
    # Smooth transition using sigmoid-like curve
    boost = np.ones_like(freqs)
    mask = freqs > (shelf_freq - transition_width)
    boost[mask] = 1.0 + (shelf_gain - 1.0) * (
        1.0 / (1.0 + np.exp(-5 * (freqs[mask] - shelf_freq) / transition_width))
    )
    
    fft *= boost[:, np.newaxis]
    
    return np.fft.irfft(fft, n=audio.shape[0], axis=0)


def adjust_stereo_width(audio, width=1.0):
    """Adjust stereo width using mid-side processing.
    
    width < 1.0: narrow
    width = 1.0: unchanged
    width > 1.0: wider
    
    Suno tracks sometimes have inconsistent stereo imaging.
    """
    if audio.shape[1] < 2 or width == 1.0:
        return audio
    
    # Convert to mid-side
    mid = (audio[:, 0] + audio[:, 1]) / 2.0
    side = (audio[:, 0] - audio[:, 1]) / 2.0
    
    # Adjust side signal
    side *= width
    
    # Convert back to left-right
    left = mid + side
    right = mid - side
    
    return np.column_stack([left, right])


def process_file(
    input_file,
    output_file,
    target_lufs=TARGET_LUFS,
    max_gain_db=MAX_GAIN_DB,
    fade_seconds=4.0,
    stereo_width=1.0,
    add_air=True
):
    print(f"\nProcessing {input_file}")

    audio, sr = sf.read(input_file)

    if audio.ndim == 1:
        audio = np.column_stack(
            [audio, audio]
        )

    meter = pyln.Meter(sr)

    original_lufs = (
        meter.integrated_loudness(audio)
    )

    print(
        f"Original LUFS: "
        f"{original_lufs:.2f}"
    )

    print(
        f"Original Peak: "
        f"{peak_db(audio):.2f} dBFS"
    )

    # Phase 1: Corrective EQ
    audio = highpass_filter(
        audio,
        sr,
        HPF_FREQ
    )

    if detect_muddiness(audio, sr):
        print(
            "Applying mud cleanup..."
        )
        audio = low_mid_cleanup(
            audio,
            sr
        )

    # Phase 2: Stereo width adjustment (before compression)
    if stereo_width != 1.0 and audio.shape[1] == 2:
        print(
            f"Adjusting stereo width: {stereo_width:.2f}"
        )
        audio = adjust_stereo_width(
            audio,
            stereo_width
        )

    # Phase 3: Dynamics (compression)
    audio = compressor(
        audio,
        sr,
        threshold_db=-18,
        ratio=2.0,
        knee_db=6.0,
        lookahead_ms=5.0
    )

    (
        audio,
        measured_lufs,
        actual_target,
        gain_added
    ) = loudness_normalize(
        audio,
        sr,
        target_lufs,
        max_gain_db
    )

    print(
        f"Requested LUFS: {target_lufs:.2f}"
    )

    print(
        f"Actual LUFS target: "
        f"{actual_target:.2f}"
    )

    print(
        f"Gain added: "
        f"{gain_added:.2f} dB"
    )

    peak_after_norm = peak_db(audio)

    print(
        f"Peak after normalization: "
        f"{peak_after_norm:.2f} dBFS"
    )

    # Phase 4: Enhancement (after normalization, before limiting)
    if add_air:
        print("Adding high-frequency air...")
        audio = enhance_high_frequency_air(audio, sr)

    # Phase 5: Limiting
    audio, limiting_db = limiter(
        audio,
        sr,
        LIMITER_CEILING_DB,
        lookahead_ms=5.0,
        release_ms=100.0
    )

    audio = hard_clip_protection(audio)

    if needs_fade_completion(audio, sr):
        print(
            "Fade appears unfinished, "
            "extending fade..."
        )

        audio = repair_fade_with_reverb(
            audio,
            sr,
            tail_seconds=fade_seconds
        )

    final_lufs = (
        meter.integrated_loudness(audio)
    )

    final_peak = peak_db(audio)

    print(
        f"Limiter reduction: "
        f"{limiting_db:.2f} dB"
    )

    print(
        f"Final LUFS: "
        f"{final_lufs:.2f}"
    )

    print(
        f"Final Peak: "
        f"{final_peak:.2f} dBFS"
    )

    print(
        f"LUFS loss from limiting: "
        f"{actual_target - final_lufs:.2f}"
    )

    sf.write(
        output_file,
        audio,
        sr,
        subtype="PCM_24"
    )

    print(
        f"Written: {output_file}"
    )

def needs_fade_completion(audio, sr):
    """
    Detect if the track is fading but hasn't reached silence.
    
    Uses dynamic fade detection to find where the fade actually starts.
    
    Checks:
    1. Consistent downward trend exists
    2. Total fade amount is significant (>3dB)
    3. Ending hasn't reached silence threshold
    """
    mono = np.mean(audio, axis=1)

    # Use dynamic fade detection
    fade_start_sample, fade_duration, rms_windows = detect_fade_start(audio, sr)
    
    # If a fade was detected, check if it's incomplete
    if fade_duration < 0.5:  # Less than half a second - probably not a fade
        return False
    
    # Get fade drop amount
    fade_drop = rms_windows[0] - rms_windows[-1] if len(rms_windows) > 1 else 0
    
    # Check both peak and RMS of the ending
    end_section = mono[-int(sr * 0.2):]
    
    end_peak = linear_to_db(
        np.max(np.abs(end_section))
    )
    
    end_rms = linear_to_db(
        np.sqrt(np.mean(end_section ** 2))
    )

    # Criteria for needing completion:
    # 1. Fade detected with at least 3 dB drop (from detect_fade_start)
    # 2. Ending must be above silence threshold
    #    - Peak above -40 dB OR RMS above -45 dB
    
    fading = fade_drop > 3.0
    unfinished = end_peak > -40.0 or end_rms > -45.0

    # Debug output
    if fading or unfinished:
        print(f"  Fade analysis:")
        print(f"    Detected fade duration: {fade_duration:.2f}s")
        print(f"    Fade drop: {fade_drop:.1f} dB")
        print(f"    End peak: {end_peak:.1f} dB")
        print(f"    End RMS: {end_rms:.1f} dB")
        print(f"    Needs completion: {fading and unfinished}")

    return fading and unfinished

def generate_impulse_response(
    sr,
    tail_seconds=5.0,
    decay_db=80.0
):

    n = int(sr * tail_seconds)

    ir = np.zeros(n)

    # direct impulse
    ir[0] = 1.0

    rng = np.random.default_rng()

    reflections = 500

    positions = rng.integers(
        1,
        n,
        size=reflections
    )

    for pos in positions:

        decay = np.exp(
            -pos / n * 8
        )

        ir[pos] += (
            rng.uniform(
                -1,
                1
            )
            * decay
        )

    ir /= np.max(
        np.abs(ir)
    )

    return ir

def generate_reverb_tail(
    audio,
    sr,
    source_seconds=2.0,
    tail_seconds=5.0,
    decay_db=80.0
):
    """
    Generate a realistic reverb tail
    using convolution.

    Returns only the generated tail.
    """

    source_len = int(
        source_seconds * sr
    )

    tail_len = int(
        tail_seconds * sr
    )

    source = audio[-source_len:]

    ir = generate_impulse_response(
        sr,
        tail_seconds,
        decay_db
    )

    channels = []

    for ch in range(
        source.shape[1]
    ):

        wet = fftconvolve(
            source[:, ch],
            ir,
            mode="full"
        )

        wet = wet[
            source_len:
            source_len + tail_len
        ]

        channels.append(wet)

    tail = np.column_stack(
        channels
    )

    # Match level to actual ending more precisely
    # Use the very last portion (50ms) for better continuity
    
    end_sample_len = min(
        int(sr * 0.05),  # 50ms
        len(source)
    )
    
    end_rms = np.sqrt(
        np.mean(
            source[-end_sample_len:] ** 2
        )
    )

    # Match to the start of the tail
    start_sample_len = min(
        int(sr * 0.05),
        len(tail)
    )
    
    start_rms = np.sqrt(
        np.mean(
            tail[:start_sample_len] ** 2
        )
    )

    if start_rms > 1e-9:
        tail *= (
            end_rms
            / start_rms
        )
    
    # Apply additional fade to ensure smooth decay to silence
    fade_curve = np.linspace(1.0, 0.0, len(tail))
    fade_curve = fade_curve ** 1.5  # Gentle exponential curve
    tail *= fade_curve[:, np.newaxis]

    return tail

def detect_fade_start(audio, sr, max_lookback=8.0):
    """
    Detect where the fade actually starts by analyzing RMS in small windows.
    
    Returns:
        - fade_start_sample: Index where fade begins
        - fade_duration: Duration of detected fade in seconds
    """
    mono = np.mean(audio, axis=1)
    
    # Look back up to max_lookback seconds
    lookback_samples = min(int(max_lookback * sr), len(mono))
    analysis_region = mono[-lookback_samples:]
    
    # Analyze in small windows (200ms)
    window_size = int(sr * 0.2)
    num_windows = max(1, lookback_samples // window_size)
    
    # Calculate RMS for each window
    rms_db = []
    for i in range(num_windows):
        start = i * window_size
        end = min((i + 1) * window_size, len(analysis_region))
        window = analysis_region[start:end]
        rms_db.append(linear_to_db(np.sqrt(np.mean(window ** 2))))
    
    # Find where consistent decline begins
    # Start from the end and work backwards
    fade_start_idx = len(rms_db) - 1  # Default: assume entire region is fading
    
    # Look for the point where the decline becomes consistent
    # We want at least 2dB drop over at least 2 windows
    for i in range(len(rms_db) - 2, -1, -1):
        # Check if there's a consistent downward trend from this point forward
        is_declining = True
        for j in range(i, len(rms_db) - 1):
            # Allow small increases (1dB tolerance)
            if rms_db[j] - rms_db[j + 1] < -1.0:
                is_declining = False
                break
        
        # Check if total drop from this point is significant
        total_drop = rms_db[i] - rms_db[-1]
        
        if is_declining and total_drop >= 3.0:
            fade_start_idx = i
            break
    
    # Convert to sample position and duration
    fade_start_sample = len(mono) - lookback_samples + (fade_start_idx * window_size)
    fade_duration = (len(mono) - fade_start_sample) / sr
    
    return fade_start_sample, fade_duration, rms_db

def calculate_fade_duration(audio, sr, max_duration=5.0, silence_threshold_db=-70.0):
    """
    Calculate how long the fade extension should be based on:
    - The existing fade rate (dB per second) - detected dynamically
    - Current ending level
    - Target silence threshold
    
    Returns the needed duration in seconds, capped at max_duration.
    """
    mono = np.mean(audio, axis=1)
    
    # Dynamically detect where the fade starts
    fade_start_sample, fade_duration, rms_windows = detect_fade_start(audio, sr)
    
    # Analyze only the fading portion
    fade_region = mono[fade_start_sample:]
    
    # Get start and end RMS of the fade
    fade_start_rms_db = linear_to_db(
        np.sqrt(np.mean(fade_region[:int(sr * 0.2)] ** 2))
    )
    
    end_samples = int(sr * 0.1)  # Last 100ms
    end_rms_db = linear_to_db(
        np.sqrt(np.mean(mono[-end_samples:] ** 2))
    )
    
    # Calculate fade rate from the actual fading portion
    fade_drop_total = fade_start_rms_db - end_rms_db
    fade_rate_db_per_sec = fade_drop_total / fade_duration if fade_duration > 0 else 0
    
    # Calculate needed drop to reach silence
    needed_drop = end_rms_db - silence_threshold_db
    
    # Calculate duration needed
    if fade_rate_db_per_sec > 0.5:  # Only if there's meaningful fade
        needed_duration = needed_drop / fade_rate_db_per_sec
        # Cap at maximum duration
        actual_duration = min(needed_duration, max_duration)
    else:
        # Not fading significantly, use max duration
        actual_duration = max_duration
    
    # Ensure minimum duration of 1 second
    actual_duration = max(1.0, actual_duration)
    
    print(f"  Adaptive fade calculation:")
    print(f"    Detected fade duration: {fade_duration:.2f}s")
    print(f"    Fade start level: {fade_start_rms_db:.1f} dB")
    print(f"    Current end level: {end_rms_db:.1f} dB")
    print(f"    Fade rate: {fade_rate_db_per_sec:.1f} dB/sec")
    print(f"    Needed drop to silence: {needed_drop:.1f} dB")
    print(f"    Calculated extension: {actual_duration:.2f}s (max: {max_duration:.1f}s)")
    
    return actual_duration

def repair_fade_with_reverb(
    audio,
    sr,
    tail_seconds=5.0
):
    """
    Append a generated reverb tail with smooth crossfade.
    Tail duration is adaptive based on existing fade rate.
    """
    
    # Calculate adaptive duration based on fade rate
    adaptive_duration = calculate_fade_duration(
        audio, sr, max_duration=tail_seconds
    )

    tail = generate_reverb_tail(
        audio,
        sr,
        source_seconds=2.0,
        tail_seconds=adaptive_duration,
        decay_db=80.0
    )

    # Add crossfade to ensure smooth transition
    # Use a longer crossfade (500ms) for more natural blending
    crossfade_len = int(sr * 0.5)
    
    if len(audio) < crossfade_len:
        # If audio is shorter than crossfade, just append
        return np.vstack([audio, tail])
    
    # Create smooth crossfade curves
    fade_out = np.linspace(1.0, 0.0, crossfade_len)
    fade_in = np.linspace(0.0, 1.0, crossfade_len)
    
    # Apply power curve for more natural fade (equal power crossfade)
    fade_out = np.sqrt(fade_out)
    fade_in = np.sqrt(fade_in)
    
    # Make a copy to avoid modifying original
    output = np.copy(audio)
    
    # Apply crossfade to the overlapping region
    output[-crossfade_len:] = (
        output[-crossfade_len:] * fade_out[:, np.newaxis]
        + tail[:crossfade_len] * fade_in[:, np.newaxis]
    )
    
    # Append the rest of the tail after the crossfade region
    remaining_tail = tail[crossfade_len:]
    
    return np.vstack([output, remaining_tail])


def repair_fade(audio, sr, fade_seconds=4.0):
    """
    Extend an unfinished fade naturally.

    Assumes the track is already fading but
    has not yet reached silence.
    """

    source_len = int(sr * 1.0)
    fade_len = int(sr * fade_seconds)

    source = audio[-source_len:]

    # Repeat enough times to create the tail
    repeats = int(
        np.ceil(fade_len / source_len)
    )

    tail = np.tile(
        source,
        (repeats, 1)
    )

    tail = tail[:fade_len]

    # Estimate current level
    end_rms = np.sqrt(
        np.mean(source ** 2)
    )

    # Fade by another 60 dB
    fade_curve_db = np.linspace(
        0,
        -60,
        fade_len
    )

    fade_curve = (
        10 ** (fade_curve_db / 20.0)
    )

    tail *= fade_curve[:, np.newaxis]

    # Crossfade first 100 ms
    crossfade_len = int(sr * 0.1)

    fade_out = np.linspace(
        1.0,
        0.0,
        crossfade_len
    )

    fade_in = np.linspace(
        0.0,
        1.0,
        crossfade_len
    )

    output = np.copy(audio)

    output[-crossfade_len:] = (
        output[-crossfade_len:] *
        fade_out[:, np.newaxis]
        +
        tail[:crossfade_len] *
        fade_in[:, np.newaxis]
    )

    tail = tail[crossfade_len:]

    return np.vstack([
        output,
        tail
    ])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file"
    )

    parser.add_argument(
        "output_file"
    )

    parser.add_argument(
        "--lufs",
        type=float,
        default=TARGET_LUFS
    )

    parser.add_argument(
        "--max-gain",
        type=float,
        default=MAX_GAIN_DB,
        help="Maximum loudness increase in dB"
    )

    parser.add_argument(
        "--fade-seconds",
        type=float,
        default=5.0,
        help="Maximum duration of the artificial fade tail in seconds"
    )

    parser.add_argument(
        "--stereo-width",
        type=float,
        default=1.0,
        help="Stereo width adjustment (0.5-1.5, where 1.0=unchanged, <1.0=narrower, >1.0=wider)"
    )

    parser.add_argument(
        "--no-air",
        action="store_true",
        help="Disable high-frequency air enhancement"
    )

    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        args.lufs,
        args.max_gain,
        args.fade_seconds,
        args.stereo_width,
        not args.no_air
    )


if __name__ == "__main__":
    main()

