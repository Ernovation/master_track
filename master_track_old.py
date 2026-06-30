#!/usr/bin/env python3

import argparse
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

from scipy.signal import butter, sosfilt


TARGET_LUFS = -12.0
LIMITER_CEILING_DB = -1.0
HPF_FREQ = 35.0


def db_to_linear(db):
    return 10 ** (db / 20.0)


def linear_to_db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def highpass_filter(audio, sr, cutoff=35.0):
    sos = butter(
        4,
        cutoff,
        btype="highpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, audio, axis=0)


def detect_muddiness(audio, sr):
    """
    Measure energy around 250-400 Hz.

    Returns True if track appears muddy.
    """

    mono = np.mean(audio, axis=1)

    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)

    total_energy = np.sum(spectrum)

    mask = (freqs >= 250) & (freqs <= 400)
    muddy_energy = np.sum(spectrum[mask])

    ratio = muddy_energy / total_energy

    return ratio > 0.08


def low_mid_cleanup(audio, sr):
    """
    Gentle notch around 320 Hz.

    Implemented as a broad dip using FFT.
    """

    mono = np.mean(audio, axis=1)

    fft = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / sr)

    center = 320.0
    width = 120.0

    gain = np.ones_like(freqs)

    dip = np.exp(
        -0.5 * ((freqs - center) / width) ** 2
    )

    gain *= (1.0 - 0.15 * dip)

    fft *= gain[:, np.newaxis]

    return np.fft.irfft(
        fft,
        n=audio.shape[0],
        axis=0
    )


def compressor(
    audio,
    threshold_db=-18.0,
    ratio=2.0,
):
    threshold = db_to_linear(threshold_db)

    output = np.copy(audio)

    envelope = np.maximum(
        np.abs(audio[:, 0]),
        np.abs(audio[:, 1])
    )

    gain = np.ones_like(envelope)

    over = envelope > threshold

    gain[over] = (
        threshold
        + (envelope[over] - threshold) / ratio
    ) / envelope[over]

    output *= gain[:, np.newaxis]

    return output


def limiter(audio, ceiling_db=-1.0):
    ceiling = db_to_linear(ceiling_db)

    peak = np.max(np.abs(audio))

    if peak > ceiling:
        audio *= ceiling / peak

    return audio


def loudness_normalize(audio, sr, target_lufs):
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(audio)

    print(
        f"Measured loudness: "
        f"{loudness:.2f} LUFS"
    )

    normalized = pyln.normalize.loudness(
        audio,
        loudness,
        target_lufs
    )

    return normalized


def process_file(
    input_file,
    output_file,
    target_lufs=-12.0
):
    audio, sr = sf.read(input_file)

    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    print("High-pass filtering...")
    audio = highpass_filter(
        audio,
        sr,
        HPF_FREQ
    )

    if detect_muddiness(audio, sr):
        print(
            "Track appears muddy, "
            "applying cleanup."
        )
        audio = low_mid_cleanup(audio, sr)

    print("Loudness normalization...")
    audio = loudness_normalize(
        audio,
        sr,
        target_lufs
    )

    print("Compression...")
    audio = compressor(audio)

    print("Limiting...")
    audio = limiter(
        audio,
        LIMITER_CEILING_DB
    )

    sf.write(
        output_file,
        audio,
        sr,
        subtype="PCM_24"
    )

    print(f"Written: {output_file}")


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

    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        args.lufs
    )


if __name__ == "__main__":
    main()
