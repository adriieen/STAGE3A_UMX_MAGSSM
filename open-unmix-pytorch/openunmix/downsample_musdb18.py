#!/usr/bin/env python3
"""
Downsample a musdb18 WAV dataset by an integer factor.

Usage:
    python downsample_musdb18.py --input /dev/shm/musdb18 --output /dev/shm/musdb18_ds4 --factor 4

This resamples every WAV stem (vocals, drums, bass, other, mixture, etc.)
from 44100 Hz → 44100/factor Hz using torchaudio.functional.resample,
which applies a proper low-pass anti-aliasing filter before decimation.

The output directory mirrors the input structure:
    output/
      train/
        TrackName/
          vocals.wav
          drums.wav
          ...
      test/
        TrackName/
          ...
"""
import argparse
from pathlib import Path

import torch
import torchaudio


def downsample_dataset(input_root: Path, output_root: Path, factor: int) -> None:
    orig_sr = 44100
    new_sr = orig_sr // factor

    print(f"Resampling {orig_sr} Hz → {new_sr} Hz  (factor ×{factor})")
    print(f"  Input:  {input_root}")
    print(f"  Output: {output_root}\n")

    for subset in ["train", "test"]:
        subset_in = input_root / subset
        if not subset_in.exists():
            print(f"[SKIP] {subset_in} does not exist")
            continue

        tracks = sorted([d for d in subset_in.iterdir() if d.is_dir()])
        print(f"[{subset}] {len(tracks)} tracks")

        for i, track_dir in enumerate(tracks):
            track_out = output_root / subset / track_dir.name
            track_out.mkdir(parents=True, exist_ok=True)

            wav_files = sorted(track_dir.glob("*.wav"))
            for wav_path in wav_files:
                out_path = track_out / wav_path.name

                # Skip if already converted
                if out_path.exists():
                    continue

                audio, sr = torchaudio.load(str(wav_path))
                assert sr == orig_sr, f"Expected {orig_sr} Hz, got {sr} Hz for {wav_path}"

                # Proper anti-aliased resample
                audio_ds = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=new_sr)

                torchaudio.save(str(out_path), audio_ds, new_sr)

            print(f"  [{i+1}/{len(tracks)}] {track_dir.name}", end="\r")

        print()  # newline after subset

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample musdb18 WAV dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to original musdb18 WAV dataset")
    parser.add_argument("--output", type=str, required=True, help="Path for downsampled output")
    parser.add_argument("--factor", type=int, default=4, help="Downsampling factor (default: 4)")
    args = parser.parse_args()

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input path not found: {input_root}")

    assert 44100 % args.factor == 0, f"44100 must be divisible by factor {args.factor}"

    downsample_dataset(input_root, output_root, args.factor)
