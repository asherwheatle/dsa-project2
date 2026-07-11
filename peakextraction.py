import argparse
import heapq
import numpy as np
import librosa
import soundata

DATA_HOME = "data/urbansound8k"


def get_spectrogram(audio_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    return spectrogram, sr


def is_local_peak(spectrogram, f, t):
    n_freqs, n_frames = spectrogram.shape
    val = spectrogram[f, t]
    for df in [-1, 0, 1]:
        for dt in [-1, 0, 1]:
            if df == 0 and dt == 0:
                continue
            nf, nt = f + df, t + dt
            if 0 <= nf < n_freqs and 0 <= nt < n_frames:
                if spectrogram[nf, nt] >= val:
                    return False
    return True


def top_k_peaks(spectrogram, k=10):
    n_freqs, n_frames = spectrogram.shape
    heap = []

    for f in range(n_freqs):
        for t in range(n_frames):
            if not is_local_peak(spectrogram, f, t):
                continue

            val = spectrogram[f, t]
            if len(heap) < k:
                heapq.heappush(heap, (val, f, t))
            elif val > heap[0][0]:
                heapq.heapreplace(heap, (val, f, t))

    return sorted(heap, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    dataset = soundata.initialize("urbansound8k", data_home=DATA_HOME)
    clip_ids = dataset.clip_ids
    sample_clip = dataset.clip(clip_ids[0])
    audio_path = sample_clip.audio_path

    print(f"Using clip: {audio_path}")
    spectrogram, sr = get_spectrogram(audio_path)
    print(f"Spectrogram shape: {spectrogram.shape}  (freq_bins x time_frames)")
    print(f"Sample rate: {sr} Hz")

    peaks = top_k_peaks(spectrogram, k=args.k)
    print(f"\nTop {args.k} peaks:")
    for amp, f, t in peaks:
        freq_hz = f * sr / 2048
        print(f"  amp={amp:.4f}  freq={freq_hz:.1f} Hz  frame={t}")

    return spectrogram


if __name__ == "__main__":
    main()
