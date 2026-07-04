import urllib.request
import numpy as np
import librosa
import soundata

DATA_HOME = "data/urbansound8k"


def get_spectrogram(audio_path, n_fft=2048, hop_length=512): # these two functions are for getting the numpy array from a sample audio file
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)  
    return spectrogram, sr


def main():
    dataset = soundata.initialize("urbansound8k", data_home=DATA_HOME)
    clip_ids = dataset.clip_ids
    sample_clip = dataset.clip(clip_ids[0])
    audio_path = sample_clip.audio_path

    print(f"Using clip: {audio_path}")
    spectrogram, sr = get_spectrogram(audio_path)
    print(f"Spectrogram shape: {spectrogram.shape}  (freq_bins x time_frames)")
    print(f"Sample rate: {sr} Hz")

    return spectrogram # we can access through spectrogram[f , t] where f is frequency, t is time frame


if __name__ == "__main__":
    main()
