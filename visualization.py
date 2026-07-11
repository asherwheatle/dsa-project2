import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundata

from peakextraction import get_spectrogram, top_k_peaks, DATA_HOME

def getSampleAudioPath():
    dataset = soundata.initialize("urbansound8k", data_home=DATA_HOME)
    clip_ids = dataset.clip_ids
    sample_clip = dataset.clip(clip_ids[0])
    return sample_clip.audio_path

def main():
    audio_path = getSampleAudioPath()
    print("Using clip:", audio_path)
    spectrogram, sr = get_spectrogram(audio_path)
    print("Spectrogram shape:", spectrogram.shape)

    k = int(input("# of peaks to extract (k): "))
    db_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    fig, (top_graph, bottom_graph) = plt.subplots(2, 1, figsize=(9, 8))

    img = librosa.display.specshow(db_spectrogram, sr=sr, x_axis="time", y_axis="log", ax=top_graph)
    top_graph.set_title("Spectrogram")
    fig.colorbar(img, ax=top_graph, format="%+2.0f dB")


    peak_amplitude = spectrogram.max(axis=0)
    peak_amplitude_db = librosa.amplitude_to_db(peak_amplitude, ref=np.max)
    peak_amplitude_db = peak_amplitude_db + 130
    times = librosa.frames_to_time(np.arange(len(peak_amplitude_db)), sr=sr)

    peaks = top_k_peaks(spectrogram, k=k)
    for amp, f, t in peaks:
        peak_time_s = times[t]
        peak_amp_value = peak_amplitude_db[t]
        bottom_graph.plot(peak_time_s, peak_amp_value, "o", markerfacecolor="none", markeredgecolor="red")

  

    bottom_graph.plot(times, peak_amplitude_db, color="blue", label="Peak amplitude (dB)")
    bottom_graph.axhline(120, color="red", linestyle="--", label="Mic Max (120 dB)")
    bottom_graph.set_title("Peak Amplitude Over Time")
    bottom_graph.set_xlabel("Time (s)")
    bottom_graph.set_ylabel("Amplitude (dB)")
  

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()