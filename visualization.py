import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundata

from peakextraction import get_spectrogram, top_k_peaks, DATA_HOME
from segment_tree import SegmentTree

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
    threshold = float(input("Amplitude threshold (dB): "))
    db_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    fig, (top_graph, bottom_graph) = plt.subplots(2, 1, figsize=(9, 8))

    img = librosa.display.specshow(db_spectrogram, sr=sr, x_axis="time", y_axis="log", ax=top_graph)
    top_graph.set_title("Spectrogram")
    fig.colorbar(img, ax=top_graph, format="%+2.0f dB")


    peak_amplitude = spectrogram.max(axis=0)
    peak_amplitude_db = librosa.amplitude_to_db(peak_amplitude, ref=np.max)
    peak_amplitude_db = peak_amplitude_db + 130
    times = librosa.frames_to_time(np.arange(len(peak_amplitude_db)), sr=sr)

    interval_lines = None
    interval_peak = None
    use_interval = input("Query peak in a time interval using segment tree? (y/n): ").strip().lower()
    if use_interval == "y":
        st = SegmentTree(peak_amplitude_db)
        max_time = float(times[-1])
        start_time = float(input(f"  Interval start time in seconds (0.00 to {max_time:.2f}): "))
        end_time = float(input(f"  Interval end time in seconds ({start_time:.2f} to {max_time:.2f}): "))
        l_idx = int(np.searchsorted(times, start_time))
        r_idx = int(np.searchsorted(times, end_time, side="right")) - 1
        l_idx = max(0, min(l_idx, len(times) - 1))
        r_idx = max(0, min(r_idx, len(times) - 1))
        if l_idx > r_idx:
            l_idx, r_idx = r_idx, l_idx
        peak_idx = st.range_max_index(l_idx, r_idx)
        interval_lines = (times[l_idx], times[r_idx])
        interval_peak = (times[peak_idx], peak_amplitude_db[peak_idx])
        print(f"  Peak in [{times[l_idx]:.2f}s, {times[r_idx]:.2f}s]: {peak_amplitude_db[peak_idx]:.1f} dB at {times[peak_idx]:.2f}s")

    peaks = top_k_peaks(spectrogram, k=k)
    for amp, f, t in peaks:
        peak_time_s = times[t]
        peak_amp_value = peak_amplitude_db[t]
        if peak_amp_value >= threshold:
            bottom_graph.plot(peak_time_s, peak_amp_value, "o", markerfacecolor="none", markeredgecolor="red")

  

    bottom_graph.plot(times, peak_amplitude_db, color="blue", label="Peak amplitude (dB)")
    bottom_graph.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.0f} dB)")

    if interval_lines is not None:
        t_start, t_end = interval_lines
        bottom_graph.axvline(t_start, color="green", linestyle=":", linewidth=1.5, label=f"Interval start ({t_start:.2f}s)")
        bottom_graph.axvline(t_end, color="green", linestyle=":", linewidth=1.5, label=f"Interval end ({t_end:.2f}s)")
        top_graph.axvline(t_start, color="green", linestyle=":", linewidth=1.5)
        top_graph.axvline(t_end, color="green", linestyle=":", linewidth=1.5)
    if interval_peak is not None:
        bottom_graph.plot(interval_peak[0], interval_peak[1], "*", markersize=14, color="green",
                          label=f"Interval peak ({interval_peak[1]:.1f} dB at {interval_peak[0]:.2f}s)")

    bottom_graph.set_title("Peak Amplitude Over Time")
    bottom_graph.set_xlabel("Time (s)")
    bottom_graph.set_ylabel("Amplitude (dB)")
    bottom_graph.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()