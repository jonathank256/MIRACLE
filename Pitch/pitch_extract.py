# Makenna

import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

"""
Function to Load Audio
"""
def load_audio(filepath, sample_rate=None):
    audio_signal, sr = librosa.load(filepath, sr=sample_rate, mono=True)
    return audio_signal, sr

"""
Extracts the pitch (fundamental frequency) of an audio signal over time.
Runs the yin algorithm which estimates the pitch for each short frame of audio
fmin/fmax define the the range of pitches we are looking for
f0 is the array of pitch estimations per frame
"""
def extract_pitch(audio_signal, sr):
    f0 = librosa.yin(
        audio_signal,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    return f0

"""
Clean pitch estimates by removing NaNs (silent frames) 
"""
def clean_pitch(f0):
    return f0[~np.isnan(f0)]


"""
Plotting Function.
Returns the figure so it can be embedded in the GUI or shown from CLI.

Note: times is derived from f0 (full length) while f0_smooth is derived
from f0_clean (NaNs removed, potentially shorter). We truncate times to
match f0_smooth to keep the two arrays aligned.
"""
def plot_pitch_analysis(f0, f0_clean, sr, title="Pitch Contour"):
    downsample_factor = 25
    times = librosa.times_like(f0, sr=sr)
    f0_smooth = savgol_filter(f0_clean, 101, 3) if len(f0_clean) > 101 else f0_clean

    # Truncate times to match f0_smooth length so indices stay aligned
    times = times[:len(f0_smooth)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times[::downsample_factor], f0_smooth[::downsample_factor],
             label="Estimated Pitch", color='purple', linestyle='-')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

"""
Saving Function
"""
def save_pitch_data(f0_clean, filepath="pitch_data.npy"):
    np.save(filepath, f0_clean)

"""
Function for GUI to Call
"""
def analyze_pitch_gui(audio_signal, sr):
    f0 = extract_pitch(audio_signal, sr)
    f0_clean = clean_pitch(f0)
    return {
        "f0": f0,
        "f0_clean": f0_clean,
        "sample_rate": sr
    }

"""
Function for CLI/Main to Call
"""
def analyze_pitch_cli(filepath):
    audio_signal, sr = load_audio(filepath)
    return analyze_pitch_gui(audio_signal, sr)

 
'''
CLI Main
Load the audio from the input
Librosa.load reads the audio file into a numpy array
sr=None keeps the original sample rate to ensure no resampling
'''
def main():

    if len(sys.argv) < 2:
        print("Usage: python pitch_extract.py <audiofile>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Analyze pitch
    results = analyze_pitch_cli(filepath)
    f0 = results["f0"]
    f0_clean = results["f0_clean"]
    sr = results["sample_rate"]

    # Print basic stats
    print("\nPitch Statistics:")
    print("Mean pitch:", round(np.mean(f0_clean), 2), "Hz")
    print("Median pitch:", round(np.median(f0_clean), 2), "Hz")
    print("Min pitch:", round(np.min(f0_clean), 2), "Hz")
    print("Max pitch:", round(np.max(f0_clean), 2), "Hz")

    # Save
    save_pitch_data(f0_clean)
    print("Saved cleaned pitch to pitch_data.npy")

    # Plot
    fig = plot_pitch_analysis(f0, f0_clean, sr)
    plt.show()

if __name__ == "__main__":
    main()