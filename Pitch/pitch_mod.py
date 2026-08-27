# Makenna

import sys
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import soundfile as sf


def load_audio(filepath, sample_rate=None):
    audio_signal, sr = librosa.load(filepath, sr=sample_rate, mono=True)
    return audio_signal, sr



def shift_pitch(audio_signal, sr, semitones):
    shifted_audio = librosa.effects.pitch_shift(audio_signal, sr=sr, n_steps=semitones)
    return shifted_audio



def extract_pitch(audio_signal, sr):
    f0 = librosa.yin(audio_signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)        #extracts new pitch using YIN algorithm, expecting lowest pitch = C2 and highest = C7
    f0_clean = f0[~np.isnan(f0)]                                                                            #removes values that aren't NaN
    f0_smooth = savgol_filter(f0_clean, 101, 3) if len(f0_clean) > 101 else f0_clean                                                                     #removes jitter and keeps the shape of the pitch
    times = librosa.times_like(f0, sr=sr)                                                                       #creates a time value in seconds for each pitch frame

    return f0_smooth, times



"""
Plot modified pitch contour.
Returns the figure so it can be embedded in the GUI or shown from CLI.

Note: times is derived from f0 (full length) while f0_smooth is derived
from f0_clean (NaNs removed, potentially shorter). We truncate times to
match f0_smooth length to keep the two arrays aligned.
"""
def plot_pitch_mod(f0_smooth, times, title="Modified Pitch Contour"):
    downsample_factor = 25

    # Truncate times to match f0_smooth length so indices stay aligned
    times = times[:len(f0_smooth)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times[::downsample_factor], f0_smooth[::downsample_factor], label="Shifted Pitch", color='pink')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

"""
Saving Function
"""
def save_audio(audio_signal, sr, filepath):
    sf.write(filepath, audio_signal, sr)
    print(f"Modified audio saved as {filepath}")

"""
Function for GUI to Call
"""
def modify_pitch_gui(audio_signal, sr, semitones):
    shifted_audio = shift_pitch(audio_signal, sr, semitones)
    f0_smooth, times = extract_pitch(shifted_audio, sr)
    return {
        "shifted_audio": shifted_audio,
        "f0_smooth": f0_smooth,
        "times": times,
        "sample_rate": sr
    }

"""
Function for CLI/Main to Call
"""
def modify_pitch_cli(filepath, semitones):
    audio_signal, sr = load_audio(filepath)
    return modify_pitch_gui(audio_signal, sr, semitones)

"""
Main - CLI
"""
def main():
    if len(sys.argv) < 3:
        print("Usage: python pitch_mod.py <audio_file> <semitones_to_shift>")
        sys.exit(1)

    filepath = sys.argv[1]
    semitones = float(sys.argv[2])
    audio_name = os.path.basename(filepath)

    # Modify pitch
    results = modify_pitch_cli(filepath, semitones)
    shifted_audio = results["shifted_audio"]
    f0_smooth = results["f0_smooth"]
    times = results["times"]
    sr = results["sample_rate"]

    # Save modified audio
    output_dir = "Modified_Audio"
    os.makedirs(output_dir, exist_ok=True)
    audio_base = os.path.splitext(audio_name)[0]
    output_file = os.path.join(output_dir, f"{audio_base}_pitch_{semitones:+.2f}.wav")
    save_audio(shifted_audio, sr, output_file)

    # Plot pitch contour
    fig = plot_pitch_mod(f0_smooth, times, title=f"Pitch-Shifted Audio ({audio_base}, {semitones:+.2f} semitones)")
    plt.show()

if __name__ == "__main__":
    main()