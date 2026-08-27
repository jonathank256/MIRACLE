# Abby

import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
import soundfile as sf


"""
PI1: Load audio and validate tempo change
"""
def load_audio(filepath, sample_rate=22050):
    audio_signal, sr = librosa.load(
        filepath,
        sr=sample_rate,
        mono=True
    )
    return audio_signal, sr

# Ensure tempo change is reasonable (0.5x to 2.0x speed)
def validate_tempo_change(change):
    if change <= 0:
        raise ValueError("Tempo change must be positive")
    if change < 0.5:
        print("Warning: Extremely slow tempo may distort audio.")
    elif change > 2.0:
        print("Warning: Extremely fast tempo may distort audio.")
    return change

"""
PI2: STFT
Time domain signal -> Time-frequency representation 
"""
def compute_stft(audio_signal, window_size=2048, hop_size=512): 
    stft = librosa.stft(
        audio_signal,
        n_fft=window_size,
        hop_length=hop_size
    )
    return stft

"""
PI3: Phase vocoder for time-stretching while preserving pitch
"""
def time_stretch_phase_vocoder(stft, rate):
    return librosa.phase_vocoder(stft, rate=rate)

"""
PI4: Inverse STFT reconstruction
Time-frequency representation -> Time domain signal 
"""
def reconstruct_audio(stft, hop_length=512):
    return librosa.istft(stft, hop_length=hop_length)

"""
PI5: Post-processing to reduce artifacts
Applies median filtering and normalization to smooth out minor time-streching artifacts 
"""
def post_process(audio_signal):
    # Median filtering
    filtered_signal = medfilt(audio_signal, kernel_size=3)

    # Normalization
    max_val = np.max(np.abs(filtered_signal))
    if max_val > 0:
        filtered_signal = filtered_signal / max_val
    return filtered_signal

"""
Plot original vs. modified audio.
Returns the figure so it can be embedded in the GUI or shown from CLI.
"""
def plot_tempo_mod(original, modified, sr, audio_name, tempo_change):
    MAX_POINTS = 4000

    def downsample(signal):
        if len(signal) > MAX_POINTS:
            step = len(signal) // MAX_POINTS
            return signal[::step]
        return signal
    
    orig_ds = downsample(original)
    mod_ds = downsample(modified)

    orig_times = np.linspace(0, len(original) / sr, len(orig_ds))
    mod_times = np.linspace(0, len(modified) / sr, len(mod_ds))

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(orig_times, orig_ds, alpha=0.6, label='Original')
    ax.plot(mod_times, mod_ds, alpha=0.6, color='red', label='Modified')

    ax.legend()
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Original vs Tempo-Modified Audio")
    fig.tight_layout()
    return fig

"""
Saving Function
"""
def save_tempo_mod(modified_audio, sr, filepath):
    # Save modified audio to disk
    sf.write(filepath, modified_audio, sr)

"""
Function for GUI to Call
"""
def modify_tempo_gui(audio_signal, sr, tempo_factor):
    # Modifiy tempo without changing pitch
    tempo_factor = validate_tempo_change(tempo_factor)

    stft_matrix = compute_stft(audio_signal)
    stretched_stft = time_stretch_phase_vocoder(stft_matrix, tempo_factor)
    modified_audio = reconstruct_audio(stretched_stft)
    modified_audio = post_process(modified_audio)

    # Return modified audio 
    return modified_audio

"""
Function for Main to Call 
"""
def modify_tempo_cli(filepath, tempo_factor):
    audio_signal, sr = load_audio(filepath)
    modified_audio = modify_tempo_gui(audio_signal, sr, tempo_factor)
    return modified_audio, sr

"""
Main - CLI
"""
def main():
    if len(sys.argv) < 3:
        print("Usage: python tempo_mod.py <audio_file> <tempo_factor>")
        sys.exit(1)

    filepath = sys.argv[1]
    tempo_factor = float(sys.argv[2])
    tempo_factor = validate_tempo_change(tempo_factor)
    audio_name = os.path.basename(filepath)

    # Modifiy audio
    modified_audio, sr = modify_tempo_cli(filepath, tempo_factor)

    # Save modified audio
    output_dir = "Modified_Audio"
    os.makedirs(output_dir, exist_ok=True)
    audio_base = os.path.splitext(audio_name)[0]
    output_file = os.path.join(output_dir, f"{audio_base}_tempo_{tempo_factor:.2f}.wav")
    save_tempo_mod(modified_audio, sr, output_file)
    print(f"Tempo-modified audio saved to {output_file}")

    # Plot original vs. modified waveform
    original_audio, _ = load_audio(filepath)
    fig = plot_tempo_mod(original_audio, modified_audio, sr, audio_name, tempo_factor)
    plt.show()


if __name__ == "__main__":
    main()