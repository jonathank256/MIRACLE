# Abby

import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os 

from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


"""
PI1
Loads an audio file and returns the audio signal and sample rate.
Converts to mono and resamples to the specified sample rate.
"""
def load_audio(filepath, sample_rate=22050):  # Default librosa sampling rate
    audio_signal, sr = librosa.load(
        filepath,
        sr=sample_rate,
        mono=True
    )
    return audio_signal, sr

"""
PI2
Computes the magnitude spectrogram of an audio signal using STFT.
Returns a 2D array of magnitude values per frequency bin and frame.
"""
def compute_magnitude_spectrogram(audio_signal, 
                                  window_size=2048, 
                                  hop_size=512): 
    stft = librosa.stft(
        audio_signal,
        n_fft=window_size,
        hop_length=hop_size
    )
    mag_spectrum = np.abs(stft)
    return mag_spectrum

"""
PI3
Calculates the spectral flux from a magnitude spectrogram.
Spectral flux measures positive changes in energy over time, which helps detect onsets.
""" 
def compute_spectral_flux(mag_spectrum):
    # Difference between magnitude spectra along time axis
    flux = np.diff(mag_spectrum, axis=1) 

    # Keep only increases in energy)
    flux = np.maximum(0, flux) 

    # Sum over all frequency bins      
    flux = np.sum(flux, axis=0)   

    return flux

"""
PI4
Estimates global tempo (BPM) and initial beat locations.
Uses librosa's onset envelope and beat tracking algorithm.
Returns tempo, beat times in seconds, and the onset envelope.
"""
def estimate_tempo_and_beats(audio_signal, sample_rate=22050):
    # Detect onset envelope (changes in energy that indicate beats)
    onset_env = librosa.onset.onset_strength(y=audio_signal, sr=sample_rate)

    # Estimate global tempo (BPM) and beat frames
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)

    # Convert beat frames indices to times in seconds 
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

    return tempo, beat_times, onset_env

"""
PI5
Refines beat locations by smoothing the onset envelope, picking peaks, and phase-locking to local maxima.
Returns the smoothed onset envelope and refined beat frame indices.
"""
def refine_beats(onset_env, smoothing_window=3, gaussian_sigma=1.0, 
                 peak_distance=3, phase_radius=2):
    
    # 1. Median smoothing
    # Reduces short-term spikes in the onset envelope
    # Helps remove isolated noise without blurring sharp peaks 
    smoothed_env = medfilt(onset_env, kernel_size=smoothing_window)
    
    # 2. Optional Gaussian smoothing to reduce high-frequency noise
    # Helps ensure detected peaks correspond to true musical beats rather than noise
    smoothed_env = gaussian_filter1d(smoothed_env, sigma=gaussian_sigma)
    
    # 3. Peak picking
    # Identify local maxima in the smoothed envelope that could correspond to beats
    peaks, _ = find_peaks(smoothed_env, distance=peak_distance)

    # 4. Phase-locking
    # For each peak, search a small window of +/- phase_radius frames
    # Find the local maximum in the smoothed envelope
    # Ensures detected beat aligns more precisely with the actual onset. 
    refined_beats = []
    for t in peaks:
        # Search window around peak
        start = max(0, t - phase_radius) 
        end = min(len(smoothed_env)-1, t + phase_radius)
        # Find local maximum within window
        local_max = start + np.argmax(smoothed_env[start:end+1])
        refined_beats.append(local_max)
    
    return smoothed_env, np.array(refined_beats) 
    # refined_beats: Array of frame indices representing refined beat positions

"""
Plotting Function
Returns the figure so it can be embedded in the GUI or shown from CLI.
"""
def plot_tempo_analysis(onset_env, smoothed_env, refined_beats, sr, title="Tempo Analysis"):
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    refined_times = librosa.frames_to_time(refined_beats, sr=sr)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, onset_env, label='Original Onset Envelope')
    ax.plot(times, smoothed_env, label='Smoothed Envelope', linestyle='--')
    ax.scatter(refined_times, smoothed_env[refined_beats], color='red', label='Refined Beats')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

"""
Saving Function
"""
def save_tempo_data(data, filepath):
    np.save(filepath, data)

"""
Function for GUI to Call 
"""
def analyze_tempo_gui(audio_signal, sr):

    # Compute magnitude spectrogram
    mag_spec = compute_magnitude_spectrogram(audio_signal)

    # Compute spectral flux
    flux = compute_spectral_flux(mag_spec)

    # Estimate tempo and beats
    tempo, beat_times, onset_env = estimate_tempo_and_beats(audio_signal, sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Refine beats
    smoothed_env, refined_beats = refine_beats(onset_env)
    refined_beat_times = librosa.frames_to_time(refined_beats, sr=sr)

    # Return everything instead of printing/saving
    return {
        "tempo_bpm": tempo,
        "beat_times": beat_times,
        "refined_beats_frames": refined_beats,
        "refined_beats_seconds": refined_beat_times,
        "onset_env": onset_env,
        "smoothed_env": smoothed_env,
        "sample_rate": sr
    }


"""
Function for CLI/Main to Call
"""
def analyze_tempo_cli(filepath):
    audio_signal, sr = load_audio(filepath)
    return analyze_tempo_gui(audio_signal, sr)

"""
CLI Main
"""
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <audiofile>")
        sys.exit(1)

    filepath = sys.argv[1]

    results = analyze_tempo_cli(filepath)

    print(f"Tempo: {results['tempo_bpm']:.2f} BPM")
    print(f"Beats detected: {len(results['beat_times'])}")

    # Save
    save_tempo_data(results, "tempo_data.npy")
    print("Saved to tempo_data.npy")

    # Plot
    fig = plot_tempo_analysis(
        results["onset_env"],
        results["smoothed_env"],
        results["refined_beats_frames"],
        results["sample_rate"]
    )
    plt.show()


if __name__ == "__main__":
    main()