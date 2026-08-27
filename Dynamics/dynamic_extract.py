# Jonathan

import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

"""
PI1
Loads an audio file and returns the audio signal and sample rate.
Converts to mono and resamples to specified sample rate.
Applies simple peak normalization to stabilize amplitude scale across files.
"""
def load_audio(filepath, sample_rate=22050):
    audio_signal, sr = librosa.load( # load audio, resample to sr, mix to mono
        filepath,
        sr=sample_rate,
        mono=True
    )
    # Peak normalization, avoids division by 0 if silence
    peak = np.max(np.abs(audio_signal)) if audio_signal.size else 0.0
    # peak normalize to max amplitude is 1.0, compare different files/scaling
    if peak > 0:
        audio_signal = audio_signal / peak
    # trims leading/trailing silence to remove boundary artifacts
    audio_signal, _ = librosa.effects.trim(audio_signal, top_db=40)

    return audio_signal, sr

"""
PI2
Segments the audio into short-time frames, compute short-time RMS energy.
Returns:
- rms (linear RMS per frame)
- rms_db (RMS converted to dB for perceptual "loudness" curve)
"""
def compute_rms(audio_signal, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms( # compute RMS energy per frame, flatten to (n_frames,)
        y=audio_signal,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True
    )[0]

    # Converts RMS amplitude to dB relative to max frame (0 dB at max)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    return rms, rms_db

"""
PI3
Extracts spectral centroid features to model perceived timbral brightness.
Returns spectral centroid in Hz per frame.
"""
def compute_spectral_centroid(audio_signal, sample_rate=22050, n_fft=2048, hop_length=512):
    # spectral centroid = "brightness", avg freq weighted by energy
    # uses STFT settings matching RMS
    centroid = librosa.feature.spectral_centroid( # compute centroid per frame in Hz
        y=audio_signal,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True
    )[0]

    return centroid

"""
PI4
Aligns loudness (RMS dB) and brightness (centroid) features on the same time axis.
Uses frame indices to time conversion via librosa.frames_to_time.
Returns times (seconds) and aligned feature arrays of the same length.
"""
def align_features(rms_db, centroid, sample_rate=22050, hop_length=512):
    # ensures arrays are same length as features may differ by 1 frame
    L = min(len(rms_db), len(centroid))
    # truncate to equal length
    rms_db = rms_db[:L]
    centroid = centroid[:L]
    # convert frame indices 0..L-1 to time in seconds, hop length: frame k
    # corresponds to k*hop_length/sr in seconds
    times = librosa.frames_to_time(
        np.arange(L),
        sr=sample_rate,
        hop_length=hop_length
    )

    return times, rms_db, centroid

"""
PI5
Applies temporal smoothing and normalization to stabilize trajectories.

Smoothing:
- Median filter (effective to remove isolated spikes)
- Gaussian smoothing (reduces high-frequency jitter)

Normalization:
- z-score using median/MAD
Returns smoothed and normalized versions
"""
def smooth_and_normalize(rms_db, centroid, median_kernel=5, gaussian_sigma=1.0, eps=1e-8):
    # replace NaNs from gating with median so filters do not have strange behaviour
    if np.isnan(centroid).any():
        centroid = np.where(np.isnan(centroid), np.nanmedian(centroid), centroid)
    # smoothing
    # median filtering
    if median_kernel is not None and median_kernel > 1: # median filter kernel size must be odd
        if median_kernel %2 == 0:
            median_kernel += 1
        # remove isolated spikes without too much blurring
        rms_s = medfilt(rms_db, kernel_size=median_kernel)
        cen_s = medfilt(centroid, kernel_size=median_kernel)
    else:
        rms_s = rms_db.copy()
        cen_s = centroid.copy()
    
    # Gaussian smoothing
    # smooths jitter across frames, sigma controls smoothing strength
    if gaussian_sigma is not None and gaussian_sigma > 0:
        rms_s = gaussian_filter1d(rms_s, sigma=gaussian_sigma)
        cen_s = gaussian_filter1d(cen_s, sigma=gaussian_sigma)

    # normalization median/MAD
    # compute robust Z score, centered by median, scaled by MAD (median absolute deviation)
    # 1.4826 makes MAD comparable to standard deviation for normal data
    # the point: "how high or low is this frame compared to the typical value of this song?"
    def robust_z(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        return (x - med) / (1.4826 * mad)
    # both features put on comparable scale
    rms_norm = robust_z(rms_s)
    cen_norm = robust_z(cen_s)

    return rms_s, cen_s, rms_norm, cen_norm

"""
Plotting function for dynamics analysis.
Returns the figure so it can be embedded in the GUI or shown from CLI.
"""
def plot_dynamics_analysis(times, rms_db_a, rms_db_s, centroid_a, centroid_s, rms_norm, centroid_norm,
                           title="Loudness (RMS dB) and Brightness (Spectral Centroid)"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # RMS (dB)
    axes[0].plot(times, rms_db_a, label="RMS (dB) raw")
    axes[0].plot(times, rms_db_s, linestyle="--", label="RMS (dB) smoothed")
    axes[0].set_ylabel("dB (ref=max)")
    axes[0].set_title(title)
    axes[0].legend()

    # Spectral Centroid
    axes[1].plot(times, centroid_a, label="Centroid (Hz) raw")
    axes[1].plot(times, centroid_s, linestyle="--", label="Centroid (Hz) smoothed")
    axes[1].set_ylabel("Hz")
    axes[1].legend()

    # Normalized
    axes[2].plot(times, rms_norm, label="RMS norm")
    axes[2].plot(times, centroid_norm, label="Centroid norm")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Robust z-score")
    axes[2].legend()

    fig.tight_layout()
    return fig

def analyze_dynamics_gui(audio_signal, sr):
    """
    Function for GUI to call.
    Runs the full dynamics extraction pipeline on a pre-loaded audio signal
    and returns a dictionary of smoothed and normalized features.
    """
    import librosa
    import numpy as np
 
    n_fft = 2048
    hop_length = 512
 
    # PI2: RMS loudness
    rms, rms_db = compute_rms(audio_signal, frame_length=n_fft, hop_length=hop_length)
 
    # PI3: Spectral brightness
    centroid = compute_spectral_centroid(
        audio_signal, sample_rate=sr, n_fft=n_fft, hop_length=hop_length
    )
 
    # PI4: Align onto a common time axis
    times, rms_db_a, centroid_a = align_features(
        rms_db, centroid, sample_rate=sr, hop_length=hop_length
    )
 
    # Gate centroid in very quiet frames (unreliable brightness estimates)
    rms_linear_a = librosa.db_to_amplitude(rms_db_a, ref=1.0)
    quiet = rms_linear_a < 1e-3
    centroid_a = centroid_a.copy()
    centroid_a[quiet] = np.nan
 
    # PI5: Smooth and normalize
    rms_db_s, centroid_s, rms_norm, centroid_norm = smooth_and_normalize(
        rms_db_a, centroid_a, median_kernel=5, gaussian_sigma=1.0
    )
 
    return {
        "times": times,
        "rms_db": rms_db_a,
        "rms_db_smoothed": rms_db_s,
        "centroid_hz": centroid_a,
        "centroid_hz_smoothed": centroid_s,
        "rms_norm": rms_norm,
        "centroid_norm": centroid_norm,
        "sample_rate": sr,
    }

def main():
    if len(sys.argv) < 2:
        print("Input does not match expected format.")
        print("Usage: python dynamic_extract.py <input_audio_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    print("About to load audio...")

    # PI1: load
    audio_signal, sr = load_audio(filepath)

    print("Finished librosa.load")

    print(f"Sample rate: {sr}")
    print(f"Audio length (samples): {len(audio_signal)}")
    print(f"Duration (seconds): {len(audio_signal)/sr:.2f}")

    # ensure RMS and centroid use consistent hop
    n_fft = 2048
    hop_length = 512

    # PI2: RMS
    rms, rms_db = compute_rms(audio_signal, frame_length=n_fft, hop_length=hop_length)

    # PI3: spectral centroid
    centroid = compute_spectral_centroid(audio_signal, sample_rate=sr, n_fft=n_fft, hop_length=hop_length)

    # PI4 align
    times, rms_db_a, centroid_a = align_features(rms_db, centroid, sample_rate=sr, hop_length=hop_length)

    # gate centroid in very quiet frames (due to unreliability)
    # centroid is unreliable when energy is small, converts dB back to amplitude-ish just to threshold
    # any frame with very low energy has centroid set to a NaN, PI5 replaces these with median pre-smoothing
    rms_linear_a = librosa.db_to_amplitude(rms_db_a, ref=1.0) # rough back conversion
    quiet = rms_linear_a < 1e-3
    centroid_a = centroid_a.copy()
    centroid_a[quiet] = np.nan

    # PI5: smooth and normalize
    rms_db_s, centroid_s, rms_norm, centroid_norm = smooth_and_normalize(
        rms_db_a,
        centroid_a,
        median_kernel=5,
        gaussian_sigma=1.0
    )

    print(f"Frames: {len(times)}")
    print(f"RMS dB (smoothed): min={rms_db_s.min():.2f}, max={rms_db_s.max():.2f}")
    print(f"Centroid Hz (smoothed): min={centroid_s.min():.2f}, max={centroid_s.max():.2f}")

    # Save extracted dynamic features to .npy file (currently overwriteable)
    dynamic_data = {
        "times_seconds": times,
        "rms_db_smoothed": rms_db_s,
        "centroid_hz_smoothed": centroid_s,
        "rms_normalized": rms_norm,
        "centroid_normalized": centroid_norm
    }

    np.save("dynamic_data.npy", dynamic_data)
    print("Dynamic feature data saved as dynamic_data.npy")

    # Plot
    fig = plot_dynamics_analysis(times, rms_db_a, rms_db_s, centroid_a, centroid_s, rms_norm, centroid_norm)
    plt.show()

if __name__ == "__main__":
    main()