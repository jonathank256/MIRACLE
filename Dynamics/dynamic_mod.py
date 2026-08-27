# Jonathan

import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

from scipy.ndimage import gaussian_filter1d

from .dynamic_extract import (
    load_audio,
    compute_rms,
    compute_spectral_centroid,
    align_features,
    smooth_and_normalize
)

"""
PI1: Validate user-specified loudness modification parameters and scaling ranges.

user_gain_db:
    Overall output gain applied after dynamic processing, in decibels.
    Positive values increase output loudness, negative values reduce it.
    Applied after normalization so the effect is preserved in the final file.

dynamic_strength:
    Controls how strongly local RMS deviations influence scaling.
    0.0 = almost no dynamic modification
    1.0 = moderate modification
    >1.0 = stronger emphasis/suppression of energy changes

brightness_strength:
    Controls how strongly timbral brightness influences the gain curve. Brightness
    is approximated with the spectral centroid, where higher values correspond to
    high-frequency energy.

    Increasing this value causes brighter sections of audio to be moderated
    slightly during scaling, helping preserve perceptual balance.
"""
def parameter_validation(user_gain_db=0.0, dynamic_strength=1.0, brightness_strength=0.25, compression_amount=0.25):
    if not isinstance(user_gain_db, (int, float)):
        raise ValueError("user_gain_db must be a numeric value.")
    
    if not isinstance(dynamic_strength, (int, float)):
        raise ValueError("dynamic_strength must be a numeric value.")
    
    if not isinstance(brightness_strength, (int, float)):
        raise ValueError("brightness_strength must be a numeric value.")
    
    if not isinstance(compression_amount, (int, float)):
        raise ValueError("compression_amount must be a numeric value.")
    
    if user_gain_db < -24 or user_gain_db > 24:
        raise ValueError("user_gain_db should be between -24 and +24 dB.")
    
    if dynamic_strength < 0.0 or dynamic_strength > 3.0:
        raise ValueError("dynamic_strength should be between 0.0 and 3.0.")
    
    if brightness_strength < 0.0 or brightness_strength > 1.0:
        raise ValueError("brightness_strength should be between 0.0 and 1.0.")
    
    if compression_amount < 0.0 or compression_amount > 1.0:
        raise ValueError("compression_amount must be between 0.0 and 1.0.")
    
    return float(user_gain_db), float(dynamic_strength), float(brightness_strength), float(compression_amount)

"""
Helper:
Converts frame-level features to signal-length sample envelope with linear interpolation.
"""
def frames_to_samples(feature, audio_len, hop_length):
    if len(feature) == 0:
        return np.ones(audio_len, dtype=np.float32)
    # create sample position of each frame
    frame_positions = np.arange(len(feature)) * hop_length
    sample_positions = np.arange(audio_len)
    # if only one frame, create array of that single value
    if len(frame_positions) < 2:
        return np.full(audio_len, feature[0] if len(feature) > 0 else 1.0, dtype=np.float32)
    # linear interpolation between frames
    return np.interp(sample_positions, frame_positions, feature).astype(np.float32)

"""
PI2: Apply gain scaling based on normalized RMS-derived loudness deviations.

Frames with lower than typical energy receive relatively higher gain, while
frames with higher than typical energy receive relatively lower gain, subject to
dynamic_strength and gain clipping.
"""
def apply_scaling(audio_signal, rms_env, hop_length, dynamic_strength=1.0):
    # quieter than typical frames (negative rms_env) get more gain
    # louder than typical frames (positive rms_env) get less gain
    gain_curve_frames = np.exp(-dynamic_strength * rms_env)

    # limit gain range to prevent extreme changes
    gain_curve_frames = np.clip(gain_curve_frames, 0.6, 1.6)
    # smooth gain curve
    gain_curve_frames = gaussian_filter1d(gain_curve_frames, sigma=3)

    return gain_curve_frames

"""
PI3: Implement dynamic range adjustment driven by the energy envelope.

compression_amount:
    0.0 -> no moderation of the gain curve
    higher values -> pull frame gains closer to 1.0
"""
def implement_adjustment(gain_curve_frames, compression_amount=0.25):
    if compression_amount < 0.0 or compression_amount > 1.0:
        raise ValueError("compression_amount must be between 0.0 and 1.0.")
    
    # move gain values partly toward 1.0 to reduce over-aggressive dynamics
    adjusted_gain = 1.0 + (gain_curve_frames - 1.0) * (1.0 - compression_amount)
    adjusted_gain = np.clip(adjusted_gain, 0.25, 4.0)

    return adjusted_gain

"""
PI4: Peak-normalize the output signal so the final waveform stays below clipping.
"""
def normalize_output(audio_signal, peak=0.98):
    max_val = np.max(np.abs(audio_signal)) if audio_signal.size else 0.0
    if max_val < 1e-12:
        return audio_signal.astype(np.float32)
    
    output = audio_signal / max_val
    output = output * peak
    return output.astype(np.float32)

"""
Apply user_gain_db as a final output level adjustment after normalization.

Because normalize_output always rescales to a fixed peak, any gain baked into the
dynamic curve before that step gets erased. Instead, we convert user_gain_db to a
linear multiplier and apply it last, then hard-clip to [-1, 1] to prevent clipping.
"""
def apply_user_gain(audio_signal, user_gain_db):
    if user_gain_db == 0.0:
        return audio_signal
    linear_gain = 10 ** (user_gain_db / 20.0)
    output = audio_signal * linear_gain
    # hard clip to prevent values exceeding digital full scale
    output = np.clip(output, -1.0, 1.0)
    return output.astype(np.float32)

"""
PI5: Implement adaptive/context-aware scaling strategies to preserve perceptual balance.

Uses brightness as a mild modifier so already bright segments are not pushed too hard.
"""
def context_aware_scaling(gain_curve_frames, centroid_norm, brightness_strength=0.25):
    # positive centroid_norm = brighter than typical, reduce gain slightly for brighter regions
    # compress centroid range to prevent extreme adjustments to brightness
    centroid_norm = np.tanh(centroid_norm)
    # compute brightness modifier
    brightness_modifier = np.exp(-brightness_strength * centroid_norm)
    adapted_gain = gain_curve_frames * brightness_modifier
    adapted_gain = np.clip(adapted_gain, 0.25, 4.0)

    return adapted_gain

"""
Plotting function for dynamics modification.
Shows RMS energy (normalized) vs the applied gain curve over time.
Returns the figure so it can be embedded in the GUI or shown from CLI.
"""
def plot_dynamics_mod(times, rms_norm, adapted_gain_frames, title="Energy vs Applied Gain"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, rms_norm, label="RMS normalized")
    ax.plot(times, adapted_gain_frames[:len(times)], label="Gain curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def modify_dynamics_gui(
    audio_signal,
    sr,
    user_gain_db=0.0,
    dynamic_strength=1.0,
    brightness_strength=0.25,
    compression_amount=0.25,
):
    """
    Function for GUI to call.
    Applies the full dynamics modification pipeline to a pre-loaded audio signal.

    Returns a dict with:
        output_audio – peak-normalised modified signal (NumPy float32 array)
        times – time axis in seconds (one value per frame)
        rms_norm – robust z-scored RMS loudness per frame
        adapted_gain_frames – final per-frame gain curve applied to the signal

    Parameters mirror the CLI arguments:
        user_gain_db – overall gain in dB (−24 to +24)
        dynamic_strength – RMS-driven gain emphasis (0.0 to 3.0)
        brightness_strength – spectral centroid moderation (0.0 to 1.0)
        compression_amount – pulls gain curve toward 1.0 (0.0 to 1.0)
    """
    import librosa
    import numpy as np
    from .dynamic_extract import (
        compute_rms,
        compute_spectral_centroid,
        align_features,
        smooth_and_normalize,
    )
 
    # PI1: validate
    user_gain_db, dynamic_strength, brightness_strength, compression_amount = (
        parameter_validation(
            user_gain_db, dynamic_strength, brightness_strength, compression_amount
        )
    )
 
    n_fft = 2048
    hop_length = 512
 
    # Extract features
    rms, rms_db = compute_rms(audio_signal, frame_length=n_fft, hop_length=hop_length)
    centroid = compute_spectral_centroid(
        audio_signal, sample_rate=sr, n_fft=n_fft, hop_length=hop_length
    )
    times, rms_db_a, centroid_a = align_features(
        rms_db, centroid, sample_rate=sr, hop_length=hop_length
    )
 
    # Gate quiet frames
    rms_linear_a = librosa.db_to_amplitude(rms_db_a, ref=1.0)
    quiet = rms_linear_a < 1e-3
    centroid_a = centroid_a.copy()
    centroid_a[quiet] = np.nan
 
    rms_db_s, centroid_s, rms_norm, centroid_norm = smooth_and_normalize(
        rms_db_a, centroid_a, median_kernel=5, gaussian_sigma=1.0
    )
 
    # PI2: pure RMS-driven gain curve (user_gain_db applied after normalization)
    gain_curve_frames = apply_scaling(
        audio_signal, rms_norm, hop_length,
        dynamic_strength=dynamic_strength
    )
 
    # PI3: dynamic range adjustment
    adjusted_gain_frames = implement_adjustment(
        gain_curve_frames, compression_amount=compression_amount
    )
 
    # PI5: brightness-aware scaling
    adapted_gain_frames = context_aware_scaling(
        adjusted_gain_frames, centroid_norm, brightness_strength=brightness_strength
    )
 
    adapted_gain_samples = frames_to_samples(
        adapted_gain_frames, len(audio_signal), hop_length
    )
 
    output_audio = audio_signal * adapted_gain_samples
 
    # PI4: normalize to consistent peak, then apply user gain last so it's preserved
    output_audio = normalize_output(output_audio, peak=0.98)
    output_audio = apply_user_gain(output_audio, user_gain_db)
 
    return {
        "output_audio": output_audio,
        "times": times,
        "rms_norm": rms_norm,
        "adapted_gain_frames": adapted_gain_frames,
    }

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 6:
         print("Usage:")
         print("   python dynamic_modify.py <input_audio_file> [user_gain_db] [dynamic_strength] [brightness_strength] [compression_amount]")
         sys.exit(1)

    filepath = sys.argv[1]
    user_gain_db = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.0
    dynamic_strength = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
    brightness_strength = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.25
    compression_amount = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.25

    user_gain_db, dynamic_strength, brightness_strength, compression_amount = parameter_validation(
        user_gain_db=user_gain_db,
        dynamic_strength=dynamic_strength,
        brightness_strength=brightness_strength,
        compression_amount=compression_amount
    )

    print("About to load audio...")
    audio_signal, sr = load_audio(filepath)
    print("Finished librosa.load")
    
    n_fft = 2048
    hop_length = 512

    rms, rms_db = compute_rms(audio_signal, frame_length=n_fft, hop_length=hop_length)
    centroid = compute_spectral_centroid(audio_signal, sample_rate=sr, n_fft=n_fft, hop_length=hop_length)
    times, rms_db_a, centroid_a = align_features(rms_db, centroid, sample_rate=sr, hop_length=hop_length)
    
    rms_linear_a = librosa.db_to_amplitude(rms_db_a, ref=1.0)
    quiet = rms_linear_a < 1e-3
    centroid_a = centroid_a.copy()
    centroid_a[quiet] = np.nan

    rms_db_s, centroid_s, rms_norm, centroid_norm = smooth_and_normalize(
        rms_db_a, 
        centroid_a,
        median_kernel=5,
        gaussian_sigma=1.0
    )

    # PI2: RMS-driven dynamic gain curve (no overall_gain here)
    gain_curve_frames = apply_scaling(
        audio_signal,
        rms_norm,
        hop_length,
        dynamic_strength=dynamic_strength
    )

    # PI3: dynamic range adjustment
    adjusted_gain_frames = implement_adjustment(
        gain_curve_frames,
        compression_amount=compression_amount
    )

    # PI5: brightness-aware scaling
    adapted_gain_frames = context_aware_scaling(
        adjusted_gain_frames,
        centroid_norm,
        brightness_strength=brightness_strength
    )

    adapted_gain_samples = frames_to_samples(
        adapted_gain_frames, 
        len(audio_signal), 
        hop_length
    )

    output_audio = audio_signal * adapted_gain_samples
    
    # PI4: normalize to consistent peak level
    output_audio = normalize_output(output_audio, peak=0.98)

    # Apply user gain after normalization so the level change is preserved in the output
    output_audio = apply_user_gain(output_audio, user_gain_db)
    
    output_path = filepath.rsplit(".", 1)[0] + "_modified.wav"
    sf.write(output_path, output_audio, sr)

    print(f"Saved modified file to: {output_path}")
    print("Gain statistics:")
    print(f"Mean gain: {np.mean(adapted_gain_frames):.3f}")
    print(f"Min gain: {np.min(adapted_gain_frames):.3f}")
    print(f"Max gain: {np.max(adapted_gain_frames):.3f}")
    print("Output peak amplitude:", np.max(np.abs(output_audio)))

    orig_rms = np.sqrt(np.mean(audio_signal**2))
    new_rms = np.sqrt(np.mean(output_audio**2))
    print(f"Original RMS energy: {orig_rms:.4f}")
    print(f"Modified RMS energy: {new_rms:.4f}")

    corr = np.corrcoef(centroid_norm[:len(adapted_gain_frames)], adapted_gain_frames)[0,1]
    print("Correlation between brightness and gain:", corr)

    fig = plot_dynamics_mod(times, rms_norm, adapted_gain_frames)
    plt.show()

if __name__ == "__main__":
    main()