# MIRACLE
MIRACLE is a desktop GUI for exploring and modifying core musical features of audio files, including tempo, pitch, and dynamics, that was built using PyQt6, librosa, and matplotlib.

Most MIR (Music Information Retrieval) tools require programming, signal processing, and music theory knowledge to use. MIRACLE wraps standard MIR techniques (such as beat tracking, pitch estimation via YIN, RMS/spectral-centroid dynamics analysis, and pitch-shifting) in a user-friendly GUI, so that anyone can upload a track, analyze it, hear and see the effect of the modifications, and export the result, all without touching code.

This project was originally built as a course project for CSC 475 (Music Information Retrieval) by Jonathan Kiss, Abby Hunter, and Makenna Clarke. This fork/repo covers the tempo, pitch, dynamics, and PyQt6 GUI modules.

<img width="746" height="772" alt="Screenshot 2026-10-08 105150" src="https://github.com/user-attachments/assets/b8e1dff0-962f-432c-aa08-bb270b99b205" />

## Features
Tempo: estimates BPM and beat positions from spectral flux/onset strength, refined via smoothing and peak-picking; time-stretch the track with a phase vocoder, preserving pitch.   

Pitch: extracts the fundamental frequency (f0) contour with the YIN algorithm, smoothed with a Savitzky–Golay filter; shift pitch up or down by semitones via phase-vocoder pitch-shifting, independent of tempo.    

Dynamics: computes short-time RMS loudness (dB) and spectral centroid (perceived brightness) curves; apply gain, RMS-adaptive dynamic scaling, brightness-aware modulation, and compression, then peak-normalizes to avoid clipping.  

Graph windows: every analysis and modification opens in its own resizable plot window with a live playback-position indicator, Play/Stop controls, and a Save Graph button (PNG/PDF/SVG export).    

Apply All: chains any combination of the three modifications (Tempo → Pitch → Dynamics) in one pass.    

Playback: play the original or modified audio directly from the app.    

Export: save the modified audio as a .wav file.    

<img width="864" height="574" alt="Screenshot 2026-10-08 105240" src="https://github.com/user-attachments/assets/dc3889fc-a468-4efc-a0ab-383f56cb78c1" />

## Installation
git clone https://github.com/jonathank256/MIRACLE.git    
cd MIRACLE    
python -m venv .venv    
.venv\Scripts\Activate.ps1    

pip install -r requirements.txt    

This project was developed and tested on Python 3.14.    

## Usage    
python mir_gui.py    

1. Click Load Audio and select a .wav or .mp3 file.    
2. Click Analyze under any section (Tempo / Pitch / Dynamics) to extract that feature.    
3. Click View Graph to inspect the analysis, or adjust the parameters and click Modify to apply a change.    
4. Click Mod Graph to compare original vs. modified.    
5. Tick Enable on any sections you want included, then click Apply All to chain them together.    
6. Use Play Original / Play Modified / Stop to preview, and Save Modified Audio to export.    

## Requirements
See requirements.txt. Core dependencies: PyQt6, librosa, soundfile, sounddevice, matplotlib, numpy, numba.
