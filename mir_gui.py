# Import functions from other scripts here 
import Dynamics.dynamic_extract as dynamic_extract
import Dynamics.dynamic_mod as dynamic_mod

import Pitch.pitch_extract as pitch_extract
import Pitch.pitch_mod as pitch_mod

import Tempo.tempo_extract as tempo_extract
import Tempo.tempo_mod as tempo_mod

# Other libraries 
import librosa
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time

# Matplotlib — use the Qt-compatible Agg canvas for embedding in QDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# Import other libraries 
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QDoubleSpinBox,
    QGroupBox, QSlider, QCheckBox, QDialog
)
from PyQt6.QtCore import Qt, QSize, QTimer


# ---------------------------------------------------------------------------
# PlotWindow
# ---------------------------------------------------------------------------

class PlotWindow(QDialog):
    """
    Embeds a matplotlib Figure inside a resizable dialog window.

    A red vertical line tracks the current playback position across every axes in the figure, updating at roughly 25 fps via a QTimer. The line
    is hidden whenever nothing is playing.

    Play/Stop buttons let the user start and stop the audio that the graph corresponds to without switching back to the main window.  Playback is
    routed through MiracleGUI._start_playback so position tracking and the anti-pop fade logic are applied consistently regardless of which window
    triggered the playback.

    A Save button exports the figure to PNG, PDF, or SVG.
    """

    def __init__(self, figure, title="Analysis", parent=None,
                 get_playback_pos=None, play_fn=None, stop_fn=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.figure = figure
        self.get_playback_pos = get_playback_pos
        self._vlines = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Embed the figure
        self.canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(self.canvas)

        # Button row: Play / Stop on the left, Save on the right
        btn_row = QHBoxLayout()

        if play_fn is not None:
            play_btn = QPushButton("Play")
            play_btn.clicked.connect(play_fn)
            btn_row.addWidget(play_btn)

        if stop_fn is not None:
            stop_btn = QPushButton("Stop")
            stop_btn.clicked.connect(stop_fn)
            btn_row.addWidget(stop_btn)

        btn_row.addStretch()

        save_btn = QPushButton("Save Graph")
        save_btn.clicked.connect(self._save_graph)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

        self.resize(860, 540)

        # Add a hidden vertical line to every axes for playback position
        for ax in figure.axes:
            vl = ax.axvline(x=0, color="red", linewidth=1.5, alpha=0.75, visible=False)
            self._vlines.append(vl)

        # QTimer refreshes the line position (~25 fps)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._update_playback_line)
        if get_playback_pos is not None:
            self._timer.start()

    def _update_playback_line(self):
        if self.get_playback_pos is None:
            return
        pos = self.get_playback_pos()
        visible = pos is not None
        for vl in self._vlines:
            if visible:
                vl.set_xdata([pos, pos])
            vl.set_visible(visible)
        self.canvas.draw_idle()

    def _save_graph(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)"
        )
        if path:
            self.figure.savefig(path, dpi=150, bbox_inches="tight")

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# MiracleGUI
# ---------------------------------------------------------------------------

class MiracleGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MIRACLE - Interactive Music Feature Tool")

        self.audio_path = ""
        self.audio_signal = None
        self.sample_rate = None
        self.processed_audio = None

        # Playback position tracking — set when sd.play() is called
        self._play_start = None # time.perf_counter() value at play start
        self._play_duration = 0.0  # duration in seconds of the audio handed to sd.play()

        # Keeps live PlotWindow references so they aren't garbage-collected
        self._plot_windows = []

        self.init_ui()

        self.setFixedSize(QSize(740, 740))

    def init_ui(self):
        """
        Builds the interface. Creates and arranges all widgets: file section, analysis/modification controls for tempo, pitch, and dynamics,
        playback controls, action buttons, and status bar.
        """
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # --------- File Section ---------
        file_group = QGroupBox("File")
        file_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Audio")
        self.load_btn.clicked.connect(self.load_audio)

        self.file_label = QLabel("No file loaded")

        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_label)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # --------- Tempo Section ---------
        tempo_group = QGroupBox("Tempo")
        tempo_layout = QHBoxLayout()

        self.tempo_enabled = QCheckBox("Enable")
        self.tempo_enabled.setToolTip("Include tempo modification in Apply All")

        self.tempo_analyze_btn = QPushButton("Analyze")
        self.tempo_analyze_btn.clicked.connect(self.analyze_tempo_clicked)

        self.tempo_graph_btn = QPushButton("View Graph")
        self.tempo_graph_btn.clicked.connect(self.view_tempo_graph)
        self.tempo_graph_btn.setToolTip("View onset envelope and beat positions (run Analyze first)")

        tempo_factor_label = QLabel("Factor:")
        self.tempo_factor_spin = QDoubleSpinBox()
        self.tempo_factor_spin.setRange(0.5, 2.0)
        self.tempo_factor_spin.setSingleStep(0.05)
        self.tempo_factor_spin.setValue(1.0)
        self.tempo_factor_spin.setToolTip("Speed multiplier: 0.5× (half speed) to 2.0× (double speed)")

        self.tempo_modify_btn = QPushButton("Modify")
        self.tempo_modify_btn.clicked.connect(self.modify_tempo_clicked)

        self.tempo_mod_graph_btn = QPushButton("Mod Graph")
        self.tempo_mod_graph_btn.clicked.connect(self.view_tempo_mod_graph)
        self.tempo_mod_graph_btn.setToolTip("View original vs modified waveform (run Modify first)")

        tempo_layout.addWidget(self.tempo_enabled)
        tempo_layout.addWidget(self.tempo_analyze_btn)
        tempo_layout.addWidget(self.tempo_graph_btn)
        tempo_layout.addStretch()
        tempo_layout.addWidget(tempo_factor_label)
        tempo_layout.addWidget(self.tempo_factor_spin)
        tempo_layout.addWidget(self.tempo_modify_btn)
        tempo_layout.addWidget(self.tempo_mod_graph_btn)

        tempo_group.setLayout(tempo_layout)
        main_layout.addWidget(tempo_group)

        # --------- Pitch Section ---------
        pitch_group = QGroupBox("Pitch")
        pitch_layout = QHBoxLayout()

        self.pitch_enabled = QCheckBox("Enable")
        self.pitch_enabled.setToolTip("Include pitch modification in Apply All")

        self.pitch_analyze_btn = QPushButton("Analyze")
        self.pitch_analyze_btn.clicked.connect(self.analyze_pitch_clicked)

        self.pitch_graph_btn = QPushButton("View Graph")
        self.pitch_graph_btn.clicked.connect(self.view_pitch_graph)
        self.pitch_graph_btn.setToolTip("View pitch contour (run Analyze first)")

        semitones_label = QLabel("Semitones:")
        self.semitones_spin = QDoubleSpinBox()
        self.semitones_spin.setRange(-12.0, 12.0)
        self.semitones_spin.setSingleStep(0.5)
        self.semitones_spin.setValue(0.0)
        self.semitones_spin.setToolTip("Shift pitch up (+) or down (−) in semitones")

        self.pitch_modify_btn = QPushButton("Modify")
        self.pitch_modify_btn.clicked.connect(self.modify_pitch_clicked)

        self.pitch_mod_graph_btn = QPushButton("Mod Graph")
        self.pitch_mod_graph_btn.clicked.connect(self.view_pitch_mod_graph)
        self.pitch_mod_graph_btn.setToolTip("View shifted pitch contour (run Modify first)")

        pitch_layout.addWidget(self.pitch_enabled)
        pitch_layout.addWidget(self.pitch_analyze_btn)
        pitch_layout.addWidget(self.pitch_graph_btn)
        pitch_layout.addStretch()
        pitch_layout.addWidget(semitones_label)
        pitch_layout.addWidget(self.semitones_spin)
        pitch_layout.addWidget(self.pitch_modify_btn)
        pitch_layout.addWidget(self.pitch_mod_graph_btn)

        pitch_group.setLayout(pitch_layout)
        main_layout.addWidget(pitch_group)

        # --------- Dynamics Section ---------
        dynamics_group = QGroupBox("Dynamics")
        dynamics_layout = QVBoxLayout()

        # Row 1: enable checkbox + analyze button + view graph button
        dynamics_row1 = QHBoxLayout()

        self.dynamics_enabled = QCheckBox("Enable")
        self.dynamics_enabled.setToolTip("Include dynamics modification in Apply All")

        self.dynamics_analyze_btn = QPushButton("Analyze")
        self.dynamics_analyze_btn.clicked.connect(self.analyze_dynamics_clicked)

        self.dynamics_graph_btn = QPushButton("View Graph")
        self.dynamics_graph_btn.clicked.connect(self.view_dynamics_graph)
        self.dynamics_graph_btn.setToolTip("View RMS loudness and spectral brightness curves (run Analyze first)")

        dynamics_row1.addWidget(self.dynamics_enabled)
        dynamics_row1.addWidget(self.dynamics_analyze_btn)
        dynamics_row1.addWidget(self.dynamics_graph_btn)
        dynamics_row1.addStretch()
        dynamics_layout.addLayout(dynamics_row1)

        # Row 2: gain + dynamic strength
        dynamics_row2 = QHBoxLayout()

        gain_label = QLabel("Gain (dB):")
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(-24.0, 24.0)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setValue(0.0)
        self.gain_spin.setToolTip("Overall output gain in dB (−24 to +24)")

        dyn_strength_label = QLabel("Dyn. Strength:")
        self.dyn_strength_spin = QDoubleSpinBox()
        self.dyn_strength_spin.setRange(0.0, 3.0)
        self.dyn_strength_spin.setSingleStep(0.1)
        self.dyn_strength_spin.setValue(1.0)
        self.dyn_strength_spin.setToolTip("How strongly RMS deviations affect scaling (0.0–3.0)")

        dynamics_row2.addWidget(gain_label)
        dynamics_row2.addWidget(self.gain_spin)
        dynamics_row2.addSpacing(12)
        dynamics_row2.addWidget(dyn_strength_label)
        dynamics_row2.addWidget(self.dyn_strength_spin)
        dynamics_layout.addLayout(dynamics_row2)

        # Row 3: brightness + compression + modify button
        dynamics_row3 = QHBoxLayout()

        brightness_label = QLabel("Brightness:")
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0.0, 1.0)
        self.brightness_spin.setSingleStep(0.05)
        self.brightness_spin.setValue(0.25)
        self.brightness_spin.setToolTip("How strongly timbral brightness moderates gain (0.0–1.0)")

        compression_label = QLabel("Compression:")
        self.compression_spin = QDoubleSpinBox()
        self.compression_spin.setRange(0.0, 1.0)
        self.compression_spin.setSingleStep(0.05)
        self.compression_spin.setValue(0.25)
        self.compression_spin.setToolTip("Pulls gain curve toward 1.0, reducing aggressiveness (0.0–1.0)")

        self.dynamics_modify_btn = QPushButton("Modify")
        self.dynamics_modify_btn.clicked.connect(self.modify_dynamics_clicked)

        self.dynamics_mod_graph_btn = QPushButton("Mod Graph")
        self.dynamics_mod_graph_btn.clicked.connect(self.view_dynamics_mod_graph)
        self.dynamics_mod_graph_btn.setToolTip("View original vs modified waveform (run Modify first)")

        dynamics_row3.addWidget(brightness_label)
        dynamics_row3.addWidget(self.brightness_spin)
        dynamics_row3.addSpacing(12)
        dynamics_row3.addWidget(compression_label)
        dynamics_row3.addWidget(self.compression_spin)
        dynamics_row3.addStretch()
        dynamics_row3.addWidget(self.dynamics_modify_btn)
        dynamics_row3.addWidget(self.dynamics_mod_graph_btn)
        dynamics_layout.addLayout(dynamics_row3)

        dynamics_group.setLayout(dynamics_layout)
        main_layout.addWidget(dynamics_group)

        # --------- Output Section ---------
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()

        self.apply_all_btn = QPushButton("Apply All")
        self.apply_all_btn.clicked.connect(self.apply_all)
        self.apply_all_btn.setToolTip("Apply all enabled modifications in order: Tempo → Pitch → Dynamics")

        self.save_btn = QPushButton("Save Modified Audio")
        self.save_btn.clicked.connect(self.save_audio)

        output_layout.addWidget(self.apply_all_btn)
        output_layout.addStretch()
        output_layout.addWidget(self.save_btn)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # --------- Playback Section ---------
        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout()

        self.play_original_btn = QPushButton("Play Original")
        self.play_original_btn.clicked.connect(self.play_original)

        self.play_modified_btn = QPushButton("Play Modified")
        self.play_modified_btn.clicked.connect(self.play_modified)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)

        playback_layout.addWidget(self.play_original_btn)
        playback_layout.addWidget(self.play_modified_btn)
        playback_layout.addWidget(self.stop_btn)

        playback_group.setLayout(playback_layout)
        main_layout.addWidget(playback_group)

        # --------- Status ---------
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

    # --------- Analysis Wrappers ---------

    def analyze_tempo_clicked(self):
        self.analyze_audio("tempo")

    def analyze_pitch_clicked(self):
        self.analyze_audio("pitch")

    def analyze_dynamics_clicked(self):
        self.analyze_audio("dynamics")

    # --------- Modification Wrappers ---------

    def modify_tempo_clicked(self):
        self.modify_audio("tempo")

    def modify_pitch_clicked(self):
        self.modify_audio("pitch")

    def modify_dynamics_clicked(self):
        self.modify_audio("dynamics")

    # --------- Load Audio ---------

    def load_audio(self):
        """
        Opens a file dialog, stores the chosen path, loads audio with librosa, resets any previously processed audio, and updates the status label.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3)"
        )

        if not file_path:
            self.status_label.setText("No file selected")
            return

        self.audio_path = file_path
        self.file_label.setText(os.path.basename(file_path))
        self.processed_audio = None

        try:
            self.audio_signal, self.sample_rate = librosa.load(
                self.audio_path, sr=22050, mono=True
            )
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_label.setText(f"Error loading audio: {e}")
            self.audio_signal, self.sample_rate = None, None

    # --------- Analyze Audio ---------

    def analyze_audio(self, mode):
        """
        Runs feature extraction for the given mode ('tempo', 'pitch', 'dynamics', or 'all'). Stores results in instance variables and summarises them in the status label.
        """
        if self.audio_signal is None:
            self.status_label.setText("No audio loaded")
            return

        output_text = ""

        if mode in ("tempo", "all"):
            try:
                self.status_label.setText("Analyzing tempo...")
                QApplication.processEvents()

                self.tempo_results = tempo_extract.analyze_tempo_gui(
                    self.audio_signal, self.sample_rate
                )
                tempo_bpm = self.tempo_results["tempo_bpm"]
                num_beats = len(self.tempo_results["beat_times"])
                output_text += f"Tempo: {tempo_bpm:.2f} BPM | Beats: {num_beats}\n"

            except Exception as e:
                output_text += f"Tempo Error: {e}\n"

        if mode in ("pitch", "all"):
            try:
                self.status_label.setText("Analyzing pitch...")
                QApplication.processEvents()

                self.pitch_results = pitch_extract.analyze_pitch_gui(
                    self.audio_signal, self.sample_rate
                )
                f0_clean = self.pitch_results["f0_clean"]

                if len(f0_clean) == 0:
                    output_text += "Pitch: No pitch detected\n"
                else:
                    output_text += (
                        f"Pitch → Mean: {f0_clean.mean():.2f} Hz | "
                        f"Min: {f0_clean.min():.2f} Hz | "
                        f"Max: {f0_clean.max():.2f} Hz\n"
                    )

            except Exception as e:
                output_text += f"Pitch Error: {e}\n"

        if mode in ("dynamics", "all"):
            try:
                self.status_label.setText("Analyzing dynamics...")
                QApplication.processEvents()

                self.dynamics_results = dynamic_extract.analyze_dynamics_gui(
                    self.audio_signal, self.sample_rate
                )
                rms_db = self.dynamics_results["rms_db_smoothed"]
                centroid = self.dynamics_results["centroid_hz_smoothed"]

                dynamic_range = rms_db.max() - rms_db.min()
                mean_brightness = np.nanmean(centroid)

                output_text += (
                    f"Dynamics → Range: {dynamic_range:.2f} dB | "
                    f"Mean Brightness: {mean_brightness:.1f} Hz\n"
                )

            except Exception as e:
                output_text += f"Dynamics Error: {e}\n"

        self.status_label.setText(output_text.strip() or "Analysis complete")

    # --------- Modify Audio ---------

    def modify_audio(self, mode, audio_in=None):
        """
        Applies the selected modification to the given audio array (or the original if none is provided). Returns the modified audio so modifications can be
        chained via apply_all. Also stores the result in self.processed_audio.
        """
        if self.audio_signal is None:
            self.status_label.setText("No audio loaded")
            return None

        # Use provided audio (for chaining) or start fresh from original
        working_audio = audio_in if audio_in is not None else self.audio_signal.copy()

        try:
            if mode == "tempo":
                tempo_factor = self.tempo_factor_spin.value()
                self.status_label.setText(f"Modifying tempo (×{tempo_factor:.2f})...")
                QApplication.processEvents()

                result = tempo_mod.modify_tempo_gui(
                    working_audio, self.sample_rate, tempo_factor
                )
                self.tempo_mod_results = {
                    "original": working_audio,
                    "modified": result,
                    "factor": tempo_factor,
                    "sr": self.sample_rate,
                }
                if audio_in is None:
                    self.status_label.setText(
                        f"Tempo modified: ×{tempo_factor:.2f} — ready to save"
                    )

            elif mode == "pitch":
                semitones = self.semitones_spin.value()
                self.status_label.setText(f"Shifting pitch ({semitones:+.1f} semitones)...")
                QApplication.processEvents()

                results = pitch_mod.modify_pitch_gui(
                    working_audio, self.sample_rate, semitones
                )
                result = results["shifted_audio"]
                self.pitch_mod_results = {
                    "shifted_audio": result,
                    "f0_smooth": results["f0_smooth"],
                    "times": results["times"],
                    "semitones": semitones,
                    "sr": self.sample_rate,
                }
                if audio_in is None:
                    self.status_label.setText(
                        f"Pitch shifted {semitones:+.1f} semitones — ready to save"
                    )

            elif mode == "dynamics":
                user_gain_db = self.gain_spin.value()
                dynamic_strength = self.dyn_strength_spin.value()
                brightness_strength = self.brightness_spin.value()
                compression_amount = self.compression_spin.value()

                self.status_label.setText("Modifying dynamics...")
                QApplication.processEvents()

                mod_result = dynamic_mod.modify_dynamics_gui(
                    working_audio,
                    self.sample_rate,
                    user_gain_db=user_gain_db,
                    dynamic_strength=dynamic_strength,
                    brightness_strength=brightness_strength,
                    compression_amount=compression_amount
                )
                result = mod_result["output_audio"]
                self.dynamics_mod_results = {
                    "original": working_audio,
                    "modified": result,
                    "times": mod_result["times"],
                    "rms_norm": mod_result["rms_norm"],
                    "adapted_gain_frames": mod_result["adapted_gain_frames"],
                    "gain_db": user_gain_db,
                    "sr": self.sample_rate,
                }
                if audio_in is None:
                    self.status_label.setText(
                        f"Dynamics modified (gain {user_gain_db:+.1f} dB, "
                        f"strength {dynamic_strength:.2f}) — ready to save"
                    )

            else:
                return None

            self.processed_audio = result
            return result

        except ValueError as e:
            self.status_label.setText(f"Parameter error: {e}")
            return None
        except Exception as e:
            self.status_label.setText(f"Modification error: {e}")
            return None

    # --------- Apply All ---------

    def apply_all(self):
        """
        Applies all enabled modifications in order: Tempo, Pitch, Dynamics. Each step feeds its output into the next, so all three are chained on
        the same audio rather than applied independently to the original.
        """
        if self.audio_signal is None:
            self.status_label.setText("No audio loaded")
            return

        enabled = {
            "tempo":    self.tempo_enabled.isChecked(),
            "pitch":    self.pitch_enabled.isChecked(),
            "dynamics": self.dynamics_enabled.isChecked(),
        }

        if not any(enabled.values()):
            self.status_label.setText("No modifications enabled — tick at least one checkbox")
            return

        audio = self.audio_signal.copy()
        applied = []

        for mode in ("tempo", "pitch", "dynamics"):
            if enabled[mode]:
                self.status_label.setText(f"Applying {mode}...")
                QApplication.processEvents()
                audio = self.modify_audio(mode, audio_in=audio)
                if audio is None:
                    # modify_audio already set an error status
                    return
                applied.append(mode)

        self.processed_audio = audio
        self.status_label.setText(f"Applied: {', '.join(applied)} — ready to save or play")

    # --------- Playback ---------

    def get_playback_pos(self):
        """
        Returns the current playback position in seconds, or None when nothing is playing.  Used by PlotWindow to drive its position indicator line.
        """
        if self._play_start is None:
            return None
        elapsed = time.perf_counter() - self._play_start
        if elapsed >= self._play_duration:
            self._play_start = None
            return None
        return elapsed

    def _prepare_audio(self, audio, source_sr):
        """
        Resamples audio to the device's native sample rate, converts to float32, and applies a short 10ms linear fade-in and fade-out to
        prevent clicks at the start and end of playback.
        """
        target_sr = int(sd.query_devices(kind="output")["default_samplerate"])
        if target_sr != source_sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
        audio = audio.astype(np.float32)

        # 10ms fade to silence out any abrupt transients at the boundaries
        fade_len = min(int(target_sr * 0.01), len(audio) // 2)
        if fade_len > 0:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            audio[:fade_len]  *= ramp
            audio[-fade_len:] *= ramp[::-1]

        return audio, target_sr

    def _start_playback(self, audio, source_sr, label="Playing..."):
        """
        Shared playback entry point used by both the main window buttons and the Play buttons inside graph windows. Prepares the audio, records
        the start time for position tracking, and hands off to sounddevice. A larger blocksize reduces the chance of buffer-underrun pops.
        """
        prepared, sr = self._prepare_audio(audio, source_sr)
        self._play_duration = len(prepared) / sr
        self._play_start = time.perf_counter()
        sd.play(prepared, sr, blocksize=4096)
        self.status_label.setText(label)

    def play_original(self):
        if self.audio_signal is None:
            self.status_label.setText("No audio loaded")
            return
        self._start_playback(self.audio_signal, self.sample_rate, "Playing original...")

    def play_modified(self):
        if self.processed_audio is None:
            self.status_label.setText("No modified audio — run a modification first")
            return
        self._start_playback(self.processed_audio, self.sample_rate, "Playing modified...")

    def stop_playback(self):
        sd.stop()
        self._play_start = None
        self.status_label.setText("Playback stopped")

    # --------- Graph Window Management ---------

    def _open_plot_window(self, figure, title, audio, sr):
        """
        Opens a new PlotWindow for the given figure.  Passes get_playback_pos for the position line and wires the window's Play/Stop buttons to
        _start_playback so the fade and buffer settings are applied consistently. Keeps a reference to the window so it isn't garbage-collected.
        """
        win = PlotWindow(
            figure,
            title=title,
            parent=self,
            get_playback_pos=self.get_playback_pos,
            play_fn=lambda: self._start_playback(audio, sr, f"Playing ({title})..."),
            stop_fn=self.stop_playback,
        )
        win.show()
        self._plot_windows.append(win)
        win.finished.connect(
            lambda _: self._plot_windows.remove(win)
            if win in self._plot_windows else None
        )

    # --------- Analysis Graph Views ---------

    def view_tempo_graph(self):
        if not hasattr(self, "tempo_results"):
            self.status_label.setText("Run Tempo Analyze first")
            return
        r = self.tempo_results
        fig = tempo_extract.plot_tempo_analysis(
            r["onset_env"], r["smoothed_env"],
            r["refined_beats_frames"], r["sample_rate"]
        )
        self._open_plot_window(fig, "Tempo Analysis", self.audio_signal, self.sample_rate)

    def view_pitch_graph(self):
        if not hasattr(self, "pitch_results"):
            self.status_label.setText("Run Pitch Analyze first")
            return
        r = self.pitch_results
        fig = pitch_extract.plot_pitch_analysis(r["f0"], r["f0_clean"], r["sample_rate"])
        self._open_plot_window(fig, "Pitch Analysis", self.audio_signal, self.sample_rate)

    def view_dynamics_graph(self):
        if not hasattr(self, "dynamics_results"):
            self.status_label.setText("Run Dynamics Analyze first")
            return
        r = self.dynamics_results
        fig = dynamic_extract.plot_dynamics_analysis(
            r["times"], r["rms_db"], r["rms_db_smoothed"],
            r["centroid_hz"], r["centroid_hz_smoothed"],
            r["rms_norm"], r["centroid_norm"]
        )
        self._open_plot_window(fig, "Dynamics Analysis", self.audio_signal, self.sample_rate)

    # --------- Modification Graph Views ---------

    def view_tempo_mod_graph(self):
        if not hasattr(self, "tempo_mod_results"):
            self.status_label.setText("Run Tempo Modify first")
            return
        r = self.tempo_mod_results
        fig = tempo_mod.plot_tempo_mod(
            r["original"], r["modified"], r["sr"],
            os.path.basename(self.audio_path), r["factor"]
        )
        self._open_plot_window(fig, "Tempo Modification", r["modified"], r["sr"])

    def view_pitch_mod_graph(self):
        if not hasattr(self, "pitch_mod_results"):
            self.status_label.setText("Run Pitch Modify first")
            return
        r = self.pitch_mod_results
        fig = pitch_mod.plot_pitch_mod(r["f0_smooth"], r["times"])
        self._open_plot_window(fig, "Pitch Modification", r["shifted_audio"], r["sr"])

    def view_dynamics_mod_graph(self):
        if not hasattr(self, "dynamics_mod_results"):
            self.status_label.setText("Run Dynamics Modify first")
            return
        r = self.dynamics_mod_results
        fig = dynamic_mod.plot_dynamics_mod(r["times"], r["rms_norm"], r["adapted_gain_frames"])
        self._open_plot_window(fig, "Dynamics Modification", r["modified"], r["sr"])

    # --------- Save Audio ---------

    def save_audio(self):
        """
        Writes self.processed_audio to disk at a user-chosen path.
        """
        if self.processed_audio is None:
            self.status_label.setText("No modified audio to save — run a modification first")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Modified Audio", "", "WAV Files (*.wav)"
        )

        if not save_path:
            self.status_label.setText("Save cancelled")
            return

        try:
            sf.write(save_path, self.processed_audio, self.sample_rate)
            self.status_label.setText(f"Saved: {os.path.basename(save_path)}")
        except Exception as e:
            self.status_label.setText(f"Error saving file: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MiracleGUI()
    window.show()
    sys.exit(app.exec())