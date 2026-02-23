#!/usr/bin/env python3
"""tysper — voice-to-keyboard daemon.

Run in foreground:  python tysper.py
Toggle recording:   kill -USR1 $(cat /tmp/tysper.pid)
"""

import io
import logging
import os
import signal
import subprocess
import threading
import wave
from enum import Enum, auto
from pathlib import Path

import gi
import numpy as np
import sounddevice as sd
from openai import OpenAI

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")
from gi.repository import AyatanaAppIndicator3 as AppIndicator3  # noqa: E402
from gi.repository import GLib, Gtk  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # 16kHz — ideal for Whisper
CHANNELS = 1
PIDFILE = Path("/tmp/tysper.pid")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
WHISPER_MODEL = "whisper-1"

# Icon names from the system icon theme (Yaru/Adwaita)
ICON_IDLE = "audio-input-microphone-symbolic"
ICON_RECORDING = "media-record-symbolic"
ICON_PROCESSING = "content-loading-symbolic"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------
class Tysper:
    def __init__(self):
        self.state = State.IDLE
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.log = logging.getLogger("tysper")

        # -- AppIndicator --------------------------------------------------
        self.indicator = AppIndicator3.Indicator.new(
            "tysper",
            ICON_IDLE,
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_title("tysper")

        # Menu (required by AppIndicator — at minimum needs one item)
        menu = Gtk.Menu()

        self.status_item = Gtk.MenuItem(label="Status: Idle")
        self.status_item.set_sensitive(False)
        menu.append(self.status_item)

        self.last_text_item = Gtk.MenuItem(label="Last: —")
        self.last_text_item.set_sensitive(False)
        menu.append(self.last_text_item)

        menu.append(Gtk.SeparatorMenuItem())

        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", lambda _: self._quit())
        menu.append(quit_item)

        menu.show_all()
        self.indicator.set_menu(menu)

    # -- UI updates (must run on GLib main thread) --------------------------

    def _set_state(self, state: State):
        self.state = state
        # Schedule icon/menu update on the main thread
        GLib.idle_add(self._update_indicator)

    def _update_indicator(self):
        if self.state == State.IDLE:
            self.indicator.set_icon_full(ICON_IDLE, "Idle")
            self.status_item.set_label("Status: Idle")
        elif self.state == State.RECORDING:
            self.indicator.set_icon_full(ICON_RECORDING, "Recording")
            self.status_item.set_label("Status: 🎙️ Recording...")
        elif self.state == State.PROCESSING:
            self.indicator.set_icon_full(ICON_PROCESSING, "Processing")
            self.status_item.set_label("Status: 🔄 Processing...")
        return False  # don't repeat

    def _quit(self):
        self.log.info("Quit requested from menu")
        if self.stream:
            self.stream.stop()
            self.stream.close()
        PIDFILE.unlink(missing_ok=True)
        Gtk.main_quit()

    # -- Audio recording ----------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            self.log.warning("Audio status: %s", status)
        self.audio_chunks.append(indata.copy())

    def start_recording(self):
        self.audio_chunks.clear()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=self._audio_callback,
        )
        self.stream.start()
        self._set_state(State.RECORDING)
        self.log.info("🎙️  Recording started")

    def stop_recording(self) -> np.ndarray | None:
        if self.stream is None:
            return None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.audio_chunks:
            self.log.warning("No audio captured")
            return None
        audio = np.concatenate(self.audio_chunks, axis=0)
        self.audio_chunks.clear()
        self.log.info("Recording stopped — captured %.1f seconds", len(audio) / SAMPLE_RATE)
        return audio

    # -- Toggle handler (called from signal) --------------------------------

    def toggle(self):
        if self.state == State.IDLE:
            self.start_recording()

        elif self.state == State.RECORDING:
            self._set_state(State.PROCESSING)
            audio = self.stop_recording()
            if audio is None:
                self._set_state(State.IDLE)
                return

            text = self._transcribe(audio)
            if text:
                self.log.info("📝 Transcription: %s", text)
                self._type_text(text)
                # Update last transcription in menu
                display = text if len(text) <= 50 else text[:47] + "..."
                GLib.idle_add(
                    lambda: self.last_text_item.set_label(f"Last: {display}") or False
                )

            self._set_state(State.IDLE)

        elif self.state == State.PROCESSING:
            self.log.info("Still processing, ignoring toggle")

    # -- Helpers ------------------------------------------------------------

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio buffer to in-memory WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        return buf.read()

    def _type_text(self, text: str):
        """Inject text into the focused window via clipboard paste (Wayland-safe)."""
        try:
            # Copy to Wayland clipboard
            subprocess.run(
                ["wl-copy", text],
                check=True,
            )
            # Simulate Ctrl+V
            subprocess.run(
                ["ydotool", "key", "ctrl+v"],
                check=True,
            )
            self.log.info("⌨️  Pasted %d characters", len(text))
        except subprocess.CalledProcessError as e:
            self.log.error("Text injection error: %s", e)
        except FileNotFoundError as e:
            self.log.error("Missing tool: %s — install wl-clipboard and ydotool", e)

    def _transcribe(self, audio: np.ndarray) -> str | None:
        """Send audio to Whisper API, return transcription text."""
        self.log.info("🔄 Sending to Whisper API...")
        try:
            wav_bytes = self._audio_to_wav_bytes(audio)
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "recording.wav"  # OpenAI needs a filename with extension

            response = self.client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
            )
            text = response.text.strip()
            if not text:
                self.log.warning("Whisper returned empty transcription")
                return None
            return text
        except Exception as e:
            self.log.error("Whisper API error: %s", e)
            return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    log = logging.getLogger("tysper")

    tysper = Tysper()

    # Write pidfile
    PIDFILE.write_text(str(os.getpid()))
    log.info("tysper daemon started (PID %d)", os.getpid())
    log.info("Toggle with: kill -USR1 %d", os.getpid())

    # Register SIGUSR1 via GLib so it integrates with the GTK main loop
    def on_sigusr1():
        threading.Thread(target=tysper.toggle, daemon=True).start()
        return True  # keep the signal handler active

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGUSR1, on_sigusr1)

    # Also handle SIGINT/SIGTERM for clean shutdown
    def on_shutdown():
        tysper._quit()
        return False

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_shutdown)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, on_shutdown)

    # Run the GTK main loop (replaces signal.pause())
    try:
        log.info("Waiting for signals... (Ctrl+C to quit)")
        Gtk.main()
    finally:
        if tysper.stream:
            tysper.stream.stop()
            tysper.stream.close()
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
