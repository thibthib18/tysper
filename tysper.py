#!/usr/bin/env python3
"""tysper — voice-to-keyboard daemon.

Run in foreground:  python tysper.py
Toggle recording:   kill -USR1 $(cat /tmp/tysper.pid)
"""

import io
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import wave
from enum import Enum, auto
from pathlib import Path

import gi
import httpx
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
MAX_RECORDING_SECS = 60

# Transcription backend: "openai" (Whisper API) or "local" (whisper.cpp server).
# Can also be switched at runtime from the tray menu.
BACKEND = os.environ.get("TYSPER_BACKEND", "openai")

# Spoken language, applied to both backends. Defaults to English: auto-detection
# mistakes an accent or filler sounds ("euh") for French and then transcribes the
# whole clip as French. Set to "auto" to detect, or any ISO-639-1 code ("de", "fr").
LANGUAGE = os.environ.get("TYSPER_LANGUAGE", "en")

# whisper.cpp — tysper spawns whisper-server once so the model stays loaded
WHISPER_CPP_DIR = Path(
    os.environ.get("WHISPER_CPP_DIR", "~/Projects/misc/whisper.cpp")
).expanduser()
WHISPER_CPP_MODEL = os.environ.get("WHISPER_CPP_MODEL", "models/ggml-base.en.bin")
WHISPER_CPP_PORT = int(os.environ.get("WHISPER_CPP_PORT", "8178"))
WHISPER_CPP_THREADS = int(
    os.environ.get("WHISPER_CPP_THREADS", str(min(8, os.cpu_count() or 4)))
)
WHISPER_CPP_STARTUP_TIMEOUT = 60
# Extra whisper-server flags, e.g. "--beam-size 5 --no-speech-thold 0.4"
WHISPER_CPP_ARGS = shlex.split(os.environ.get("WHISPER_CPP_ARGS", ""))

# Initial prompt — a vocabulary hint (names, jargon) for both backends. Whisper
# only looks at the last ~224 tokens of it, so keep it to a line or two.
PROMPT_FILE = Path(
    os.environ.get("TYSPER_PROMPT_FILE", "~/.config/tysper/prompt.txt")
).expanduser()


def transcription_prompt() -> str:
    """Re-read on every transcription, so editing the word list needs no restart."""
    env = os.environ.get("TYSPER_PROMPT")
    if env:
        return env.strip()
    try:
        return PROMPT_FILE.read_text().strip()
    except OSError:
        return ""


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
# Local whisper.cpp server
# ---------------------------------------------------------------------------
def _die_with_parent():
    """Ask the kernel to SIGTERM the child if tysper dies (no orphaned server)."""
    import ctypes

    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class WhisperCppServer:
    """Owns a whisper-server child process and talks to its /inference endpoint."""

    def __init__(self):
        self.log = logging.getLogger("tysper.whisper-cpp")
        self.url = f"http://127.0.0.1:{WHISPER_CPP_PORT}"
        self.proc: subprocess.Popen | None = None
        self.ready = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Spawn the server (idempotent) and wait for the model to load in the background."""
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return
            binary = WHISPER_CPP_DIR / "build/bin/whisper-server"
            model = WHISPER_CPP_DIR / WHISPER_CPP_MODEL
            for path in (binary, model):
                if not path.exists():
                    self.log.error("Not found: %s", path)
                    return
            self.ready.clear()
            self.proc = subprocess.Popen(
                [
                    str(binary),
                    "--model", str(model),
                    "--host", "127.0.0.1",
                    "--port", str(WHISPER_CPP_PORT),
                    "--threads", str(WHISPER_CPP_THREADS),
                    "--language", LANGUAGE,
                    *WHISPER_CPP_ARGS,
                ],
                cwd=WHISPER_CPP_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=_die_with_parent,
            )
            self.log.info("Starting whisper-server (PID %d, model %s)", self.proc.pid, model.name)
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _wait_ready(self):
        t0 = time.monotonic()
        while time.monotonic() - t0 < WHISPER_CPP_STARTUP_TIMEOUT:
            if self.proc.poll() is not None:
                self.log.error("whisper-server exited with code %d", self.proc.returncode)
                return
            try:
                httpx.get(self.url, timeout=1)
                self.ready.set()
                self.log.info("whisper-server ready in %.1fs", time.monotonic() - t0)
                return
            except httpx.TransportError:
                time.sleep(0.2)
        self.log.error("whisper-server not ready after %ds", WHISPER_CPP_STARTUP_TIMEOUT)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.ready.clear()

    def wait_ready(self):
        self.start()
        if not self.ready.wait(WHISPER_CPP_STARTUP_TIMEOUT):
            raise RuntimeError("whisper-server is not running")

    def transcribe(self, wav_bytes: bytes, prompt: str = "") -> str:
        self.wait_ready()
        response = httpx.post(
            f"{self.url}/inference",
            files={"file": ("recording.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json", "temperature": "0.0", "prompt": prompt},
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(body["error"])
        return body["text"]


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------
class Tysper:
    def __init__(self):
        self.state = State.IDLE
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.log = logging.getLogger("tysper")
        self._openai: OpenAI | None = None  # created lazily; local-only needs no key
        self.whisper_cpp = WhisperCppServer()
        self.backend = BACKEND if BACKEND in ("openai", "local") else "openai"
        if self.backend == "local":
            self.whisper_cpp.start()

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

        openai_item = Gtk.RadioMenuItem(label="Backend: OpenAI API")
        local_item = Gtk.RadioMenuItem(label="Backend: whisper.cpp (local)", group=openai_item)
        (local_item if self.backend == "local" else openai_item).set_active(True)
        openai_item.connect("toggled", self._on_backend_toggled, "openai")
        local_item.connect("toggled", self._on_backend_toggled, "local")
        menu.append(openai_item)
        menu.append(local_item)

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

    def _on_backend_toggled(self, item: Gtk.RadioMenuItem, backend: str):
        if not item.get_active():
            return
        self.backend = backend
        self.log.info("Backend switched to %s", backend)
        if backend == "local":
            self.whisper_cpp.start()  # warm up now, not on first transcription

    def _quit(self):
        self.log.info("Quit requested from menu")
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.whisper_cpp.stop()
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

        # Schedule auto-discard after MAX_RECORDING_SECS
        self._recording_timer = threading.Timer(
            MAX_RECORDING_SECS, self._auto_discard
        )
        self._recording_timer.daemon = True
        self._recording_timer.start()

    def _auto_discard(self):
        if self.state != State.RECORDING:
            return
        self.log.warning("⏰ Max recording duration reached (%ds) — discarding", MAX_RECORDING_SECS)
        self.stop_recording()
        self._set_state(State.IDLE)

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
            self._recording_timer.cancel()
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
        """Transcribe audio with the selected backend, return transcription text."""
        backend = self.backend
        self.log.info("🔄 Transcribing with %s...", backend)
        try:
            wav_bytes = self._audio_to_wav_bytes(audio)
            prompt = transcription_prompt()
            if backend == "local":
                self.whisper_cpp.wait_ready()  # keep model load out of the timing
            t0 = time.monotonic()
            if backend == "local":
                text = self.whisper_cpp.transcribe(wav_bytes, prompt)
            else:
                text = self._transcribe_openai(wav_bytes, prompt)
            self.log.info(
                "⏱️  %s took %.2fs for %.1fs of audio",
                backend, time.monotonic() - t0, len(audio) / SAMPLE_RATE,
            )
            text = text.strip()
            if not text:
                self.log.warning("Whisper returned empty transcription")
                return None
            return text
        except Exception as e:
            self.log.error("Transcription error (%s): %s", backend, e)
            return None

    def _transcribe_openai(self, wav_bytes: bytes, prompt: str = "") -> str:
        if self._openai is None:
            self._openai = OpenAI()  # reads OPENAI_API_KEY from env
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "recording.wav"  # OpenAI needs a filename with extension
        kwargs = {} if LANGUAGE == "auto" else {"language": LANGUAGE}
        response = self._openai.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            prompt=prompt,
            **kwargs,
        )
        return response.text


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
        tysper.whisper_cpp.stop()
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
