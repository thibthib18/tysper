#!/usr/bin/env python3
"""tysper — voice-to-keyboard daemon.

Run in foreground:  python tysper.py
Toggle recording:   kill -USR1 $(cat /tmp/tysper.pid)
"""

import signal
import sys
import os
import logging
import tempfile
import threading
from pathlib import Path
from enum import Enum, auto

import subprocess

import numpy as np
import sounddevice as sd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # 16kHz — ideal for Whisper
CHANNELS = 1
PIDFILE = Path("/tmp/tysper.pid")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
WHISPER_MODEL = "whisper-1"

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
        self.state = State.RECORDING
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
            self.state = State.PROCESSING
            audio = self.stop_recording()
            if audio is None:
                self.state = State.IDLE
                return

            text = self._transcribe(audio)
            if text:
                self.log.info("📝 Transcription: %s", text)
                self._type_text(text)

            self.state = State.IDLE

        elif self.state == State.PROCESSING:
            self.log.info("Still processing, ignoring toggle")

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio buffer to in-memory WAV bytes."""
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        return buf.read()

    def _type_text(self, text: str):
        """Inject text into the focused window via xdotool."""
        try:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", text],
                check=True,
            )
            self.log.info("⌨️  Typed %d characters", len(text))
        except subprocess.CalledProcessError as e:
            self.log.error("xdotool error: %s", e)
        except FileNotFoundError:
            self.log.error("xdotool not found — install with: sudo apt install xdotool")

    def _transcribe(self, audio: np.ndarray) -> str | None:
        """Send audio to Whisper API, return transcription text."""
        self.log.info("🔄 Sending to Whisper API...")
        try:
            wav_bytes = self._audio_to_wav_bytes(audio)
            import io
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

    # Register SIGUSR1 handler
    def on_sigusr1(signum, frame):
        # Run toggle on a thread to avoid blocking the signal handler
        threading.Thread(target=tysper.toggle, daemon=True).start()

    signal.signal(signal.SIGUSR1, on_sigusr1)

    # Keep the process alive
    try:
        log.info("Waiting for signals... (Ctrl+C to quit)")
        while True:
            signal.pause()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        if tysper.stream:
            tysper.stream.stop()
            tysper.stream.close()
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
