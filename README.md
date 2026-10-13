# tysper

Voice-to-keyboard for Linux. Press a hotkey, speak, press again — your words are transcribed via OpenAI's Whisper API and typed into whatever window is focused.

tysper runs as a background daemon with a system tray indicator showing its current state (idle, recording, processing).

## Requirements

- Ubuntu 24.04 (GNOME, Wayland)
- Python 3.12+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

### 1. System dependencies

```bash
sudo apt install -y \
    python3.12-venv \
    python3-gi \
    gir1.2-ayatanaappindicator3-0.1 \
    gnome-shell-extension-appindicator \
    wl-clipboard \
    ydotool
```

### 2. ydotool permissions

ydotool needs access to `/dev/uinput` to simulate keyboard input. Add yourself to the `input` group and create a udev rule so permissions persist across reboots:

```bash
sudo usermod -aG input $USER
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/80-uinput.rules
```

**Reboot** for the group change to take effect.

Verify after reboot:

```bash
groups | grep input
ls -la /dev/uinput   # should show group "input" with rw permissions
```

### 3. GNOME AppIndicator extension

Make sure the AppIndicator extension is enabled (needed for the tray icon):

```bash
gnome-extensions list --enabled | grep appindicator
```

If not listed:

```bash
gnome-extensions enable ubuntu-appindicators@ubuntu.com
```

### 4. Python environment

```bash
cd /path/to/tysper
python3 -m venv .venv --system-site-packages
.venv/bin/pip install sounddevice numpy openai
```

The `--system-site-packages` flag is required so the venv can access the system-installed `gi` (PyGObject) bindings.

### 5. systemd user service

Create `~/.config/systemd/user/tysper.service`:

```ini
[Unit]
Description=tysper — voice-to-keyboard daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=/path/to/tysper/.venv/bin/python3 /path/to/tysper/tysper.py
Environment=OPENAI_API_KEY=sk-your-key-here
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical-session.target
```

Replace `/path/to/tysper` with the actual path and set your OpenAI API key.

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable tysper
systemctl --user start tysper
```

### 6. Keyboard shortcut

Go to **Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts** and add:

| Field | Value |
|-------|-------|
| Name | tysper toggle |
| Command | `bash -c 'kill -USR1 $(cat /tmp/tysper.pid)'` |
| Shortcut | Your choice (e.g. F3) |

## Usage

Press your hotkey once to start recording — the tray icon changes to a record symbol. Speak, then press the hotkey again. tysper sends the audio to Whisper, and the transcribed text is pasted into whatever window has focus.

The tray icon shows the current state:

| Icon | State |
|------|-------|
| 🎤 Microphone | Idle — ready to record |
| 🔴 Record | Recording — speak now |
| ⏳ Loading | Processing — waiting for Whisper |

Right-click the tray icon to see the last transcription or quit the daemon.

## Running manually

For development or debugging, run in the foreground:

```bash
export OPENAI_API_KEY="sk-your-key-here"
.venv/bin/python3 tysper.py
```

Toggle from another terminal:

```bash
kill -USR1 $(cat /tmp/tysper.pid)
```

## Local transcription (whisper.cpp)

Instead of the OpenAI API, tysper can transcribe locally with [whisper.cpp](https://github.com/ggml-org/whisper.cpp). tysper starts `whisper-server` as a child process, so the model stays loaded between recordings, and stops it when tysper quits.

Build whisper.cpp and download a model:

```bash
cd ~/Projects/misc/whisper.cpp
cmake -B build && cmake --build build -j --config Release
./models/download-ggml-model.sh base.en
```

Pick the backend with `TYSPER_BACKEND=local` (the default is `openai`), or switch at any time from the tray menu. Only the OpenAI backend needs `OPENAI_API_KEY`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `TYSPER_BACKEND` | `openai` | `openai` or `local` |
| `WHISPER_CPP_DIR` | `~/Projects/misc/whisper.cpp` | whisper.cpp checkout (with `build/bin/whisper-server`) |
| `WHISPER_CPP_MODEL` | `models/ggml-base.en.bin` | Model path, relative to `WHISPER_CPP_DIR` |
| `TYSPER_LANGUAGE` | `en` | Spoken language, both backends: ISO-639-1 code, or `auto` to detect |
| `WHISPER_CPP_PORT` | `8178` | Port the local server listens on (127.0.0.1 only) |
| `WHISPER_CPP_THREADS` | min(8, CPU count) | Inference threads |

Each transcription logs how long it took, for either backend, so you can compare them:

```
⏱️  local took 0.51s for 5.0s of audio
```

`*.en` models are English-only. For other languages, use a multilingual model such as `ggml-small.bin` (`./models/download-ggml-model.sh small`).

## Logs

When running as a systemd service:

```bash
# Follow logs in real time
journalctl --user -u tysper -f

# View recent logs
journalctl --user -u tysper --since "5 minutes ago"
```

## How it works

tysper is a Python daemon that:

1. Listens for `SIGUSR1` signals to toggle recording
2. Records audio from the default microphone via `sounddevice` (16kHz mono)
3. Sends the recording to OpenAI's Whisper API for transcription
4. Copies the transcribed text to the Wayland clipboard (`wl-copy`)
5. Simulates Ctrl+V via `ydotool` to paste into the focused window
6. Shows status via an AppIndicator tray icon

## Known limitations

- **Overwrites clipboard**: The paste-based text injection replaces your current clipboard contents.
- **Wayland only**: Text injection uses `wl-copy` + `ydotool`, which are Wayland-specific. For X11 sessions, `xdotool type` would work but doesn't handle Unicode well.
- **Requires network** with the OpenAI backend. The local whisper.cpp backend works offline.
