"""Optional local voice helpers for desktop demos.

The web app now uses browser-based voice input and output for deployment-ready
behavior. This module remains available only for local desktop experiments.
"""

from __future__ import annotations

import threading

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None


_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine

    if pyttsx3 is None:
        return None

    with _engine_lock:
        if _engine is None:
            _engine = pyttsx3.init()
        return _engine


def speak_locally(text: str) -> bool:
    """Speak text on the host machine if pyttsx3 is installed."""

    engine = _get_engine()
    if engine is None:
        return False

    def run() -> None:
        with _engine_lock:
            engine.stop()
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()
    return True
