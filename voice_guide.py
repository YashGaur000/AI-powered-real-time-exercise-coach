"""Non-blocking text-to-speech for coaching cues (Windows-friendly pyttsx3)."""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional


class VoiceGuide:
    def __init__(self, enabled: bool = True, min_repeat_interval_s: float = 3.5) -> None:
        self.enabled = enabled
        self._min_repeat_interval_s = min_repeat_interval_s
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._last_spoken: str = ""
        self._last_time: float = 0.0
        self._engine = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        if enabled:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 175)
            except Exception:
                self._engine = None
                self.enabled = False

        if self.enabled and self._engine is not None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while not self._stop.is_set():
            msg = self._q.get()
            if msg is None:
                break
            try:
                if self._engine is not None:
                    self._engine.say(msg)
                    self._engine.runAndWait()
            except Exception:
                pass

    def speak(self, text: str, force: bool = False) -> None:
        if not self.enabled or not text.strip():
            return
        now = time.monotonic()
        if not force and text == self._last_spoken and (now - self._last_time) < self._min_repeat_interval_s:
            return
        self._last_spoken = text
        self._last_time = now
        self._q.put(text)

    def speak_priority(self, text: str) -> None:
        self.speak(text, force=True)
