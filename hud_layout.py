"""Shared HUD geometry: non-overlapping text zones and line wrapping."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def help_panel_width(frame_w: int) -> int:
    """Keep ~52% of width for the main view + left HUD; remainder for shortcuts."""
    reserved_left = int(frame_w * 0.52)
    cap = frame_w - reserved_left - 16
    return int(max(130, min(420, cap)))


def help_panel_left_x(frame_w: int) -> int:
    return frame_w - help_panel_width(frame_w) - 10


def content_right_bound(frame_w: int, gap: int = 14) -> int:
    """Rightmost x for left-column text (stays clear of the help panel)."""
    edge = help_panel_left_x(frame_w) - gap
    return max(48, edge)


def bottom_status_band_h() -> int:
    return 44


def line_height(scale: float, thickness: int, gap: int = 6) -> int:
    _, th = cv2.getTextSize("Testyg", FONT, scale, thickness)[0]
    return int(th + gap)


def wrap_text(
    text: str,
    max_width_px: int,
    scale: float,
    thickness: int,
) -> List[str]:
    text = text.strip()
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        (tw, _), _ = cv2.getTextSize(trial, FONT, scale, thickness)
        if tw <= max_width_px:
            current = trial
        else:
            if current:
                lines.append(current)
            (ww, _), _ = cv2.getTextSize(word, FONT, scale, thickness)
            if ww > max_width_px:
                chunk = ""
                for ch in word:
                    t2 = chunk + ch
                    (cw, _), _ = cv2.getTextSize(t2, FONT, scale, thickness)
                    if cw <= max_width_px:
                        chunk = t2
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                current = chunk
            else:
                current = word
    if current:
        lines.append(current)
    return lines


def draw_bottom_status_bar(
    frame_bgr,
    text: str,
    *,
    ok: bool = True,
    scale: float = 0.52,
    thickness: int = 2,
) -> None:
    h, w = frame_bgr.shape[:2]
    band = bottom_status_band_h()
    y0 = h - band
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (25, 28, 32), -1)
    cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0, frame_bgr)
    cv2.line(frame_bgr, (0, y0), (w, y0), (90, 90, 100), 1)

    color = (210, 255, 215) if ok else (200, 210, 255)
    max_x = content_right_bound(w) - 8
    lines = wrap_text(text, max_x - 12, scale, thickness)
    if not lines:
        lines = [text]
    lh = line_height(scale, thickness, gap=4)
    base_y = h - 10
    for i, line in enumerate(reversed(lines)):
        y = base_y - i * lh
        cv2.putText(frame_bgr, line, (12, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_help_panel(frame_bgr, lines: Sequence[str]) -> None:
    h, w = frame_bgr.shape[:2]
    panel_w = help_panel_width(w)
    x0 = help_panel_left_x(w)
    y0 = 10
    scale = 0.45
    thickness = 1
    lh = line_height(scale, thickness, gap=5)
    ph = 14 + len(lines) * lh
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 10, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    cv2.rectangle(frame_bgr, (x0, y0), (w - 10, y0 + ph), (100, 100, 100), 1)
    y = y0 + lh + 2
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (x0 + 8, y),
            FONT,
            scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )
        y += lh
