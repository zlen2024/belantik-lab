---
name: manim-explain-video
description: Fast Manim Community Edition explainer videos whose only job is to explain a concept visually. Use when the user says "buat video explain", "visualkan", "explain guna manim", "video untuk faham", or wants a quick visual walkthrough of an idea (spins, graphs, matrices, algorithms, architectures) and does NOT want a polished deliverable. No storyboard, no draft round, no HyperFrames, no voice-over. Renders once at 480p. NOT for polished or vertical productions (use manim-hyperframes-vertical) and NOT for API questions about Manim (use manimce-best-practices).
---

# Manim explain video

A throwaway-quality, single-pass visual explainer. Goal: the viewer understands one idea. Speed beats polish.

## Rules (the user's standing choices)

- Framework is **Manim CE** (`from manim import *`). Do not route through `/hyperframes` and do not run its intent interview.
- **No draft, no storyboard, no review rounds.** Plan in your head, write the script, render once, check frames, report.
- **Render at 480p only**: `-ql` (854x480, 15 fps). No 1080p unless the user asks.
- **Text captions only**, no voice-over, no music, unless asked.
- Language: Bahasa Melayu with English technical terms, matching the user's notes. Keep identifiers and math notation unchanged.
- Length: 45-90 s, 3-5 parts. One idea per part.

## Workflow

1. Read `manimce-best-practices/SKILL.md` only if you need an API detail; the template covers the common cases.
2. Copy `template.py` from this skill to `<project>/<topic>_video.py` and replace the `part_*` methods. Keep the helpers (`say`, `clear`, `spin`).
3. Render to a scratch/results dir so the project stays clean:
   ```bash
   python -m manim -ql --media_dir <results>/manim_media <topic>_video.py <SceneName>
   ```
4. Verify: `ffprobe` the duration, extract 4-5 frames with `ffmpeg -ss <t> -i <mp4> -frames:v 1`, and read them. Check for text/formula overlap and anything off-screen. Fix and re-render only if something is actually broken.
5. Copy the final `480p15/<Scene>.mp4` to `<project>/results/<name>_480p.mp4` and give the user the path.

## Visual grammar

- One caption line at the bottom (`say`), the title shrunk at the top, content in the middle band (y from -2.5 to 2.8).
- Colour carries meaning: teal = `+1`/OK, purple = `-1`/neutral state, amber = the key relation/formula, red = constraint violated.
- Build up objects with `LaggedStart` so the eye follows the order of the argument; end each part with `wait(2-3)` then `clear()`.
- Show 5-8 representative items plus a "..." marker rather than the full count; say the real count in text.
- Prefer Text for captions and MathTex for formulas (LaTeX is installed via TinyTeX). Avoid exotic Unicode in `Text` (use `MathTex(r"\vdots")` for dots); use plain `x` and `<=` in captions.

## Pitfalls seen before

- `DashedLine.point_from_proportion` raises (it is a group). Use `get_center()`.
- Formula placed at y > 2.6 collides with the shrunk title; keep it <= 2.3.
- A matching diagram that already exists as an SVG/widget is not reusable in Manim; redraw with shapes.
- Run from the script's directory or pass absolute paths; `--media_dir` keeps output out of the project root.
- Do not report a visual as verified unless you actually looked at its frame.

## Honesty

Label illustrative numbers and assumptions inside the video (for example "andaian: degree 16"). Do not present an unverified hardware spec as fact.
