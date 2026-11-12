"""Template for a quick Manim explainer. Copy, rename, replace the part_* methods.

Render (480p, single pass):
    python -m manim -ql --media_dir results/manim_media <file>.py Explainer
"""

from manim import *

TEAL_C = "#1D9E75"    # +1 / ok
PURPLE_C = "#7F77DD"  # -1 / neutral
AMBER_C = "#EF9F27"   # key relation / formula
RED_C = "#E24B4A"     # constraint violated


def spin(sign, r=0.22):
    col = TEAL_C if sign > 0 else PURPLE_C
    return Circle(radius=r, color=col, stroke_width=2).set_fill(col, 0.35)


class Explainer(Scene):
    TITLE = "Tajuk: satu soalan yang dijawab video ini"

    def construct(self):
        self.cap = None
        title = Text(self.TITLE, font_size=34)
        self.play(Write(title), run_time=1.5)
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP, buff=0.3))
        self.keep = [title]

        self.part_one()
        self.part_two()
        self.part_summary()

    # ---- helpers -------------------------------------------------------
    def say(self, txt, color=GREY_A):
        """Replace the bottom caption (max ~65 chars)."""
        new = Text(txt, font_size=26, color=color)
        if new.width > 12.5:
            new.scale_to_fit_width(12.5)
        new.to_edge(DOWN, buff=0.5)
        if self.cap is None:
            self.play(FadeIn(new, shift=UP * 0.2))
        else:
            self.play(ReplacementTransform(self.cap, new))
        self.cap = new

    def clear(self):
        """Fade out everything except the title and caption."""
        keep = self.keep + ([self.cap] if self.cap is not None else [])
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone])

    # ---- parts (replace) ----------------------------------------------
    def part_one(self):
        row = VGroup(*[spin(s) for s in [1, -1, 1, 1, -1]]).arrange(RIGHT, buff=0.5)
        self.say("Bahagian 1: kenalkan objek")
        self.play(LaggedStart(*[FadeIn(c, scale=0.5) for c in row], lag_ratio=0.2))
        self.wait(2)
        self.clear()

    def part_two(self):
        eq = MathTex(r"E=-J\sum_i a_i b_i", font_size=40, color=AMBER_C).move_to([0, 0.5, 0])
        self.say("Bahagian 2: hubungan utama")
        self.play(Write(eq))
        self.wait(2.5)
        self.clear()

    def part_summary(self):
        points = [
            ("Poin 1", WHITE),
            ("Poin 2", WHITE),
            ("Catatan / had: nyatakan andaian di sini", AMBER_C),
        ]
        grp = VGroup(*[Text(t, font_size=28, color=c) for t, c in points]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, 0.2, 0])
        self.say("Ringkasan")
        for g in grp:
            self.play(FadeIn(g, shift=RIGHT * 0.3), run_time=0.8)
            self.wait(0.6)
        self.wait(3)
