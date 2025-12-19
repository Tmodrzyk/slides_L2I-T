from manim import *
from manim_slides import Slide
import seaborn as sns
import numpy as np

rng = np.random.default_rng(0)
palette = sns.color_palette()

# ---- Shared typography / style constants ----
LARGE_FONTSIZE = 38
DEFAULT_FONTSIZE = 28
SMALL_FONTSIZE = 18
TITLE_FONTSIZE = 42
SUBTITLE_FONTSIZE = 38
TEXT_FONT = "CMU Sans Serif"
# ---- Colors ----
BLUE_CREATIS = "#2e9bddff"
BLUE_CREATIS_FADED = "#c4ced4ff"
BLUE_SEABORN = rgb_to_color(palette[0])
ORANGE_SEABORN = rgb_to_color(palette[1])
GREEN_SEABORN = rgb_to_color(palette[2])
RED_SEABORN = rgb_to_color(palette[3])
PURPLE_SEABORN = rgb_to_color(palette[4])
BROWN_SEABORN = rgb_to_color(palette[5])
PINK_SEABORN = rgb_to_color(palette[6])
GRAY_SEABORN = rgb_to_color(palette[7])
YELLOW_SEABORN = rgb_to_color(palette[8])
TEAL_SEABORN = rgb_to_color(palette[9])

# ---- Shared positions ----
SUBTITLE_POSITION = UP * 2.75 + LEFT * 6

# ---- Global defaults for Text ----
Text.set_default(
    color=BLACK,
    font=TEXT_FONT,
    font_size=DEFAULT_FONTSIZE,
)

# ---- Global defaults for Tex ----
template = TexTemplate()
template.add_to_preamble(r"\usepackage{cmbright}")

Mobject.set_default(
    color=BLACK,
)
MathTex.set_default(
    color=BLACK,
    font_size=DEFAULT_FONTSIZE,
    tex_template=template,
)
Tex.set_default(
    color=BLACK,
    font_size=DEFAULT_FONTSIZE,
    tex_template=template,
)
Axes.set_default(
    axis_config={
        "color": BLACK,
        "stroke_width": 2,
        "include_tip": True,
        "tip_width": 0.25,
        "tip_height": 0.25,
    },
)
ParametricFunction.set_default(
    stroke_width=3,
)
