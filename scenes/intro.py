from manim import *  # or: from manimlib import *

from manim_slides import Slide

DEFAULT_FONT_SIZE = 22
SMALL_FONT_SIZE = 18
TITLE_FONT_SIZE = 42
SUBTITLE_FONT_SIZE = 26
TEXT_FONT = "CMU Sans Serif"
Text.set_default(color=BLACK, font=TEXT_FONT, font_size=DEFAULT_FONT_SIZE)

template = TexTemplate()
template.add_to_preamble(r"\usepackage{cmbright}")

Tex.set_default(color=BLACK, font_size=DEFAULT_FONT_SIZE)
Tex.set_default(tex_template=template)


class Intro(Slide):
    def construct(self):
        self.camera.background_color = WHITE

        logo_creatis = SVGMobject(
            "./assets/img/logo/creatis_quadri_logo.svg", height=0.4
        ).to_corner(UL)

        logo_udl = (
            SVGMobject("./assets/img/logo/Logo_Université_de_Lyon.svg", height=0.5)
            .to_corner(UR)
            .align_to(logo_creatis, DOWN)
        )
        upper_border = VGroup(logo_creatis, logo_udl)

        # self.play(FadeIn(logos, text_column, affiliations))
        self.add(upper_border)
        self.wait()

        # self.play(dot.animate.move_to(ORIGIN))
