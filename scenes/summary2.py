from manim import *  # or: from manimlib import *
from manim_slides import Slide
import sys

sys.path.append(".")
from src.common import *


class Summary2(Slide):
    def __init__(self):
        super().__init__()
        self.counter = 9

    def update_slide_number(self):
        self.counter += 1
        old_slide_number = self.canvas["slide_number"]
        new_slide_number = Text(f"{self.counter}").move_to(old_slide_number)
        self.play(Transform(old_slide_number, new_slide_number))

    def construct(self):
        self.camera.background_color = WHITE
        slide_number = Text(str(self.counter)).to_corner(DR)
        self.add_to_canvas(slide_number=slide_number)
        self.add(slide_number)

        #  Fixed upper border with logos

        logo_creatis = SVGMobject(
            "./assets/img/logo/creatis_quadri_logo.svg", height=0.4
        ).to_corner(UL)

        logo_udl = (
            SVGMobject("./assets/img/logo/Logo_Université_de_Lyon.svg", height=0.5)
            .to_corner(UR)
            .align_to(logo_creatis, DOWN)
        )
        upper_border = VGroup(logo_creatis, logo_udl)

        self.add(upper_border)

        # Slide 1: What is an inverse problem

        section1 = Tex(
            r"What is an inverse problem?",
            font_size=SUBTITLE_FONTSIZE,
            color=BLUE_CREATIS_FADED,
        )
        section2 = Tex(
            r"First order optimization",
            font_size=SUBTITLE_FONTSIZE,
            color=BLUE_CREATIS,
        )
        section3 = Tex(
            r"Deep learning for inverse problems",
            font_size=SUBTITLE_FONTSIZE,
            color=BLUE_CREATIS_FADED,
        )

        section4 = Tex(
            r"Plug-and-Play for Poisson inverse problems",
            font_size=SUBTITLE_FONTSIZE,
            color=BLUE_CREATIS_FADED,
        )

        section5 = Tex(
            r"MLEM as a Bregman mirror descent",
            font_size=SUBTITLE_FONTSIZE,
            color=BLUE_CREATIS_FADED,
        )

        section6 = Tex(
            r"Conclusion and perspectives",
            font_size=SUBTITLE_FONTSIZE,
            color=BLUE_CREATIS_FADED,
        )

        sections = (
            VGroup(section1, section2, section3, section4, section5, section6)
            .arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
            .align_to(SUBTITLE_POSITION, LEFT)
            .align_to(SUBTITLE_POSITION, UP)
            .shift(DOWN * MED_SMALL_BUFF)
        )

        self.add(section1, section3, section4, section5, section6)

        self.next_slide()

        self.play(Write(section2), run_time=0.75)
