from manim import *  # or: from manimlib import *
from manim_slides import Slide
import sys

sys.path.append(".")
from src.common import *


class Title(Slide):
    def __init__(self):
        super().__init__()
        self.counter = 1

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

        logo_creatis = SVGMobject(
            "./assets/img/logo/creatis_quadri_logo.svg", height=0.4
        ).to_corner(UL)

        logo_udl = (
            SVGMobject("./assets/img/logo/Logo_Université_de_Lyon.svg", height=0.5)
            .to_corner(UR)
            .align_to(logo_creatis, DOWN)
        )
        logo_insa = SVGMobject(
            "./assets/img/logo/Logo_INSALyon-pantone.svg", height=0.5
        )
        logo_cnrs = SVGMobject("./assets/img/logo/Logo_CNRS.svg", height=1)
        logo_inserm = SVGMobject("./assets/img/logo/Logo_Inserm.svg", height=0.75)
        logo_labex = SVGMobject("./assets/img/logo/LABEX_PRIMES.svg", height=0.5)
        logos_tutelles = (
            VGroup(logo_insa, logo_cnrs, logo_inserm, logo_labex)
            .arrange(DOWN, buff=MED_LARGE_BUFF)
            .next_to(logo_creatis, DOWN, buff=LARGE_BUFF)
            .to_edge(LEFT)
        )
        logos = VGroup(logo_creatis, logo_udl, logos_tutelles)

        title = Tex(
            r"Hybrid methods for computational imaging\\with applications to Poisson inverse problems",
            font_size=TITLE_FONTSIZE,
        )
        subtitle = Tex(
            "LP2I-Toulouse Seminar",
            font_size=SUBTITLE_FONTSIZE,
        )
        date = Tex(
            "20/01/2025",
            color=GREY,
        )
        author = Tex(
            r"Thibaut Modrzyk$^{1}$",
        )
        directors = VGroup(
            Tex(
                r"Ane Etxebeste$^{1}$",
            ),
            Tex(
                r"Elie Bretin$^{2}$",
            ),
            Tex(
                r"Voichita Maxim$^{1}$",
            ),
        ).arrange(RIGHT)

        author_block = VGroup(author, directors).arrange(DOWN, buff=MED_SMALL_BUFF)
        affiliations = VGroup(
            Tex(
                r"\mbox{$^{1}$ INSA-Lyon, Université Claude Bernard Lyon 1, CNRS, Inserm, \textbf{CREATIS}, UMR 5220}",
                font_size=SMALL_FONTSIZE,
            ),
            Tex(
                r"\mbox{$^{2}$ INSA-Lyon, Université Claude Bernard Lyon 1, CNRS, \textbf{Institut Camille Jordan}, UMR 5208}",
                font_size=SMALL_FONTSIZE,
            ),
        ).arrange(DOWN, buff=SMALL_BUFF, aligned_edge=LEFT)

        text_column = (
            VGroup(title, subtitle, date, author_block)
            .arrange(DOWN, buff=MED_LARGE_BUFF)
            .to_edge(RIGHT, buff=1.5 * LARGE_BUFF)
        )
        affiliations.next_to(text_column, DOWN, buff=LARGE_BUFF).to_edge(
            DOWN, buff=LARGE_BUFF
        )

        # self.play(FadeIn(logos, text_column, affiliations))
        self.add(logos, text_column, affiliations)
        self.wait(0.1)
        self.next_slide()
