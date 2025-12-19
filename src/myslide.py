from manim import *
from .common import *
from manim_slides import Slide


class MySlide(Slide):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 1

    def update_slide_number(self):
        self.counter += 1
        old_slide_number = self.canvas["slide_number"]
        new_slide_number = Text(f"{self.counter}").move_to(old_slide_number)
        self.play(Transform(old_slide_number, new_slide_number))
