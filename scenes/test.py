from manim import *
from manim_slides import Slide


class CanvasExample(Slide):
    def update_slide_number(self):
        self.counter += 1
        old_slide_number = self.canvas["slide_number"]
        new_slide_number = Text(f"{self.counter}").move_to(old_slide_number)
        self.play(Transform(old_slide_number, new_slide_number))

    def construct(self):
        title = Text("My Title").to_corner(UL)

        self.counter = 1
        slide_number = Text("1").to_corner(DL)

        self.add_to_canvas(title=title, slide_number=slide_number)

        self.play(FadeIn(title), FadeIn(slide_number))
        self.next_slide()

        circle = Circle(radius=2)
        dot = Dot()

        self.update_slide_number()
        self.play(Create(circle))
        self.play(MoveAlongPath(dot, circle))

        self.next_slide()
        self.update_slide_number()

        square = Square()

        self.wipe(self.mobjects_without_canvas, square)
        self.next_slide()

        self.update_slide_number()
        self.play(Transform(self.canvas["title"], Text("New Title").to_corner(UL)))
        self.next_slide()

        self.remove_from_canvas("title", "slide_number")
        self.wipe(self.mobjects_without_canvas, [])
