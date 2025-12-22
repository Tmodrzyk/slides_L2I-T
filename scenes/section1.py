from manim import *  # or: from manimlib import *
from manim_slides import Slide
import sys

sys.path.append(".")
from src.common import *


class Section1(Slide):
    def __init__(self):
        super().__init__()
        self.counter = 3

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

        ################################################################################
        # Slide 1: What is an inverse problem
        ################################################################################
        self.next_section(skip_animations=False)
        self.next_slide()

        slide_title = (
            Tex(
                r"What is an inverse problem?",
                font_size=SUBTITLE_FONTSIZE,
                color=BLUE_CREATIS,
            )
            .align_to(SUBTITLE_POSITION, LEFT)
            .align_to(SUBTITLE_POSITION, UP)
        )

        self.add(slide_title)
        axes = Axes(
            x_range=[0, 2, 0.2],
            y_range=[0, 1, 0.2],
            x_length=8,
            y_length=4,
            tips=False,
        )

        x_true = lambda x: np.piecewise(
            x,
            [x < 0.5, (x >= 0.5) & (x < 1.0), (x >= 1.0) & (x < 1.5), x >= 1.5],
            [0.9, 0.2, 0.7, 0.1],
        )
        graph_true = axes.plot(
            x_true,
            use_smoothing=False,
            color=BLUE_SEABORN,
        )

        x_true_label = axes.get_graph_label(
            graph_true,
            label="x",
            x_val=1.2,
            color=BLUE_SEABORN,
            direction=UR,
            buff=MED_LARGE_BUFF,
        )
        x_true_label[0].font_size = SUBTITLE_FONTSIZE

        self.add(axes)
        self.play(LaggedStart(Create(graph_true), FadeIn(x_true_label), lag_ratio=0.5))

        self.next_slide()

        k_size = 100  # large support (# of points)
        sigma = 0.85  # very smooth
        xk = np.linspace(-2, 2, k_size)
        kernel = np.exp(-(xk**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        n_points = 400
        t = np.linspace(0, 2, n_points)
        x_vals = x_true(t)  # sampled piecewise-constant signal
        x_blurred = np.convolve(x_vals, kernel, mode="same")  # blurred Ax

        graph_blur = axes.plot_line_graph(
            x_values=t,
            y_values=x_blurred,
            add_vertex_dots=False,
            stroke_width=3,
            stroke_color=GREEN_SEABORN,
        )
        blur_label = axes.get_graph_label(
            graph_blur["line_graph"],
            label=r"Ax",
            x_val=1.7,
            color=GREEN_SEABORN,
            buff=MED_LARGE_BUFF,
            direction=UR,
        )
        blur_label[0].font_size = SUBTITLE_FONTSIZE

        self.play(LaggedStart(Create(graph_blur), FadeIn(blur_label), lag_ratio=0.5))

        self.next_slide()

        noise_std = 0.03
        noise = rng.normal(0.0, noise_std, size=x_blurred.shape)
        x_blurred_noisy = np.clip(x_blurred + noise, 0.0, 1.0)

        noisy_graph = axes.plot_line_graph(
            x_values=t,
            y_values=x_blurred_noisy,
            add_vertex_dots=False,
            stroke_width=3,
            stroke_color=ORANGE_SEABORN,
        )
        noisy_label = axes.get_graph_label(
            noisy_graph["line_graph"],
            label=r"Ax + \epsilon",
            x_val=0.6,
            color=ORANGE_SEABORN,
            direction=UR,
            buff=MED_LARGE_BUFF,
        )
        noisy_label[0].font_size = SUBTITLE_FONTSIZE

        self.play(LaggedStart(Create(noisy_graph), FadeIn(noisy_label), lag_ratio=0.5))

        self.next_slide()

        # Shrink and move the graph to the left
        graph_group = VGroup(
            axes,
            graph_true,
            x_true_label,
            graph_blur,
            blur_label,
            noisy_graph,
            noisy_label,
        )

        self.play(graph_group.animate.scale(0.6).to_edge(LEFT, buff=1))

        # Create descriptions
        gt_desc = Tex("Ground truth", font_size=DEFAULT_FONTSIZE, color=BLUE_SEABORN)
        operator_desc = Tex(
            "Forward operator", font_size=DEFAULT_FONTSIZE, color=GREEN_SEABORN
        )
        data_desc = Tex(
            "Measured data", font_size=DEFAULT_FONTSIZE, color=ORANGE_SEABORN
        )
        descs = (
            VGroup(gt_desc, operator_desc, data_desc)
            .arrange(RIGHT, buff=0.5)
            .to_edge(RIGHT, buff=2)
            .align_to(graph_group, UP)
        )

        # Create labels
        gt_label = MathTex("x", font_size=LARGE_FONTSIZE, color=BLUE_SEABORN).next_to(
            gt_desc, DOWN, buff=MED_SMALL_BUFF
        )
        operator_label = MathTex(
            "A", font_size=LARGE_FONTSIZE, color=GREEN_SEABORN
        ).next_to(operator_desc, DOWN, buff=MED_SMALL_BUFF)
        data_label = MathTex(
            r"y", font_size=LARGE_FONTSIZE, color=ORANGE_SEABORN
        ).next_to(data_desc, DOWN, buff=MED_SMALL_BUFF)

        labels = VGroup(gt_label, operator_label, data_label)

        # First row of images (CT example)
        img_width = 1.5
        gt_img = (
            ImageMobject("./assets/img/section1/slide1/ground_truth.png")
            .scale_to_fit_width(img_width)
            .next_to(gt_label, DOWN, buff=MED_SMALL_BUFF)
        )
        sino_img = (
            ImageMobject("./assets/img/section1/slide1/sinogram.png")
            .scale_to_fit_width(img_width)
            .next_to(data_label, DOWN, buff=MED_SMALL_BUFF)
        )
        scanner_img = (
            ImageMobject("./assets/img/section1/slide1/scanner.png")
            .scale_to_fit_width(img_width)
            .next_to(operator_label, DOWN, buff=MED_SMALL_BUFF)
        )
        row1_images = (
            Group(gt_img, scanner_img, sino_img)
            .arrange(RIGHT, buff=0.75)
            .next_to(labels, DOWN, buff=MED_LARGE_BUFF)
        )

        # Second row of images (Astronomy example)
        moon_img = (
            ImageMobject("./assets/img/section1/slide1/moon.png")
            .scale_to_fit_width(img_width)
            .next_to(gt_img, DOWN, buff=MED_SMALL_BUFF)
        )
        blurred_img = (
            (ImageMobject("./assets/img/section1/slide1/moon_blur.png"))
            .scale_to_fit_width(img_width)
            .next_to(sino_img, DOWN, buff=MED_SMALL_BUFF)
        )
        telescope_img = (
            ImageMobject("./assets/img/section1/slide1/telescope.png")
            .scale_to_fit_width(img_width)
            .next_to(scanner_img, DOWN, buff=MED_SMALL_BUFF)
        )
        row2_images = (
            Group(moon_img, telescope_img, blurred_img)
            .arrange(RIGHT, buff=0.75)
            .next_to(row1_images, DOWN, buff=MED_LARGE_BUFF)
        )

        right_side = (
            Group(descs, labels, row1_images, row2_images)
            .align_to(graph_group, UP)
            .shift(UP)
            .to_edge(RIGHT, buff=1)
        )

        # Animate first row
        self.play(FadeIn(labels, descs, row1_images))

        self.next_slide()

        # Animate second row
        self.play(FadeIn(row2_images))

        self.next_slide()
        self.play(FadeOut(graph_group, right_side))
        self.update_slide_number()

        ################################################################################
        # Slide 2: Convolution and its matrix
        ################################################################################
        self.next_section(skip_animations=False)

        image_values = [1, 2, 3, 4, 5]
        kernel_values = [1, 2, 1]

        # Create the image array visualization
        image_squares = VGroup()
        for i, val in enumerate(image_values):
            square = Square(side_length=0.8, color=BLUE_SEABORN, fill_opacity=0.3)
            text = Text(str(val), font_size=DEFAULT_FONTSIZE, color=BLACK)
            text.move_to(square)
            cell = VGroup(square, text)
            image_squares.add(cell)
        image_squares.arrange(RIGHT, buff=0)

        image_label = Tex(
            "Image $x$", font_size=LARGE_FONTSIZE, color=BLUE_SEABORN
        ).next_to(image_squares, UP, buff=MED_SMALL_BUFF)

        # Create the kernel array visualization
        kernel_squares = VGroup()
        for i, val in enumerate(kernel_values):
            square = Square(side_length=0.8, color=ORANGE_SEABORN, fill_opacity=0.5)
            text = Text(str(val), font_size=DEFAULT_FONTSIZE, color=BLACK)
            text.move_to(square)
            cell = VGroup(square, text)
            kernel_squares.add(cell)
        kernel_squares.arrange(RIGHT, buff=0)

        kernel_label = Tex(
            "Kernel $k$", font_size=LARGE_FONTSIZE, color=ORANGE_SEABORN
        ).next_to(kernel_squares, DOWN, buff=MED_SMALL_BUFF)

        image_group = VGroup(image_label, image_squares).to_edge(LEFT, buff=2)
        kernel_squares.move_to(image_squares[0:3]).shift(DOWN)
        kernel_label.next_to(kernel_squares, DOWN, buff=MED_SMALL_BUFF)
        kernel_group = VGroup(kernel_squares, kernel_label)

        toeplitz_data = [
            [1, 2, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 2, 1],
        ]

        matrix_squares = VGroup()
        for row_idx, row in enumerate(toeplitz_data):
            for col_idx, val in enumerate(row):
                fill_color = ORANGE_SEABORN if val != 0 else WHITE
                fill_opacity = 0.5 if val != 0 else 0.1
                square = Square(
                    side_length=0.8,
                    color=BLACK,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_width=1,
                )
                text = Text(str(val), font_size=DEFAULT_FONTSIZE, color=BLACK)
                text.move_to(square)
                cell = VGroup(square, text)
                cell.move_to([col_idx * 0.8, -row_idx * 0.8, 0])
                matrix_squares.add(cell)

        matrix_squares.move_to(
            VGroup(image_label, image_squares, kernel_label, kernel_squares)
        ).to_edge(RIGHT, buff=2)

        matrix_label = Tex(
            "Toeplitz Matrix $A$", font_size=LARGE_FONTSIZE, color=GREEN_SEABORN
        ).next_to(matrix_squares, UP, buff=MED_SMALL_BUFF)

        matrix_group = VGroup(matrix_label, matrix_squares)

        slide2_title = (
            Tex(
                r"What does $A$ look like?",
                font_size=SUBTITLE_FONTSIZE,
                color=BLUE_CREATIS,
            )
            .align_to(SUBTITLE_POSITION, LEFT)
            .align_to(SUBTITLE_POSITION, UP)
        )

        # Animate
        self.play(Transform(slide_title, slide2_title), FadeIn(image_group))
        self.next_slide()

        self.play(FadeIn(kernel_group))
        self.next_slide()

        self.play(FadeIn(matrix_group))
        self.next_slide()

        self.next_slide()

        # 2D Image (3x3)
        image_2d_values = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        image_2d_squares = VGroup()
        for row_idx, row in enumerate(image_2d_values):
            for col_idx, val in enumerate(row):
                square = Square(side_length=0.6, color=BLUE_SEABORN, fill_opacity=0.3)
                text = Text(str(val), font_size=DEFAULT_FONTSIZE * 0.8, color=BLACK)
                text.move_to(square)
                cell = VGroup(square, text)
                cell.move_to([col_idx * 0.6, -row_idx * 0.6, 0])
                image_2d_squares.add(cell)

        image_2d_label = Tex(
            r"Image $x$ (N$\times$N)", font_size=LARGE_FONTSIZE, color=BLUE_SEABORN
        ).next_to(image_2d_squares, UP, buff=MED_SMALL_BUFF)

        image_2d_group = VGroup(image_2d_label, image_2d_squares).move_to(image_group)

        # 2D Kernel (2x2)
        kernel_2d_values = [
            [1, 2],
            [3, 4],
        ]

        kernel_2d_squares = VGroup()
        for row_idx, row in enumerate(kernel_2d_values):
            for col_idx, val in enumerate(row):
                square = Square(side_length=0.6, color=ORANGE_SEABORN, fill_opacity=0.5)
                text = Text(str(val), font_size=DEFAULT_FONTSIZE * 0.8, color=BLACK)
                text.move_to(square)
                cell = VGroup(square, text)
                cell.move_to([col_idx * 0.6, -row_idx * 0.6, 0])
                kernel_2d_squares.add(cell)

        kernel_2d_label = Tex(
            r"Kernel $k$ (K$\times$K)", font_size=LARGE_FONTSIZE, color=ORANGE_SEABORN
        ).next_to(kernel_2d_squares, DOWN, buff=MED_SMALL_BUFF)

        kernel_2d_squares.next_to(image_2d_squares, DOWN, buff=LARGE_BUFF)
        kernel_2d_label.next_to(kernel_2d_squares, DOWN, buff=MED_SMALL_BUFF)
        kernel_2d_group = VGroup(kernel_2d_squares, kernel_2d_label).next_to(
            image_2d_group, DOWN, buff=MED_SMALL_BUFF
        )

        VGroup(image_2d_group, kernel_2d_group).move_to(image_group).shift(DOWN)

        # Block-Toeplitz Matrix
        block_toeplitz_data = [
            [1, 2, 0, 3, 4, 0, 0, 0, 0],
            [0, 1, 2, 0, 3, 4, 0, 0, 0],
            [0, 0, 0, 1, 2, 0, 3, 4, 0],
            [0, 0, 0, 0, 1, 2, 0, 3, 4],
        ]

        matrix_2d_squares = VGroup()
        cell_size = 0.4
        for row_idx, row in enumerate(block_toeplitz_data):
            for col_idx, val in enumerate(row):
                fill_color = ORANGE_SEABORN if val != 0 else WHITE
                fill_opacity = 0.5 if val != 0 else 0.1
                square = Square(
                    side_length=cell_size,
                    color=BLACK,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_width=1,
                )
                text = Text(str(val), font_size=DEFAULT_FONTSIZE * 0.6, color=BLACK)
                text.move_to(square)
                cell = VGroup(square, text)
                cell.move_to([col_idx * cell_size, -row_idx * cell_size, 0])
                matrix_2d_squares.add(cell)

        matrix_2d_squares.move_to(matrix_squares).to_edge(RIGHT, buff=1.5)

        matrix_2d_label = Tex(
            "Block-Toeplitz Matrix $A$", font_size=LARGE_FONTSIZE, color=GREEN_SEABORN
        ).next_to(matrix_2d_squares, UP, buff=MED_SMALL_BUFF)

        # Add brace on the left side showing number of rows (4)
        left_brace = Brace(matrix_2d_squares, LEFT, color=GREEN_SEABORN)
        left_brace_label = left_brace.get_tex(r"(N-K+1)^2").set_color(GREEN_SEABORN)

        # Add brace on the bottom showing number of columns (9)
        bottom_brace = Brace(matrix_2d_squares, DOWN, color=GREEN_SEABORN)
        bottom_brace_label = bottom_brace.get_tex(r"N^2").set_color(GREEN_SEABORN)

        matrix_2d_group = VGroup(
            matrix_2d_label,
            matrix_2d_squares,
            left_brace,
            left_brace_label,
            bottom_brace,
            bottom_brace_label,
        )

        # Animate 2D version
        self.play(
            Transform(image_group, image_2d_group),
            Transform(kernel_group, kernel_2d_group),
        )
        self.next_slide()

        self.play(Transform(matrix_group, matrix_2d_group))
        self.next_slide()

        self.play(FadeOut(image_group, kernel_group))

        left_title = Tex(r"$\bf{In~practice}$", font_size=SUBTITLE_FONTSIZE)
        image_size = Tex(
            r"Image Size: N $\approx$ 512",
            font_size=LARGE_FONTSIZE,
            color=BLUE_SEABORN,
        )
        matrix_size = Tex(
            r"Matrix Size: N$^2$ $\approx$ 260k rows",
            font_size=LARGE_FONTSIZE,
            color=GREEN_SEABORN,
        )
        matrix_elements = Tex(
            r"Matrix Elements: N$^4$ $\approx 6 \times 10^{10}$ !",
            font_size=LARGE_FONTSIZE,
        )
        matrix_memory = Tex(
            r"Memory $\approx$ $\bf{250}$ GB (float32) !",
            font_size=LARGE_FONTSIZE,
        )

        left_text = (
            VGroup(left_title, image_size, matrix_size, matrix_elements, matrix_memory)
            .arrange(DOWN)
            .to_edge(LEFT, buff=1)
            .align_to(matrix_2d_group, UP)
            .shift(UP)
        )
        VGroup(image_size, matrix_size, matrix_elements, matrix_memory).arrange(
            DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF
        ).next_to(left_title, DOWN, buff=LARGE_BUFF)

        self.play(FadeIn(left_text))

        self.next_slide()
        self.play(FadeOut(left_text, matrix_group))

        self.update_slide_number()

        ################################################################################
        # Slide 3: Tomography and its matrix
        ################################################################################
        self.next_section(skip_animations=False)

        # Create the circular object (phantom)
        phantom_radius_1 = 1.5
        phantom = Circle(radius=phantom_radius_1, color=BLUE_SEABORN, fill_opacity=0.3)
        phantom_label = Tex(
            "Object", font_size=LARGE_FONTSIZE, color=BLUE_SEABORN
        ).next_to(phantom, DOWN, buff=MED_SMALL_BUFF)

        # Create the detector lines (source and detector)
        detector_length = 2
        detector_distance = 2.5

        # Source line (left side)
        source_line = Line(
            start=UP * detector_length / 2 + LEFT * detector_distance,
            end=DOWN * detector_length / 2 + LEFT * detector_distance,
            color=ORANGE_SEABORN,
            stroke_width=4,
        )
        source_label = Tex(
            "Source", font_size=DEFAULT_FONTSIZE, color=ORANGE_SEABORN
        ).next_to(source_line, LEFT, buff=SMALL_BUFF)

        # Detector line (right side)
        detector_line = Line(
            start=UP * detector_length / 2 + RIGHT * detector_distance,
            end=DOWN * detector_length / 2 + RIGHT * detector_distance,
            color=ORANGE_SEABORN,
            stroke_width=4,
        )
        detector_label = Tex(
            "Detector", font_size=DEFAULT_FONTSIZE, color=ORANGE_SEABORN
        ).next_to(detector_line, RIGHT, buff=SMALL_BUFF)

        # Create parallel rays
        num_rays = 7
        rays = VGroup()
        for i in range(num_rays):
            t = (i / (num_rays - 1)) - 0.5  # Range from -0.5 to 0.5
            y_pos = t * detector_length
            ray = Line(
                start=LEFT * detector_distance + UP * y_pos,
                end=RIGHT * detector_distance + UP * y_pos,
                color=YELLOW,
                stroke_width=2,
                stroke_opacity=0.7,
            )
            rays.add(ray)

        # Group everything for rotation
        tomography_setup = VGroup(
            source_line, detector_line, rays, source_label, detector_label
        )
        tomography_group = VGroup(phantom, tomography_setup)
        tomography_group.move_to(ORIGIN)

        phantom_label.next_to(phantom, DOWN, buff=MED_SMALL_BUFF)

        # Animate the phantom appearing
        self.play(Create(phantom), FadeIn(phantom_label))
        self.next_slide()

        # Animate source and detector appearing
        self.play(
            Create(source_line),
            Create(detector_line),
            FadeIn(source_label, detector_label),
        )

        # Animate rays shooting
        self.play(Create(rays, lag_ratio=0.1))
        self.next_slide()

        # Rotate the setup around the phantom
        # First, we need to make labels follow the rotation properly
        # Remove labels for cleaner rotation
        self.play(FadeOut(source_label, detector_label, phantom_label))

        # Create a rotation group (source, detector, rays)
        rotation_group = VGroup(source_line, detector_line, rays)

        # Full rotation animation
        self.play(
            Rotate(
                rotation_group,
                angle=PI,
                about_point=phantom.get_center(),
                rate_func=linear,
            ),
            run_time=3,
        )
        self.next_slide()

        # Add grid on top of phantom
        num_voxels = 5
        voxel_size = (phantom_radius_1 * 2) / num_voxels
        grid = VGroup()

        # Create grid lines
        for i in range(num_voxels + 1):
            # Vertical lines
            v_line = Line(
                start=phantom.get_center()
                + LEFT * phantom_radius_1
                + RIGHT * i * voxel_size
                + UP * phantom_radius_1,
                end=phantom.get_center()
                + LEFT * phantom_radius_1
                + RIGHT * i * voxel_size
                + DOWN * phantom_radius_1,
                color=BLACK,
                stroke_width=1,
            )
            grid.add(v_line)
            # Horizontal lines
            h_line = Line(
                start=phantom.get_center()
                + UP * phantom_radius_1
                + DOWN * i * voxel_size
                + LEFT * phantom_radius_1,
                end=phantom.get_center()
                + UP * phantom_radius_1
                + DOWN * i * voxel_size
                + RIGHT * phantom_radius_1,
                color=BLACK,
                stroke_width=1,
            )
            grid.add(h_line)

        # Create bounding square for the grid
        grid_square = Square(
            side_length=phantom_radius_1 * 2,
            color=BLACK,
            stroke_width=2,
        ).move_to(phantom.get_center())

        # Add brace under the grid for "j voxels"
        grid_brace = Brace(grid_square, DOWN, color=BLUE_SEABORN)
        grid_brace_label = grid_brace.get_tex(r"j~\text{voxels}").set_color(
            BLUE_SEABORN
        )

        # Add "i detector bins" label next to detector
        num_detector_bins = num_rays
        detector_bins_label = MathTex(
            r"M~\text{detector bins}", font_size=DEFAULT_FONTSIZE, color=ORANGE_SEABORN
        ).next_to(detector_line, LEFT, buff=SMALL_BUFF)

        self.play(
            Create(grid),
            Create(grid_square),
            FadeIn(grid_brace, grid_brace_label),
            FadeIn(detector_bins_label),
        )
        self.next_slide()

        # Move diagram to the left
        left_diagram = VGroup(
            phantom,
            rotation_group,
            grid,
            grid_square,
            grid_brace,
            grid_brace_label,
            detector_bins_label,
        )
        self.play(left_diagram.animate.scale(0.7).to_edge(LEFT, buff=1))
        self.next_slide()

        # Create matrix visualization
        num_cols = num_voxels * num_voxels  # j^2 voxels (flattened)
        num_rows_per_angle = num_detector_bins  # i detector bins per angle

        cell_size = 0.25
        matrix_values_1 = [
            [1, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 1],
        ]

        # Create first set of rows (first angle)
        matrix_squares_tomo = VGroup()
        for row_idx, row in enumerate(matrix_values_1):
            for col_idx, val in enumerate(row):
                fill_color = ORANGE_SEABORN if val != 0 else WHITE
                fill_opacity = 0.5 if val != 0 else 0.1
                square = Square(
                    side_length=cell_size,
                    color=BLACK,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_width=1,
                )
                text = Text(str(val), font_size=DEFAULT_FONTSIZE * 0.4, color=BLACK)
                text.move_to(square)
                cell = VGroup(square, text)
                cell.move_to([col_idx * cell_size, -row_idx * cell_size, 0])
                matrix_squares_tomo.add(cell)

        matrix_squares_tomo.to_edge(RIGHT, buff=2).shift(UP * 1.5)

        dots_right = Tex(r"$\dots$", font_size=LARGE_FONTSIZE, color=BLACK).next_to(
            VGroup(matrix_squares_tomo), RIGHT, buff=SMALL_BUFF
        )

        matrix_tomo_label = Tex(
            "System Matrix $A$", font_size=LARGE_FONTSIZE, color=GREEN_SEABORN
        ).next_to(matrix_squares_tomo, UP, buff=MED_SMALL_BUFF)

        self.play(FadeIn(matrix_squares_tomo, matrix_tomo_label, dots_right))
        self.next_slide()

        # Add brace on the left side for detector bins (first angle only)
        left_matrix_brace = Brace(matrix_squares_tomo, LEFT, color=ORANGE_SEABORN)
        left_matrix_brace_label = left_matrix_brace.get_tex(
            r"M~\text{detector bins}"
        ).set_color(ORANGE_SEABORN)

        # Add brace on bottom for voxels
        bottom_matrix_brace = Brace(
            VGroup(matrix_squares_tomo, dots_right), DOWN, color=BLUE_SEABORN
        )
        bottom_matrix_brace_label = bottom_matrix_brace.get_tex(r"j^2").set_color(
            BLUE_SEABORN
        )

        self.play(
            FadeIn(left_matrix_brace, left_matrix_brace_label),
            FadeIn(bottom_matrix_brace, bottom_matrix_brace_label),
        )
        self.next_slide()

        # Add second set of rows (second angle)
        matrix_values_2 = [
            [0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
        ]

        matrix_squares_tomo_2 = VGroup()
        start_row = len(matrix_values_1)
        for row_idx, row in enumerate(matrix_values_2):
            for col_idx, val in enumerate(row):
                fill_color = ORANGE_SEABORN if val != 0 else WHITE
                fill_opacity = 0.5 if val != 0 else 0.1
                square = Square(
                    side_length=cell_size,
                    color=BLACK,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_width=1,
                )
                text = Text(str(val), font_size=DEFAULT_FONTSIZE * 0.4, color=BLACK)
                text.move_to(square)
                cell = VGroup(square, text)
                cell.move_to(
                    [col_idx * cell_size, -(start_row + row_idx) * cell_size, 0]
                )
                matrix_squares_tomo_2.add(cell)

        matrix_squares_tomo_2.move_to(matrix_squares_tomo, aligned_edge=UL).shift(
            DOWN * len(matrix_values_1) * cell_size
        )

        new_dots_right = Tex(r"$\dots$", font_size=LARGE_FONTSIZE, color=BLACK).next_to(
            VGroup(matrix_squares_tomo, matrix_squares_tomo_2),
            RIGHT,
            buff=MED_SMALL_BUFF,
        )

        # Add three dots to show continuation
        dots = Tex(r"$\vdots$", font_size=LARGE_FONTSIZE, color=BLACK).next_to(
            VGroup(matrix_squares_tomo, matrix_squares_tomo_2), DOWN, buff=SMALL_BUFF
        )

        # Update bottom brace position after adding dots
        final_bottom_matrix_brace = Brace(
            VGroup(matrix_squares_tomo, dots_right), DOWN, color=BLUE_SEABORN
        ).shift(DOWN * (len(matrix_values_2) * cell_size + 0.5))
        final_bottom_matrix_brace_label = final_bottom_matrix_brace.get_tex(
            r"N^2"
        ).set_color(BLUE_SEABORN)

        # Transform left brace to wrap around all values + dots
        full_matrix = VGroup(matrix_squares_tomo, matrix_squares_tomo_2, dots)
        new_left_matrix_brace = Brace(full_matrix, LEFT, color=ORANGE_SEABORN)
        new_left_matrix_brace_label = new_left_matrix_brace.get_tex(
            r"M~\text{detector bins} \times \text{angles}"
        ).set_color(ORANGE_SEABORN)

        self.next_slide()

        self.play(
            FadeIn(matrix_squares_tomo_2),
            Transform(dots_right, new_dots_right),
            FadeIn(dots),
            Transform(bottom_matrix_brace, final_bottom_matrix_brace),
            Transform(bottom_matrix_brace_label, final_bottom_matrix_brace_label),
            Transform(left_matrix_brace, new_left_matrix_brace),
            Transform(left_matrix_brace_label, new_left_matrix_brace_label),
        )

        self.next_slide()

        # Fade out the diagram on the left
        self.play(FadeOut(left_diagram))
        self.next_slide()

        left_title = Tex(r"$\bf{In~practice}$", font_size=SUBTITLE_FONTSIZE)
        num_voxels = Tex(
            r"Number of voxels: $N^2 \approx 260k$",
            font_size=LARGE_FONTSIZE,
            color=BLUE_SEABORN,
        )
        num_detector_bins = Tex(
            r"Detector bins: $M \approx 800$",
            font_size=LARGE_FONTSIZE,
            color=ORANGE_SEABORN,
        )
        num_angles = Tex(
            r"Angles: 192",
            font_size=LARGE_FONTSIZE,
        )
        matrix_memory = Tex(
            r"Memory $\approx \bf{200 GB}$ (float32) !",
            font_size=LARGE_FONTSIZE,
        )

        left_text = (
            VGroup(left_title, num_voxels, num_detector_bins, num_angles, matrix_memory)
            .arrange(DOWN)
            .to_edge(LEFT, buff=1)
            .align_to(matrix_2d_group, UP)
            .shift(UP)
        )
        VGroup(num_voxels, num_detector_bins, num_angles, matrix_memory).arrange(
            DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF
        ).next_to(left_title, DOWN, buff=LARGE_BUFF)

        self.play(FadeIn(left_text))

        self.next_slide()

        # Clean up
        self.play(
            FadeOut(
                matrix_squares_tomo,
                matrix_squares_tomo_2,
                dots,
                dots_right,
                matrix_tomo_label,
                left_matrix_brace,
                left_matrix_brace_label,
                bottom_matrix_brace,
                bottom_matrix_brace_label,
                left_text,
            )
        )
        self.update_slide_number()

        ################################################################################
        # Slide 4: Conditioning
        ################################################################################
        self.next_section(skip_animations=False)

        slide4_title = (
            Tex(
                r"Why is this hard ?",
                font_size=SUBTITLE_FONTSIZE,
                color=BLUE_CREATIS,
            )
            .align_to(SUBTITLE_POSITION, LEFT)
            .align_to(SUBTITLE_POSITION, UP)
        )

        self.play(Transform(slide_title, slide4_title))

        svd = Tex("$A$ can be decomposed using Singular Value Decomposition (SVD):")
        svd_eq = MathTex(r"A = U \Sigma V^{\top}", font_size=LARGE_FONTSIZE)
        svd_group = (
            VGroup(svd, svd_eq)
            .arrange(DOWN, buff=MED_SMALL_BUFF)
            .next_to(slide_title, DOWN, buff=MED_LARGE_BUFF)
            .align_to(slide_title, LEFT)
        )
        svd_eq.set_x(0)

        # Create Sigma matrix (diagonal)
        sigma_matrix = Matrix(
            [
                [r"\sigma_0", "0", r"\cdots", "0"],
                ["0", r"\sigma_1", r"\cdots", "0"],
                [r"\vdots", r"\vdots", r"\ddots", r"\vdots"],
                ["0", "0", r"\cdots", r"\sigma_{r-1}"],
            ],
            h_buff=MED_LARGE_BUFF,
            v_buff=MED_LARGE_BUFF,
            element_to_mobject=lambda m: MathTex(m, font_size=SMALL_FONTSIZE),
            left_bracket="(",
            right_bracket=")",
        ).next_to(svd_eq, DOWN, buff=MED_LARGE_BUFF)

        self.play(FadeIn(svd_group))
        self.next_slide()

        self.play(FadeIn(sigma_matrix))
        self.next_slide()

        inverse_svd = Tex("Let's assume we can invert $A$. Then:")
        inverse_svd_eq = MathTex(
            r"A^{-1} = V \Sigma^{-1} U^{\top}", font_size=LARGE_FONTSIZE
        )
        inverse_svd_group = (
            VGroup(inverse_svd, inverse_svd_eq)
            .arrange(DOWN, buff=MED_SMALL_BUFF)
            .next_to(svd_eq, DOWN, buff=MED_LARGE_BUFF)
            .align_to(slide_title, LEFT)
        )
        inverse_svd_eq.set_x(0)
        self.play(FadeOut(sigma_matrix))
        self.play(FadeIn(inverse_svd_group))
        self.next_slide()

        instability = (
            Tex(
                "$\\sigma_i$ very small $\\rightarrow$ $\\Sigma^{-1}$ contains very large values $\\rightarrow$ causes instability.",
            )
            .next_to(inverse_svd_group, DOWN, buff=LARGE_BUFF)
            .align_to(slide_title, LEFT)
        )

        self.play(FadeIn(instability))
        self.next_slide()

        self.play(FadeOut(inverse_svd_group, instability, svd_group))

        ################################################################################
        # Slide 5: Ill-conditioning
        ################################################################################
        self.next_section(skip_animations=False)
        self.update_slide_number()

        condition_nb_def = VGroup(
            Tex(
                r"Let $A \in \mathbb{R}^{m \times n}$ and $\sigma_{0}, \dots, \sigma_{r-1}$ its singular values.",
                font_size=DEFAULT_FONTSIZE,
            ),
            Tex(
                r"The condition number of $A$ is defined as:",
                font_size=DEFAULT_FONTSIZE,
            ),
            MathTex(
                r"\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}",
                font_size=DEFAULT_FONTSIZE,
            ),
            Tex(
                r"with $\sigma_{\max}$ its largest singular value and $\sigma_{\min}$ its smallest non-zero singular value.",
                font_size=DEFAULT_FONTSIZE,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF)
        condition_nb_def[2].set_x(condition_nb_def.get_x())

        defbox_contidition_nb = (
            DefinitionBox(content=condition_nb_def, title="Condition Number")
            .align_to(slide_title, LEFT)
            .shift(0.5 * UP)
        )

        self.play(FadeIn(defbox_contidition_nb))
        self.next_slide()

        illcond = (
            Tex(
                "When $\\kappa(A)$ is large, $A$ is said to be \\bf{ill-conditioned}.",
                font_size=LARGE_FONTSIZE,
            )
            .next_to(defbox_contidition_nb, DOWN, buff=MED_LARGE_BUFF)
            .set_x(0)
        )

        self.play(FadeIn(illcond))
        self.next_slide()

        self.play(
            FadeOut(defbox_contidition_nb),
            illcond.animate.next_to(slide_title, DOWN, buff=MED_LARGE_BUFF).align_to(
                slide_title, LEFT
            ),
        )

        deconv = (
            Tex("Deconvolution is ill-conditioned.", font_size=LARGE_FONTSIZE)
            .next_to(illcond, DOWN, buff=MED_LARGE_BUFF)
            .align_to(slide_title, LEFT)
        )

        img_deconv = (
            ImageMobject("./assets/img/section1/slide4/moon-pseudoinverse.png")
            .scale_to_fit_width(10)
            .next_to(deconv, DOWN, buff=MED_LARGE_BUFF)
            .set_x(0)
        )

        self.play(FadeIn(deconv, img_deconv))
        self.next_slide()

        tomo = (
            Tex(
                "Tomographic reconstruction is ill-conditioned.",
                font_size=LARGE_FONTSIZE,
            )
            .next_to(illcond, DOWN, buff=MED_LARGE_BUFF)
            .align_to(slide_title, LEFT)
        )
        img_tomo = (
            ImageMobject("./assets/img/section1/slide4/tomo-pseudoinverse.png")
            .scale_to_fit_width(10)
            .move_to(img_deconv)
        )
        self.play(FadeOut(img_deconv), ReplacementTransform(deconv, tomo))
        self.play(FadeIn(img_tomo))
        self.next_slide()

        self.update_slide_number()
        self.play(FadeOut(tomo, img_tomo, illcond))

        ################################################################################
        # Slide 6: Modelisation
        ################################################################################
        self.next_section(skip_animations=False)
        modelisation_title = (
            Tex(
                r"Variational formulation",
                font_size=SUBTITLE_FONTSIZE,
                color=BLUE_CREATIS,
            )
            .align_to(SUBTITLE_POSITION, LEFT)
            .align_to(SUBTITLE_POSITION, UP)
        )
        self.play(Transform(slide_title, modelisation_title))

        intro = (
            Tex("We model the reconstruction problem as:")
            .next_to(slide_title, DOWN, buff=MED_LARGE_BUFF)
            .align_to(slide_title, LEFT)
        )
        eq_minimization = (
            MathTex(
                r"\hat{x} = \arg\min_{x} \| Ax - y \|^2_2",
                font_size=LARGE_FONTSIZE,
            )
            .next_to(intro, DOWN, buff=LARGE_BUFF)
            .set_x(0)
        )
        self.play(FadeIn(intro, eq_minimization))
        self.next_slide()

        eq_minimization_reg = (
            MathTex(
                r"\hat{x} = \arg\min_{x} \| Ax - y \|^2_2 + \lambda \mathcal{R}(x)",
                font_size=LARGE_FONTSIZE,
            )
            .next_to(intro, DOWN, buff=LARGE_BUFF)
            .set_x(0)
        )
        self.play(Transform(eq_minimization, eq_minimization_reg))
        self.next_slide()

        # Add braces under the equation terms
        data_fidelity_brace = Brace(
            eq_minimization_reg[0][10:16],  # ||Ax - y||_2^2 part
            DOWN,
            color=GREEN_SEABORN,
        )
        data_fidelity_label = data_fidelity_brace.get_tex(
            "\\text{data-fidelity}"
        ).set_color(GREEN_SEABORN)

        regularization_brace = Brace(
            eq_minimization_reg[0][19:], UP, color=ORANGE_SEABORN  # R(x) part
        )
        regularization_label = regularization_brace.get_tex(
            "\\text{regularization}"
        ).set_color(ORANGE_SEABORN)

        self.play(
            FadeIn(data_fidelity_brace, data_fidelity_label),
            FadeIn(regularization_brace, regularization_label),
        )
        self.next_slide()

        neg_log_likelihood_label = data_fidelity_brace.get_tex(
            "\\text{negative log likelihood}"
        ).set_color(GREEN_SEABORN)
        prior_label = regularization_brace.get_tex("\\text{prior}   ").set_color(
            ORANGE_SEABORN
        )

        self.play(
            Transform(data_fidelity_label, neg_log_likelihood_label),
            Transform(regularization_label, prior_label),
        )
        self.next_slide()
        self.play(
            FadeOut(
                eq_minimization_reg,
                intro,
                data_fidelity_brace,
                data_fidelity_label,
                regularization_brace,
                regularization_label,
            )
        )

        self.wait()
