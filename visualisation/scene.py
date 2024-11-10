from manim import *


class MainScene(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)
        self.play(Create(grid))

# run: manim -pql scene.py MainScene
