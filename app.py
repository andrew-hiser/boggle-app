from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.graphics import Color, Line


class CameraOverlay(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Create camera preview
        self.camera = Camera(play=True, resolution=(640, 480))
        self.camera.allow_stretch = True

        # Create capture button
        self.capture_btn = Button(text="Capture Image", size_hint_y=None, height=60)
        self.capture_btn.bind(on_press=self.capture_image)

        # Add everything
        self.add_widget(self.camera)
        self.add_widget(self.capture_btn)

        # Add overlay square
        with self.camera.canvas.after:
            Color(1, 0, 0, 0.8)  # Red, semi-transparent
            self.square = Line(rectangle=(150, 100, 340, 280), width=2)

    def capture_image(self, instance):
        self.camera.export_to_png("captured.png")
        print("Image saved as captured.png")


class CameraApp(App):
    def build(self):
        return CameraOverlay()


if __name__ == "__main__":
    CameraApp().run()
