# apps/drawing_app.py
import cv2
import numpy as np
from color_detector.finger_tracker import FingerTracker

class DrawingApp:
    def __init__(self):
        self.finger_tracker = FingerTracker()
        self.canvas = self.setup_canvas()
        self.drawing = False
        self.prev_point = None
        self.color_index = 0
        self.colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.color_text = ["BLACK", "RED", "GREEN", "BLUE"]

    def setup_canvas(self):
        paint_window = np.zeros((471, 636, 3)) + 255
        return paint_window

    def change_color(self):
        self.color_index = (self.color_index + 1) % len(self.colors)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            finger_position = self.finger_tracker.track_finger(frame)

            if finger_position is not None:
                cv2.circle(frame, finger_position, 10, (0, 0, 255), -1)
                self.finger_tracker.draw_finger_movement(frame)

                if self.drawing:
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, finger_position, self.colors[self.color_index], 2)
                    self.prev_point = finger_position
            else:
                self.prev_point = None

            cv2.putText(frame, f"Color: {self.color_text[self.color_index]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Tracking", frame)
            cv2.imshow("Paint", self.canvas)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            elif key == ord("c"):
                self.canvas = self.setup_canvas()
            elif key == ord("d"):
                self.drawing = not self.drawing
            elif key == ord("n"):  # Press 'n' to change color
                self.change_color()

        cap.release()
        cv2.destroyAllWindows()
