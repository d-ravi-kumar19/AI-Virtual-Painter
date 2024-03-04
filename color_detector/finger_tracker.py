# color_detector/finger_tracker.py
import cv2
import mediapipe as mp
from collections import deque

class FingerTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.results = None
        self.finger_history = deque(maxlen=20)

    def track_finger(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            index_finger_tip = (
                int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            )
            self.finger_history.appendleft(index_finger_tip)
            return index_finger_tip

        return None
    

    def draw_finger_movement(self, frame):
        for i in range(1, len(self.finger_history)):
            if self.finger_history[i - 1] is not None and self.finger_history[i] is not None:
                cv2.line(frame, self.finger_history[i - 1], self.finger_history[i], (0, 255, 0), 2)
