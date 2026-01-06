import os
import json
import cv2
import numpy as np
from typing import NamedTuple
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_landmarks_on_frame(image, results):
    draw_landmarks(image, results.face_landmarks, FACEMESH_CONTOURS,
                   DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                   DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    draw_landmarks(image, results.pose_landmarks, POSE_CONNECTIONS,
                   DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                   DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    draw_landmarks(image, results.left_hand_landmarks, HAND_CONNECTIONS,
                   DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                   DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    draw_landmarks(image, results.right_hand_landmarks, HAND_CONNECTIONS,
                   DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                   DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
