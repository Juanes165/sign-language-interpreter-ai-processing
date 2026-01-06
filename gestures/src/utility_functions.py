# Copia mínima de utilidades originales para gesture_collector
import os
import json
import cv2
import numpy as np
from typing import NamedTuple
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from app_constants import KEYPOINTS_PATH


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results


def hand_detected(results: NamedTuple) -> bool:
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


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


def store_frames(frames, output_folder):
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"{i+1}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
