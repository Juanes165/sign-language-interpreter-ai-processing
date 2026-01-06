import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from mediapipe.python.solutions.holistic import Holistic
from app_constants import ASSETS_PATH, KEYPOINTS_PATH, FRAME_ACTIONS_PATH, MODEL_FRAMES, WORDS_JSON_PATH
from utility import ensure_dir, extract_keypoints


def process_all_sequences(gesture_labels=None):
    if gesture_labels is None:
        try:
            with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
                gesture_labels = json.load(f)["word_ids"]
        except Exception:
            gesture_labels = []

    base_dir = Path(ASSETS_PATH) / 'frame_actions'
    ensure_dir(KEYPOINTS_PATH)

    with Holistic() as holistic:
        for label_idx, gesture in enumerate(gesture_labels):
            gesture_dir = base_dir / gesture
            if not gesture_dir.exists():
                print(f"⚠️  No existe carpeta para gesto: {gesture}")
                continue

            sequences = []
            for sample_dir in sorted(gesture_dir.glob('sample_*')):
                seq = []
                frames = sorted(sample_dir.glob('*.jpg'))[:MODEL_FRAMES]
                for frame_path in frames:
                    frame = cv2.imread(str(frame_path))
                    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    kp = extract_keypoints(results)
                    seq.append(kp)
                if len(seq) == MODEL_FRAMES:
                    sequences.append(seq)

            sequences = np.array(sequences, dtype=np.float32)
            out_path = Path(KEYPOINTS_PATH) / f"{gesture}.npy"
            np.save(out_path, sequences)
            print(f"✅ Guardado: {out_path} -> {sequences.shape}")


if __name__ == '__main__':
    process_all_sequences()
