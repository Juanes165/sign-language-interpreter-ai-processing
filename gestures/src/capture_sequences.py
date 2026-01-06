import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from utility_functions import create_directory, draw_landmarks_on_frame, mediapipe_detection, store_frames, hand_detected
from app_constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime

def gesture_collector(output_directory, pre_capture_frames=1, min_required_frames=5, frame_delay=3):
    create_directory(output_directory)
    frame_counter = 0
    frames_buffer = []
    delayed_frames = 0
    capturing = False

    with Holistic() as holistic_detector:
        video_capture = cv2.VideoCapture(0)
        
        while video_capture.isOpened():
            success, current_frame = video_capture.read()
            if not success:
                break

            frame_copy = current_frame.copy()
            detection_results = mediapipe_detection(current_frame, holistic_detector)
            
            if hand_detected(detection_results) or capturing:
                capturing = False
                frame_counter += 1
                if frame_counter > pre_capture_frames:
                    cv2.putText(frame_copy, 'Capturing...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames_buffer.append(np.asarray(current_frame))
            else:
                if len(frames_buffer) >= min_required_frames + pre_capture_frames:
                    delayed_frames += 1
                    if delayed_frames < frame_delay:
                        capturing = True
                        continue
                    frames_buffer = frames_buffer[: - (pre_capture_frames + frame_delay)]
                    timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
                    sample_folder = os.path.join(output_directory, f"sample_{timestamp}")
                    create_directory(sample_folder)
                    store_frames(frames_buffer, sample_folder)

                capturing, delayed_frames = False, 0
                frames_buffer, frame_counter = [], 0
                cv2.putText(frame_copy, 'Ready to capture...', FONT_POS, FONT, FONT_SIZE, (0, 220, 100))
            
            draw_landmarks_on_frame(frame_copy, detection_results)
            cv2.imshow(f'Sample Collection for "{os.path.basename(output_directory)}"', frame_copy)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_name = "adios-gen"
    gesture_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, gesture_name)
    gesture_collector(gesture_path)
