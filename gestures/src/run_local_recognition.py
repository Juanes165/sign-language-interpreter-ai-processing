import cv2
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from pathlib import Path
import argparse
from mediapipe.python.solutions.holistic import Holistic
from app_constants import MODEL_FRAMES, MIN_LENGTH_FRAMES, MODEL_DIR, KERAS_MODEL_PATH, WORDS_JSON_PATH, FONT, FONT_SIZE, words_text
from voice_output import generate_audio_from_text
from utility import extract_keypoints, draw_landmarks_on_frame
from utility_functions import mediapipe_detection, hand_detected


def pick_model_path() -> Path:
    """Elige un modelo por defecto disponible (.keras siempre primero)."""
    candidates = [
        Path(MODEL_DIR) / 'actions_15.keras',   # Siempre preferir .keras
        # Path(KERAS_MODEL_PATH),
        Path(MODEL_DIR) / 'actions_15.h5',      # Fallback a .h5
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError('No se encontró actions_15.keras en models/. Entrena uno primero con train_lstm_actions.py')


def load_labels():
    try:
        with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)["word_ids"]
    except Exception:
        return ["gesto_1", "gesto_2", "gesto_3"]

def interpolate_keypoints(keypoints, target_length=15):
    cur = len(keypoints)
    if cur == target_length:
        return keypoints
    idxs = np.linspace(0, cur - 1, target_length)
    out = []
    for i in idxs:
        lo = int(np.floor(i))
        hi = int(np.ceil(i))
        w = i - lo
        if lo == hi:
            out.append(keypoints[lo])
        else:
            p = (1 - w) * np.array(keypoints[lo]) + w * np.array(keypoints[hi])
            out.append(p.tolist())
    return out


def normalize_keypoints(keypoints, target_length=15):
    cur = len(keypoints)
    if cur < target_length:
        return interpolate_keypoints(keypoints, target_length)
    if cur > target_length:
        step = cur / target_length
        idxs = np.arange(0, cur, step).astype(int)[:target_length]
        return [keypoints[i] for i in idxs]
    return keypoints


def main(
    threshold: float = 0.2,
    margin_frame: int = 1,
    delay_frames: int = 3,
    device: int = 0,
    speak: bool = True,
    model_file: str | None = None,
):
    labels = load_labels()
    model_path = Path(model_file) if model_file else pick_model_path()
    print(f"📦 Cargando modelo: {model_path}")
    
    try:
        model = load_model(str(model_path))
    except (ValueError, KeyError) as e:
        print(f"⚠️  Error cargando modelo con compile=True: {e}")
        print("🔄 Intentando cargar con compile=False...")
        model = load_model(str(model_path), compile=False)
        # Recompilar manualmente
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("➡️ Modelo LSTM cargado")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la cámara')

    kp_seq = []
    last_pred = ''
    sentence = []
    count_frame = 0
    fix_frames = 0
    recording = False

    with Holistic() as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)

            # Modo por segmentos (igual al recognition original/LSTM)
            if hand_detected(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
                    cv2.putText(frame, 'Capturando...', (5, 30), FONT, FONT_SIZE, (255, 50, 0))
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        cv2.putText(frame, 'Capturando...', (5, 30), FONT, FONT_SIZE, (255, 50, 0))
                    # recortar frames del margen y la demora
                    if margin_frame + delay_frames > 0:
                        kp_seq = kp_seq[:-(margin_frame + delay_frames)] if len(kp_seq) > (margin_frame + delay_frames) else kp_seq
                    # normalizar a longitud del modelo
                    kp_norm = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    seq = np.expand_dims(np.array(kp_norm, dtype=np.float32), axis=0)
                    preds = model.predict(seq, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    conf = float(preds[idx])
                    if conf > threshold:
                        # Buscar primero con el label completo (incluye guiones)
                        label_full = labels[idx]
                        word_id_first = label_full.split('-')[0]
                        # Intentar primero con el label completo, luego con la primera parte, luego usar el label como fallback
                        spoken = words_text.get(label_full, words_text.get(word_id_first, label_full))
                        last_pred = f"{label_full} ({conf:.2f})"
                        if speak:
                            generate_audio_from_text(spoken)
                        # log opcional a archivo
                        try:
                            with open("words.txt", "a", encoding="utf-8") as f:
                                f.write(f"{spoken} - {conf}\n")
                        except Exception:
                            pass
                        # acumular frase (máximo 6 elementos para legibilidad)
                        sentence.insert(0, spoken)
                        if len(sentence) > 6:
                            sentence = sentence[:6]

                # reset estado
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
                cv2.putText(frame, 'Listo para capturar...', (5, 30), FONT, FONT_SIZE, (0, 220, 100))

            draw_landmarks_on_frame(frame, results)
            # Overlay superior (frase acumulada)
            cv2.rectangle(frame, (0, 0), (640, 35), (0, 0, 255), -1)
            cv2.putText(frame, ' | '.join(sentence) if sentence else '', (5, 25), FONT, FONT_SIZE, (255, 255, 255))
            # Overlay inferior (última predicción)
            h = frame.shape[0]
            cv2.rectangle(frame, (0, h-35), (640, h), (0, 0, 0), -1)
            cv2.putText(frame, last_pred or 'Escuchando...', (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow('Reconocimiento Local (LSTM)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconocimiento local (.keras) con MediaPipe Holistic')
    parser.add_argument('--threshold', type=float, default=0.7, help='Umbral de confianza para aceptar la predicción')
    parser.add_argument('--margin-frame', type=int, default=1, help='Frames iniciales a ignorar tras detectar manos')
    parser.add_argument('--delay-frames', type=int, default=3, help='Frames extra tras perder las manos para cerrar la muestra')
    parser.add_argument('--device', type=int, default=0, help='Índice de cámara para OpenCV (0 por defecto)')
    parser.add_argument('--mute', action='store_true', help='Silencia la salida de voz (TTS)')
    parser.add_argument('--model', type=str, default=None, help='Ruta a un modelo .keras a usar')
    args = parser.parse_args()

    main(
        threshold=args.threshold,
        margin_frame=args.margin_frame,
        delay_frames=args.delay_frames,
        device=args.device,
        speak=not args.mute,
        model_file=args.model,
    )
