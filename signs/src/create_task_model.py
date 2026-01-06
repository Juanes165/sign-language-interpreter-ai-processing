import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, 'temp_training')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
EXPORT_DIR = os.path.join(BASE_DIR, 'data', 'export')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LABELS = {
    -1: "None",
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "ene", 15: "O", 16: "P",
    17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X",
    25: "Y", 26: "Z",
}


def prepare_folder_from_export():
    # Limpiar temp
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Crear carpetas por clase
    for idx, name in LABELS.items():
        class_dir = os.path.join(TEMP_DIR, str(idx))
        os.makedirs(class_dir, exist_ok=True)

    # Copiar imágenes exportadas a carpetas por clase
    moved = 0
    for fname in os.listdir(EXPORT_DIR):
        if not (fname.endswith('.jpg') or fname.endswith('.png')):
            continue
        # nombre esperado: "A_0.jpg" o "0_0_0.jpg" -> extraer primer token
        token = fname.split('_')[0]
        class_idx = None
        # mapear token a índice
        for idx, label in LABELS.items():
            if label.lower() == token.lower():
                class_idx = idx
                break
        if class_idx is None:
            try:
                class_idx = int(token)
            except Exception:
                continue
        src = os.path.join(EXPORT_DIR, fname)
        dst = os.path.join(TEMP_DIR, str(class_idx), fname)
        shutil.copy2(src, dst)
        moved += 1

    none_dir = os.path.join(TEMP_DIR, 'None')
    os.makedirs(none_dir, exist_ok=True)
    if len(os.listdir(none_dir)) == 0:
        import numpy as np
        import cv2
        for i in range(10):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(none_dir, f'none_{i}.jpg'), img)
        logging.info('Imágenes de clase None generadas.')

    logging.info(f'Imágenes preparadas: {moved}')


def ensure_mmm_installed():
    try:
        import mediapipe_model_maker  # noqa: F401
        return True
    except Exception:
        logging.info('Instalando mediapipe-model-maker...')
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mediapipe-model-maker'])
            return True
        except subprocess.CalledProcessError:
            return False


def train_and_export():
    if not ensure_mmm_installed():
        logging.error('No se pudo instalar mediapipe-model-maker')
        return False

    from mediapipe_model_maker import gesture_recognizer

    data = gesture_recognizer.Dataset.from_folder(TEMP_DIR)
    logging.info(f'Clases: {data.label_names}')
    train_data, val_data = data.split(0.8)

    hparams = gesture_recognizer.HParams(
        export_dir=MODEL_DIR,
        batch_size=64,
        epochs=50,
        learning_rate=0.001,
        shuffle=True,
    )
    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=0.01,
        layer_widths=[50]
    )
    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options,
        hparams=hparams
    )

    model = gesture_recognizer.GestureRecognizer.create(
        options=options,
        train_data=train_data,
        validation_data=val_data
    )

    result = model.evaluate(val_data)
    logging.info(f'Exactitud validación: {result[1]}')
    model.export_model()

    out_path = os.path.join(MODEL_DIR, 'gesture_recognizer.task')
    ok = os.path.exists(out_path)
    logging.info(f'Modelo exportado: {ok} -> {out_path}')
    return ok


if __name__ == '__main__':
    if not os.path.exists(EXPORT_DIR):
        logging.error('No existe data/export. Ejecuta datasetCreator.py primero.')
        sys.exit(1)
    prepare_folder_from_export()
    ok = train_and_export()
    sys.exit(0 if ok else 2)
