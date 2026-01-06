import os
import logging
import mediapipe as mp
import cv2
from models.hand_gesture_model import HandGestures, session
import pickle
import json
import sys

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorios
DATA_DIR = '../data/raw'
EXPORT_DIR = '../data/export'
ANNOTATIONS_DIR = os.path.join(EXPORT_DIR, 'annotations')
MODEL_DIR = '../models'

# Crear directorios si no existen
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Información sobre modelos MediaPipe
HAND_LANDMARKER_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
USE_MEDIAPIPE_TASKS = False

# Verificar si existe el modelo hand_landmarker.task
if os.path.exists(HAND_LANDMARKER_PATH):
    try:
        # Intentar cargar el modelo con MediaPipe Tasks
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Configurar detector de manos con MediaPipe Tasks
        base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.3
        )
        detector = vision.HandLandmarker.create_from_options(options)
        USE_MEDIAPIPE_TASKS = True
        logging.info("Usando MediaPipe Tasks con el modelo hand_landmarker.task")
    except Exception as e:
        logging.error(f"Error al cargar el modelo MediaPipe Tasks: {e}")
        USE_MEDIAPIPE_TASKS = False
else:
    logging.warning(f"No se encontró el modelo {HAND_LANDMARKER_PATH}. Usando la API clásica de MediaPipe Hands.")
    print("\n====================== ATENCIÓN ======================")
    print(f"El modelo hand_landmarker.task no se encuentra en {MODEL_DIR}")
    print("Para usar MediaPipe Tasks, descarga el modelo desde:")
    print("https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models")
    print("Y colócalo en la carpeta models/ con el nombre 'hand_landmarker.task'")
    print("Continuando con la API clásica de MediaPipe...\n")

# Configurar MediaPipe Hands (método tradicional como respaldo)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def process_image_with_mp_tasks(img_path, class_name, img_id):
    """Procesa la imagen con MediaPipe Tasks."""
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"No se pudo leer la imagen {img_path}.")
        return None

    # Convertir a RGB para MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Detectar manos
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        # Guardar la imagen en el directorio de exportación
        export_img_path = os.path.join(EXPORT_DIR, f"{class_name}_{img_id}.jpg")
        cv2.imwrite(export_img_path, img)
        
        # Crear anotación para esta imagen
        annotation = {
            "image_path": export_img_path,
            "label": class_name,
            "hand_landmarks": []
        }
        
        for hand_landmarks in detection_result.hand_landmarks:
            # Extraer landmark points
            landmarks_list = []
            x_coords = []
            y_coords = []
            
            for i, landmark in enumerate(hand_landmarks):
                x, y, z = landmark.x, landmark.y, landmark.z
                landmarks_list.append({"x": x, "y": y, "z": z})
                x_coords.append(x)
                y_coords.append(y)
            
            annotation["hand_landmarks"].append(landmarks_list)
            
            # Guardar en BD
            try:
                gesture = HandGestures(class_name=class_name)
                for i in range(21):
                    setattr(gesture, f'x_{i}', hand_landmarks[i].x)
                    setattr(gesture, f'y_{i}', hand_landmarks[i].y)
                session.add(gesture)
            except Exception as e:
                logging.error(f"Error al guardar en la BD: {e}")
        
        return annotation
    return None

def process_image_with_mp_hands(img_path, class_name, img_id):
    """Procesa la imagen con MediaPipe Hands (API clásica)."""
    img = cv2.imread(img_path)
    if img is None:
        logging.error(f"No se pudo leer la imagen {img_path}.")
        return None

    # Procesar con MediaPipe Hands
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Guardar la imagen en el directorio de exportación
        export_img_path = os.path.join(EXPORT_DIR, f"{class_name}_{img_id}.jpg")
        cv2.imwrite(export_img_path, img)
        
        # Crear anotación para esta imagen
        annotation = {
            "image_path": export_img_path,
            "label": class_name,
            "hand_landmarks": []
        }
        
        # También recolectar datos para pickle (modelo tradicional)
        data_aux = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer coordenadas
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            # Para anotaciones JSON
            landmarks_list = []
            for i, lm in enumerate(hand_landmarks.landmark):
                landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z})
            
            annotation["hand_landmarks"].append(landmarks_list)
            
            # Para datos pickle (formato tradicional)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            
            # Guardar en BD
            try:
                gesture = HandGestures(class_name=class_name)
                for i in range(21):
                    setattr(gesture, f'x_{i}', x_[i])
                    setattr(gesture, f'y_{i}', y_[i])
                session.add(gesture)
            except Exception as e:
                logging.error(f"Error al guardar en la BD: {e}")
            
        # Guardar datos auxiliares para entrenamiento
        return annotation, data_aux
    return None, None

def main():
    annotations = []
    pickle_data = []
    pickle_labels = []
    count = 0
    
    for dir_ in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(class_dir):
            continue
            
        logging.info(f"Procesando clase: {dir_}")
        
        for idx, img_name in enumerate(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)
            
            # Procesar la imagen según el método disponible
            if USE_MEDIAPIPE_TASKS:
                annotation = process_image_with_mp_tasks(img_path, dir_, f"{count}_{idx}")
                if annotation:
                    annotations.append(annotation)
                    count += 1
            else:
                annotation, data_aux = process_image_with_mp_hands(img_path, dir_, f"{count}_{idx}")
                if annotation:
                    annotations.append(annotation)
                    if data_aux:  # Si hay datos auxiliares para pickle
                        pickle_data.append(data_aux)
                        pickle_labels.append(dir_)
                    count += 1
            
            # Mostrar progreso
            if count % 50 == 0:
                logging.info(f"Procesadas {count} imágenes")
    
    # Guardar las anotaciones en un archivo JSON
    annotations_file = os.path.join(ANNOTATIONS_DIR, "annotations.json")
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Guardar datos en formato pickle para modelo tradicional
    if pickle_data:
        pickle_file = os.path.join(MODEL_DIR, "data.pickle")
        with open(pickle_file, 'wb') as f:
            pickle.dump({'data': pickle_data, 'labels': pickle_labels}, f)
        logging.info(f"Datos para entrenamiento guardados en: {pickle_file}")
    
    # Guardar la sesión de la BD
    try:
        session.commit()
        logging.info(f"Insertados {count} registros en la BD.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error al insertar en la BD, se hizo rollback: {e}")
    finally:
        session.close()
        
    logging.info(f"Dataset creado. Se procesaron {count} imágenes y se guardaron en {EXPORT_DIR}")
    logging.info(f"Anotaciones guardadas en {annotations_file}")

if __name__ == "__main__":
    main()