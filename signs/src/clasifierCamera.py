import cv2
import numpy as np
import threading
import time
import logging
import os
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import sys

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Intentar importar transformers y torch
text_generator = None
try:
    from transformers import pipeline
    import torch
    
    # Comprobar disponibilidad de CUDA para el modelo de texto
    if torch.cuda.is_available():
        device = 0
    else:
        device = -1
    
    # Inicializar el generador de texto
    try:
        text_generator = pipeline('text-generation', model='dccuchile/bert-base-spanish-wwm-cased', device=device)
        logging.info("Generador de texto inicializado correctamente.")
    except Exception as e:
        logging.warning(f"No se pudo cargar el modelo de texto: {e}")
        logging.warning("La funcionalidad de predicción de palabras no estará disponible.")

except ImportError:
    logging.warning("No se encontró la librería 'transformers'. La funcionalidad de predicción de palabras estará desactivada.")
    logging.info("Para habilitar esta función, instale las librerías necesarias con: pip install transformers torch")
    print("\n====================== AVISO ======================")
    print("La librería 'transformers' no está instalada.")
    print("Para habilitar la predicción de palabras, ejecute:")
    print("pip install transformers torch")
    print("===================================================\n")

# Directorios y rutas de modelos (usar rutas absolutas)
BASEDIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASEDIR, '..', 'models'))
GESTURE_RECOGNIZER_PATH = os.path.join(MODEL_DIR, 'gesture_recognizer.task')
PICKLE_MODEL_PATH = os.path.join(MODEL_DIR, 'model.p')

# Diccionario de etiquetas de letras
labels_dict = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "ene", 15: "O", 16: "P",
    17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X",
    25: "Y", 26: "Z",
}

# Variables globales
USE_MEDIAPIPE_TASKS = False
recognized_letters = []
predicted_word = ""
last_recognized_letter = None
last_recognition_time = time.time()
timeout = 1

# Verificar disponibilidad de modelos
if os.path.exists(GESTURE_RECOGNIZER_PATH):
    try:
        # Importar correctamente las clases de MediaPipe Tasks (API moderna)
        BaseOptions = mp_python.BaseOptions
        GestureRecognizer = mp_vision.GestureRecognizer
        GestureRecognizerOptions = mp_vision.GestureRecognizerOptions
        VisionRunningMode = mp_vision.RunningMode
        
        # Verificar explícitamente la existencia del archivo
        if not os.path.isfile(GESTURE_RECOGNIZER_PATH):
            logging.error(f"El archivo del modelo no existe: {GESTURE_RECOGNIZER_PATH}")
            USE_MEDIAPIPE_TASKS = False
        else:
            logging.info(f"Modelo encontrado en: {GESTURE_RECOGNIZER_PATH}")
            
            # Función de callback para gestos
            def process_result(result, output_image, timestamp_ms):
                global last_recognized_letter
                
                if result.gestures:
                    top_gesture = result.gestures[0][0]
                    category_name = top_gesture.category_name
                    score = top_gesture.score
                    
                    if score > 0.7:
                        if category_name.isdigit():
                            category_idx = int(category_name)
                            if category_idx in labels_dict:
                                last_recognized_letter = labels_dict[category_idx]
                            else:
                                last_recognized_letter = category_name
                        else:
                            last_recognized_letter = category_name
            
            # Configuración para GestureRecognizer
            def create_recognizer():
                base_options = BaseOptions(model_asset_path=os.path.abspath(GESTURE_RECOGNIZER_PATH))
                options = GestureRecognizerOptions(
                    base_options=base_options,
                    running_mode=VisionRunningMode.LIVE_STREAM,
                    result_callback=process_result
                )
                return GestureRecognizer.create_from_options(options)
            
            USE_MEDIAPIPE_TASKS = True
            logging.info("Usando modelo MediaPipe Tasks para reconocimiento de gestos.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo MediaPipe Tasks: {e}")
        USE_MEDIAPIPE_TASKS = False
elif os.path.exists(PICKLE_MODEL_PATH):
    logging.info(f"Usando modelo pickle tradicional: {PICKLE_MODEL_PATH}")
else:
    logging.error("No se encontró ningún modelo. Por favor, entrena un modelo primero.")
    print("\n====================== ERROR ======================")
    print("No se encontró ningún modelo de reconocimiento válido.")
    print("Verifica que exista al menos uno de estos archivos:")
    print(f"1. {GESTURE_RECOGNIZER_PATH}")
    print(f"2. {PICKLE_MODEL_PATH}")
    print("Ejecuta 'Trainer.py' para entrenar un modelo primero.")
    print("===================================================\n")
    sys.exit(1)

# Función para cargar el modelo tradicional (pickle)
def load_pickle_model():
    try:
        with open(PICKLE_MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
        logging.info("Modelo pickle cargado correctamente.")
        return model_dict['model']
    except Exception as e:
        logging.error(f"Error al cargar el modelo pickle: {e}")
        sys.exit(1)

# Función para generar texto
def generate_text_async(recognized_text):
    global predicted_word
    if text_generator:
        try:
            prediction = text_generator(recognized_text, max_length=20, num_return_sequences=1)
            predicted_word = prediction[0]['generated_text']
            logging.info(f"Palabra predicha: {predicted_word}")
        except Exception as e:
            logging.error(f"Error al generar texto: {e}")
    else:
        logging.debug("Generador de texto no disponible, se omite la predicción.")

def main():
    print('HOLAAAA')
    global last_recognition_time, predicted_word, recognized_letters, last_recognized_letter, USE_MEDIAPIPE_TASKS
    
    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: No se pudo abrir la cámara.")
        return
    
    # Configuración según el tipo de modelo disponible
    if USE_MEDIAPIPE_TASKS:
        # Usar MediaPipe Tasks
        try:
            recognizer = create_recognizer()
            frame_timestamp_ms = int(time.time() * 1000)
            logging.info("Reconocedor de gestos inicializado correctamente.")
        except Exception as e:
            logging.error(f"Error al inicializar el reconocedor: {e}")
            logging.info("Cambiando al modelo tradicional...")
            USE_MEDIAPIPE_TASKS = False
            model = load_pickle_model()
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
    else:
        # Configuración para el modelo tradicional
        model = load_pickle_model()
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: No se pudo capturar el fotograma.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if USE_MEDIAPIPE_TASKS:
                # Procesar con MediaPipe Tasks
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    recognizer.recognize_async(mp_image, frame_timestamp_ms)
                    # Avanzar timestamp en ms para el modo LIVE_STREAM
                    frame_timestamp_ms += 33  # ~30 FPS
                except Exception as e:
                    logging.error(f"Error al procesar el fotograma: {e}")
            else:
                # Procesar con el método tradicional
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Dibujar landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Extraer coordenadas
                        x_ = [lm.x for lm in hand_landmarks.landmark]
                        y_ = [lm.y for lm in hand_landmarks.landmark]
                        
                        # Preparar datos para predicción
                        data_aux = []
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(x_[i] - min(x_))
                            data_aux.append(y_[i] - min(y_))
                        
                        # Realizar predicción
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_class = int(prediction[0])
                        
                        # Obtener la letra reconocida
                        if predicted_class in labels_dict:
                            current_time = time.time()
                            if (current_time - last_recognition_time) > timeout:
                                recognized_letter = labels_dict[predicted_class]
                                recognized_letters.append(recognized_letter)
                                last_recognition_time = current_time
                            
                            # Mostrar rectángulo y letra
                            H, W, _ = frame.shape
                            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, labels_dict[predicted_class], (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Comprobar si hay una nueva letra reconocida (para MediaPipe Tasks)
            if USE_MEDIAPIPE_TASKS:
                current_time = time.time()
                if last_recognized_letter and (current_time - last_recognition_time) > timeout:
                    recognized_letters.append(last_recognized_letter)
                    last_recognition_time = current_time
                    last_recognized_letter = None
            
            # Mostrar letras reconocidas
            cv2.putText(frame, f"Letras: {''.join(recognized_letters)}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Generar predicción de texto
            if len(recognized_letters) >= 3 and threading.active_count() == 1 and text_generator:
                threading.Thread(target=generate_text_async, args=("".join(recognized_letters),)).start()
            
            # Mostrar predicción
            if predicted_word:
                cv2.putText(frame, f"Predicción: {predicted_word}", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Mostrar instrucciones
            cv2.putText(frame, "Presiona 'q' para salir, 'c' para limpiar", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Mostrar vista previa
            cv2.imshow('Reconocimiento de Gestos', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                recognized_letters = []
                predicted_word = ""
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        if USE_MEDIAPIPE_TASKS:
            recognizer.close()
        else:
            hands.close()

if __name__ == "__main__":
    main()