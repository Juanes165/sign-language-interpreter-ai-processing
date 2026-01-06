import mediapipe as mp
import cv2
import os
import time # Importa time para timestamp

# Importa utilidades de dibujo y soluciones de manos
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# --- Utilidades de dibujo --- (Puedes mover esto a un archivo separado si lo prefieres)
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3
_RED = (0, 0, 255)
_GREEN = (0, 255, 0)
_BLUE = (255, 0, 0)
_THICKNESS_WRIST_MCP = 5
_THICKNESS_FINGER = 3
_THICKNESS_DOT = -1 # Círculo relleno

def draw_landmarks_on_image(rgb_image, detection_result):
  """Dibuja los landmarks y conexiones de la mano en la imagen."""
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Itera a través de las manos detectadas
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Dibuja los landmarks
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image
# --- Fin de utilidades de dibujo ---


# Obtiene la ruta absoluta del directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construye la ruta absoluta al modelo
model_path = os.path.join(script_dir, '../models/gesture_recognizer.task')

# Verifica que el archivo exista
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

# Opciones Base y GestureRecognizer (usando las importaciones directas)
BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

# Variable global para almacenar el último resultado
latest_result = None

def process_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback para procesar y almacenar el resultado del reconocimiento."""
    global latest_result
    latest_result = result
    # Opcional: Imprimir aquí si solo quieres la salida en consola
    # if result.gestures:
    #     top_gesture = result.gestures[0][0]
    #     print(f"Timestamp: {timestamp_ms} - Gesto: {top_gesture.category_name} ({top_gesture.score:.2f})")


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path), # Usa la ruta absoluta
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=process_result,
    num_hands=1 # Puedes ajustar esto si necesitas detectar más manos
)

recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
# timestamp = 0 # Usaremos time.time() para timestamps más precisos

print("Presiona 'q' para salir.")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    # Voltea el frame horizontalmente para una vista tipo espejo
    frame = cv2.flip(frame, 1)

    # Convierte el frame BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Obtiene el timestamp actual en milisegundos
    timestamp_ms = int(time.time() * 1000)

    # Realiza el reconocimiento asíncrono
    recognizer.recognize_async(mp_image, timestamp_ms)

    # Dibuja los resultados en el frame original (BGR)
    current_frame = frame
    gesture_text = ""
    if latest_result:
        # Dibuja los landmarks si existen
        if latest_result.hand_landmarks:
             # Convierte la imagen de vuelta a RGB para dibujar, luego de vuelta a BGR
             # O dibuja directamente en el frame BGR si las utilidades lo permiten
             # Aquí dibujaremos sobre el frame BGR directamente
             current_frame = draw_landmarks_on_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), latest_result)
             current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR) # Convierte de vuelta a BGR para mostrar con OpenCV

        # Obtiene y muestra el nombre del gesto
        if latest_result.gestures:
            top_gesture = latest_result.gestures[0][0]
            gesture_text = f"Gesto: {top_gesture.category_name} ({top_gesture.score:.2f})"
            # Dibuja el texto del gesto en la imagen
            cv2.putText(current_frame, gesture_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("Reconocimiento de Gestos", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Cierra el reconocedor explícitamente
recognizer.close()
