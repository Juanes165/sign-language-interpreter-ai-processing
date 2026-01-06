import os
import cv2

# Frames y keypoints
MODEL_FRAMES = 15
LENGTH_KEYPOINTS = 1662
MIN_LENGTH_FRAMES = 5

# Rutas
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_PATH = os.path.join(ROOT_PATH, 'assets')
DATA_PATH = os.path.join(ASSETS_PATH, 'data')
KEYPOINTS_PATH = os.path.join(DATA_PATH, 'keypoints')
FRAME_ACTIONS_PATH = os.path.join(ASSETS_PATH, 'frame_actions')
MODEL_DIR = os.path.join(ROOT_PATH, 'models')
WORDS_JSON_PATH = os.path.join(MODEL_DIR, 'words.json')
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, f'actions_{MODEL_FRAMES}.keras')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'gestos_web_compatible.tflite')
SAVEDMODEL_DIR = os.path.join(MODEL_DIR, 'saved_model_web')

# Ajustes de texto para visualización (compatibles con gesture_collector original)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# Config de voz (TTS)
SPEECH_LANGUAGE = 'es'
AUDIO_OUTPUT_FILE = 'speech.mp3'

# Mapeo palabra->texto hablado (ajusta según tus etiquetas)
words_text = {
	"bien": "BIEN",
	"feliz-cumpleanos": "FELIZ CUMPLEAÑOS",
	"buenas-noches": "BUENAS NOCHES",
	"mal": "MAL",
	"como-estas": "COMO ESTAS",
	"gracias": "GRACIAS",
	"mas-o-menos": "MAS O MENOS",
	"con-gusto": "CON GUSTO",
	"lo-siento": "LO SIENTO",
	"buenos-dias": "BUENOS DIAS",
	"bienvenido": "BIENVENIDO",
	"buenas-tardes": "BUENAS TARDES",
	"permiso": "PERMISO",
	"adios": "ADIÓS",
	"perdon": "PERDÓN",
	"sordo": "SORDO",
	"hola": "HOLA",
	"por-favor": "POR FAVOR"
}
