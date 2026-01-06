import os
import logging
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import mediapipe as mp
import cv2
import subprocess
import sys
import shutil
import platform

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorios con rutas compatibles con múltiples sistemas operativos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
EXPORT_DIR = os.path.join(BASE_DIR, 'data', 'export')
ANNOTATIONS_DIR = os.path.join(EXPORT_DIR, 'annotations')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TEMP_DIR = os.path.join(BASE_DIR, 'temp_training')

# Asegurar que los directorios existen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Etiquetas
labels_dict = {
    -1: "None",  # Clase especial requerida por MediaPipe
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "ene", 15: "O", 16: "P",
    17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X",
    25: "Y", 26: "Z",
}

def collect_dataset():
    """Recolecta y procesa las imágenes para crear el dataset."""
    logging.info("Recolectando dataset desde las imágenes...")
    
    data = []
    labels = []
    
    # Configurar MediaPipe para detección de manos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    # Procesar cada directorio (cada letra/clase)
    for dir_ in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(class_dir):
            continue
            
        logging.info(f"Procesando clase: {dir_}")
        
        # Obtener índice de clase
        class_idx = None
        for idx, label in labels_dict.items():
            if label.lower() == dir_.lower():
                class_idx = idx
                break
        
        if class_idx is None:
            logging.warning(f"Clase {dir_} no encontrada en labels_dict. Asignando valor numérico.")
            try:
                class_idx = int(dir_)
            except ValueError:
                class_idx = len(set(labels))  # Asignar siguiente índice disponible
        
        # Procesar cada imagen del directorio
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Leer imagen
            img = cv2.imread(img_path)
            if img is None:
                logging.error(f"No se pudo leer la imagen: {img_path}")
                continue
            
            # Procesar con MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extraer coordenadas
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    
                    # Normalizar y aplanar características
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(x_[i] - min(x_))
                        data_aux.append(y_[i] - min(y_))
                    
                    # Verificar dimensiones correctas (21 landmarks x 2 coordenadas)
                    if len(data_aux) == 42:
                        data.append(data_aux)
                        labels.append(class_idx)
    
    # Cerrar MediaPipe
    hands.close()
    
    logging.info(f"Dataset recolectado: {len(data)} muestras de {len(set(labels))} clases")
    return np.array(data), np.array(labels)

def train_sklearn_model(data, labels):
    """Entrena un modelo de scikit-learn y lo guarda como archivo pickle."""
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Entrenar modelo MLP (mejor para gestos de manos)
    params = {
        'hidden_layer_sizes': (100,),
        'max_iter': 500,
        'alpha': 0.0001,
        'solver': 'adam',
        'verbose': 10,
        'random_state': 42,
        'learning_rate_init': 0.001
    }
    model = MLPClassifier(**params)
    
    logging.info("Entrenando modelo MLP...")
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Precisión del modelo: {accuracy:.4f}")
    
    # Guardar modelo junto con el diccionario de etiquetas
    pickle_path = os.path.join(MODEL_DIR, 'model.p')
    with open(pickle_path, 'wb') as f:
        pickle.dump({'model': model, 'labels_dict': labels_dict}, f)
    
    logging.info(f"Modelo guardado en: {pickle_path}")
    
    # También guardar el dataset procesado para uso futuro
    data_path = os.path.join(MODEL_DIR, 'data.pickle')
    with open(data_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    logging.info(f"Dataset guardado en: {data_path}")
    return model, accuracy

def get_label_from_index(index):
    """Convierte un índice numérico a su etiqueta de texto correspondiente."""
    return labels_dict.get(index, f"Desconocido({index})")

# Función auxiliar para cargar el modelo y hacer predicciones con etiquetas
def load_model_and_predict(features):
    """Carga el modelo y predice la etiqueta en formato texto."""
    model_path = os.path.join(MODEL_DIR, 'model.p')
    if not os.path.exists(model_path):
        logging.error(f"No se encontró el modelo en {model_path}")
        return "Modelo no encontrado"
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        # Usar el diccionario de etiquetas guardado con el modelo, o el global si no existe
        labels_map = data.get('labels_dict', labels_dict)
    
    # Hacer la predicción y convertir a etiqueta de texto
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return labels_map.get(prediction, f"Desconocido({prediction})")

def check_mediapipe_model_maker():
    """Verifica si MediaPipe Model Maker está instalado."""
    try:
        import mediapipe_model_maker
        logging.info("MediaPipe Model Maker está instalado.")
        return True
    except ImportError:
        logging.warning("MediaPipe Model Maker no está instalado.")
        return False

def prepare_directory_structure():
    """Prepara la estructura de directorios para entrenar un modelo MediaPipe."""
    # Limpiar el directorio temporal
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Crear directorios para cada clase
    for idx, label in labels_dict.items():
        os.makedirs(os.path.join(TEMP_DIR, str(idx)), exist_ok=True)
    
    # Crear directorio especial "None" requerido por MediaPipe
    none_dir = os.path.join(TEMP_DIR, "None")
    os.makedirs(none_dir, exist_ok=True)
    
    # Buscar imágenes en EXPORT_DIR y copiarlas a la estructura correcta
    contador_images = 0
    for item in os.listdir(EXPORT_DIR):
        if item.endswith(".jpg") or item.endswith(".png"):
            # Obtener la clase de la imagen
            parts = item.split("_")
            if len(parts) > 0:
                class_name = parts[0]
                
                # Encontrar el índice de la clase
                class_idx = None
                for idx, label in labels_dict.items():
                    if label.lower() == class_name.lower():
                        class_idx = idx
                        break
                
                if class_idx is None:
                    try:
                        class_idx = int(class_name)
                    except ValueError:
                        logging.warning(f"No se pudo determinar la clase para la imagen: {item}")
                        continue
                
                # Copiar la imagen al directorio correspondiente
                source = os.path.join(EXPORT_DIR, item)
                target_dir = os.path.join(TEMP_DIR, str(class_idx))
                target = os.path.join(target_dir, item)
                shutil.copy2(source, target)
                contador_images += 1
    
    # Si no existen imágenes para la clase "None", crear algunas artificiales
    none_images_count = len(os.listdir(none_dir))
    if none_images_count == 0:
        logging.warning("No se encontraron imágenes para la clase 'None'. Creando imágenes artificiales...")
        # Crear 10 imágenes artificiales sin manos para la clase "None"
        for i in range(10):
            # Crear una imagen negra
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            # Guardar la imagen
            cv2.imwrite(os.path.join(none_dir, f"none_{i}.jpg"), img)
        logging.info(f"Se crearon 10 imágenes artificiales para la clase 'None'")
        contador_images += 10
    
    logging.info(f"Se prepararon {contador_images} imágenes para el entrenamiento.")
    return contador_images > 0

def train_mediapipe_task_model():
    """Entrena un modelo MediaPipe GestureRecognizer y genera el archivo .task."""
    # Verificar si MediaPipe Model Maker está instalado
    if not check_mediapipe_model_maker():
        logging.error("MediaPipe Model Maker no está instalado. Instalándolo...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe-model-maker"])
            logging.info("MediaPipe Model Maker instalado correctamente.")
        except subprocess.CalledProcessError:
            logging.error("No se pudo instalar MediaPipe Model Maker. El modelo .task no se generará.")
            return False
    
    # Preparar la estructura de directorios
    if not prepare_directory_structure():
        logging.error("No se pudieron preparar las imágenes para el entrenamiento. Verifica que existan imágenes en data/export/")
        return False
    
    # Generar y ejecutar el script de entrenamiento
    script_path = os.path.join(TEMP_DIR, "train_script.py")
    with open(script_path, 'w') as f:
        f.write("""
import os
from mediapipe_model_maker import gesture_recognizer

# Define la ruta a tus datos
data_dir = '{}'
model_dir = '{}'  # Directorio para guardar el modelo exportado

try:
    # Carga los datos - MediaPipe requiere una clase "None" en el dataset
    data = gesture_recognizer.Dataset.from_folder(data_dir)
    
    # Imprimir clases encontradas para depuración
    print("Clases encontradas:", data.label_names)
    
    # Dividir manualmente en conjuntos de entrenamiento y validación (80/20)
    train_data, validation_data = data.split(0.8)
    
    # Crear objeto de opciones como se muestra en el notebook
    hparams = gesture_recognizer.HParams(
        export_dir=model_dir,               
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
    
    # Intentar crear el modelo con las opciones correctas
    print("Creando modelo con API actual de MediaPipe...")
    model = gesture_recognizer.GestureRecognizer.create(
        options=options,
        train_data=train_data,
        validation_data=validation_data
    )
    
    # Evalúa el modelo
    result = model.evaluate(validation_data)
    print(f'Exactitud: {{result[1]}}')
    
    # Exporta el modelo
    model.export_model()
    print("Modelo exportado correctamente.")
except Exception as e:
    print(f"Error durante el entrenamiento: {{e}}")
    import traceback
    traceback.print_exc()
    raise e
""".format(TEMP_DIR, MODEL_DIR))
    
    logging.info("Ejecutando script de entrenamiento MediaPipe Model Maker...")
    try:
        subprocess.check_call([sys.executable, script_path])
        logging.info("Modelo MediaPipe entrenado y exportado correctamente.")
        
        # Verificar si el modelo se generó
        if os.path.exists(os.path.join(MODEL_DIR, "gesture_recognizer.task")):
            logging.info("Archivo .task generado correctamente.")
            return True
        else:
            logging.error("El archivo .task no se generó. Revisa los errores anteriores.")
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error al entrenar el modelo MediaPipe: {e}")
        return False

def main():
    """Función principal que ejecuta el proceso de entrenamiento completo."""
    logging.info("Iniciando proceso de entrenamiento...")
    
    # 1. Recolectar dataset para el modelo scikit-learn
    data, labels = collect_dataset()
    
    if len(data) == 0 or len(labels) == 0:
        logging.error("No se pudieron recolectar datos. Verifica que existan imágenes en data/raw/")
        return
    
    # 2. Entrenar modelo scikit-learn (para uso inmediato)
    model, accuracy = train_sklearn_model(data, labels)
    
    # 3. Intentar entrenar modelo MediaPipe (para generar .task)
    logging.info("Iniciando entrenamiento de modelo MediaPipe GestureRecognizer...")
    if train_mediapipe_task_model():
        logging.info("¡Éxito! El modelo gesture_recognizer.task ha sido creado.")
    else:
        logging.warning("No se pudo generar el modelo .task automáticamente.")
        
        # Generar instrucciones manuales como alternativa
        mediapipe_instructions_path = os.path.join(MODEL_DIR, "mediapipe_instructions.txt")
        with open(mediapipe_instructions_path, 'w') as f:
            f.write("# Instrucciones para entrenar un modelo MediaPipe GestureRecognizer\n\n")
            f.write("Para entrenar un modelo personalizado de reconocimiento de gestos, sigue estos pasos:\n\n")
            f.write("1. Instala las herramientas de MediaPipe Model Maker:\n")
            f.write("   ```\n   pip install mediapipe-model-maker\n   ```\n\n")
            f.write("2. Utiliza las imágenes procesadas en: " + EXPORT_DIR + "\n\n")
            f.write("3. IMPORTANTE: MediaPipe requiere una categoría especial llamada 'None' con imágenes que representan la ausencia de gestos.\n\n")
            f.write("4. Ejecuta el siguiente código Python:\n")
            f.write("   ```python\n")
            f.write("   from mediapipe_model_maker import gesture_recognizer\n")
            f.write("   import os\n")
            f.write("   import numpy as np\n")
            f.write("   import cv2\n\n")
            f.write("   # Define la ruta a tus datos\n")
            f.write("   data_dir = '" + TEMP_DIR + "'\n\n")
            f.write("   # Asegúrate de tener una carpeta 'None' con imágenes\n")
            f.write("   none_dir = os.path.join(data_dir, 'None')\n")
            f.write("   os.makedirs(none_dir, exist_ok=True)\n")
            f.write("   if len(os.listdir(none_dir)) == 0:\n")
            f.write("       # Crear imágenes artificiales para la clase None\n")
            f.write("       for i in range(10):\n")
            f.write("           img = np.zeros((224, 224, 3), dtype=np.uint8)\n")
            f.write("           cv2.imwrite(os.path.join(none_dir, f'none_{i}.jpg'), img)\n")
            f.write("       print('Creadas imágenes para clase None')\n\n")
            f.write("   # Carga los datos\n")
            f.write("   data = gesture_recognizer.Dataset.from_folder(data_dir)\n")
            f.write("   print('Clases encontradas:', data.label_names)\n\n")
            f.write("   # Dividir manualmente en conjuntos de entrenamiento y validación\n")
            f.write("   train_data, validation_data = data.split(0.8)\n\n")
            f.write("   # Configurar opciones del modelo y entrenamiento\n")
            f.write("   hparams = gesture_recognizer.HParams(\n")
            f.write("       export_dir='" + os.path.join(MODEL_DIR) + "',\n")
            f.write("       batch_size=4,\n")
            f.write("       epochs=20\n")
            f.write("   )\n")
            f.write("   model_options = gesture_recognizer.ModelOptions(dropout_rate=0.1)\n")
            f.write("   options = gesture_recognizer.GestureRecognizerOptions(\n")
            f.write("       model_options=model_options,\n")
            f.write("       hparams=hparams\n")
            f.write("   )\n\n")
            f.write("   # Entrena el modelo con options como primer parámetro\n")
            f.write("   model = gesture_recognizer.GestureRecognizer.create(\n")
            f.write("       options=options,\n")
            f.write("       train_data=train_data,\n")
            f.write("       validation_data=validation_data\n")
            f.write("   )\n\n")
            f.write("   # Evalúa el modelo\n")
            f.write("   result = model.evaluate(validation_data)\n")
            f.write("   print(f'Exactitud: {result[1]}')\n\n")
            f.write("   # Exporta el modelo\n")
            f.write("   model.export_model()\n")
            f.write("   ```\n\n")
            f.write("5. Una vez completado, coloca el archivo gesture_recognizer.task en: " + MODEL_DIR + "\n")
        
        logging.info(f"Se han generado instrucciones manuales en: {mediapipe_instructions_path}")
    
    logging.info("Proceso de entrenamiento completado.")
    if os.path.exists(os.path.join(MODEL_DIR, "model.p")):
        logging.info(f"Modelo scikit-learn entrenado con precisión: {accuracy:.4f}")
    
    # Realizar limpieza final
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        logging.info("Archivos temporales eliminados.")

if __name__ == "__main__":
    main()
