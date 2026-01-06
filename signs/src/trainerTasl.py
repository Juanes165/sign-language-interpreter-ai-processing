import os
import logging
import numpy as np
import cv2
import shutil
import subprocess
import sys

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, 'data', 'export')  # Directorio con las imágenes exportadas
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # Donde se guardará el modelo .task
TEMP_DIR = os.path.join(BASE_DIR, 'temp_training')  # Directorio temporal para el entrenamiento

# Asegurar que los directorios existen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Etiquetas (diccionario de clases)
labels_dict = {
    -1: "None",  # Clase especial requerida por MediaPipe
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "ene", 15: "O", 16: "P",
    17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X",
    25: "Y", 26: "Z",
}

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
        for i in range(10):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(none_dir, f"none_{i}.jpg"), img)
        logging.info("Se crearon 10 imágenes artificiales para la clase 'None'")
        contador_images += 10
    
    logging.info(f"Se prepararon {contador_images} imágenes para el entrenamiento.")
    return contador_images > 0

def create_task_model():
    """Crea un modelo MediaPipe GestureRecognizer y genera el archivo .task."""
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
        f.write(f"""
import os
from mediapipe_model_maker import gesture_recognizer

# Define la ruta a los datos
data_dir = '{TEMP_DIR}'

try:
    # Carga los datos
    data = gesture_recognizer.Dataset.from_folder(data_dir)
    print("Clases encontradas:", data.label_names)
    
    # División de datos (80% entrenamiento, 20% validación)
    train_data, validation_data = data.split(0.8)
    
    # Configuración del modelo
    hparams = gesture_recognizer.HParams(
        export_dir='{MODEL_DIR}',
        batch_size=64,               
        epochs=500,                  
        learning_rate=0.01,          
        shuffle=True,                
        use_augmentation=True,       
        warmup_epochs=5,             
        momentum=0.9                 
    )

    model_options = gesture_recognizer.ModelOptions(
        dropout_rate=0.01,           
        layer_widths=[50]            
    )
    
    options = gesture_recognizer.GestureRecognizerOptions(
        model_options=model_options, 
        hparams=hparams
    )
    
    # Creación y entrenamiento del modelo
    model = gesture_recognizer.GestureRecognizer.create(
        options=options,
        train_data=train_data,
        validation_data=validation_data
    )
    
    # Evaluación
    result = model.evaluate(validation_data)
    print(f'Exactitud: {{result[1]}}')
    
    # Exportación del modelo
    model.export_model()
    print("Modelo exportado correctamente.")
except Exception as e:
    print(f"Error durante el entrenamiento: {{e}}")
    import traceback
    traceback.print_exc()
    raise e
""")
    
    logging.info("Ejecutando script de entrenamiento...")
    try:
        subprocess.check_call([sys.executable, script_path])
        
        # Verificar si el modelo se generó
        if os.path.exists(os.path.join(MODEL_DIR, "gesture_recognizer.task")):
            logging.info("¡Éxito! El archivo gesture_recognizer.task ha sido creado.")
            return True
        else:
            logging.error("El archivo .task no se generó. Revisa los errores anteriores.")
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error al entrenar el modelo: {e}")
        return False

if __name__ == "__main__":
    success = create_task_model()
    
    # Limpiar archivos temporales
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        logging.info("Archivos temporales eliminados.")
    
    if success:
        logging.info(f"Modelo .task guardado en: {os.path.join(MODEL_DIR, 'gesture_recognizer.task')}")
    else:
        logging.error("No se pudo crear el modelo .task")