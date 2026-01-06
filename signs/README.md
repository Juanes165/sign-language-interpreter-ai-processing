# Sistema de Reconocimiento de Lenguaje de Señas

Este sistema utiliza MediaPipe GestureRecognizer para detectar y reconocer letras del lenguaje de señas en tiempo real.

## Requisitos previos

1. Python 3.8 o superior
2. Dependencias (instalar con pip):

```
pip install mediapipe opencv-python kivy transformers torch sqlalchemy python-dotenv
```

## Estructura del proyecto

- `src/` - Código fuente
  - `menu.py` - Interfaz gráfica del sistema
  - `clasifierCamera.py` - Reconocimiento en tiempo real
  - `datasetCreator.py` - Creación de datasets
  - `Trainer.py` - Simulación de entrenamiento
  - `models/` - Modelos de datos
- `models/` - Directorio para guardar modelos entrenados
- `data/` - Datasets e imágenes

## Modelo GestureRecognizer

Para que el sistema funcione, necesitas un modelo GestureRecognizer en formato .task. Puedes:

1. Descargar un modelo preentrenado desde MediaPipe:
   - https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer

2. Entrenar tu propio modelo:
   - https://developers.google.com/mediapipe/solutions/model_maker/gesture_recognizer

Coloca el archivo .task en la carpeta `models/` con el nombre `gesture_recognizer.task`.

## Ejecución

1. Ejecuta `python src/menu.py` para iniciar la interfaz
2. Usa los botones para acceder a las diferentes funcionalidades

## Funcionalidades

- **Capturar Imágenes**: Captura de imágenes para el dataset
- **Crear DataSet**: Procesa imágenes y crea anotaciones
- **Entrenar Modelo**: Simulación del proceso de entrenamiento
- **Reconocer Gestos**: Detecta lenguaje de señas en tiempo real

2. **Crea y activa un entorno virtual:**

    ```bash
    py -3.10 -m venv env
    python -m venv venv
    # En Windows
    env\Scripts\activate
    # En macOS y Linux
    source venv/bin/activate
    ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

```bash
python collector.py

python datasetCreator.py

python Trainer.py

python clasifierCamera.py


cd /mnt/c/Users/geide/Documents/GitHub/Tesis/Señas/src