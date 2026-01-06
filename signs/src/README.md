# Guía para Crear tu Propio Modelo de Reconocimiento de Señas

Este documento explica cómo crear y entrenar tu propio modelo de reconocimiento de señas
utilizando las imágenes en `data/raw`.

## Preparación

Asegúrate de tener las siguientes carpetas:

- `data/raw`: Debe contener subcarpetas, cada una nombrada con una letra ("A", "B", etc.)
  y dentro de cada subcarpeta, imágenes de manos haciendo esa seña.
- `models`: Carpeta donde se guardarán los modelos entrenados.

## Flujo de Trabajo

### 1. Prepara tu conjunto de datos

Organiza tus imágenes en `data/raw` de la siguiente manera:
```
data/
└── raw/
    ├── A/
    │   ├── imagen1.jpg
    │   ├── imagen2.jpg
    │   └── ...
    ├── B/
    │   ├── imagen1.jpg
    │   └── ...
    └── ...
```

### 2. Procesa las imágenes

Ejecuta `datasetCreator.py` para procesar tus imágenes y preparar los datos:

```bash
python datasetCreator.py
```

Este script:
- Detecta manos en cada imagen
- Extrae los puntos de referencia (landmarks)
- Guarda las imágenes procesadas en `data/export`
- Crea anotaciones JSON en `data/export/annotations`
- Guarda los landmarks en la base de datos

### 3. Entrena el modelo

Ejecuta `Trainer.py` para entrenar tu modelo:

```bash
python Trainer.py
```

Este script:
- Recolecta datos de tus imágenes
- Entrena un modelo scikit-learn (guardado como `model.p` en `models/`)
- Proporciona instrucciones para entrenar un modelo MediaPipe avanzado

### 4. Utiliza tu modelo entrenado

Hay dos opciones:

#### Opción 1: Usar el modelo scikit-learn (`model.p`)

El reconocimiento funcionará automáticamente con este modelo, pero es menos avanzado.

#### Opción 2: Crear un modelo MediaPipe avanzado

Para crear un modelo `.task` avanzado:
1. Sigue las instrucciones en `models/mediapipe_instructions.txt`
2. Instala MediaPipe Model Maker
3. Ejecuta el código Python proporcionado
4. Coloca el modelo resultante como `gesture_recognizer.task` en `models/`

### 5. Prueba tu modelo

Ejecuta `clasifierCamera.py` para probar tu modelo con la cámara:

```bash
python clasifierCamera.py
```

## Crear/actualizar el modelo .task (MediaPipe)

- Requisitos (Windows PowerShell):
  - Python 3.10–3.11 recomendado
  - Instalar dependencias: `pip install -r "requirements Stable.txt"`

- Pasos:
  1) Procesar imágenes (si no lo hiciste):
     - `python datasetCreator.py`
  2) Entrenar y exportar el modelo MediaPipe (.task):
     - `python create_task_model.py`
     - El archivo `models/gesture_recognizer.task` será creado/actualizado.
  3) Probar con la cámara:
     - `python clasifierCamera.py`

Notas:
- El código usa directamente MediaPipe Tasks (API moderna) cuando `models/gesture_recognizer.task` existe.
- Si no existe el .task, se usa el modelo tradicional `models/model.p` si está disponible.

## Mejorando tu Modelo

Para mejorar tu modelo:

1. **Más datos**: Añade más imágenes a tus carpetas en `data/raw`
2. **Diversidad**: Incluye variaciones en iluminación, fondos y posiciones
3. **Equilibrio**: Asegúrate de tener un número similar de imágenes para cada letra
4. **Precisión**: Asegúrate de que las manos estén claramente visibles en las imágenes



// nuevo farma
Organizar tus imágenes por letra/clase en data/raw
Procesar estas imágenes con datasetCreator.py
Entrenar un modelo con Trainer.py (se creará automáticamente un modelo scikit-learn)
SOLO.TASK python create_task_model.py
Opcionalmente, seguir las instrucciones para crear un modelo MediaPipe avanzado
Probar el modelo con clasifierCamera.py