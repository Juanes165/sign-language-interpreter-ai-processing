# Guía para Descargar y Usar Modelos MediaPipe

Para usar el sistema de reconocimiento de señas, necesitas descargar los modelos necesarios de MediaPipe.

## Modelos Necesarios

1. **Hand Landmarker**: Para detectar los landmarks de las manos
2. **Gesture Recognizer**: Para reconocer gestos (opcional)

## Cómo Descargar los Modelos

### Hand Landmarker

1. Visita la [página de Hand Landmarker de MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
2. Desplázate hasta la sección "Models"
3. Descarga el archivo "hand_landmarker.task"
4. Coloca este archivo en la carpeta `models/` de tu proyecto

### Gesture Recognizer (Opcional)

1. Visita la [página de Gesture Recognizer de MediaPipe](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)
2. Desplázate hasta la sección "Models"
3. Descarga el archivo "gesture_recognizer.task"
4. Coloca este archivo en la carpeta `models/` de tu proyecto

## Estructura de Carpetas

Asegúrate de que tu estructura de carpetas se vea así:

```
Señas/
├── models/
│   ├── hand_landmarker.task  (Necesario para datasetCreator.py)
│   ├── gesture_recognizer.task  (Opcional, para clasifierCamera.py)
│   ├── model.p  (Generado por Trainer.py)
│   └── data.pickle  (Generado por datasetCreator.py)
├── data/
│   ├── raw/  (Tus imágenes de entrenamiento)
│   └── export/  (Generado por datasetCreator.py)
└── src/
    ├── clasifierCamera.py
    ├── datasetCreator.py
    ├── Trainer.py
    └── ...
```

## Alternativa

Si no quieres usar los modelos .task de MediaPipe, el sistema también puede funcionar con el modelo tradicional:

1. Entrena tu propio modelo con `Trainer.py`
2. Esto generará un archivo `model.p` en la carpeta `models/`
3. El sistema usará automáticamente este modelo si no encuentra los modelos de MediaPipe
