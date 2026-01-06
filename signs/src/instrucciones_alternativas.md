# Instrucciones Alternativas para Crear el Modelo gesture_recognizer.task

Si el entrenamiento automático del modelo falló, puedes seguir estas instrucciones manuales para crear el archivo `gesture_recognizer.task`.

## Requisitos Previos

1. Asegúrate de tener instalado Python 3.8 o superior
2. Instala MediaPipe Model Maker:

```bash
pip install mediapipe-model-maker
```

## Pasos para Crear el Modelo

1. **Prepara tus datos**: Verifica que ya hayas ejecutado `datasetCreator.py` para procesar tus imágenes

2. **Crea la estructura de directorios**: 
   - Crea un directorio temporal (por ejemplo, `temp_training`)
   - Dentro, crea subdirectorios numerados para cada clase (0, 1, 2, etc.)
   - Copia tus imágenes de `data/export/` a los directorios correspondientes según su clase

3. **Crea un script de entrenamiento**: Crea un archivo `train_model.py` con el siguiente contenido:

```python
import os
from mediapipe_model_maker import gesture_recognizer

# Define la ruta a tus datos
data_dir = 'ruta/a/tu/temp_training'

# Carga los datos
data = gesture_recognizer.Dataset.from_folder(
    data_dir,
    validation_split=0.2
)

# Entrena el modelo
model = gesture_recognizer.GestureRecognizer.create(
    train_data=data.train_data,
    validation_data=data.validation_data,
    batch_size=32,
    epochs=50
)

# Evalúa el modelo
result = model.evaluate(data.validation_data)
print(f'Exactitud: {result[1]}')

# Exporta el modelo
model.export_model('models/gesture_recognizer.task')
print("Modelo exportado correctamente.")
```

4. **Ejecuta el script**:

```bash
python train_model.py
```

5. **Verifica el modelo**: Una vez completado, deberías tener el archivo `gesture_recognizer.task` en la carpeta `models/`

## Solución de Problemas

Si encuentras errores durante el entrenamiento:

1. **Problemas de memoria**: Reduce el `batch_size` a un valor menor (por ejemplo, 16 u 8)
2. **Errores de dependencias**: Verifica que tienes instaladas las versiones correctas de tensorflow y otras dependencias
3. **Formato de imágenes**: Asegúrate de que todas las imágenes son válidas y están en formato JPG o PNG
4. **Número de muestras**: Cada clase debe tener al menos 10 imágenes para un entrenamiento efectivo

## Uso del Modelo Alternativo

Si sigues teniendo problemas, recuerda que el sistema también puede funcionar con el modelo `model.p` que ya se generó automáticamente. Este modelo no es tan avanzado como el `.task`, pero te permitirá usar el sistema mientras resuelves los problemas con MediaPipe Model Maker.
