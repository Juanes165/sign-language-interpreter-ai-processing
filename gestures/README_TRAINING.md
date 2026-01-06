# 🧠 Guía Completa de Entrenamiento - Desde Web Contributions hasta Modelo Final

**Guía paso a paso para procesar contribuciones web y entrenar modelos LSTM con normalización**

---

## 📋 Resumen del Flujo

```
1. Descargar Contribuciones Web (Google Drive)
   ↓
2. Convertir JSON → .npy
   ↓
3. Calcular Estadísticas de Normalización ⭐ IMPORTANTE
   ↓
4. Entrenar Modelo (Node.js v6 o Python v2)
   ↓
5. Modelo Listo para Uso
```

---

## 🚀 Paso 1: Descargar Contribuciones Web

Las contribuciones vienen desde el frontend web y se almacenan en Google Drive.

### Opción A: Desde Google Drive (Recomendado)

```bash
cd gesto_releasev1
node scripts/download_from_drive.js
```

**Configuración:**
- Credenciales: `unavoz-bb3744af7f68.json` (en la raíz)
- Service Account: `unavoz@unavoz.iam.gserviceaccount.com`
- Folder ID: `1zkP5QPXCZU1nM2hL11r6VIzK0053yNtb`

**Salida esperada:**
```
🔐 Autenticando con Google Drive...
📥 Descargando archivos...
✅ Descarga completa: 100 archivos
```

**Archivos guardados en:** `assets/web_contributions/*.json`

### Opción B: Desde Frontend Local

Si tienes el frontend corriendo localmente, los archivos están en:
```
sign-language-interpreter-frontend/captured_samples/*.json
```

---

## 🔄 Paso 2: Convertir JSON → .npy

Convierte las contribuciones web a formato .npy para entrenamiento:

```bash
# Si las contribuciones están en assets/web_contributions/
python scripts/convert_frontend_samples_to_npy.py assets/web_contributions

# O desde el directorio del frontend
python scripts/convert_frontend_samples_to_npy.py ../../sign-language-interpreter-frontend
```

**Proceso:**
- ✅ Lee todos los JSON del directorio
- ✅ Agrupa por gesto
- ✅ Normaliza a 15 frames (padding o muestreo uniforme)
- ✅ Guarda como `.npy` en `assets/data/keypoints/`
- ✅ Actualiza `models/words.json` con los gestos encontrados

**Salida esperada:**
```
🌐 Conversor: Frontend Samples → NumPy

📊 Total de archivos: 100
✅ Muestras válidas: 100

📋 Resumen por gesto:
   - hola: 15 muestras
   - bien: 12 muestras
   - gracias: 18 muestras
   ...

📦 Procesando gestos...
✅ Conversión completada
```

**Archivos generados:**
```
assets/data/keypoints/
├── hola.npy
├── bien.npy
├── gracias.npy
└── ...
```

---

## 📊 Paso 3: Calcular Estadísticas de Normalización ⭐ IMPORTANTE

**Este paso es CRÍTICO** para un buen entrenamiento. Calcula las estadísticas necesarias para normalizar los datos:

```bash
python src/calculate_normalization_stats.py
```

**¿Por qué normalizar?**
- ✅ Mejora la convergencia del modelo
- ✅ Reduce el tiempo de entrenamiento
- ✅ Mejora la precisión final
- ✅ Normaliza por componente (Pose, Face, Hands)

**Salida esperada:**
```
📊 Calculando estadísticas de normalización...

📁 Procesando archivos .npy en: assets/data/keypoints
✅ Archivos encontrados: 18

📊 Calculando estadísticas por componente:
   - Pose: 132 keypoints
   - Face: 1404 keypoints
   - Left Hand: 63 keypoints
   - Right Hand: 63 keypoints

✅ Estadísticas guardadas en: models/normalization_stats.json
```

**Archivo generado:**
```
models/normalization_stats.json
```

**⚠️ IMPORTANTE:** Este paso debe ejecutarse ANTES de entrenar. El entrenador lo requiere.

---

## 🧠 Paso 4: Entrenar Modelo

Ahora puedes entrenar usando cualquiera de los dos trainers disponibles.

---

### 🌐 Opción A: Node.js v6 (Recomendado para Web)

**Trainer:** `src/train_lstm_node_v6.js`

#### Instalación

```bash
npm install
```

#### Ejecución

```bash
# Auto-detección de backend (GPU → CPU → JS)
node src/train_lstm_node_v6.js

# Forzar GPU (si tienes NVIDIA + cuDNN)
node src/train_lstm_node_v6.js --gpu

# Forzar CPU (acelerado con TensorFlow C++)
node src/train_lstm_node_v6.js --cpu

# Forzar JavaScript puro (más lento)
node src/train_lstm_node_v6.js --js
```

#### Arquitectura del Modelo

```
Input: [batch, 15 frames, 1662 keypoints]
   ↓
LSTM(256) + Dropout(0.3) + BatchNorm
   ↓
LSTM(128) + Dropout(0.3) + BatchNorm
   ↓
Dense(32, ReLU) + Dropout(0.3)
   ↓
Dense(num_classes, Softmax)
```

**Parámetros optimizados (TOP 1 Grid Search):**
- LSTM 1: 256 units
- LSTM 2: 128 units
- Dense: 32 units
- Dropout: 0.3
- Recurrent Dropout: 0.2
- L2 Regularization: 0.001
- Learning Rate: 0.0001
- Batch Size: 32

#### Salida Esperada

```
🧠 Entrenamiento LSTM v6.0 - Parámetros Optimizados

📋 Gestos: 18
📊 Distribución estratificada por gesto:
   hola: Train=50, Val=15, Test=8 (Total=73)
   bien: Train=45, Val=13, Test=7 (Total=65)
   ...

🏗️  Construyendo modelo LSTM v6.0...
   Arquitectura: LSTM[256, 128] → Dense[32] → Softmax[18]

🚀 Iniciando entrenamiento...

Epoch 125/200
loss: 0.239 - acc: 0.979 - val_loss: 0.047 - val_acc: 0.995

⏹️  EARLY STOPPING ACTIVADO
   Mejor val_loss: 0.0467 (epoch 125)

📊 Evaluando modelo en conjunto de prueba...
✅ Test Accuracy: 97.94%

💾 Guardando modelo...
✅ Modelo guardado en: models/modelo_tfjs_node
✅ Matriz de confusión guardada en: models/confusion_matrix_v6.json
✅ Reporte guardado en: models/training_report_v6.json
```

#### Archivos Generados

```
models/
├── modelo_tfjs_node/
│   ├── model.json              # Arquitectura del modelo
│   ├── weights.bin             # Pesos del modelo
│   └── words.json              # Etiquetas de gestos
├── normalization_stats.json     # Estadísticas (ya existente)
├── training_report_v6.json     # Reporte completo de entrenamiento
└── confusion_matrix_v6.json    # Matriz de confusión
```

---

### 🐍 Opción B: Python v2 (Recomendado para GPU)

**Trainer:** `src/train_lstm_actions_v2.py`

#### Instalación

```bash
# Crear entorno virtual
python -m venv env
source env/bin/activate  # Linux/Mac
.\env\Scripts\Activate.ps1  # Windows

# Instalar dependencias
pip install -r requirements_v2.txt
```

#### Ejecución

```bash
python src/train_lstm_actions_v2.py
```

El script detecta automáticamente GPU si está disponible.

#### Arquitectura del Modelo

Igual que Node.js v6 (mismos parámetros optimizados).

#### Salida Esperada

```
🧠 Entrenamiento de LSTM para reconocimiento de gestos en Python
VERSIÓN v2.0

🚀 GPU detectada y configurada para uso
📋 Gestos: 18

📊 Evaluando modelo...
✅ Test Accuracy: 97.94%

📊 INFORME DETALLADO POR GESTO
   - Métricas por gesto (precision, recall, F1)
   - Matriz de confusión
   - Análisis de rendimiento

💾 Informe guardado en: models/training_report.json
✅ Modelo Keras guardado en: models/actions_15.keras
✅ Modelo exportado a: models/modelo_tfjs_node/
```

#### Archivos Generados

```
models/
├── actions_15.keras            # Modelo Keras (solo Python)
├── modelo_tfjs_node/           # Modelo TensorFlow.js
│   ├── model.json
│   ├── weights.bin
│   └── words.json
└── training_report.json        # Reporte completo
```

---

## 📊 Paso 5: Verificar Resultados

### Reporte de Entrenamiento

Ambos trainers generan reportes detallados:

**Node.js v6:** `models/training_report_v6.json`
**Python v2:** `models/training_report.json`

**Contenido del reporte:**
- Métricas generales (accuracy, loss)
- Métricas por gesto (precision, recall, F1, support)
- Historial de entrenamiento
- Matriz de confusión (solo Node.js v6)

### Matriz de Confusión

Solo Node.js v6 genera: `models/confusion_matrix_v6.json`

```json
{
  "version": "v6.0",
  "gestures": ["hola", "bien", ...],
  "matrix": [[...], [...]]
}
```

---

## 🔧 Configuración Avanzada

### Ajustar Parámetros de Entrenamiento

#### Node.js v6

Edita `src/train_lstm_node_v6.js`:

```javascript
const CONFIG = {
  EPOCHS: 200,
  BATCH_SIZE: 32,
  LEARNING_RATE: 0.0001,
  VALIDATION_SPLIT: 0.2,
  TEST_SPLIT: 0.1,
  EARLY_STOPPING_PATIENCE: 10,
};
```

#### Python v2

Edita `src/train_lstm_actions_v2.py`:

```python
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
```

---

## 🐛 Solución de Problemas

### Error: "Archivo de normalización no encontrado"

**Solución:**
```bash
# Ejecutar paso 3 primero
python src/calculate_normalization_stats.py
```

### Support muy bajo en test

**Causa:** División no estratificada (solo en versiones antiguas)

**Solución:** Node.js v6 ya usa split estratificado automáticamente.

### Entrenamiento muy lento (Node.js)

**Soluciones:**
1. Instalar `@tensorflow/tfjs-node` para aceleración CPU
2. Instalar `@tensorflow/tfjs-node-gpu` para GPU (requiere cuDNN)
3. Usar Python v2 con GPU (más rápido)

### GPU no funciona (Python)

**Verificar:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Si no aparece GPU:**
1. Verificar CUDA instalado: `nvidia-smi`
2. Reinstalar TensorFlow con soporte GPU:
   ```bash
   pip install tensorflow[and-cuda]
   ```

---

## 📈 Métricas de Calidad

### Dataset Saludable

```
✅ Mínimo 30 muestras por gesto
✅ Total: 500+ muestras
✅ Balance: ±20% entre clases
✅ Normalización aplicada
```

### Modelo Entrenado

```
✅ Test accuracy > 95%
✅ Validation accuracy > 90%
✅ Loss decreciente y estable
✅ No overfitting (gap train/val < 10%)
✅ Support equilibrado en test (split estratificado)
```

---

## 🎯 Workflow Completo (Resumen)

```bash
# 1. Descargar contribuciones web
node scripts/download_from_drive.js

# 2. Convertir JSON → .npy
python scripts/convert_frontend_samples_to_npy.py assets/web_contributions

# 3. Calcular estadísticas de normalización
python src/calculate_normalization_stats.py

# 4a. Entrenar con Node.js v6
npm install
node src/train_lstm_node_v6.js

# O 4b. Entrenar con Python v2
pip install -r requirements_v2.txt
python src/train_lstm_actions_v2.py

# 5. Verificar resultados
# Revisar: models/training_report_v6.json o models/training_report.json
```

---

## 📚 Referencias

- [TensorFlow.js](https://www.tensorflow.org/js)
- [Keras](https://keras.io/)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)

---

**Última actualización**: Noviembre 2025  
**Versión**: v6.0  
**Estado**: Producción ✅
