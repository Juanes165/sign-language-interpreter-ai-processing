# 🚀 Pipeline de Entrenamiento LSTM - Reconocimiento de Gestos

Pipeline completo para entrenar modelos LSTM de reconocimiento de gestos en lengua de señas desde contribuciones web hasta modelos listos para producción.

---

## 📋 Flujo Completo

```
1. Descargar Contribuciones Web (Google Drive)
   ↓
2. Convertir JSON → .npy
   ↓
3. Calcular Estadísticas de Normalización
   ↓
4. Entrenar Modelo (Node.js v6 o Python v2)
   ↓
5. Modelo Listo para Uso
```

**📖 [Ver Guía Completa de Entrenamiento →](README_TRAINING.md)**

---

## 🎯 Dos Opciones de Entrenamiento

### 🌐 Node.js v6 (Recomendado para Web)

**Trainer:** `src/train_lstm_node_v6.js`

**Ideal para:**
- ✅ Aplicaciones web/navegador
- ✅ Integración con Next.js/React
- ✅ Modelo TensorFlow.js nativo
- ✅ Sin conversión de formatos

**Ventajas:**
- Exporta directo a TensorFlow.js
- Mismo ecosistema JavaScript
- Integración simple con frontend

**Desventajas:**
- Más lento que Python (sin GPU)
- Requiere Node.js instalado

---

### 🐍 Python v2 (Recomendado para GPU)

**Trainer:** `src/train_lstm_actions_v2.py`

**Ideal para:**
- ✅ Máxima velocidad (GPU NVIDIA)
- ✅ Aplicaciones de escritorio
- ✅ Prototipado rápido
- ✅ Entrenamiento intensivo

**Ventajas:**
- Muy rápido con GPU
- Soporte completo de CUDA
- Métricas detalladas

**Desventajas:**
- Requiere conversión a TF.js para web
- Configuración más compleja

---

## 🚀 Inicio Rápido

### Opción 1: Node.js v6

```bash
# 1. Instalar dependencias
npm install

# 2. Seguir guía completa desde web_contributions
# Ver: README_TRAINING.md
```

### Opción 2: Python v2

```bash
# 1. Crear entorno virtual
python -m venv env
source env/bin/activate  # Linux/Mac
.\env\Scripts\Activate.ps1  # Windows

# 2. Instalar dependencias
pip install -r requirements_v2.txt

# 3. Seguir guía completa desde web_contributions
# Ver: README_TRAINING.md
```

---

## 📊 Comparación Rápida

| Característica | Node.js v6 | Python v2 |
|----------------|------------|-----------|
| **Velocidad** | 🐌 Lento (CPU) | 🚀🚀🚀 Muy rápido (GPU) |
| **GPU NVIDIA** | ⚠️ Requiere cuDNN | ✅ Excelente soporte |
| **Facilidad** | ⭐⭐⭐⭐⭐ Muy simple | ⭐⭐⭐ Simple |
| **Output** | TF.js nativo | Keras + TF.js |
| **Métricas** | ✅ Completas | ✅ Completas |
| **Web Ready** | ✅ Directo | ⚠️ Requiere conversión |

---

## 📁 Estructura del Proyecto

```
gesto_releasev1/
├── README.md                    # Este archivo
├── README_TRAINING.md          # Guía completa de entrenamiento
│
├── src/
│   ├── train_lstm_node_v6.js   # Trainer Node.js v6
│   ├── train_lstm_actions_v2.py # Trainer Python v2
│   └── ...
│
├── scripts/
│   ├── download_from_drive.js  # Descargar contribuciones web
│   └── convert_frontend_samples_to_npy.py  # Convertir JSON → .npy
│
├── assets/
│   ├── web_contributions/      # Contribuciones JSON del frontend
│   └── data/keypoints/         # Archivos .npy para entrenamiento
│
└── models/
    ├── modelo_tfjs_node/        # Modelo TensorFlow.js
    ├── normalization_stats.json # Estadísticas de normalización
    ├── training_report_v6.json  # Reporte de entrenamiento
    └── confusion_matrix_v6.json # Matriz de confusión
```

---

## 📚 Documentación

- **[README_TRAINING.md](README_TRAINING.md)** - Guía completa desde web_contributions hasta modelo final
- **[docs/](docs/)** - Documentación técnica adicional

---

## ✅ Checklist Rápido

- [ ] Node.js v14+ o Python 3.8-3.11 instalado
- [ ] Dependencias instaladas (`npm install` o `pip install`)
- [ ] Contribuciones web descargadas
- [ ] Archivos .npy generados
- [ ] Estadísticas de normalización calculadas
- [ ] Modelo entrenado
- [ ] Reporte de entrenamiento generado

---

## 🆘 Soporte

Para problemas o preguntas:
1. Revisa [README_TRAINING.md](README_TRAINING.md) para guía detallada
2. Consulta la documentación en `docs/`
3. Verifica los logs de entrenamiento

---

**Última actualización**: Noviembre 2025  
**Versión**: v6.0  
**Estado**: Producción ✅
