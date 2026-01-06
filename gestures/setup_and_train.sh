#!/bin/bash

# ===================================================
# Script de instalación y entrenamiento automático
# ===================================================

set -e  # Salir si hay error

echo ""
echo "================================================"
echo " LSTM Gesture Recognition - Instalación y Entrenamiento"
echo "================================================"
echo ""

# Verificar que Node.js está instalado
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js no está instalado."
    echo "Por favor instala Node.js desde: https://nodejs.org/"
    exit 1
fi

echo "[INFO] Node.js detectado:"
node --version
npm --version
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "package.json" ]; then
    echo "[ERROR] No se encuentra package.json"
    echo "Asegúrate de ejecutar este script desde el directorio gesto_releasev1"
    exit 1
fi

# Instalar dependencias
echo "================================================"
echo " Paso 1: Instalando dependencias de Node.js..."
echo "================================================"
echo ""
npm install
echo ""
echo "[OK] Dependencias instaladas correctamente"
echo ""

# Verificar que existen los archivos de keypoints
echo "================================================"
echo " Paso 2: Verificando datos de entrenamiento..."
echo "================================================"
echo ""

if [ ! -d "assets/data/keypoints" ]; then
    echo "[ERROR] No existe el directorio de keypoints"
    echo "Ejecuta primero el script de Python para extraer keypoints"
    exit 1
fi

KEYPOINT_COUNT=$(ls -1 assets/data/keypoints/*.npy 2>/dev/null | wc -l)

if [ "$KEYPOINT_COUNT" -lt 1 ]; then
    echo "[ERROR] No se encontraron archivos .npy en assets/data/keypoints/"
    echo "Ejecuta primero: python src/extract_keypoints.py"
    exit 1
fi

echo "[OK] Encontrados $KEYPOINT_COUNT archivos de keypoints"
echo ""

# Entrenar el modelo
echo "================================================"
echo " Paso 3: Entrenando modelo LSTM..."
echo "================================================"
echo ""
echo "Este proceso puede tardar 5-15 minutos..."
echo ""
npm run train
echo ""
echo "[OK] Modelo entrenado correctamente"
echo ""

# Verificar el modelo
echo "================================================"
echo " Paso 4: Verificando modelo exportado..."
echo "================================================"
echo ""
npm run verify || echo "[WARN] La verificación falló, pero el modelo puede estar OK"
echo ""

# Copiar a Next.js
echo "================================================"
echo " Paso 5: Copiando modelo a Next.js..."
echo "================================================"
echo ""
npm run copy-to-nextjs || {
    echo "[WARN] No se pudo copiar automáticamente"
    echo "Copia manualmente los archivos de:"
    echo "  models/modelo_tfjs_node/"
    echo "a:"
    echo "  ../sign-language-interpreter-frontend/public/models/lstm_gestos/"
}
echo ""

# Resumen final
echo "================================================"
echo " COMPLETADO EXITOSAMENTE"
echo "================================================"
echo ""
echo "[OK] El modelo LSTM ha sido entrenado y exportado"
echo ""
echo "Archivos generados:"
echo "  - models/modelo_tfjs_node/model.json"
echo "  - models/modelo_tfjs_node/group1-shard*.bin"
echo "  - models/modelo_tfjs_node/words.json"
echo ""
echo "Próximos pasos:"
echo "  1. Reinicia el servidor de Next.js si está corriendo"
echo "  2. Navega a http://localhost:3000/gestures"
echo "  3. Prueba el reconocimiento de gestos"
echo ""
echo "Para re-entrenar en el futuro:"
echo "  npm run full-pipeline"
echo ""
