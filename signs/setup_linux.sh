#!/bin/bash

# Actualizar el sistema
sudo apt-get update

# Instalar dependencias del sistema para OpenCV
sudo apt-get install -y python3-dev python3-pip 
sudo apt-get install -y libgl1-mesa-glx

# Crear un entorno virtual (opcional pero recomendado)
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate

# Instalar dependencias de Python
pip install numpy scikit-learn opencv-python mediapipe

# Intentar instalar mediapipe-model-maker (opcional)
pip install mediapipe-model-maker || echo "No se pudo instalar mediapipe-model-maker automáticamente"

echo "Configuración completada. Activa el entorno virtual con: source venv/bin/activate"
