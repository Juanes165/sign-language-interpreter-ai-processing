"""
Convertidor de modelo Keras a TensorFlow.js
Convierte un modelo .keras entrenado a formato TF.js

Uso:
    python src/convert_keras_to_tfjs.py
"""

import os
import sys
# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path

# Intentar cargar tensorflowjs
try:
    import tensorflowjs as tfjs
    TFJS_AVAILABLE = True
except ImportError:
    TFJS_AVAILABLE = False
    print('AVISO: tensorflowjs no esta instalado')

# Rutas
ROOT_PATH = Path(__file__).parent.parent
MODEL_DIR = ROOT_PATH / 'models'
KERAS_MODEL_PATH = MODEL_DIR / 'actions_15.keras'
TFJS_OUTPUT_PATH = MODEL_DIR / 'modelo_tfjs_node'
WORDS_JSON_PATH = MODEL_DIR / 'words.json'


def convert_keras_to_tfjs():
    """Convierte un modelo Keras a TensorFlow.js"""
    
    if not TFJS_AVAILABLE:
        print('\nERROR: tensorflowjs no esta instalado')
        print('\nPara instalar:')
        print('   pip install tensorflowjs==4.11.0')
        print('\nNOTA: Este paquete puede tener conflictos de dependencias')
        print('Alternativa: Usa train_lstm_node_v5.js directamente')
        return False
    
    # Verificar que existe el modelo Keras
    if not KERAS_MODEL_PATH.exists():
        print(f'\nERROR: No se encontro el modelo Keras en: {KERAS_MODEL_PATH}')
        print('Primero ejecuta: python src/train_lstm_actions_v2.py')
        return False
    
    print('\nConvirtiendo modelo Keras a TensorFlow.js...')
    print(f'   Input:  {KERAS_MODEL_PATH}')
    print(f'   Output: {TFJS_OUTPUT_PATH}')
    
    # Crear directorio de salida
    TFJS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convertir modelo
        import tensorflow as tf
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print('OK: Modelo Keras cargado exitosamente')
        
        tfjs.converters.save_keras_model(model, str(TFJS_OUTPUT_PATH))
        print('OK: Modelo exportado a TensorFlow.js')
        print(f'   Archivos guardados en: {TFJS_OUTPUT_PATH}')
        
        # Copiar words.json si existe
        if WORDS_JSON_PATH.exists():
            import json
            words_data = json.loads(WORDS_JSON_PATH.read_text(encoding='utf-8'))
            words_output = TFJS_OUTPUT_PATH / 'words.json'
            words_output.write_text(json.dumps(words_data, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'OK: Copiado words.json a: {words_output}')
        
        print('\nConversion completada exitosamente!')
        print(f'\nArchivos generados en: {TFJS_OUTPUT_PATH}')
        print('   - model.json (arquitectura)')
        print('   - weights.bin (pesos del modelo)')
        print('   - words.json (etiquetas)')
        
        return True
        
    except Exception as e:
        print(f'\nERROR durante la conversion: {e}')
        return False


if __name__ == '__main__':
    print('=' * 60)
    print('Convertidor Keras -> TensorFlow.js')
    print('=' * 60)
    
    success = convert_keras_to_tfjs()
    
    if not success:
        print('\nRECOMENDACION:')
        print('   Usa train_lstm_node_v5.js para obtener output TF.js')
        print('   - Ya esta instalado')
        print('   - Sin conflictos de dependencias')
        print('   - Mismo resultado')

