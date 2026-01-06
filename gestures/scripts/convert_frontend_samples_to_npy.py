"""
🔄 Conversor: Frontend Samples → NumPy (.npy)

Este script procesa los archivos JSON individuales capturados desde el frontend
y los convierte al formato .npy esperado por el entrenador LSTM.

Uso:
    python scripts/convert_frontend_samples_to_npy.py <directorio-frontend>

Ejemplo:
    python scripts/convert_frontend_samples_to_npy.py ../../sign-language-interpreter-frontend
"""

import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib

# Intentar importar tqdm (opcional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print('💡 Instala tqdm para barras de progreso: pip install tqdm')

# Configuración
MODEL_FRAMES = 15
LENGTH_KEYPOINTS = 1662
ROOT_PATH = Path(__file__).parent.parent
KEYPOINTS_OUTPUT_PATH = ROOT_PATH / 'assets' / 'data' / 'keypoints'
WORDS_JSON_PATH = ROOT_PATH / 'models' / 'words.json'


def normalize_gesture_name(gesture_name):
    """Normaliza el nombre del gesto para usar como nombre de archivo"""
    mapping = {
        '¿': '', '?': '', ' ': '-',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'
    }
    result = gesture_name.lower()
    for old, new in mapping.items():
        result = result.replace(old, new)
    return result


def pad_sequence_uniform(sequence, target_length, offset=0):
    """
    Realiza padding o muestreo uniforme de una secuencia a longitud fija
    
    Args:
        sequence: Lista de frames, cada frame es una lista de 1662 valores
        target_length: Longitud objetivo (15 frames)
        offset: Offset inicial para el muestreo (0, 1, o 2 para augmentation)
    
    Returns:
        Lista de frames con exactamente target_length frames
    """
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    elif current_length < target_length:
        # Pad al inicio con ceros
        padding = [[0.0] * LENGTH_KEYPOINTS for _ in range(target_length - current_length)]
        return padding + sequence
    else:
        # 🎯 MUESTREO UNIFORME: Distribuye frames a lo largo de toda la secuencia
        # Con offset para data augmentation
        indices = np.linspace(offset, current_length - 1, target_length, dtype=int)
        return [sequence[i] for i in indices]


def pad_sequence(sequence, target_length):
    """
    Wrapper para compatibilidad: usa muestreo uniforme sin offset
    """
    return pad_sequence_uniform(sequence, target_length, offset=0)


def validate_sample(sample):
    """Valida que una muestra tenga la estructura correcta"""
    if not sample.get('gesture') or not sample.get('keypoints'):
        print(f'⚠️  Muestra sin gesture o keypoints')
        return False
    
    keypoints = sample['keypoints']
    if not isinstance(keypoints, list) or len(keypoints) == 0:
        print(f"⚠️  {sample['gesture']}: keypoints vacíos")
        return False
    
    # Validar que cada frame tenga la cantidad correcta de keypoints
    for i, frame in enumerate(keypoints):
        if not isinstance(frame, list) or len(frame) != LENGTH_KEYPOINTS:
            print(f"⚠️  {sample['gesture']}: frame {i} tiene {len(frame) if isinstance(frame, list) else 0} keypoints (esperado: {LENGTH_KEYPOINTS})")
            return False
    
    return True


def analyze_data(np_array):
    """Analiza la calidad de los datos"""
    analysis = {
        'min': float(np_array.min()),
        'max': float(np_array.max()),
        'mean': float(np_array.mean()),
        'std': float(np_array.std()),
        'nan_count': int(np.isnan(np_array).sum()),
        'inf_count': int(np.isinf(np_array).sum())
    }
    
    # Detectar valores anómalos (fuera de rango razonable)
    outliers = np.abs(np_array) > 10
    analysis['outliers'] = int(outliers.sum())
    
    return analysis


def verify_npy_file(file_path, expected_shape):
    """Verifica que el archivo .npy es correcto"""
    try:
        loaded = np.load(file_path)
        
        # Verificar shape
        if loaded.shape != expected_shape:
            print(f"   ❌ Shape incorrecto: {loaded.shape} vs esperado {expected_shape}")
            return False
        
        # Verificar NaN
        if np.isnan(loaded).any():
            print(f"   ❌ Contiene NaN")
            return False
        
        # Verificar infinitos
        if np.isinf(loaded).any():
            print(f"   ❌ Contiene infinitos")
            return False
        
        print(f"   ✅ Archivo verificado correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error en verificación: {e}")
        return False


def analyze_gesture_quality(samples):
    """Analiza la calidad y consistencia de muestras de un gesto"""
    frame_lengths = [len(s['keypoints']) for s in samples]
    avg_frames = np.mean(frame_lengths)
    frame_std = np.std(frame_lengths)
    min_frames = min(frame_lengths)
    max_frames = max(frame_lengths)
    
    print(f"   📊 Estadísticas:")
    print(f"      - Frames promedio: {avg_frames:.1f}")
    print(f"      - Desviación: {frame_std:.1f}")
    print(f"      - Rango: {min_frames}-{max_frames}")
    
    if frame_std > 10:
        print(f"   ⚠️  Alta variabilidad en longitud de frames")
    
    return {
        'avg': avg_frames,
        'std': frame_std,
        'min': min_frames,
        'max': max_frames
    }


def detect_duplicates(samples):
    """Detecta muestras duplicadas basadas en hash de keypoints"""
    seen = set()
    duplicates = []
    
    for sample in samples:
        # Crear hash del primer frame de keypoints
        keypoints_str = str(sample['keypoints'][0]) if sample['keypoints'] else ""
        sample_hash = hashlib.md5(keypoints_str.encode()).hexdigest()
        
        if sample_hash in seen:
            duplicates.append(sample.get('gesture', 'unknown'))
        else:
            seen.add(sample_hash)
    
    return duplicates


def process_samples_directory(samples_dir):
    """Procesa todos los archivos JSON de un directorio"""
    print('🌐 Conversor: Frontend Samples → NumPy\n')
    print(f'📂 Procesando: {samples_dir}\n')
    
    samples_dir = Path(samples_dir)
    
    # Leer todos los archivos JSON
    json_files = list(samples_dir.glob('*.json'))
    
    if len(json_files) == 0:
        print('❌ No se encontraron archivos JSON en el directorio')
        sys.exit(1)
    
    print(f'📊 Total de archivos: {len(json_files)}\n')
    
    # Cargar y validar todas las muestras
    all_samples = []
    invalid_files = []
    
    # Usar barra de progreso si está disponible
    iterator = tqdm(json_files, desc="Validando archivos") if HAS_TQDM else json_files
    
    for json_file in iterator:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sample = json.load(f)
            
            if validate_sample(sample):
                all_samples.append(sample)
            else:
                invalid_files.append(json_file.name)
        except Exception as error:
            print(f'❌ Error procesando {json_file.name}: {error}')
            invalid_files.append(json_file.name)
    
    print(f'✅ Muestras válidas: {len(all_samples)}')
    if invalid_files:
        print(f'⚠️  Archivos inválidos: {len(invalid_files)}')
        for f in invalid_files:
            print(f'   - {f}')
    print()
    
    if len(all_samples) == 0:
        print('❌ No hay muestras válidas para procesar')
        sys.exit(1)
    
    # Agrupar por gesto
    samples_by_gesture = defaultdict(list)
    for sample in all_samples:
        gesture_name = sample.get('gestureName', sample['gesture'])
        samples_by_gesture[gesture_name].append(sample)
    
    # Mostrar resumen
    print('📋 Resumen por gesto:')
    for name, samples in samples_by_gesture.items():
        print(f'   - {name}: {len(samples)} muestras')
    print()
    
    # Convertir a .npy
    KEYPOINTS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f'📦 Procesando {len(samples_by_gesture)} gestos...\n')
    
    # Contadores para el resumen
    total_augmented_samples = 0
    
    # Iterar con barra de progreso si está disponible
    gesture_iterator = tqdm(samples_by_gesture.items(), desc="Convirtiendo gestos") if HAS_TQDM else samples_by_gesture.items()
    
    for gesture_name, samples in gesture_iterator:
        normalized_name = normalize_gesture_name(gesture_name)
        print(f'🔄 {gesture_name} ({len(samples)} muestras)')
        
        # Detectar duplicados
        duplicates = detect_duplicates(samples)
        if duplicates:
            print(f'   ⚠️  Detectados {len(duplicates)} duplicados (se mantienen)')
        
        # Analizar calidad del gesto
        _quality_stats = analyze_gesture_quality(samples)  # Para evitar warning de variable no usada
        
        # 🎯 DATA AUGMENTATION: Generar 3 muestras por cada captura con diferentes offsets
        # Esto triplica el dataset y mejora la robustez del modelo
        augmented_sequences = []
        num_offsets = 3  # Generar 3 versiones de cada muestra
        
        for sample in samples:
            keypoints = sample['keypoints']
            current_length = len(keypoints)
            
            # Si la secuencia es muy corta, solo generar 1 muestra
            if current_length < MODEL_FRAMES + 2:
                augmented_sequences.append(pad_sequence_uniform(keypoints, MODEL_FRAMES, offset=0))
            else:
                # Generar 3 muestras con offsets diferentes
                for offset in range(num_offsets):
                    augmented_sequences.append(pad_sequence_uniform(keypoints, MODEL_FRAMES, offset=offset))
        
        num_augmented = len(augmented_sequences)
        total_augmented_samples += num_augmented
        print(f'   📊 Muestras originales: {len(samples)} → Muestras augmentadas: {num_augmented} (x{num_augmented/len(samples):.1f})')
        
        # Convertir a numpy array
        # Primero a lista plana, luego reshape
        flat_data = []
        for sequence in augmented_sequences:
            for frame in sequence:
                flat_data.extend(frame)
        
        # Crear array NumPy con forma (N_augmented, 15, 1662)
        shape = (num_augmented, MODEL_FRAMES, LENGTH_KEYPOINTS)
        np_array = np.array(flat_data, dtype=np.float32).reshape(shape)
        
        # Analizar estadísticas de datos
        analysis = analyze_data(np_array)
        print(f'   📈 Datos: min={analysis["min"]:.4f}, max={analysis["max"]:.4f}, mean={analysis["mean"]:.4f}')
        
        if analysis['nan_count'] > 0:
            print(f'   ⚠️  {analysis["nan_count"]} valores NaN detectados')
        if analysis['inf_count'] > 0:
            print(f'   ⚠️  {analysis["inf_count"]} valores infinitos detectados')
        if analysis['outliers'] > 0:
            print(f'   ⚠️  {analysis["outliers"]} valores anómalos (|x| > 10)')
        
        # Guardar como .npy
        try:
            output_path = KEYPOINTS_OUTPUT_PATH / f'{normalized_name}.npy'
            np.save(output_path, np_array)
            print(f'   ✅ Guardado: {normalized_name}.npy ({shape})')
            
            # Verificar archivo creado
            verify_npy_file(output_path, shape)
            
        except Exception as error:
            print(f'   ❌ Error creando .npy: {error}')
    
    # Actualizar words.json
    gesture_ids = [normalize_gesture_name(name) for name in samples_by_gesture.keys()]
    
    existing_words = []
    if WORDS_JSON_PATH.exists():
        try:
            with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_words = existing_data.get('word_ids', [])
        except Exception as error:
            print(f'⚠️  Error leyendo words.json existente: {error}')
    
    # Combinar con gestos existentes (sin duplicados)
    all_gestures = list(set(existing_words + gesture_ids))
    words_data = {'word_ids': all_gestures}
    
    WORDS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(WORDS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(words_data, f, indent=2, ensure_ascii=False)
    
    print(f'\n✅ Actualizado: {WORDS_JSON_PATH}')
    print(f'   Gestos totales: {len(all_gestures)}')
    
    # Resumen final
    print('\n' + '='*60)
    print('🎯 Conversión completada!')
    print('='*60)
    print(f'\n📁 Archivos .npy guardados en:')
    print(f'   {KEYPOINTS_OUTPUT_PATH}')
    print(f'\n📊 Resumen de conversión:')
    print(f'   - Gestos procesados: {len(samples_by_gesture)}')
    print(f'   - Muestras originales: {len(all_samples)}')
    print(f'   - Muestras augmentadas: {total_augmented_samples} (x{total_augmented_samples/len(all_samples):.1f})')
    print(f'   - Archivos inválidos: {len(invalid_files)}')
    print(f'\n🎯 Data Augmentation aplicado:')
    print(f'   ✅ Muestreo uniforme distribuido')
    print(f'   ✅ 3 offsets por muestra (0, 1, 2)')
    print(f'   ✅ Dataset triplicado para mejor precisión')
    print(f'\n🚀 Próximo paso:')
    print(f'   cd {ROOT_PATH}')
    print(f'   node src/train_lstm_node_v2.js')
    print('='*60)


def main():
    if len(sys.argv) < 2:
        print('❌ Uso: python scripts/convert_frontend_samples_to_npy.py <directorio-con-json>')
        print('')
        print('Opciones:')
        print('  1. Desde frontend:')
        print('     python scripts/convert_frontend_samples_to_npy.py ../../sign-language-interpreter-frontend')
        print('  2. Desde web_contributions:')
        print('     python scripts/convert_frontend_samples_to_npy.py assets/web_contributions')
        print('  3. Desde cualquier directorio con archivos JSON:')
        print('     python scripts/convert_frontend_samples_to_npy.py /ruta/a/archivos/json/')
        sys.exit(1)
    
    input_dir = Path(sys.argv[1]).resolve()
    
    if not input_dir.exists():
        print(f'❌ Directorio no encontrado: {input_dir}')
        sys.exit(1)
    
    # Detectar si es directorio de frontend o de web_contributions
    captured_samples_dir = None
    
    # Opción 1: Es el directorio frontend (buscar captured_samples dentro)
    if (input_dir / 'captured_samples').exists():
        captured_samples_dir = input_dir / 'captured_samples'
        print('📂 Detectado: directorio del frontend, usando captured_samples/')
    
    # Opción 2: Es directamente web_contributions o cualquier directorio con JSONs
    elif input_dir.exists() and list(input_dir.glob('*.json')):
        captured_samples_dir = input_dir
        print('📂 Detectado: directorio con archivos JSON')
    
    # Opción 3: Es un directorio pero sin archivos JSON
    elif input_dir.exists():
        print(f'❌ No se encontraron archivos JSON en: {input_dir}')
        print(f'   Archivos en el directorio: {list(input_dir.iterdir())}')
        sys.exit(1)
    
    else:
        print(f'❌ Directorio no encontrado: {input_dir}')
        sys.exit(1)
    
    print(f'📁 Procesando desde: {captured_samples_dir}\n')
    
    process_samples_directory(captured_samples_dir)


if __name__ == '__main__':
    main()

