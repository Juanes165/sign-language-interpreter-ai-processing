"""
Script para calcular estadísticas de normalización de keypoints
Calcula mean y std por componente (pose, face, hands) para normalización estandarizada
"""

import os
import json
import numpy as np
from pathlib import Path

# Importar constantes directamente sin cv2
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYPOINTS_PATH = os.path.join(ROOT_PATH, 'assets', 'data', 'keypoints')
MODEL_DIR = os.path.join(ROOT_PATH, 'models')
WORDS_JSON_PATH = os.path.join(MODEL_DIR, 'words.json')
LENGTH_KEYPOINTS = 1662

# Índices de cada componente en el array de keypoints
POSE_START = 0
POSE_END = 132  # 33 landmarks × 4 valores (x, y, z, visibility)
FACE_START = 132
FACE_END = 1536  # 468 landmarks × 3 valores + 132 de pose
LEFT_HAND_START = 1536
LEFT_HAND_END = 1599  # 21 landmarks × 3 valores + 1536 anterior
RIGHT_HAND_START = 1599
RIGHT_HAND_END = 1662  # 21 landmarks × 3 valores + 1599 anterior


def calculate_stats(all_keypoints):
    """
    Calcula estadísticas de normalización por componente
    """
    print("\n📊 Calculando estadísticas por componente...")
    
    # Separar por componente
    pose_data = all_keypoints[:, POSE_START:POSE_END]
    face_data = all_keypoints[:, FACE_START:FACE_END]
    left_hand_data = all_keypoints[:, LEFT_HAND_START:LEFT_HAND_END]
    right_hand_data = all_keypoints[:, RIGHT_HAND_START:RIGHT_HAND_END]
    
    stats = {
        'pose': {
            'mean': np.mean(pose_data, axis=0).tolist(),
            'std': np.std(pose_data, axis=0).tolist()
        },
        'face': {
            'mean': np.mean(face_data, axis=0).tolist(),
            'std': np.std(face_data, axis=0).tolist()
        },
        'left_hand': {
            'mean': np.mean(left_hand_data, axis=0).tolist(),
            'std': np.std(left_hand_data, axis=0).tolist()
        },
        'right_hand': {
            'mean': np.mean(right_hand_data, axis=0).tolist(),
            'std': np.std(right_hand_data, axis=0).tolist()
        },
        'global': {
            'mean': np.mean(all_keypoints, axis=0).tolist(),
            'std': np.std(all_keypoints, axis=0).tolist()
        }
    }
    
    # Mostrar resumen
    print(f"   Pose: mean={np.mean(pose_data):.4f}, std={np.std(pose_data):.4f}")
    print(f"   Face: mean={np.mean(face_data):.4f}, std={np.std(face_data):.4f}")
    print(f"   Left Hand: mean={np.mean(left_hand_data):.4f}, std={np.std(left_hand_data):.4f}")
    print(f"   Right Hand: mean={np.mean(right_hand_data):.4f}, std={np.std(right_hand_data):.4f}")
    
    # Detectar std muy pequeños (casi constantes) que podrían causar problemas
    pose_std = np.std(pose_data, axis=0)
    face_std = np.std(face_data, axis=0)
    lh_std = np.std(left_hand_data, axis=0)
    rh_std = np.std(right_hand_data, axis=0)
    
    small_std_threshold = 1e-6
    small_std_count = np.sum(pose_std < small_std_threshold) + \
                      np.sum(face_std < small_std_threshold) + \
                      np.sum(lh_std < small_std_threshold) + \
                      np.sum(rh_std < small_std_threshold)
    
    if small_std_count > 0:
        print(f"\n   ⚠️  {small_std_count} características tienen std < {small_std_threshold}")
        print(f"   Estas serán reemplazadas por 1.0 para evitar división por cero")
    
    return stats


def main():
    print("🔍 Calculando estadísticas de normalización para keypoints")
    print("=" * 70)
    
    # Cargar lista de gestos
    if os.path.exists(WORDS_JSON_PATH):
        with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
            gestures = json.load(f)["word_ids"]
    else:
        print(f"⚠️  {WORDS_JSON_PATH} no encontrado")
        print("   Buscando archivos .npy directamente...")
        npy_files = list(Path(KEYPOINTS_PATH).glob('*.npy'))
        gestures = [f.stem for f in npy_files]
    
    print(f"\n📁 Directorio de keypoints: {KEYPOINTS_PATH}")
    print(f"🏷️  Gestos encontrados: {len(gestures)}")
    
    # Cargar todos los keypoints
    all_keypoints = []
    total_sequences = 0
    
    for gesture in gestures:
        npy_path = Path(KEYPOINTS_PATH) / f"{gesture}.npy"
        
        if not npy_path.exists():
            print(f"   ⚠️  Falta {npy_path}")
            continue
        
        try:
            data = np.load(npy_path, allow_pickle=True)
            
            # Asegurar forma correcta
            if data.ndim == 2:
                # Una secuencia: (T, D)
                if data.shape[0] == 1 or data.shape == (1, LENGTH_KEYPOINTS):
                    data = data.reshape(1, -1, LENGTH_KEYPOINTS)
                else:
                    data = data.reshape(1, *data.shape)
            elif data.ndim == 3:
                # Múltiples secuencias: (N, T, D)
                pass
            else:
                print(f"   ⚠️  Forma inesperada en {gesture}.npy: {data.shape}")
                continue
            
            # Aplanar a (N*T, D) para calcular estadísticas
            n, t, d = data.shape
            flattened = data.reshape(-1, d)  # (N*T, D)
            all_keypoints.append(flattened)
            total_sequences += n
            
            print(f"   ✅ {gesture}: {data.shape} -> {flattened.shape[0]} frames")
            
        except Exception as e:
            print(f"   ❌ Error cargando {gesture}.npy: {e}")
            continue
    
    if len(all_keypoints) == 0:
        print("\n❌ No se encontraron datos válidos para calcular estadísticas")
        return
    
    # Combinar todos los keypoints
    all_keypoints = np.concatenate(all_keypoints, axis=0)
    print(f"\n📊 Total de frames procesados: {all_keypoints.shape[0]}")
    print(f"📊 Total de secuencias: {total_sequences}")
    print(f"📊 Dimensiones: {all_keypoints.shape}")
    
    # Verificar dimensiones
    if all_keypoints.shape[1] != LENGTH_KEYPOINTS:
        print(f"\n❌ Error: Se esperaban {LENGTH_KEYPOINTS} keypoints, pero se encontraron {all_keypoints.shape[1]}")
        return
    
    # Calcular estadísticas
    stats = calculate_stats(all_keypoints)
    
    # Guardar estadísticas
    output_path = Path(MODEL_DIR) / 'normalization_stats.json'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Estadísticas guardadas en: {output_path}")
    print("\n" + "=" * 70)
    print("📋 Resumen:")
    print(f"   - Archivos procesados: {len(gestures)}")
    print(f"   - Total frames: {all_keypoints.shape[0]}")
    print(f"   - Total secuencias: {total_sequences}")
    print(f"   - Estadísticas por componente: pose, face, left_hand, right_hand")
    print(f"   - Estadísticas globales: también incluidas")
    print("=" * 70)
    print("\n💡 Usa estas estadísticas en los scripts de entrenamiento para normalizar los keypoints")


if __name__ == '__main__':
    main()

