"""
🧠 Entrenamiento de LSTM para reconocimiento de gestos en Python
VERSIÓN v2.0 - CON INFORME DETALLADO POR GESTO

Mejoras sobre v1:
✅ Informe detallado de métricas por gesto (precisión, recall, F1-score)
✅ Matriz de confusión
✅ Análisis de rendimiento individual por clase
✅ Guardado de reporte JSON con todas las métricas
✅ Detección automática de GPU/CPU
✅ Todos los parámetros optimizados del TOP 1 grid search

Parámetros optimizados (GRID SEARCH - TOP 1):
- LSTM 1: 256 units (MEJOR - 100% val/test accuracy)
- LSTM 2: 128 units
- Dense: 32 units (MEJOR para esta arquitectura)
- Dropout: 0.3
- L2 regularization: 0.001
- Learning rate: 0.0001
- Batch size: 32

Uso:
    python src/train_lstm_actions_v2.py
"""

import json
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# Detectar GPU
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print('🚀 GPU detectada y configurada para uso')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device[0], True)
        backend = 'GPU'
    else:
        print('⚡ Usando CPU (GPU no disponible)')
        backend = 'CPU'
except Exception as e:
    print(f'⚠️  No se pudo configurar GPU: {e}')
    backend = 'CPU'

# Configuración (igual que train_lstm_node_v5.js)
MODEL_FRAMES = 15
LENGTH_KEYPOINTS = 1662
EPOCHS = 60
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.3
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.0001

# Rutas
ROOT_PATH = Path(__file__).parent.parent
KEYPOINTS_PATH = ROOT_PATH / 'assets' / 'data' / 'keypoints'
MODEL_DIR = ROOT_PATH / 'models'
WORDS_JSON_PATH = MODEL_DIR / 'words.json'


def load_sequences(gestures: List[str]) -> Tuple[List[np.ndarray], List[int]]:
    """Carga secuencias de keypoints desde archivos .npy"""
    seqs, labels = [], []
    for idx, g in enumerate(gestures):
        file = KEYPOINTS_PATH / f"{g}.npy"
        if not file.exists():
            print(f"⚠️  Falta {file}")
            continue
        arr = np.load(file, allow_pickle=True)
        # Normalizar a (N, T, D)
        if arr.ndim == 2 and arr.shape == (MODEL_FRAMES, LENGTH_KEYPOINTS):
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3 or arr.shape[-1] != LENGTH_KEYPOINTS:
            print(f"⚠️  {file.name} forma inesperada {arr.shape}, se omite")
            continue
        for seq in arr:
            seqs.append(seq)
            labels.append(idx)
    print(f"✅ Cargadas {len(seqs)} secuencias de {len(gestures)} gestos")
    return seqs, labels


def analyze_data_distribution(sequences: List[np.ndarray], labels: List[int], 
                              gestures: List[str]) -> Dict[str, int]:
    """Analiza la distribución de datos y detecta problemas"""
    print('\n📊 Análisis de Distribución de Datos:\n')
    
    class_counts = {}
    total_samples = len(sequences)
    
    for idx, gesture in enumerate(gestures):
        count = labels.count(idx)
        class_counts[gesture] = count
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"   - {gesture}: {count} muestras ({percentage:.1f}%)")
    
    # Detectar desbalance
    counts = list(class_counts.values())
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if ratio > 2:
            print(f"\n   ⚠️  Desbalance de datos detectado (ratio: {ratio:.2f}x)")
            print(f"   Sugerencia: Considerar data augmentation o class weights")
    
    # Estadísticas de validación
    validation_count = int(total_samples * VALIDATION_SPLIT)
    training_count = total_samples - validation_count
    
    print(f"\n   📈 Muestras de entrenamiento: {training_count}")
    print(f"   📈 Muestras de validación: {validation_count}")
    
    if total_samples < 20:
        print(f"\n   ⚠️  Dataset muy pequeño ({total_samples} muestras)")
        print(f"   Recomendación: Considerar aumentar el dataset")
    
    return class_counts


def build_lstm(num_classes: int) -> Sequential:
    """
    🏆 Modelo LSTM optimizado - TOP 1 del Grid Search
    100% Val/Test Accuracy!
    """
    model = Sequential()
    model.add(Input(shape=(MODEL_FRAMES, LENGTH_KEYPOINTS)))
    
    # LSTM 1 - TOP 1 del Grid Search
    model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    # LSTM 2 - TOP 1 del Grid Search
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    # Dense - TOP 1 del Grid Search
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    
    # Output
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                gestures: List[str]) -> Dict:
    """Calcula métricas detalladas por gesto: precisión, recall, F1-score"""
    num_classes = len(gestures)
    
    # Convertir one-hot a labels
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    
    # Calcular métricas por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    
    # Calcular confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # Construir métricas por clase
    per_class_metrics = []
    for i in range(num_classes):
        # Calcular TP, FP, FN
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        
        per_class_metrics.append({
            'gesture': gestures[i],
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i]),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })
    
    # Calcular promedios
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    return {
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'macro': {
            'precision': float(macro_prec),
            'recall': float(macro_rec),
            'f1_score': float(macro_f1)
        },
        'weighted': {
            'precision': float(weighted_prec),
            'recall': float(weighted_rec),
            'f1_score': float(weighted_f1)
        }
    }


def print_per_class_report(metrics: Dict, gestures: List[str]) -> None:
    """Imprime informe detallado por gesto"""
    print('\n' + '=' * 80)
    print('📊 INFORME DETALLADO POR GESTO')
    print('=' * 80)
    
    # Tabla de métricas por clase
    print('\n📈 Métricas por Clase:')
    print('-' * 80)
    print(f"{'Gesto':<20} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12} {'Soporte':<10}")
    print('-' * 80)
    
    for m in metrics['per_class_metrics']:
        print(f"{m['gesture'][:19]:<20} {m['precision']*100:>10.2f}% "
              f"{m['recall']*100:>10.2f}% {m['f1_score']*100:>10.2f}% "
              f"{m['support']:>10}")
    
    print('-' * 80)
    
    # Promedios
    print('\n📊 Promedios:')
    print('   Macro promedio:')
    print(f"      Precisión: {metrics['macro']['precision']*100:.2f}%")
    print(f"      Recall: {metrics['macro']['recall']*100:.2f}%")
    print(f"      F1-Score: {metrics['macro']['f1_score']*100:.2f}%")
    
    print('   Promedio ponderado:')
    print(f"      Precisión: {metrics['weighted']['precision']*100:.2f}%")
    print(f"      Recall: {metrics['weighted']['recall']*100:.2f}%")
    print(f"      F1-Score: {metrics['weighted']['f1_score']*100:.2f}%")
    
    # Matriz de confusión
    print('\n🔍 Matriz de Confusión:')
    print('   (Filas: verdadero, Columnas: predicción)')
    print('\n   ' + ' '.join([f"{g[:10]:<10}" for g in gestures]))
    
    for i, row in enumerate(metrics['confusion_matrix']):
        row_str = f"{gestures[i][:10]:<10} " + ' '.join([f"{str(val):<10}" for val in row])
        print('   ' + row_str)
    
    # Análisis de clases problemáticas
    print('\n🔎 Análisis de Rendimiento:')
    f1_scores = [m['f1_score'] for m in metrics['per_class_metrics']]
    worst_f1 = min(f1_scores)
    best_f1 = max(f1_scores)
    
    if worst_f1 < 0.5:
        worst = metrics['per_class_metrics'][f1_scores.index(worst_f1)]
        print(f"   ⚠️  Gesto con menor F1-Score: {worst['gesture']} ({worst_f1*100:.2f}%)")
        print(f"      Precisión: {worst['precision']*100:.2f}%, Recall: {worst['recall']*100:.2f}%")
    else:
        print(f"   ✅ Todas las clases tienen F1-Score > 50%")
    
    if best_f1 > 0.95:
        best = metrics['per_class_metrics'][f1_scores.index(best_f1)]
        print(f"   🏆 Gesto con mejor F1-Score: {best['gesture']} ({best_f1*100:.2f}%)")
    
    print('=' * 80)


def main():
    print('🚀 Iniciando entrenamiento de LSTM en Python v2.0')
    print('✨ Informe Detallado por Gesto\n')
    print(f'Backend: {backend}')
    print('📋 Configuración:')
    print(f'   - Frames por secuencia: {MODEL_FRAMES}')
    print(f'   - Keypoints por frame: {LENGTH_KEYPOINTS}')
    print(f'   - Epochs: {EPOCHS}')
    print(f'   - Batch size: {BATCH_SIZE}')
    print(f'   - Early stopping patience: {EARLY_STOPPING_PATIENCE}\n')
    
    # 1. Cargar etiquetas
    if WORDS_JSON_PATH.exists():
        with open(WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
            gestures = json.load(f)['word_ids']
    else:
        gestures = ['hola-der', 'dias-gen', 'paz-der']
    
    print(f'🏷️  Gestos a entrenar: {", ".join(gestures)}\n')
    
    # 2. Cargar secuencias
    seqs, labels = load_sequences(gestures)
    if not seqs:
        raise ValueError('❌ No hay secuencias válidas. Ejecuta extracción de keypoints primero.')
    
    # 3. Analizar distribución de datos
    class_counts = analyze_data_distribution(seqs, labels, gestures)
    
    # 4. Preparar datos
    X = pad_sequences(seqs, maxlen=MODEL_FRAMES, padding='pre', truncating='post', dtype='float32')
    y = to_categorical(labels, num_classes=len(gestures))
    
    print(f'\n📦 Forma de datos:')
    print(f'   X: {X.shape} (samples, frames, keypoints)')
    print(f'   y: {y.shape} (samples, classes)\n')
    
    # 5. Construir modelo
    print('🏗️  Construyendo modelo...')
    model = build_lstm(num_classes=len(gestures))
    
    # Analizar modelo
    total_params = model.count_params()
    print(f'\n📊 Estadísticas del Modelo:')
    print(f'   - Parámetros totales: {total_params:,}')
    estimated_size_mb = (total_params * 4) / (1024 * 1024)
    print(f'   - Tamaño estimado: {estimated_size_mb:.2f} MB')
    
    print('\n🎯 Parámetros del modelo (TOP 1 GRID SEARCH - 100% Val/Test Acc!):')
    print(f'   - LSTM 1: 256 units 🏆 (TOP 1)')
    print(f'   - LSTM 2: 128 units 🏆 (TOP 1)')
    print(f'   - Dense: 32 units 🏆 (TOP 1)')
    print(f'   - Dropout: 0.3')
    print(f'   - L2 regularization: 0.001')
    print(f'   - Learning rate: {LEARNING_RATE}')
    print(f'   - Batch size: {BATCH_SIZE}')
    
    print('\n🏗️  Arquitectura del modelo:')
    model.summary()
    
    # 6. Dividir datos
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    # 7. Entrenar
    print('\n🎓 Entrenando modelo...\n')
    start_time = datetime.now()
    
    es = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, 
                       restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # 8. Evaluar en datos de validación
    print('\n📊 Evaluando modelo para informe detallado...\n')
    val_predictions = model.predict(X_val, verbose=0)
    metrics = calculate_per_class_metrics(y_val, val_predictions, gestures)
    
    # 9. Guardar modelo en formato Keras
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keras_model_path = MODEL_DIR / f'actions_{MODEL_FRAMES}.keras'
    model.save(keras_model_path)
    print(f'✅ Modelo Keras guardado en: {keras_model_path}')
    
    # 10. Resumen final
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    total_epochs = len(history.history['loss'])
    
    print('\n' + '=' * 60)
    print('📊 Resultados finales:')
    print('=' * 60)
    print(f'   - Training Loss: {final_loss:.4f}')
    print(f'   - Training Accuracy: {final_acc*100:.2f}%')
    print(f'   - Validation Loss: {final_val_loss:.4f}')
    print(f'   - Validation Accuracy: {final_val_acc*100:.2f}%')
    print(f'   - Epochs completados: {total_epochs}/{EPOCHS}')
    print(f'   - Tiempo de entrenamiento: {training_time:.2f}s')
    
    # Análisis de resultados
    overfit = final_acc - final_val_acc
    print('\n📈 Análisis de Resultados:')
    
    if overfit > 0.3:
        print(f'   🚨 OVERFITTING SEVERO detectado')
        print(f'      Gap train/val accuracy: {overfit*100:.1f}%')
    elif overfit > 0.15:
        print(f'   ⚠️  Overfitting moderado detectado')
        print(f'      Gap train/val accuracy: {overfit*100:.1f}%')
    else:
        print(f'   ✅ Buena generalización')
        print(f'      Gap train/val accuracy: {overfit*100:.1f}%')
    
    if final_val_acc < 0.1 and final_acc > 0.5:
        print(f'\n   🚨 ALERTA CRÍTICA: Modelo NO está generalizando')
        print(f'      Val accuracy muy bajo ({final_val_acc*100:.1f}%)')
    
    print('=' * 60)
    
    # 13. Imprimir informe detallado por gesto
    print_per_class_report(metrics, gestures)
    
    # 14. Guardar informe detallado en archivo JSON
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'backend': backend,
        'configuration': {
            'epochs': total_epochs,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'validation_split': VALIDATION_SPLIT,
            'training_time_seconds': float(training_time)
        },
        'overall_metrics': {
            'training_loss': float(final_loss),
            'training_accuracy': float(final_acc),
            'validation_loss': float(final_val_loss),
            'validation_accuracy': float(final_val_acc),
            'overfit_gap': float(overfit)
        },
        'per_class_metrics': metrics['per_class_metrics'],
        'averages': {
            'macro': metrics['macro'],
            'weighted': metrics['weighted']
        },
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    report_path = MODEL_DIR / 'training_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f'\n💾 Informe guardado en: {report_path}')
    
    print('\n🧹 Entrenamiento completado correctamente')


if __name__ == '__main__':
    main()

