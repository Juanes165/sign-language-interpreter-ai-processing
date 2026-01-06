"""
Script de búsqueda de hiperparámetros para el modelo LSTM de reconocimiento de gestos.
Soporta Grid Search, Random Search y Optuna (optimización bayesiana).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle
from datetime import datetime

from app_constants import KEYPOINTS_PATH, MODEL_FRAMES, LENGTH_KEYPOINTS, MODEL_DIR, WORDS_JSON_PATH


# ============================================
# CONFIGURACIÓN DE HIPERPARÁMETROS A BUSCAR
# ============================================

SEARCH_SPACE = {
    'lstm_units_1': [32, 64, 128, 256],
    'lstm_units_2': [64, 128, 256, 512],
    'dense_units': [32, 64, 128, 256],
    'dropout_rate': [0.3, 0.4, 0.5, 0.6],
    'l2_lstm': [0.001, 0.01, 0.1],
    'l2_dense': [0.001, 0.01, 0.1],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'use_bidirectional': [True, False],
    'num_lstm_layers': [1, 2, 3],
}


def load_sequences(gestures: List[str]):
    """Carga las secuencias de keypoints desde archivos .npy"""
    seqs, labels = [], []
    for idx, g in enumerate(gestures):
        file = Path(KEYPOINTS_PATH) / f"{g}.npy"
        if not file.exists():
            print(f"⚠️  Falta {file}")
            continue
        arr = np.load(file, allow_pickle=True)
        
        if arr.ndim == 2 and arr.shape == (MODEL_FRAMES, LENGTH_KEYPOINTS):
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3 or arr.shape[-1] != LENGTH_KEYPOINTS:
            print(f"⚠️  {file.name} forma inesperada {arr.shape}, se omite")
            continue
            
        for seq in arr:
            seqs.append(seq)
            labels.append(idx)
    
    return seqs, labels


def build_model(num_classes: int, params: Dict[str, Any]):
    """
    Construye el modelo LSTM con los hiperparámetros especificados.
    
    Args:
        num_classes: Número de clases a predecir
        params: Diccionario con hiperparámetros
    """
    model = Sequential()
    model.add(Input(shape=(int(MODEL_FRAMES), LENGTH_KEYPOINTS)))
    
    # Primera capa LSTM
    lstm_layer_1 = LSTM(
        params['lstm_units_1'],
        return_sequences=params['num_lstm_layers'] > 1,
        kernel_regularizer=l2(params['l2_lstm'])
    )
    
    if params.get('use_bidirectional', False):
        model.add(Bidirectional(lstm_layer_1))
    else:
        model.add(lstm_layer_1)
    
    model.add(Dropout(params['dropout_rate']))
    model.add(BatchNormalization())
    
    # Capas LSTM adicionales
    if params['num_lstm_layers'] >= 2:
        lstm_layer_2 = LSTM(
            params['lstm_units_2'],
            return_sequences=params['num_lstm_layers'] > 2,
            kernel_regularizer=l2(params['l2_lstm'])
        )
        
        if params.get('use_bidirectional', False):
            model.add(Bidirectional(lstm_layer_2))
        else:
            model.add(lstm_layer_2)
        
        model.add(Dropout(params['dropout_rate']))
        model.add(BatchNormalization())
    
    if params['num_lstm_layers'] >= 3:
        model.add(LSTM(
            params['lstm_units_2'] // 2,
            return_sequences=False,
            kernel_regularizer=l2(params['l2_lstm'])
        ))
        model.add(Dropout(params['dropout_rate']))
        model.add(BatchNormalization())
    
    # Capas densas
    model.add(Dense(
        params['dense_units'],
        activation='relu',
        kernel_regularizer=l2(params['l2_dense'])
    ))
    model.add(Dropout(params['dropout_rate']))
    
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar con learning rate personalizado
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def evaluate_model(params: Dict[str, Any], X_train, y_train, X_val, y_val, num_classes: int, trial=None):
    """
    Entrena y evalúa un modelo con los hiperparámetros dados.
    
    Returns:
        val_accuracy: Accuracy en validación
    """
    # Limpiar sesión de Keras para evitar acumulación de memoria
    tf.keras.backend.clear_session()
    
    model = build_model(num_classes, params)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    # Callback para Optuna pruning
    if trial is not None:
        from optuna.integration import TFKerasPruningCallback
        callbacks.append(TFKerasPruningCallback(trial, 'val_accuracy'))
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=params['batch_size'],
        callbacks=callbacks,
        verbose=0
    )
    
    # Obtener mejor accuracy
    val_accuracy = max(history.history['val_accuracy'])
    val_loss = min(history.history['val_loss'])
    
    return val_accuracy, val_loss, model


# ============================================
# 1. GRID SEARCH
# ============================================

def grid_search(X_train, y_train, X_val, y_val, num_classes: int):
    """
    Búsqueda exhaustiva de hiperparámetros.
    ⚠️ Puede ser MUY lento dependiendo del espacio de búsqueda.
    """
    print("🔍 Iniciando Grid Search...")
    
    # Reducir espacio para grid search (sería muy lento con todos)
    grid_params = {
        'lstm_units_1': [64, 128],
        'lstm_units_2': [128, 256],
        'dense_units': [64, 128],
        'dropout_rate': [0.4, 0.5],
        'l2_lstm': [0.001, 0.01],
        'l2_dense': [0.001, 0.01],
        'learning_rate': [0.0001, 0.001],
        'batch_size': [32],
        'use_bidirectional': [False],
        'num_lstm_layers': [2],
    }
    
    from itertools import product
    
    # Generar todas las combinaciones
    keys = grid_params.keys()
    values = grid_params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"📊 Total de combinaciones: {len(combinations)}")
    
    best_accuracy = 0
    best_params = None
    results = []
    
    for i, params in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Probando: {params}")
        
        try:
            val_acc, val_loss, _ = evaluate_model(params, X_train, y_train, X_val, y_val, num_classes)
            print(f"✅ Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            results.append({
                'params': params,
                'val_accuracy': val_acc,
                'val_loss': val_loss
            })
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = params
                print(f"🎯 ¡Nuevo mejor modelo! Accuracy: {best_accuracy:.4f}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return best_params, best_accuracy, results


# ============================================
# 2. RANDOM SEARCH
# ============================================

def random_search(X_train, y_train, X_val, y_val, num_classes: int, n_iter=50):
    """
    Búsqueda aleatoria de hiperparámetros.
    Más rápido que Grid Search y a menudo igual de efectivo.
    """
    print(f"🎲 Iniciando Random Search con {n_iter} iteraciones...")
    
    best_accuracy = 0
    best_params = None
    results = []
    
    for i in range(n_iter):
        # Samplear parámetros aleatoriamente
        params = {
            'lstm_units_1': np.random.choice(SEARCH_SPACE['lstm_units_1']),
            'lstm_units_2': np.random.choice(SEARCH_SPACE['lstm_units_2']),
            'dense_units': np.random.choice(SEARCH_SPACE['dense_units']),
            'dropout_rate': np.random.choice(SEARCH_SPACE['dropout_rate']),
            'l2_lstm': np.random.choice(SEARCH_SPACE['l2_lstm']),
            'l2_dense': np.random.choice(SEARCH_SPACE['l2_dense']),
            'learning_rate': np.random.choice(SEARCH_SPACE['learning_rate']),
            'batch_size': np.random.choice(SEARCH_SPACE['batch_size']),
            'use_bidirectional': np.random.choice(SEARCH_SPACE['use_bidirectional']),
            'num_lstm_layers': np.random.choice(SEARCH_SPACE['num_lstm_layers']),
        }
        
        print(f"\n[{i+1}/{n_iter}] Probando: {params}")
        
        try:
            val_acc, val_loss, _ = evaluate_model(params, X_train, y_train, X_val, y_val, num_classes)
            print(f"✅ Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            results.append({
                'params': params,
                'val_accuracy': val_acc,
                'val_loss': val_loss
            })
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = params
                print(f"🎯 ¡Nuevo mejor modelo! Accuracy: {best_accuracy:.4f}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return best_params, best_accuracy, results


# ============================================
# 3. OPTUNA (Optimización Bayesiana)
# ============================================

def objective(trial, X_train, y_train, X_val, y_val, num_classes):
    """Función objetivo para Optuna"""
    
    # Sugerir hiperparámetros
    params = {
        'lstm_units_1': trial.suggest_categorical('lstm_units_1', SEARCH_SPACE['lstm_units_1']),
        'lstm_units_2': trial.suggest_categorical('lstm_units_2', SEARCH_SPACE['lstm_units_2']),
        'dense_units': trial.suggest_categorical('dense_units', SEARCH_SPACE['dense_units']),
        'dropout_rate': trial.suggest_categorical('dropout_rate', SEARCH_SPACE['dropout_rate']),
        'l2_lstm': trial.suggest_categorical('l2_lstm', SEARCH_SPACE['l2_lstm']),
        'l2_dense': trial.suggest_categorical('l2_dense', SEARCH_SPACE['l2_dense']),
        'learning_rate': trial.suggest_categorical('learning_rate', SEARCH_SPACE['learning_rate']),
        'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
        'use_bidirectional': trial.suggest_categorical('use_bidirectional', SEARCH_SPACE['use_bidirectional']),
        'num_lstm_layers': trial.suggest_categorical('num_lstm_layers', SEARCH_SPACE['num_lstm_layers']),
    }
    
    val_acc, val_loss, _ = evaluate_model(params, X_train, y_train, X_val, y_val, num_classes, trial)
    
    return val_acc


def optuna_search(X_train, y_train, X_val, y_val, num_classes: int, n_trials=100):
    """
    Búsqueda con Optuna (optimización bayesiana).
    Más eficiente que Random Search, aprende de trials anteriores.
    """
    print(f"🔬 Iniciando Optuna con {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, num_classes),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*60)
    print("📊 RESULTADOS DE OPTUNA")
    print("="*60)
    print(f"🎯 Mejor accuracy: {study.best_value:.4f}")
    print(f"📋 Mejores parámetros:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params, study.best_value, study


# ============================================
# VALIDACIÓN CRUZADA
# ============================================

def cross_validation_search(X, y, num_classes: int, params: Dict[str, Any], n_splits=5):
    """
    Evalúa un conjunto de parámetros usando validación cruzada.
    Más robusto que un solo split train/val.
    """
    print(f"🔄 Validación cruzada con {n_splits} folds...")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n📊 Fold {fold}/{n_splits}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        val_acc, val_loss, _ = evaluate_model(
            params, X_train_fold, y_train_fold, X_val_fold, y_val_fold, num_classes
        )
        
        accuracies.append(val_acc)
        print(f"✅ Fold {fold} - Val Accuracy: {val_acc:.4f}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n📈 Accuracy promedio: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc, accuracies


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """Script principal de búsqueda de hiperparámetros"""
    
    print("="*60)
    print("🚀 BÚSQUEDA DE HIPERPARÁMETROS LSTM")
    print("="*60)
    
    # Cargar datos
    if Path(WORDS_JSON_PATH).exists():
        gestures = json.load(open(WORDS_JSON_PATH, 'r', encoding='utf-8'))["word_ids"]
    else:
        gestures = ['hola-der', 'dias-gen', 'paz-der']
    
    print(f"\n📊 Gestos cargados: {gestures}")
    print(f"📊 Total de clases: {len(gestures)}")
    
    seqs, labels = load_sequences(gestures)
    if not seqs:
        raise ValueError("❌ No hay secuencias válidas. Ejecuta extracción de keypoints primero.")
    
    print(f"📊 Secuencias cargadas: {len(seqs)}")
    
    # Preparar datos
    X = pad_sequences(seqs, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float32')
    y = to_categorical(labels, num_classes=len(gestures))
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Datos de entrenamiento: {X_train.shape}")
    print(f"📊 Datos de validación: {X_val.shape}")
    
    # Seleccionar método de búsqueda
    print("\n" + "="*60)
    print("Selecciona el método de búsqueda:")
    print("="*60)
    print("1. Random Search (recomendado para empezar - rápido)")
    print("2. Optuna (optimización bayesiana - más eficiente)")
    print("3. Grid Search (búsqueda exhaustiva - muy lento)")
    print("4. Validación cruzada con parámetros actuales")
    
    choice = input("\nElige una opción (1-4): ").strip()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(MODEL_DIR) / "hyperparameter_search"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if choice == "1":
        n_iter = int(input("¿Cuántas iteraciones? (recomendado: 50-100): ") or "50")
        best_params, best_acc, results = random_search(
            X_train, y_train, X_val, y_val, len(gestures), n_iter
        )
        method = "random_search"
        
    elif choice == "2":
        n_trials = int(input("¿Cuántos trials? (recomendado: 100-200): ") or "100")
        best_params, best_acc, study = optuna_search(
            X_train, y_train, X_val, y_val, len(gestures), n_trials
        )
        method = "optuna"
        
        # Guardar estudio de Optuna
        study_file = results_dir / f"optuna_study_{timestamp}.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        print(f"\n💾 Estudio Optuna guardado en: {study_file}")
        
        # Visualizaciones de Optuna (opcionales)
        try:
            import optuna.visualization as vis
            fig = vis.plot_optimization_history(study)
            fig.write_html(results_dir / f"optuna_history_{timestamp}.html")
            
            fig = vis.plot_param_importances(study)
            fig.write_html(results_dir / f"optuna_importance_{timestamp}.html")
            
            print(f"📊 Visualizaciones guardadas en: {results_dir}")
        except Exception as e:
            print(f"⚠️  No se pudieron crear visualizaciones: {e}")
        
        results = None
        
    elif choice == "3":
        confirm = input("⚠️  Grid Search puede tardar HORAS. ¿Continuar? (s/n): ")
        if confirm.lower() != 's':
            print("❌ Cancelado")
            return
        best_params, best_acc, results = grid_search(
            X_train, y_train, X_val, y_val, len(gestures)
        )
        method = "grid_search"
        
    elif choice == "4":
        # Parámetros actuales del modelo
        current_params = {
            'lstm_units_1': 64,
            'lstm_units_2': 128,
            'dense_units': 64,
            'dropout_rate': 0.5,
            'l2_lstm': 0.01,
            'l2_dense': 0.001,
            'learning_rate': 0.001,
            'batch_size': 32,
            'use_bidirectional': False,
            'num_lstm_layers': 2,
        }
        mean_acc, std_acc, fold_accs = cross_validation_search(
            X, y, len(gestures), current_params
        )
        
        # Guardar resultados
        results_file = results_dir / f"cross_validation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'params': current_params,
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'fold_accuracies': [float(a) for a in fold_accs]
            }, f, indent=2)
        
        print(f"\n💾 Resultados guardados en: {results_file}")
        return
    
    else:
        print("❌ Opción inválida")
        return
    
    # Guardar resultados
    print("\n" + "="*60)
    print("💾 GUARDANDO RESULTADOS")
    print("="*60)
    
    results_file = results_dir / f"{method}_{timestamp}.json"
    
    save_data = {
        'method': method,
        'timestamp': timestamp,
        'best_params': best_params,
        'best_accuracy': float(best_acc),
        'num_classes': len(gestures),
        'gestures': gestures,
        'search_space': SEARCH_SPACE,
    }
    
    if results is not None:
        save_data['all_results'] = results
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✅ Resultados guardados en: {results_file}")
    
    # Entrenar modelo final con mejores parámetros
    print("\n" + "="*60)
    print("🎯 ENTRENANDO MODELO FINAL")
    print("="*60)
    
    train_final = input("\n¿Entrenar modelo final con mejores parámetros? (s/n): ")
    if train_final.lower() == 's':
        print("\n🏋️  Entrenando modelo final...")
        
        _, _, final_model = evaluate_model(
            best_params, X_train, y_train, X_val, y_val, len(gestures)
        )
        
        # Guardar modelo
        final_model_path = Path(MODEL_DIR) / f"best_model_{timestamp}.keras"
        final_model.save(final_model_path)
        print(f"✅ Modelo final guardado en: {final_model_path}")
        
        # Guardar parámetros
        params_file = Path(MODEL_DIR) / f"best_params_{timestamp}.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"✅ Parámetros guardados en: {params_file}")
    
    print("\n" + "="*60)
    print("✅ BÚSQUEDA COMPLETADA")
    print("="*60)
    print(f"🎯 Mejor accuracy alcanzado: {best_acc:.4f}")
    print(f"📋 Mejores parámetros:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")


if __name__ == '__main__':
    main()
