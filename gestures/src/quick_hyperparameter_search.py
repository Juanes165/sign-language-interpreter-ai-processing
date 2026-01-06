"""
Script simplificado de búsqueda de hiperparámetros usando Optuna.
Versión rápida y directa - solo lo esencial.
"""

import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import optuna

from app_constants import KEYPOINTS_PATH, MODEL_FRAMES, LENGTH_KEYPOINTS, MODEL_DIR, WORDS_JSON_PATH


def load_data():
    """Carga y prepara los datos"""
    if Path(WORDS_JSON_PATH).exists():
        gestures = json.load(open(WORDS_JSON_PATH, 'r', encoding='utf-8'))["word_ids"]
    else:
        gestures = ['hola-der', 'dias-gen', 'paz-der']
    
    seqs, labels = [], []
    for idx, g in enumerate(gestures):
        file = Path(KEYPOINTS_PATH) / f"{g}.npy"
        if not file.exists():
            continue
        arr = np.load(file, allow_pickle=True)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3 or arr.shape[-1] != LENGTH_KEYPOINTS:
            continue
        for seq in arr:
            seqs.append(seq)
            labels.append(idx)
    
    X = pad_sequences(seqs, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float32')
    y = to_categorical(labels, num_classes=len(gestures))
    
    return train_test_split(X, y, test_size=0.2, random_state=42), gestures


def create_model(trial, num_classes):
    """Crea modelo con hiperparámetros sugeridos por Optuna"""
    tf.keras.backend.clear_session()
    
    # Hiperparámetros a optimizar
    lstm_units = trial.suggest_categorical('lstm_units', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.3, 0.6, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-4, 1e-1)
    use_bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    
    # Construir modelo
    model = Sequential([
        Input(shape=(int(MODEL_FRAMES), LENGTH_KEYPOINTS)),
    ])
    
    # Primera LSTM
    lstm1 = LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg))
    model.add(Bidirectional(lstm1) if use_bidirectional else lstm1)
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    
    # Segunda LSTM
    lstm2 = LSTM(lstm_units * 2, return_sequences=False, kernel_regularizer=l2(l2_reg))
    model.add(Bidirectional(lstm2) if use_bidirectional else lstm2)
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    
    # Dense
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def objective(trial, X_train, y_train, X_val, y_val, num_classes):
    """Función objetivo para Optuna"""
    model = create_model(trial, num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0),
    ]
    
    # Entrenar
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    return max(history.history['val_accuracy'])


def main():
    print("="*60)
    print("🚀 BÚSQUEDA RÁPIDA DE HIPERPARÁMETROS (OPTUNA)")
    print("="*60)
    
    # Cargar datos
    print("\n📊 Cargando datos...")
    (X_train, X_val, y_train, y_val), gestures = load_data()
    num_classes = len(gestures)
    
    print(f"✅ Gestos: {gestures}")
    print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Configurar Optuna
    n_trials = int(input("\n¿Cuántos trials quieres ejecutar? (recomendado: 50-100): ") or "50")
    
    print(f"\n🔬 Iniciando búsqueda con {n_trials} trials...")
    print("⏳ Esto puede tardar un rato...\n")
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, num_classes),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Resultados
    print("\n" + "="*60)
    print("🎯 RESULTADOS")
    print("="*60)
    print(f"Mejor accuracy: {study.best_value:.4f}")
    print(f"\nMejores parámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Guardar
    results_dir = Path(MODEL_DIR) / "hyperparameter_search"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = results_dir / f"optuna_simple_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_accuracy': study.best_value,
            'n_trials': n_trials,
            'gestures': gestures,
        }, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: {results_file}")
    
    # Entrenar modelo final
    if input("\n¿Entrenar modelo final? (s/n): ").lower() == 's':
        print("\n🏋️  Entrenando modelo final...")
        
        # Crear modelo con mejores parámetros
        class MockTrial:
            def __init__(self, params):
                self.params = params
            def suggest_categorical(self, name, _):
                return self.params[name]
            def suggest_float(self, name, *args, **kwargs):
                return self.params[name]
            def suggest_loguniform(self, name, *args, **kwargs):
                return self.params[name]
        
        mock_trial = MockTrial(study.best_params)
        final_model = create_model(mock_trial, num_classes)
        
        final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=study.best_params['batch_size'],
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ],
            verbose=1
        )
        
        # Guardar
        model_file = Path(MODEL_DIR) / f"best_model_{timestamp}.keras"
        final_model.save(model_file)
        print(f"\n✅ Modelo guardado en: {model_file}")
    
    print("\n✅ ¡Búsqueda completada!")


if __name__ == '__main__':
    main()
