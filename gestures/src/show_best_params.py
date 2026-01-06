"""
Script para actualizar train_lstm_actions.py con los mejores parámetros encontrados.
Basado en los resultados de: optuna_simple_20251025_235212.json
"""

# MEJORES PARÁMETROS ENCONTRADOS POR OPTUNA
# Accuracy: 100%
# Trial: 0

BEST_PARAMS = {
    'lstm_units_1': 64,
    'lstm_units_2': 128,  # lstm_units_1 * 2
    'dense_units': 64,
    'dropout_rate': 0.4,
    'l2_lstm': 0.0016978765039737565,
    'l2_dense': 0.0016978765039737565,
    'learning_rate': 0.0006164347038978914,
    'batch_size': 16,
    'use_bidirectional': False,
}

print("="*60)
print("🎯 MEJORES HIPERPARÁMETROS ENCONTRADOS")
print("="*60)
print(f"\n✨ Accuracy alcanzado: 100%\n")
print("📋 Parámetros optimizados:")
for key, value in BEST_PARAMS.items():
    print(f"   {key}: {value}")

print("\n" + "="*60)
print("📝 CÓDIGO PARA train_lstm_actions.py")
print("="*60)

code = f"""
# ⚡ Modelo optimizado con Optuna (Accuracy: 100%)
# Parámetros encontrados automáticamente

def build_lstm_optimized(num_classes: int):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from keras.regularizers import l2
    from app_constants import MODEL_FRAMES, LENGTH_KEYPOINTS
    
    model = Sequential()
    model.add(Input(shape=(int(MODEL_FRAMES), LENGTH_KEYPOINTS)))
    
    # Primera capa LSTM
    model.add(LSTM({BEST_PARAMS['lstm_units_1']}, 
                   return_sequences=True, 
                   kernel_regularizer=l2({BEST_PARAMS['l2_lstm']})))
    model.add(Dropout({BEST_PARAMS['dropout_rate']}))
    model.add(BatchNormalization())
    
    # Segunda capa LSTM
    model.add(LSTM({BEST_PARAMS['lstm_units_2']}, 
                   return_sequences=False, 
                   kernel_regularizer=l2({BEST_PARAMS['l2_lstm']})))
    model.add(Dropout({BEST_PARAMS['dropout_rate']}))
    model.add(BatchNormalization())
    
    # Capas densas
    model.add(Dense({BEST_PARAMS['dense_units']}, 
                    activation='relu', 
                    kernel_regularizer=l2({BEST_PARAMS['l2_dense']})))
    model.add(Dropout({BEST_PARAMS['dropout_rate']}))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar con learning rate optimizado
    optimizer = tf.keras.optimizers.Adam(learning_rate={BEST_PARAMS['learning_rate']})
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Usar batch_size optimizado
OPTIMIZED_BATCH_SIZE = {BEST_PARAMS['batch_size']}
"""

print(code)

print("\n" + "="*60)
print("💡 INSTRUCCIONES")
print("="*60)
print("""
1. Abre train_lstm_actions.py
2. Reemplaza la función build_lstm() con build_lstm_optimized()
3. En main(), usa: model = build_lstm_optimized(num_classes=len(gestures))
4. Cambia batch_size=32 por batch_size=16
5. Entrena el modelo: python train_lstm_actions.py

O ejecuta este script para actualizar automáticamente:
python apply_optuna_params.py
""")

# Guardar en archivo
output_file = "optimized_params_code.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(code)

print(f"\n✅ Código guardado en: {output_file}")
