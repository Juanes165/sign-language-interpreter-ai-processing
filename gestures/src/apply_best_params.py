"""
Script para aplicar los mejores hiperparámetros encontrados al modelo de entrenamiento.
Lee los resultados de la búsqueda y actualiza el modelo.
"""

import json
from pathlib import Path
from app_constants import MODEL_DIR

def load_best_params(results_file=None):
    """
    Carga los mejores parámetros desde un archivo de resultados.
    
    Args:
        results_file: Ruta al archivo JSON de resultados. Si es None, busca el más reciente.
    """
    results_dir = Path(MODEL_DIR) / "hyperparameter_search"
    
    if results_file is None:
        # Buscar el archivo más reciente
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError("No se encontraron archivos de resultados")
        
        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Usando archivo más reciente: {results_file.name}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data['best_params'], data['best_accuracy']


def generate_model_code(params):
    """
    Genera el código del modelo con los parámetros optimizados.
    """
    code = f"""
def build_lstm_optimized(num_classes: int):
    \"\"\"
    Modelo LSTM optimizado usando búsqueda de hiperparámetros.
    Parámetros encontrados con accuracy: {params.get('accuracy', 'N/A')}
    \"\"\"
    model = Sequential()
    model.add(Input(shape=(int(MODEL_FRAMES), LENGTH_KEYPOINTS)))
    
    # Primera capa LSTM
    lstm_1 = LSTM(
        {params.get('lstm_units_1', 64)},
        return_sequences={params.get('num_lstm_layers', 2) > 1},
        kernel_regularizer=l2({params.get('l2_lstm', 0.01)})
    )
    """
    
    if params.get('use_bidirectional', False):
        code += """
    model.add(Bidirectional(lstm_1))"""
    else:
        code += """
    model.add(lstm_1)"""
    
    code += f"""
    model.add(Dropout({params.get('dropout_rate', 0.5)}))
    model.add(BatchNormalization())
    """
    
    if params.get('num_lstm_layers', 2) >= 2:
        code += f"""
    # Segunda capa LSTM
    lstm_2 = LSTM(
        {params.get('lstm_units_2', 128)},
        return_sequences={params.get('num_lstm_layers', 2) > 2},
        kernel_regularizer=l2({params.get('l2_lstm', 0.01)})
    )
    """
        
        if params.get('use_bidirectional', False):
            code += """
    model.add(Bidirectional(lstm_2))"""
        else:
            code += """
    model.add(lstm_2)"""
        
        code += f"""
    model.add(Dropout({params.get('dropout_rate', 0.5)}))
    model.add(BatchNormalization())
    """
    
    code += f"""
    # Capas densas
    model.add(Dense(
        {params.get('dense_units', 64)},
        activation='relu',
        kernel_regularizer=l2({params.get('l2_dense', 0.001)})
    ))
    model.add(Dropout({params.get('dropout_rate', 0.5)}))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar con learning rate optimizado
    optimizer = tf.keras.optimizers.Adam(learning_rate={params.get('learning_rate', 0.001)})
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Parámetros de entrenamiento optimizados
OPTIMIZED_BATCH_SIZE = {params.get('batch_size', 32)}
OPTIMIZED_EPOCHS = 100  # Con EarlyStopping
"""
    
    return code


def generate_training_code(params):
    """Genera el código de entrenamiento con los parámetros optimizados"""
    return f"""
# Entrenar con parámetros optimizados
model = build_lstm_optimized(num_classes=len(gestures))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size={params.get('batch_size', 32)},
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ],
    verbose=1
)
"""


def main():
    print("="*60)
    print("📊 APLICAR MEJORES HIPERPARÁMETROS")
    print("="*60)
    
    try:
        params, accuracy = load_best_params()
        
        print(f"\n🎯 Mejor accuracy encontrado: {accuracy:.4f}")
        print(f"\n📋 Parámetros optimizados:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Generar código
        params_with_acc = {**params, 'accuracy': accuracy}
        model_code = generate_model_code(params_with_acc)
        training_code = generate_training_code(params)
        
        # Guardar en archivo
        output_file = Path(MODEL_DIR) / "optimized_model_code.py"
        
        full_code = """# Código generado automáticamente con parámetros optimizados
# NO editar manualmente - generado por apply_best_params.py

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from app_constants import MODEL_FRAMES, LENGTH_KEYPOINTS

""" + model_code + "\n\n" + training_code
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        print(f"\n✅ Código generado en: {output_file}")
        print("\n📝 Próximos pasos:")
        print("   1. Revisa el código generado")
        print("   2. Copia la función build_lstm_optimized() a train_lstm_actions.py")
        print("   3. Reemplaza build_lstm() por build_lstm_optimized()")
        print("   4. Actualiza BATCH_SIZE si es necesario")
        print("   5. Entrena el modelo final: python train_lstm_actions.py")
        
        # Opción de actualizar automáticamente
        auto_update = input("\n¿Quieres actualizar train_lstm_actions.py automáticamente? (s/n): ")
        
        if auto_update.lower() == 's':
            update_training_script(params_with_acc)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Ejecuta primero:")
        print("   python hyperparameter_tuning.py")
        print("   o")
        print("   python quick_hyperparameter_search.py")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")


def update_training_script(params):
    """Actualiza train_lstm_actions.py con los nuevos parámetros"""
    
    from pathlib import Path
    
    script_path = Path(__file__).parent / "train_lstm_actions.py"
    
    if not script_path.exists():
        print(f"❌ No se encontró {script_path}")
        return
    
    # Leer archivo actual
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Crear respaldo
    backup_path = script_path.parent / f"{script_path.stem}_backup.py"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Respaldo creado en: {backup_path}")
    
    # Generar nueva función
    new_function = generate_model_code(params).strip()
    
    # Buscar y reemplazar la función build_lstm
    import re
    
    # Patrón para encontrar la función build_lstm
    pattern = r'def build_lstm\(num_classes: int\):.*?(?=\n\ndef |\n\nif __name__|$)'
    
    if re.search(pattern, content, re.DOTALL):
        # Reemplazar
        new_content = re.sub(
            pattern,
            new_function.replace('build_lstm_optimized', 'build_lstm'),
            content,
            flags=re.DOTALL
        )
        
        # Guardar
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ train_lstm_actions.py actualizado con nuevos parámetros")
        print(f"📊 Accuracy esperado: ~{params.get('accuracy', 'N/A')}")
        
    else:
        print("⚠️  No se pudo encontrar la función build_lstm para reemplazar")
        print("   Por favor, copia manualmente la función desde optimized_model_code.py")


if __name__ == '__main__':
    main()
