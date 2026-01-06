import json
import numpy as np
from pathlib import Path
from typing import List
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from app_constants import (
    KEYPOINTS_PATH, MODEL_FRAMES, LENGTH_KEYPOINTS,
    MODEL_DIR, WORDS_JSON_PATH, KERAS_MODEL_PATH, SAVEDMODEL_DIR
)

def load_sequences(gestures: List[str]):
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

def build_lstm(num_classes: int):
    model = Sequential()
    # 🔹 Primera capa LSTM define el input_shape (sin Input() explícito)
    model.add(LSTM(64, return_sequences=True, input_shape=(int(MODEL_FRAMES), LENGTH_KEYPOINTS), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    if Path(WORDS_JSON_PATH).exists():
        gestures = json.load(open(WORDS_JSON_PATH, 'r', encoding='utf-8'))["word_ids"]
    else:
        gestures = ['hola-der', 'dias-gen', 'paz-der']

    seqs, labels = load_sequences(gestures)
    if not seqs:
        raise ValueError("No hay secuencias válidas. Ejecuta extracción de keypoints primero.")

    X = pad_sequences(seqs, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float32')
    y = to_categorical(labels, num_classes=len(gestures))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    model = build_lstm(num_classes=len(gestures))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es])

    out = Path(KERAS_MODEL_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 🔹 Guardar .h5 (para TensorFlow.js)
    # Keras 3 infiere el formato desde la extensión, no necesita save_format
    out_h5 = out.parent / f"actions_{int(MODEL_FRAMES)}.h5"
    model.save(out_h5)
    print(f"✅ Guardado modelo H5 (TFJS compatible): {out_h5}")

    # 🔹 Guardar .keras (nativo)
    model.save(out)
    print(f"✅ Guardado modelo KERAS nativo: {out}")

    # 🔹 Exportar en formato SavedModel (para .tflite o despliegue directo)
    # En Keras 3, usar model.export() para SavedModel
    savedmodel_path = Path(SAVEDMODEL_DIR)
    savedmodel_path.mkdir(parents=True, exist_ok=True)
    model.export(savedmodel_path)
    print(f"✅ Exportado modelo SavedModel: {savedmodel_path}")

if __name__ == '__main__':
    main()
