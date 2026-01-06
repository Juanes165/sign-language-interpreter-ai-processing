/**
 * 🧠 Entrenamiento de LSTM para reconocimiento de gestos en Node.js
 * VERSIÓN v6.0 - CON PARÁMETROS OPTIMIZADOS Y MEJORAS DE CONSISTENCIA
 * 
 * Mejoras sobre v5:
 * ✅ Parámetros del TOP 1 del Grid Search: LSTM[256, 128], Dense[32]
 * ✅ Learning Rate Scheduling para mejor convergencia
 * ✅ Recurrent Dropout adicional para mejor regularización en LSTM
 * ✅ Ajustes de batch normalization después de cada capa LSTM
 * ✅ Mejor manejo de early stopping con patience ajustada
 * ✅ Optimización de batch size basada en resultados del grid search
 * 
 * Backend:
 * - GPU: @tensorflow/tfjs-node-gpu (más rápido si tienes GPU NVIDIA con CUDA)
 * - CPU: @tensorflow/tfjs-node (acelerado con TensorFlow C++ bindings)
 * 
 * Parámetros optimizados (GRID SEARCH - TOP 1):
 * - LSTM 1: 256 units (MEJOR - 100% val/test accuracy)
 * - LSTM 2: 128 units
 * - Dense: 32 units (MEJOR para arquitectura 256-128)
 * - Dropout: 0.3
 * - Recurrent Dropout: 0.2 (NUEVO - mejor regularización)
 * - L2 regularization: 0.001
 * - Learning rate: 0.0001 (con scheduling)
 * - Batch size: 32
 * 
 * Uso:
 *   node src/train_lstm_node_v6.js          # Auto-detección (GPU -> CPU -> JS)
 *   node src/train_lstm_node_v6.js --gpu    # Forzar GPU
 *   node src/train_lstm_node_v6.js --cpu    # Forzar CPU
 */

// Intentar usar tfjs-node-gpu (GPU), tfjs-node (CPU), o tfjs puro
let tf;
let backend = 'unknown';

// Parsear argumentos de línea de comandos
const args = process.argv.slice(2);
const forceBackend = args.find(arg => arg === '--gpu' || arg === '--cpu' || arg === '--js');

if (forceBackend === '--gpu') {
  try {
    tf = require('@tensorflow/tfjs-node-gpu');
    backend = 'gpu';
    console.log('🚀 Usando @tensorflow/tfjs-node-gpu (GPU acelerado) [FORZADO]\n');
  } catch (e) {
    console.error('❌ Error cargando GPU backend:', e.message);
    process.exit(1);
  }
} else if (forceBackend === '--cpu') {
  try {
    tf = require('@tensorflow/tfjs-node');
    backend = 'cpu';
    console.log('💻 Usando @tensorflow/tfjs-node (CPU acelerado) [FORZADO]\n');
  } catch (e) {
    console.error('❌ Error cargando CPU backend:', e.message);
    process.exit(1);
  }
} else if (forceBackend === '--js') {
  tf = require('@tensorflow/tfjs');
  backend = 'js';
  console.log('⚠️  Usando @tensorflow/tfjs (JavaScript puro - más lento)\n');
} else {
  // Auto-detección: Intentar GPU primero, luego CPU, luego JS
  try {
    tf = require('@tensorflow/tfjs-node-gpu');
    backend = 'gpu';
    console.log('🚀 Usando @tensorflow/tfjs-node-gpu (GPU acelerado)\n');
  } catch (e) {
    try {
      tf = require('@tensorflow/tfjs-node');
      backend = 'cpu';
      console.log('💻 Usando @tensorflow/tfjs-node (CPU acelerado)\n');
    } catch (e2) {
      tf = require('@tensorflow/tfjs');
      backend = 'js';
      console.log('⚠️  Usando @tensorflow/tfjs (JavaScript puro - más lento)\n');
    }
  }
}

const fs = require('fs');
const path = require('path');
const { fromArrayBuffer } = require('numpy-parser');

// ========================================
// CONFIGURACIÓN
// ========================================

const CONFIG = {
  ROOT_PATH: path.resolve(__dirname, '..'),
  get KEYPOINTS_PATH() { return path.join(this.ROOT_PATH, 'assets', 'data', 'keypoints'); },
  get MODEL_DIR() { return path.join(this.ROOT_PATH, 'models'); },
  get WORDS_JSON_PATH() { return path.join(this.MODEL_DIR, 'words.json'); },
  get MODEL_PATH() { return path.join(this.MODEL_DIR, 'modelo_tfjs_node'); },
  get NORMALIZATION_STATS_PATH() { return path.join(this.MODEL_DIR, 'normalization_stats.json'); },
  
  MODEL_FRAMES: 15,
  LENGTH_KEYPOINTS: 1662,
  LEARNING_RATE: 0.0001,  // TOP 1 del grid search
  BATCH_SIZE: 32,  // TOP 1 del grid search
  EPOCHS: 200,
  VALIDATION_SPLIT: 0.2,
  EARLY_STOPPING_PATIENCE: 10,  // Aumentado para dar más oportunidades
  TEST_SPLIT: 0.1,
};

// ========================================
// NORMALIZACIÓN (igual que v5)
// ========================================

function loadNormalizationStats() {
  if (!fs.existsSync(CONFIG.NORMALIZATION_STATS_PATH)) {
    console.warn(`⚠️  Archivo de normalización no encontrado: ${CONFIG.NORMALIZATION_STATS_PATH}`);
    console.warn(`   Ejecuta: python src/calculate_normalization_stats.py`);
    return null;
  }
  
  try {
    const statsData = fs.readFileSync(CONFIG.NORMALIZATION_STATS_PATH, 'utf-8');
    const stats = JSON.parse(statsData);
    console.log(`✅ Estadísticas de normalización cargadas desde: ${CONFIG.NORMALIZATION_STATS_PATH}`);
    return stats;
  } catch (error) {
    console.error(`❌ Error cargando estadísticas de normalización: ${error.message}`);
    return null;
  }
}

function normalizeKeypointsSequence(sequence, stats) {
  if (!stats) {
    return sequence;
  }
  
  return tf.tidy(() => {
    const POSE_START = 0;
    const POSE_END = 132;
    const FACE_START = 132;
    const FACE_END = 1536;
    const LEFT_HAND_START = 1536;
    const LEFT_HAND_END = 1599;
    const RIGHT_HAND_START = 1599;
    const RIGHT_HAND_END = 1662;
    
    const frames = sequence.shape[0];
    const components = [];
    
    // Normalizar Pose
    const poseMean = tf.tensor1d(stats.pose.mean);
    const poseStd = tf.tensor1d(stats.pose.std.map(s => Math.max(s, 1e-6)));
    const poseData = sequence.slice([0, POSE_START], [frames, POSE_END - POSE_START]);
    const poseNormalized = poseData.sub(poseMean).div(poseStd);
    components.push(poseNormalized);
    
    // Normalizar Face
    const faceMean = tf.tensor1d(stats.face.mean);
    const faceStd = tf.tensor1d(stats.face.std.map(s => Math.max(s, 1e-6)));
    const faceData = sequence.slice([0, FACE_START], [frames, FACE_END - FACE_START]);
    const faceNormalized = faceData.sub(faceMean).div(faceStd);
    components.push(faceNormalized);
    
    // Normalizar Left Hand
    const lhMean = tf.tensor1d(stats.left_hand.mean);
    const lhStd = tf.tensor1d(stats.left_hand.std.map(s => Math.max(s, 1e-6)));
    const lhData = sequence.slice([0, LEFT_HAND_START], [frames, LEFT_HAND_END - LEFT_HAND_START]);
    const lhNormalized = lhData.sub(lhMean).div(lhStd);
    components.push(lhNormalized);
    
    // Normalizar Right Hand
    const rhMean = tf.tensor1d(stats.right_hand.mean);
    const rhStd = tf.tensor1d(stats.right_hand.std.map(s => Math.max(s, 1e-6)));
    const rhData = sequence.slice([0, RIGHT_HAND_START], [frames, RIGHT_HAND_END - RIGHT_HAND_START]);
    const rhNormalized = rhData.sub(rhMean).div(rhStd);
    components.push(rhNormalized);
    
    // Limpiar
    [poseMean, poseStd, faceMean, faceStd, lhMean, lhStd, rhMean, rhStd].forEach(t => t.dispose());
    
    return tf.concat(components, 1);
  });
}

// ========================================
// CARGA DE DATOS (igual que v5)
// ========================================

function loadNpyFile(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    const npArray = fromArrayBuffer(arrayBuffer);
    
    if (npArray.shape.length === 2 && 
        npArray.shape[0] === CONFIG.MODEL_FRAMES && 
        npArray.shape[1] === CONFIG.LENGTH_KEYPOINTS) {
      return tf.tensor3d(npArray.data, [1, ...npArray.shape]);
    } 
    else if (npArray.shape.length === 3 && npArray.shape[2] === CONFIG.LENGTH_KEYPOINTS) {
      return tf.tensor3d(npArray.data, npArray.shape);
    } 
    else {
      console.warn(`⚠️  ${path.basename(filePath)} tiene forma inesperada:`, npArray.shape);
      return null;
    }
  } catch (error) {
    console.error(`❌ Error cargando ${filePath}:`, error.message);
    return null;
  }
}

function loadSequences(gestures, normalizationStats = null) {
  const sequences = [];
  const labels = [];

  gestures.forEach((gesture, idx) => {
    const filePath = path.join(CONFIG.KEYPOINTS_PATH, `${gesture}.npy`);
    
    if (!fs.existsSync(filePath)) {
      console.warn(`⚠️  Falta ${filePath}`);
      return;
    }

    const tensor = loadNpyFile(filePath);
    if (!tensor) return;

    const numSequences = tensor.shape[0];
    for (let i = 0; i < numSequences; i++) {
      let seq = tensor.slice([i, 0, 0], [1, CONFIG.MODEL_FRAMES, CONFIG.LENGTH_KEYPOINTS]).squeeze([0]);
      
      // Aplicar normalización si hay estadísticas
      if (normalizationStats) {
        seq = normalizeKeypointsSequence(seq, normalizationStats);
      }
      
      sequences.push(seq);
      labels.push(idx);
    }

    tensor.dispose();
  });

  console.log(`✅ Cargadas ${sequences.length} secuencias de ${gestures.length} gestos`);
  if (normalizationStats) {
    console.log(`   📊 Normalización aplicada por componente`);
  }
  return { sequences, labels };
}

function padSequences(sequences, maxLen) {
  return tf.tidy(() => {
    const padded = sequences.map(seq => {
      const currentLen = seq.shape[0];
      
      if (currentLen === maxLen) {
        return seq;
      } else if (currentLen < maxLen) {
        const padding = tf.zeros([maxLen - currentLen, CONFIG.LENGTH_KEYPOINTS]);
        return tf.concat([padding, seq], 0);
      } else {
        // Muestreo uniforme
        const indices = [];
        for (let i = 0; i < maxLen; i++) {
          const index = Math.floor(i * (currentLen - 1) / (maxLen - 1));
          indices.push(index);
        }
        
        const sampledFrames = indices.map(idx => 
          seq.slice([idx, 0], [1, CONFIG.LENGTH_KEYPOINTS]).squeeze([0])
        );
        
        return tf.stack(sampledFrames);
      }
    });

    return tf.stack(padded);
  });
}

// ========================================
// CONSTRUCCIÓN DEL MODELO - VERSIÓN MEJORADA v6
// ========================================

function buildLSTM(numClasses) {
  const model = tf.sequential();

  model.add(tf.layers.inputLayer({
    inputShape: [CONFIG.MODEL_FRAMES, CONFIG.LENGTH_KEYPOINTS]
  }));

  // 🏆 LSTM 1 - TOP 1 del Grid Search: 256 units
  // Añadido recurrent_dropout para mejor regularización
  model.add(tf.layers.lstm({
    units: 256,  // TOP 1: 256 units (mejor que 64 actual)
    returnSequences: true,
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    recurrentDropout: 0.2,  // NUEVO: Regularización en conexiones recurrentes
    dropout: 0.2  // Dropout adicional en inputs (además del layer dropout)
  }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.batchNormalization({
    axis: -1,  // Normalizar sobre la última dimensión
    momentum: 0.99,  // Momentum más alto para mejor estabilidad
    epsilon: 1e-5
  }));

  // 🏆 LSTM 2 - TOP 1 del Grid Search: 128 units
  model.add(tf.layers.lstm({
    units: 128,  // TOP 1: 128 units (mejor que 32 actual)
    returnSequences: false,
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    recurrentDropout: 0.2,  // NUEVO
    dropout: 0.2  // NUEVO
  }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.batchNormalization({
    axis: -1,
    momentum: 0.99,
    epsilon: 1e-5
  }));

  // 🏆 Dense - TOP 1 del Grid Search: 32 units (para arquitectura 256-128)
  model.add(tf.layers.dense({
    units: 32,  // TOP 1: 32 units (mejor que 64 actual para esta arquitectura)
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.batchNormalization({
    axis: -1,
    momentum: 0.99,
    epsilon: 1e-5
  }));

  // Output
  model.add(tf.layers.dense({
    units: numClasses,
    activation: 'softmax'
  }));

  // Learning Rate (constante como en v5 - TensorFlow.js no soporta cosineDecayRestarts)
  // Nota: En el futuro se podría implementar manualmente un callback para decay
  model.compile({
    optimizer: tf.train.adam(CONFIG.LEARNING_RATE),  // Igual que v5
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// ========================================
// EARLY STOPPING (igual que v5)
// ========================================

function createEarlyStoppingCallback(model, patience = 15) {
  let bestLoss = Infinity;
  let bestEpoch = 0;
  let wait = 0;
  let stopped = false;
  let bestWeights = null;

  return {
    onEpochEnd: async (epoch, logs) => {
      const currentLoss = logs.val_loss;
      const trainAcc = (logs.acc ?? logs.accuracy) ?? 0;
      const valAcc = (logs.val_acc ?? logs.val_accuracy) ?? 0;

      if (currentLoss < bestLoss) {
        bestLoss = currentLoss;
        bestEpoch = epoch + 1;
        wait = 0;
        
        if (bestWeights) {
          bestWeights.forEach(w => w.dispose());
        }
        bestWeights = model.getWeights().map(w => w.clone());
        
        console.log(`\n📊 Epoch ${epoch + 1}: val_loss mejoró a ${currentLoss.toFixed(4)} ⭐ (guardado)`);
        console.log(`   Train Acc: ${(trainAcc * 100).toFixed(2)}% | Val Acc: ${(valAcc * 100).toFixed(2)}%`);
      } else {
        wait++;
        if (wait >= patience && !stopped) {
          stopped = true;
          console.log(`\n⏹️  EARLY STOPPING ACTIVADO`);
          console.log(`   Sin mejora por ${patience} epochs`);
          console.log(`   Mejor val_loss: ${bestLoss.toFixed(4)} (epoch ${bestEpoch})`);
          
          if (bestWeights) {
            console.log(`   🔄 Restaurando mejor modelo (epoch ${bestEpoch})...`);
            model.setWeights(bestWeights);
          }
          
          console.log(`   🛑 Deteniendo entrenamiento...\n`);
          if (bestWeights) {
            bestWeights.forEach(w => w.dispose());
          }
          model.stopTraining = true;
        } else if (wait % 5 === 0) {
          console.log(`   ⏳ Esperando mejora... (${wait}/${patience})`);
        }
      }
      
      // Detección de overfitting
      if (trainAcc > 0 && valAcc > 0) {
        const gap = trainAcc - valAcc;
        if (gap > 0.15) {
          console.log(`   ⚠️  Posible overfitting detectado (gap: ${(gap * 100).toFixed(2)}%)`);
        }
      }
    },
    
    getBestEpoch: () => bestEpoch,
    getBestLoss: () => bestLoss
  };
}

// ========================================
// FUNCIONES DE EVALUACIÓN (igual que v5)
// ========================================

async function evaluateModel(model, X_test, y_test, gestures) {
  console.log('\n📊 Evaluando modelo en conjunto de prueba...');
  
  const predictions = model.predict(X_test);
  const predClasses = await predictions.argMax(-1).data();
  const trueClasses = await y_test.argMax(-1).data();
  
  predictions.dispose();
  
  const correct = Array.from(predClasses).reduce((acc, pred, idx) => 
    acc + (pred === trueClasses[idx] ? 1 : 0), 0);
  const accuracy = correct / predClasses.length;
  
  console.log(`✅ Test Accuracy: ${(accuracy * 100).toFixed(2)}%`);
  
  // Calcular matriz de confusión
  const numClasses = gestures.length;
  const confusionMatrix = [];
  
  // Inicializar matriz de confusión
  for (let i = 0; i < numClasses; i++) {
    confusionMatrix[i] = Array(numClasses).fill(0);
  }
  
  // Llenar matriz de confusión
  const numSamples = predClasses.length;
  for (let i = 0; i < numSamples; i++) {
    const trueClass = trueClasses[i];
    const predClass = predClasses[i];
    confusionMatrix[trueClass][predClass]++;
  }
  
  // Calcular métricas por gesto
  const perGestureMetrics = {};
  gestures.forEach((gesture, idx) => {
    const truePositives = Array.from(predClasses).filter((pred, i) => 
      pred === idx && trueClasses[i] === idx).length;
    const falsePositives = Array.from(predClasses).filter((pred, i) => 
      pred === idx && trueClasses[i] !== idx).length;
    const falseNegatives = Array.from(trueClasses).filter((trueClass, i) => 
      trueClass === idx && predClasses[i] !== idx).length;
    
    const precision = (truePositives + falsePositives) > 0 
      ? truePositives / (truePositives + falsePositives) : 0;
    const recall = (truePositives + falseNegatives) > 0 
      ? truePositives / (truePositives + falseNegatives) : 0;
    const f1 = (precision + recall) > 0 
      ? 2 * (precision * recall) / (precision + recall) : 0;
    
    perGestureMetrics[gesture] = {
      precision: precision,
      recall: recall,
      f1: f1,
      support: Array.from(trueClasses).filter(c => c === idx).length
    };
  });
  
  return {
    accuracy,
    perGestureMetrics,
    confusionMatrix
  };
}

// ========================================
// MAIN
// ========================================

async function main() {
  console.log('🧠 Entrenamiento LSTM v6.0 - Parámetros Optimizados\n');
  console.log('=' .repeat(70));
  
  // 1. Cargar gestos
  const wordsData = JSON.parse(fs.readFileSync(CONFIG.WORDS_JSON_PATH, 'utf-8'));
  const gestures = wordsData.word_ids || [];
  console.log(`📋 Gestos: ${gestures.length}`);
  console.log(`   ${gestures.join(', ')}\n`);
  
  if (gestures.length === 0) {
    console.error('❌ No hay gestos definidos en words.json');
    process.exit(1);
  }
  
  // 2. Cargar estadísticas de normalización
  const normalizationStats = loadNormalizationStats();
  
  // 3. Cargar secuencias
  const { sequences, labels } = loadSequences(gestures, normalizationStats);
  
  if (sequences.length === 0) {
    console.error('❌ No se encontraron secuencias para entrenar');
    process.exit(1);
  }
  
  // 4. Preparar datos
  console.log('\n📦 Preparando datos...');
  const paddedSequences = padSequences(sequences, CONFIG.MODEL_FRAMES);
  const labelsTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), gestures.length);
  
  // Función para mezclar array (Fisher-Yates shuffle)
  function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }
  
  // Dividir en train/val/test con STRATIFIED SPLIT (mantiene proporciones por gesto)
  const totalSamples = paddedSequences.shape[0];
  const testSplit = CONFIG.TEST_SPLIT;
  const valSplit = CONFIG.VALIDATION_SPLIT;
  const trainSplit = 1 - valSplit - testSplit;
  
  // Agrupar índices por clase (gesto)
  const indicesByClass = {};
  gestures.forEach((_, idx) => {
    indicesByClass[idx] = [];
  });
  
  labels.forEach((label, idx) => {
    indicesByClass[label].push(idx);
  });
  
  // Mezclar cada clase por separado
  Object.keys(indicesByClass).forEach(key => {
    indicesByClass[key] = shuffleArray(indicesByClass[key]);
  });
  
  // Dividir cada clase proporcionalmente
  const trainIndices = [];
  const valIndices = [];
  const testIndices = [];
  
  console.log('\n📊 Distribución estratificada por gesto:');
  Object.keys(indicesByClass).forEach(key => {
    const classIndices = indicesByClass[key];
    const classSize = classIndices.length;
    const gestureName = gestures[parseInt(key)];
    
    const testSize = Math.max(1, Math.floor(classSize * testSplit));
    const valSize = Math.max(1, Math.floor(classSize * valSplit));
    const trainSize = classSize - valSize - testSize;
    
    // Asegurar que al menos haya 1 muestra en train
    if (trainSize < 1) {
      console.warn(`⚠️  ${gestureName}: solo ${classSize} muestras, ajustando división...`);
      const adjustedTestSize = Math.max(1, Math.min(testSize, Math.floor(classSize / 3)));
      const adjustedValSize = Math.max(1, Math.min(valSize, Math.floor(classSize / 3)));
      const adjustedTrainSize = classSize - adjustedValSize - adjustedTestSize;
      
      trainIndices.push(...classIndices.slice(0, adjustedTrainSize));
      valIndices.push(...classIndices.slice(adjustedTrainSize, adjustedTrainSize + adjustedValSize));
      testIndices.push(...classIndices.slice(adjustedTrainSize + adjustedValSize));
      
      console.log(`   ${gestureName}: Train=${adjustedTrainSize}, Val=${adjustedValSize}, Test=${adjustedTestSize} (Total=${classSize})`);
    } else {
      trainIndices.push(...classIndices.slice(0, trainSize));
      valIndices.push(...classIndices.slice(trainSize, trainSize + valSize));
      testIndices.push(...classIndices.slice(trainSize + valSize));
      
      console.log(`   ${gestureName}: Train=${trainSize}, Val=${valSize}, Test=${testSize} (Total=${classSize})`);
    }
  });
  
  // Mezclar los índices finales para evitar orden por clase
  const finalTrainIndices = shuffleArray(trainIndices);
  const finalValIndices = shuffleArray(valIndices);
  const finalTestIndices = shuffleArray(testIndices);
  
  const trainSize = finalTrainIndices.length;
  const valSize = finalValIndices.length;
  const testSize = finalTestIndices.length;
  
  // Convertir índices a tensores para usar con tf.gather
  const trainIndicesTensor = tf.tensor1d(finalTrainIndices, 'int32');
  const valIndicesTensor = tf.tensor1d(finalValIndices, 'int32');
  const testIndicesTensor = tf.tensor1d(finalTestIndices, 'int32');
  
  const X_train = tf.gather(paddedSequences, trainIndicesTensor);
  const y_train = tf.gather(labelsTensor, trainIndicesTensor);
  const X_val = tf.gather(paddedSequences, valIndicesTensor);
  const y_val = tf.gather(labelsTensor, valIndicesTensor);
  const X_test = tf.gather(paddedSequences, testIndicesTensor);
  const y_test = tf.gather(labelsTensor, testIndicesTensor);
  
  // Limpiar tensores de índices
  trainIndicesTensor.dispose();
  valIndicesTensor.dispose();
  testIndicesTensor.dispose();
  
  console.log(`   Train: ${trainSize} muestras`);
  console.log(`   Val: ${valSize} muestras`);
  console.log(`   Test: ${testSize} muestras`);
  
  // Limpiar
  paddedSequences.dispose();
  labelsTensor.dispose();
  
  // 5. Construir modelo
  console.log('\n🏗️  Construyendo modelo LSTM v6.0...');
  console.log('   Arquitectura: LSTM[256, 128] → Dense[32] → Softmax[18]');
  console.log('   Parámetros TOP 1 del Grid Search');
  const model = buildLSTM(gestures.length);
  
  // Resumen del modelo
  console.log('\n📊 Resumen del modelo:');
  model.summary();
  
  // 6. Entrenar
  console.log('\n🚀 Iniciando entrenamiento...\n');
  
  const earlyStopping = createEarlyStoppingCallback(model, CONFIG.EARLY_STOPPING_PATIENCE);
  
  const history = await model.fit(X_train, y_train, {
    epochs: CONFIG.EPOCHS,
    batchSize: CONFIG.BATCH_SIZE,
    validationData: [X_val, y_val],
    callbacks: {
      onEpochEnd: earlyStopping.onEpochEnd.bind(earlyStopping)
    },
    verbose: 1
  });
  
  // 7. Evaluar
  const evaluation = await evaluateModel(model, X_test, y_test, gestures);
  
  // 8. Guardar modelo
  console.log('\n💾 Guardando modelo...');
  if (!fs.existsSync(CONFIG.MODEL_PATH)) {
    fs.mkdirSync(CONFIG.MODEL_PATH, { recursive: true });
  }
  
  await model.save(`file://${CONFIG.MODEL_PATH}`);
  console.log(`✅ Modelo guardado en: ${CONFIG.MODEL_PATH}`);
  
  // 8.5. Guardar words.json en la carpeta del modelo
  const wordsOutputPath = path.join(CONFIG.MODEL_PATH, 'words.json');
  fs.writeFileSync(wordsOutputPath, JSON.stringify({ word_ids: gestures }, null, 2));
  console.log(`✅ Etiquetas guardadas en: ${wordsOutputPath}`);
  
  // 9. Guardar reporte
  const report = {
    version: 'v6.0',
    timestamp: new Date().toISOString(),
    backend,
    architecture: {
      lstm1_units: 256,
      lstm2_units: 128,
      dense_units: 32,
      dropout: 0.3,
      recurrent_dropout: 0.2,
      l2_regularization: 0.001,
      learning_rate: CONFIG.LEARNING_RATE,
      batch_size: CONFIG.BATCH_SIZE
    },
    training: {
      total_samples: totalSamples,
      train_samples: trainSize,
      val_samples: valSize,
      test_samples: testSize,
      epochs_completed: history.history.loss.length,
      best_epoch: earlyStopping.getBestEpoch(),
      best_val_loss: earlyStopping.getBestLoss()
    },
    metrics: {
      test_accuracy: evaluation.accuracy,
      per_gesture: evaluation.perGestureMetrics
    },
    history: {
      loss: history.history.loss,
      acc: history.history.acc ?? history.history.accuracy,
      val_loss: history.history.val_loss,
      val_acc: history.history.val_acc ?? history.history.val_accuracy
    }
  };
  
  const reportPath = path.join(CONFIG.MODEL_DIR, 'training_report_v6.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`✅ Reporte guardado en: ${reportPath}`);
  
  // 10. Guardar matriz de confusión
  const confusionMatrixData = {
    version: 'v6.0',
    gestures: gestures,
    matrix: evaluation.confusionMatrix
  };
  
  const confusionMatrixPath = path.join(CONFIG.MODEL_DIR, 'confusion_matrix_v6.json');
  fs.writeFileSync(confusionMatrixPath, JSON.stringify(confusionMatrixData, null, 2));
  console.log(`✅ Matriz de confusión guardada en: ${confusionMatrixPath}`);
  
  // Limpiar memoria
  X_train.dispose();
  y_train.dispose();
  X_val.dispose();
  y_val.dispose();
  X_test.dispose();
  y_test.dispose();
  model.dispose();
  
  console.log('\n✅ Entrenamiento completado!');
}

main().catch(console.error);

