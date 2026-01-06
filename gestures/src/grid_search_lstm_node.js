/**
 * 🔍 Grid Search de Hiperparámetros para LSTM en Node.js
 * VERSIÓN COMPLETA - Búsqueda exhaustiva de hiperparámetros
 * 
 * Basado en: LSTM_Hyperparameter_Search_Local.ipynb
 * 
 * Este script realiza una búsqueda exhaustiva de hiperparámetros
 * para el modelo LSTM de reconocimiento de gestos.
 * 
 * Uso:
 *   node src/grid_search_lstm_node.js
 *   node src/grid_search_lstm_node.js --gpu    # Forzar GPU
 *   node src/grid_search_lstm_node.js --cpu    # Forzar CPU
 *   node src/grid_search_lstm_node.js --fast   # Búsqueda rápida (menos combinaciones)
 */

// Intentar usar tfjs-node-gpu, tfjs-node, o tfjs puro
let tf;
let backend = 'unknown';

// Parsear argumentos de línea de comandos
const args = process.argv.slice(2);
const forceBackend = args.find(arg => arg === '--gpu' || arg === '--cpu' || arg === '--js');
const fastMode = args.includes('--fast');

if (forceBackend === '--gpu') {
  try {
    tf = require('@tensorflow/tfjs-node-gpu');
    backend = 'gpu';
    console.log('🚀 Usando @tensorflow/tfjs-node-gpu (GPU acelerado) [FORZADO]\n');
  } catch (e) {
    console.error('❌ Error: No se pudo cargar @tensorflow/tfjs-node-gpu');
    console.error(`   Detalles: ${e.message}`);
    console.error('   Instala con: npm install @tensorflow/tfjs-node-gpu');
    process.exit(1);
  }
} else if (forceBackend === '--cpu') {
  try {
    tf = require('@tensorflow/tfjs-node');
    backend = 'cpu';
    console.log('⚡ Usando @tensorflow/tfjs-node (CPU acelerado) [FORZADO]\n');
  } catch (e) {
    console.error('❌ Error: No se pudo cargar @tensorflow/tfjs-node');
    process.exit(1);
  }
} else if (forceBackend === '--js') {
  tf = require('@tensorflow/tfjs');
  backend = 'js';
  console.log('⚠️  Usando @tensorflow/tfjs (JavaScript puro - más lento) [FORZADO]\n');
} else {
  // Auto-detección
  try {
    tf = require('@tensorflow/tfjs-node-gpu');
    backend = 'gpu';
    console.log('🚀 Usando @tensorflow/tfjs-node-gpu (GPU acelerado)\n');
  } catch (e) {
    try {
      tf = require('@tensorflow/tfjs-node');
      backend = 'cpu';
      console.log('⚡ Usando @tensorflow/tfjs-node (CPU acelerado)\n');
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
  MODEL_FRAMES: 15,
  LENGTH_KEYPOINTS: 1662,
  ROOT_PATH: path.join(__dirname, '..'),
  EPOCHS: 100,
  VALIDATION_SPLIT: 0.2,
  EARLY_STOPPING_PATIENCE: 10,
};

CONFIG.KEYPOINTS_PATH = path.join(CONFIG.ROOT_PATH, 'assets', 'data', 'keypoints');
CONFIG.MODEL_DIR = path.join(CONFIG.ROOT_PATH, 'models');
CONFIG.WORDS_JSON_PATH = path.join(CONFIG.MODEL_DIR, 'words.json');
CONFIG.NORMALIZATION_STATS_PATH = path.join(CONFIG.MODEL_DIR, 'normalization_stats.json');

// ========================================
// GRID DE PARÁMETROS
// ========================================
// Mismo grid que en el notebook de Python
const paramGrid = fastMode ? {
  // Versión rápida para pruebas
  'lstm_units_1': [64, 128],
  'lstm_units_2': [32, 64],
  'dense_units': [32, 64],
  'dropout_rate': [0.3, 0.4],
  'l2_regularizer': [0.001, 0.01],
  'learning_rate': [0.0001, 0.001],
  'batch_size': [32],
} : {
  // Grid completo (432 combinaciones)
  'lstm_units_1': [64, 128, 256],
  'lstm_units_2': [32, 64, 128],
  'dense_units': [32, 64],
  'dropout_rate': [0.3, 0.4, 0.5],
  'l2_regularizer': [0.001, 0.01],
  'learning_rate': [0.0001, 0.001],
  'batch_size': [32, 64],
};

// Calcular total de combinaciones
function calculateTotalCombinations(grid) {
  return Object.values(grid).reduce((acc, values) => acc * values.length, 1);
}

const totalCombinations = calculateTotalCombinations(paramGrid);

// ========================================
// GENERAR COMBINACIONES (cartesian product)
// ========================================
function cartesianProduct(obj) {
  const keys = Object.keys(obj);
  const values = Object.values(obj);
  
  // Función para combinar arrays
  function combine(arrays) {
    if (arrays.length === 0) return [[]];
    const first = arrays[0];
    const rest = combine(arrays.slice(1));
    const result = [];
    
    for (const item of first) {
      for (const combination of rest) {
        result.push([item, ...combination]);
      }
    }
    
    return result;
  }
  
  const combinations = combine(values);
  
  return combinations.map(combo => {
    const result = {};
    keys.forEach((key, idx) => {
      result[key] = combo[idx];
    });
    return result;
  });
}

// ========================================
// NORMALIZACIÓN DE KEYPOINTS
// ========================================

/**
 * Carga las estadísticas de normalización desde archivo JSON
 */
function loadNormalizationStats() {
  if (!fs.existsSync(CONFIG.NORMALIZATION_STATS_PATH)) {
    console.warn(`⚠️  Archivo de normalización no encontrado: ${CONFIG.NORMALIZATION_STATS_PATH}`);
    console.warn(`   Ejecuta: python src/calculate_normalization_stats.py`);
    return null;
  }
  
  try {
    const statsData = fs.readFileSync(CONFIG.NORMALIZATION_STATS_PATH, 'utf-8');
    const stats = JSON.parse(statsData);
    return stats;
  } catch (error) {
    console.error(`❌ Error cargando estadísticas de normalización: ${error.message}`);
    return null;
  }
}

/**
 * Normaliza una secuencia de keypoints por componente usando estadísticas
 * @param {tf.Tensor} sequence - Tensor de forma (frames, keypoints)
 * @param {Object} stats - Estadísticas de normalización
 * @returns {tf.Tensor} Secuencia normalizada
 */
function normalizeKeypointsSequence(sequence, stats) {
  if (!stats) {
    return sequence; // Sin normalización si no hay estadísticas
  }
  
  return tf.tidy(() => {
    // Índices de cada componente
    const POSE_START = 0;
    const POSE_END = 132; // 33 landmarks × 4
    const FACE_START = 132;
    const FACE_END = 1536; // 468 landmarks × 3
    const LEFT_HAND_START = 1536;
    const LEFT_HAND_END = 1599; // 21 landmarks × 3
    const RIGHT_HAND_START = 1599;
    const RIGHT_HAND_END = 1662; // 21 landmarks × 3
    
    const frames = sequence.shape[0];
    const components = [];
    
    // Normalizar Pose
    const poseMean = tf.tensor1d(stats.pose.mean);
    const poseStd = tf.tensor1d(stats.pose.std.map(s => Math.max(s, 1e-6))); // Evitar división por cero
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
    
    // Limpiar tensores temporales
    poseMean.dispose();
    poseStd.dispose();
    faceMean.dispose();
    faceStd.dispose();
    lhMean.dispose();
    lhStd.dispose();
    rhMean.dispose();
    rhStd.dispose();
    
    // Concatenar componentes normalizados
    return tf.concat(components, 1);
  });
}

// ========================================
// UTILIDADES DE CARGA
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
// CONSTRUCCIÓN DEL MODELO
// ========================================

function buildLSTMModel(numClasses, params) {
  tf.disposeVariables(); // Limpiar sesión previa
  
  const model = tf.sequential();
  
  model.add(tf.layers.inputLayer({
    inputShape: [CONFIG.MODEL_FRAMES, CONFIG.LENGTH_KEYPOINTS]
  }));
  
  // LSTM 1
  model.add(tf.layers.lstm({
    units: params.lstm_units_1,
    returnSequences: true,
    kernelRegularizer: tf.regularizers.l2({ l2: params.l2_regularizer })
  }));
  model.add(tf.layers.dropout({ rate: params.dropout_rate }));
  model.add(tf.layers.batchNormalization());
  
  // LSTM 2
  model.add(tf.layers.lstm({
    units: params.lstm_units_2,
    returnSequences: false,
    kernelRegularizer: tf.regularizers.l2({ l2: params.l2_regularizer })
  }));
  model.add(tf.layers.dropout({ rate: params.dropout_rate }));
  model.add(tf.layers.batchNormalization());
  
  // Dense
  model.add(tf.layers.dense({
    units: params.dense_units,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: params.l2_regularizer })
  }));
  model.add(tf.layers.dropout({ rate: params.dropout_rate }));
  
  // Output
  model.add(tf.layers.dense({
    units: numClasses,
    activation: 'softmax'
  }));
  
  // Compilar
  const optimizer = tf.train.adam(params.learning_rate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

// ========================================
// EARLY STOPPING
// ========================================

function createEarlyStoppingCallback(model, patience = 10) {
  let bestLoss = Infinity;
  let bestEpoch = 0;
  let wait = 0;
  let stopped = false;
  let bestWeights = null;

  return {
    onEpochEnd: async (epoch, logs) => {
      const currentLoss = logs.val_loss;

      if (currentLoss < bestLoss) {
        bestLoss = currentLoss;
        bestEpoch = epoch + 1;
        wait = 0;
        
        // Guardar mejores pesos
        if (bestWeights) {
          bestWeights.forEach(w => w.dispose());
        }
        bestWeights = model.getWeights().map(w => w.clone());
      } else {
        wait++;
        if (wait >= patience && !stopped) {
          stopped = true;
          
          // Restaurar mejores pesos
          if (bestWeights) {
            model.setWeights(bestWeights);
          }
          
          if (bestWeights) {
            bestWeights.forEach(w => w.dispose());
          }
          
          return true;
        }
      }
    },
    
    onTrainEnd: async () => {
      if (bestWeights) {
        bestWeights.forEach(w => w.dispose());
      }
    }
  };
}

// ========================================
// EVALUACIÓN DE PARÁMETROS
// ========================================

async function evaluateParams(params, X_train, y_train, X_test, y_test, numClasses) {
  try {
    const model = buildLSTMModel(numClasses, params);
    
    // Callbacks
    const earlyStopping = createEarlyStoppingCallback(model, CONFIG.EARLY_STOPPING_PATIENCE);
    
    // Entrenar
    const history = await model.fit(
      X_train, y_train,
      {
        validationData: [X_test, y_test],
        epochs: CONFIG.EPOCHS,
        batchSize: params.batch_size,
        callbacks: [earlyStopping],
        verbose: 0
      }
    );
    
    // Obtener mejores métricas (usando acceso directo como en train_lstm_node_v5.js)
    const bestValLoss = history.history.val_loss ? Math.min(...history.history.val_loss) : NaN;
    const bestValAcc = history.history.val_acc ? Math.max(...history.history.val_acc) : NaN;
    const bestTrainAcc = history.history.acc ? Math.max(...history.history.acc) : (history.history.accuracy ? Math.max(...history.history.accuracy) : NaN);
    const bestTrainLoss = history.history.loss ? Math.min(...history.history.loss) : NaN;
    const epochsUsed = history.history.loss ? history.history.loss.length : 0;
    
    // Evaluar en test
    const testResults = model.evaluate(X_test, y_test, { verbose: 0 });
    const testLoss = await testResults[0].data();
    const testAcc = await testResults[1].data();
    
    testResults.forEach(r => r.dispose());
    
    // Limpiar
    model.dispose();
    tf.disposeVariables();
    
    return {
      val_loss: bestValLoss,
      val_accuracy: bestValAcc,
      train_loss: bestTrainLoss,
      train_accuracy: bestTrainAcc,
      test_loss: testLoss[0],
      test_accuracy: testAcc[0],
      epochs_used: epochsUsed,
      status: 'success'
    };
    
  } catch (error) {
    return {
      val_loss: NaN,
      val_accuracy: NaN,
      train_loss: NaN,
      train_accuracy: NaN,
      test_loss: NaN,
      test_accuracy: NaN,
      epochs_used: 0,
      status: `error: ${error.message}`
    };
  }
}

// ========================================
// MAIN
// ========================================

async function main() {
  console.log('🔍 Grid Search de Hiperparámetros para LSTM');
  console.log('Versión: Completa (Grid Search Exhaustivo)\n');
  
  console.log('Backend:', backend.toUpperCase());
  console.log('📁 Rutas configuradas:');
  console.log(`   KEYPOINTS_PATH: ${CONFIG.KEYPOINTS_PATH}`);
  console.log(`   WORDS_JSON_PATH: ${CONFIG.WORDS_JSON_PATH}`);
  console.log(`   MODEL_FRAMES: ${CONFIG.MODEL_FRAMES}`);
  console.log(`   LENGTH_KEYPOINTS: ${CONFIG.LENGTH_KEYPOINTS}\n`);
  
  // Mostrar grid de parámetros
  console.log('📋 Grid de parámetros definido:');
  Object.entries(paramGrid).forEach(([key, values]) => {
    console.log(`   - ${key}: [${values.join(', ')}]`);
  });
  console.log(`\n📊 Total de combinaciones: ${totalCombinations}`);
  
  if (fastMode) {
    console.log('⚡ Modo rápido activado (--fast)\n');
  }
  
  // Estimar tiempo
  const timePerCombination = backend === 'gpu' ? 0.5 : (backend === 'cpu' ? 1.5 : 5.0);
  const estimatedHours = (totalCombinations * timePerCombination) / 3600;
  console.log(`🖥️  Dispositivo: ${backend.toUpperCase()}`);
  console.log(`⏱️  Tiempo estimado: ~${estimatedHours.toFixed(1)} horas (${(estimatedHours*60).toFixed(0)} minutos)`);
  if (backend === 'cpu' || backend === 'js') {
    console.log(`   💡 Usar GPU NVIDIA puede reducir el tiempo significativamente\n`);
  } else {
    console.log();
  }
  
  // 1. Cargar etiquetas
  let gestures;
  if (fs.existsSync(CONFIG.WORDS_JSON_PATH)) {
    const wordsData = JSON.parse(fs.readFileSync(CONFIG.WORDS_JSON_PATH, 'utf-8'));
    gestures = wordsData.word_ids;
  } else {
    gestures = ['bien', 'mal', 'hola'];
  }
  console.log(`🏷️  Gestos a entrenar: ${gestures.join(', ')}`);
  console.log(`📊 Total de clases: ${gestures.length}\n`);
  
  // 2. Cargar estadísticas de normalización
  const normalizationStats = loadNormalizationStats();
  if (normalizationStats) {
    console.log(`📊 Normalización activada: mean=0, std=1 por componente\n`);
  } else {
    console.log(`⚠️  Normalización desactivada (no se encontraron estadísticas)\n`);
  }
  
  // 3. Cargar secuencias (con normalización si está disponible)
  const { sequences, labels } = loadSequences(gestures, normalizationStats);
  if (sequences.length === 0) {
    throw new Error('❌ No hay secuencias válidas.');
  }
  
  // 4. Preparar datos
  const X = padSequences(sequences, CONFIG.MODEL_FRAMES);
  const y = tf.oneHot(tf.tensor1d(labels, 'int32'), gestures.length);
  
  console.log(`\n📦 Forma de datos:`);
  console.log(`   X: [${X.shape.join(', ')}] (samples, frames, keypoints)`);
  console.log(`   y: [${y.shape.join(', ')}] (samples, classes)\n`);
  
  // 5. Split train/test
  const totalSamples = sequences.length;
  const testSize = Math.floor(totalSamples * CONFIG.VALIDATION_SPLIT);
  const trainSize = totalSamples - testSize;
  
  const X_train = X.slice([0, 0, 0], [trainSize, CONFIG.MODEL_FRAMES, CONFIG.LENGTH_KEYPOINTS]);
  const X_test = X.slice([trainSize, 0, 0], [testSize, CONFIG.MODEL_FRAMES, CONFIG.LENGTH_KEYPOINTS]);
  const y_train = y.slice([0, 0], [trainSize, gestures.length]);
  const y_test = y.slice([trainSize, 0], [testSize, gestures.length]);
  
  console.log(`📊 Datos de entrenamiento: [${trainSize}, ${CONFIG.MODEL_FRAMES}, ${CONFIG.LENGTH_KEYPOINTS}]`);
  console.log(`📊 Datos de prueba: [${testSize}, ${CONFIG.MODEL_FRAMES}, ${CONFIG.LENGTH_KEYPOINTS}]\n`);
  
  // 6. Generar todas las combinaciones
  const combinations = cartesianProduct(paramGrid);
  console.log(`🔍 Iniciando Grid Search con ${combinations.length} combinaciones...\n`);
  
  // 7. Ejecutar grid search
  const results = [];
  let bestAccuracy = 0;
  let bestParams = null;
  
  const startTime = Date.now();
  
  for (let i = 0; i < combinations.length; i++) {
    const params = combinations[i];
    console.log(`[${i + 1}/${combinations.length}] Probando:`, params);
    
    const metrics = await evaluateParams(
      params, X_train, y_train, X_test, y_test,
      numClasses = gestures.length
    );
    
    // Combinar params y metrics
    const result = { ...params, ...metrics };
    results.push(result);
    
    if (metrics.status === 'success') {
      console.log(`   ✅ Val Acc: ${metrics.val_accuracy.toFixed(4)}, Test Acc: ${metrics.test_accuracy.toFixed(4)}`);
      if (metrics.val_accuracy > bestAccuracy) {
        bestAccuracy = metrics.val_accuracy;
        bestParams = params;
        console.log(`   🎯 ¡Nuevo mejor modelo! Val Acc: ${bestAccuracy.toFixed(4)}\n`);
      } else {
        console.log();
      }
    } else {
      console.log(`   ❌ Error: ${metrics.status}\n`);
    }
    
    // Mostrar progreso cada 10 combinaciones
    if ((i + 1) % 10 === 0) {
      const elapsed = (Date.now() - startTime) / 1000;
      const perCombination = elapsed / (i + 1);
      const remaining = perCombination * (combinations.length - i - 1);
      console.log(`⏳ Progreso: ${i + 1}/${combinations.length} (${((i + 1) / combinations.length * 100).toFixed(1)}%)`);
      console.log(`⏱️  Tiempo restante estimado: ~${(remaining / 60).toFixed(1)} minutos\n`);
    }
  }
  
  const endTime = Date.now();
  const totalTime = ((endTime - startTime) / 1000 / 60).toFixed(2);
  
  console.log(`\n✅ Grid Search completado`);
  console.log(`🎯 Mejor val_accuracy: ${bestAccuracy.toFixed(4)}`);
  console.log(`📋 Mejores parámetros:`, bestParams);
  
  // 8. Convertir resultados a formato CSV
  const csvRows = [];
  
  // Headers
  const headers = [
    'rank',
    'lstm_units_1', 'lstm_units_2', 'dense_units', 'dropout_rate',
    'l2_regularizer', 'learning_rate', 'batch_size',
    'val_accuracy', 'test_accuracy', 'train_accuracy',
    'val_loss', 'test_loss', 'train_loss',
    'epochs_used', 'status'
  ];
  csvRows.push(headers.join(','));
  
  // Ordenar por val_accuracy
  const sortedResults = [...results].sort((a, b) => b.val_accuracy - a.val_accuracy);
  
  // Data rows
  sortedResults.forEach((result, idx) => {
    const values = [
      idx + 1, // rank
      result.lstm_units_1,
      result.lstm_units_2,
      result.dense_units,
      result.dropout_rate,
      result.l2_regularizer,
      result.learning_rate,
      result.batch_size,
      result.val_accuracy.toFixed(4),
      result.test_accuracy.toFixed(4),
      result.train_accuracy.toFixed(4),
      result.val_loss.toFixed(4),
      result.test_loss.toFixed(4),
      result.train_loss.toFixed(4),
      result.epochs_used,
      result.status
    ];
    csvRows.push(values.join(','));
  });
  
  // 9. Guardar resultados
  const outputFile = 'resultsLSTM.csv';
  const outputPath = path.join(CONFIG.MODEL_DIR, outputFile);
  fs.writeFileSync(outputPath, csvRows.join('\n'), 'utf-8');
  console.log(`\n✅ Resultados guardados en: ${outputPath}`);
  
  // 10. Guardar también resumen
  const summaryLines = [];
  summaryLines.push('='.repeat(80));
  summaryLines.push('RESUMEN DE GRID SEARCH LSTM');
  summaryLines.push('='.repeat(80));
  summaryLines.push('');
  summaryLines.push(`Total de combinaciones probadas: ${results.length}`);
  summaryLines.push(`Combinaciones exitosas: ${results.filter(r => r.status === 'success').length}`);
  summaryLines.push(`Combinaciones con error: ${results.filter(r => r.status !== 'success').length}`);
  summaryLines.push(`Tiempo total: ${totalTime} minutos`);
  summaryLines.push('');
  summaryLines.push('Top 10 mejores modelos:');
  summaryLines.push('-'.repeat(80));
  summaryLines.push(
    'Rank'.padEnd(6) +
    'LSTM1'.padEnd(6) +
    'LSTM2'.padEnd(6) +
    'Dense'.padEnd(6) +
    'Drop'.padEnd(6) +
    'L2'.padEnd(8) +
    'LR'.padEnd(8) +
    'BS'.padEnd(4) +
    'Val Acc'.padEnd(10) +
    'Test Acc'
  );
  summaryLines.push('-'.repeat(80));
  
  sortedResults.slice(0, 10).forEach((result, idx) => {
    if (result.status === 'success') {
      summaryLines.push(
        (idx + 1).toString().padEnd(6) +
        result.lstm_units_1.toString().padEnd(6) +
        result.lstm_units_2.toString().padEnd(6) +
        result.dense_units.toString().padEnd(6) +
        result.dropout_rate.toString().padEnd(6) +
        result.l2_regularizer.toString().padEnd(8) +
        result.learning_rate.toString().padEnd(8) +
        result.batch_size.toString().padEnd(4) +
        (result.val_accuracy * 100).toFixed(2) + '%'.padEnd(6) +
        (result.test_accuracy * 100).toFixed(2) + '%'
      );
    }
  });
  
  const summaryFile = 'resultsLSTM_summary.txt';
  const summaryPath = path.join(CONFIG.MODEL_DIR, summaryFile);
  fs.writeFileSync(summaryPath, summaryLines.join('\n'), 'utf-8');
  console.log(`✅ Resumen guardado en: ${summaryPath}`);
  
  // 11. Mostrar resultados en consola
  console.log('\n' + '='.repeat(80));
  console.log('📊 RESUMEN DE RESULTADOS');
  console.log('='.repeat(80));
  console.log(`Total de combinaciones probadas: ${results.length}`);
  console.log(`Combinaciones exitosas: ${results.filter(r => r.status === 'success').length}`);
  console.log(`Combinaciones con error: ${results.filter(r => r.status !== 'success').length}`);
  
  if (results.filter(r => r.status === 'success').length > 0) {
    const successfulResults = results.filter(r => r.status === 'success');
    const valAccs = successfulResults.map(r => r.val_accuracy);
    console.log(`\n📈 Estadísticas de Validation Accuracy:`);
    console.log(`   Min: ${(Math.min(...valAccs) * 100).toFixed(2)}%`);
    console.log(`   Max: ${(Math.max(...valAccs) * 100).toFixed(2)}%`);
    console.log(`   Mean: ${(valAccs.reduce((a, b) => a + b, 0) / valAccs.length * 100).toFixed(2)}%`);
    
    console.log(`\n🏆 Top 5 mejores modelos:`);
    sortedResults.slice(0, 5).forEach((result, idx) => {
      if (result.status === 'success') {
        console.log(`\n${idx + 1}. Val Acc: ${(result.val_accuracy * 100).toFixed(2)}% | Test Acc: ${(result.test_accuracy * 100).toFixed(2)}%`);
        console.log(`   Params: LSTM[${result.lstm_units_1}, ${result.lstm_units_2}] Dense[${result.dense_units}] Drop[${result.dropout_rate}] L2[${result.l2_regularizer}] LR[${result.learning_rate}] BS[${result.batch_size}]`);
      }
    });
  }
  
  console.log('='.repeat(80));
  console.log(`\n⏱️  Tiempo total: ${totalTime} minutos`);
  console.log(`\n💾 ARCHIVOS GENERADOS:`);
  console.log(`   1. ${outputPath} - Resultados completos en CSV`);
  console.log(`   2. ${summaryPath} - Resumen de mejores modelos`);
  console.log('='.repeat(80));
  
  // Limpiar memoria
  X.dispose();
  y.dispose();
  X_train.dispose();
  X_test.dispose();
  y_train.dispose();
  y_test.dispose();
  sequences.forEach(seq => seq.dispose());
  
  console.log('\n🧹 Memoria limpiada correctamente');
}

// ========================================
// EJECUCIÓN
// ========================================

if (require.main === module) {
  main().catch(error => {
    console.error('❌ Error durante el grid search:', error);
    process.exit(1);
  });
}

module.exports = { 
  buildLSTMModel, 
  evaluateParams,
  loadNormalizationStats,
  normalizeKeypointsSequence
};

