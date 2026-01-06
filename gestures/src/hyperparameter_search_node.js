/**
 * 🔍 Búsqueda de Hiperparámetros en Node.js
 * 
 * Busca los mejores hiperparámetros para el modelo LSTM usando:
 * - Random Search (búsqueda aleatoria)
 * - Evaluación con validación cruzada o simple train/val split
 * 
 * Uso:
 *   node src/hyperparameter_search_node.js --trials 20
 */

// Cargar TensorFlow.js
let tf;
let backend = 'cpu';
try {
  tf = require('@tensorflow/tfjs-node');
  backend = 'node';
  console.log('✅ Usando @tensorflow/tfjs-node (CPU acelerado)\n');
} catch (e) {
  tf = require('@tensorflow/tfjs');
  backend = 'cpu';
  console.log('⚠️  Usando @tensorflow/tfjs (JavaScript puro)\n');
}

const fs = require('fs');
const path = require('path');
const { fromArrayBuffer } = require('numpy-parser');

// ========================================
// CONFIGURACIÓN BASE
// ========================================
const BASE_CONFIG = {
  MODEL_FRAMES: 15,
  LENGTH_KEYPOINTS: 1662,
  ROOT_PATH: path.join(__dirname, '..'),
  EPOCHS: 50,  // Reducido para búsqueda rápida
  VALIDATION_SPLIT: 0.2,
  EARLY_STOPPING_PATIENCE: 10,
};

BASE_CONFIG.KEYPOINTS_PATH = path.join(BASE_CONFIG.ROOT_PATH, 'assets', 'data', 'keypoints');
BASE_CONFIG.MODEL_DIR = path.join(BASE_CONFIG.ROOT_PATH, 'models');
BASE_CONFIG.WORDS_JSON_PATH = path.join(BASE_CONFIG.MODEL_DIR, 'words.json');
BASE_CONFIG.RESULTS_DIR = path.join(BASE_CONFIG.MODEL_DIR, 'hyperparameter_search');

// ========================================
// ESPACIO DE BÚSQUEDA
// ========================================
const SEARCH_SPACE = {
  lstm_units_1: [32, 64, 128],
  lstm_units_2: [64, 128, 256],
  dense_units: [32, 64, 128],
  dropout: [0.3, 0.4, 0.5, 0.6],
  l2_reg: [0.001, 0.005, 0.01, 0.05],
  learning_rate: [0.0001, 0.0005, 0.001, 0.005],
  batch_size: [8, 16, 32],
};

// ========================================
// UTILIDADES DE CARGA (simplificadas)
// ========================================

function loadNpyFile(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    const npArray = fromArrayBuffer(arrayBuffer);
    
    if (npArray.shape.length === 2 && 
        npArray.shape[0] === BASE_CONFIG.MODEL_FRAMES && 
        npArray.shape[1] === BASE_CONFIG.LENGTH_KEYPOINTS) {
      return tf.tensor3d(npArray.data, [1, ...npArray.shape]);
    } 
    else if (npArray.shape.length === 3 && npArray.shape[2] === BASE_CONFIG.LENGTH_KEYPOINTS) {
      return tf.tensor3d(npArray.data, npArray.shape);
    }
    return null;
  } catch (error) {
    return null;
  }
}

function loadSequences(gestures) {
  const sequences = [];
  const labels = [];

  gestures.forEach((gesture, idx) => {
    const filePath = path.join(BASE_CONFIG.KEYPOINTS_PATH, `${gesture}.npy`);
    
    if (!fs.existsSync(filePath)) return;

    const tensor = loadNpyFile(filePath);
    if (!tensor) return;

    const numSequences = tensor.shape[0];
    for (let i = 0; i < numSequences; i++) {
      sequences.push(tensor.slice([i, 0, 0], [1, BASE_CONFIG.MODEL_FRAMES, BASE_CONFIG.LENGTH_KEYPOINTS]).squeeze([0]));
      labels.push(idx);
    }

    tensor.dispose();
  });

  return { sequences, labels };
}

function padSequences(sequences, maxLen) {
  return tf.tidy(() => {
    const padded = sequences.map(seq => {
      const currentLen = seq.shape[0];
      
      if (currentLen === maxLen) {
        return seq;
      } else if (currentLen < maxLen) {
        const padding = tf.zeros([maxLen - currentLen, BASE_CONFIG.LENGTH_KEYPOINTS]);
        return tf.concat([padding, seq], 0);
      } else {
        return seq.slice([currentLen - maxLen, 0], [maxLen, BASE_CONFIG.LENGTH_KEYPOINTS]);
      }
    });

    return tf.stack(padded);
  });
}

// ========================================
// CONSTRUCCIÓN DE MODELO PARAMETRIZABLE
// ========================================

function buildModel(numClasses, params) {
  const model = tf.sequential();

  model.add(tf.layers.inputLayer({
    inputShape: [BASE_CONFIG.MODEL_FRAMES, BASE_CONFIG.LENGTH_KEYPOINTS]
  }));

  // LSTM 1
  model.add(tf.layers.lstm({
    units: params.lstm_units_1,
    returnSequences: true,
    kernelRegularizer: tf.regularizers.l2({ l2: params.l2_reg })
  }));
  model.add(tf.layers.dropout({ rate: params.dropout }));
  model.add(tf.layers.batchNormalization());

  // LSTM 2
  model.add(tf.layers.lstm({
    units: params.lstm_units_2,
    returnSequences: false,
    kernelRegularizer: tf.regularizers.l2({ l2: params.l2_reg })
  }));
  model.add(tf.layers.dropout({ rate: params.dropout }));
  model.add(tf.layers.batchNormalization());

  // Dense
  model.add(tf.layers.dense({
    units: params.dense_units,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: params.l2_reg })
  }));
  model.add(tf.layers.dropout({ rate: params.dropout }));

  // Output
  model.add(tf.layers.dense({
    units: numClasses,
    activation: 'softmax'
  }));

  model.compile({
    optimizer: tf.train.adam(params.learning_rate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// ========================================
// EARLY STOPPING
// ========================================

function createEarlyStoppingCallback(patience = 10) {
  let bestLoss = Infinity;
  let wait = 0;

  return {
    onEpochEnd: async (epoch, logs) => {
      const currentLoss = logs.val_loss;

      if (currentLoss < bestLoss) {
        bestLoss = currentLoss;
        wait = 0;
      } else {
        wait++;
        if (wait >= patience) {
          // En TensorFlow.js, retornar true detiene el entrenamiento
          return true;
        }
      }
    }
  };
}

// ========================================
// FUNCIÓN DE EVALUACIÓN
// ========================================

async function evaluateParams(params, X, y, numClasses, trialNum) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`🔬 Trial ${trialNum}`);
  console.log(`${'='.repeat(60)}`);
  console.log('Parámetros:');
  console.log(`  - LSTM 1: ${params.lstm_units_1} units`);
  console.log(`  - LSTM 2: ${params.lstm_units_2} units`);
  console.log(`  - Dense: ${params.dense_units} units`);
  console.log(`  - Dropout: ${params.dropout}`);
  console.log(`  - L2 reg: ${params.l2_reg}`);
  console.log(`  - Learning rate: ${params.learning_rate}`);
  console.log(`  - Batch size: ${params.batch_size}`);

  const model = buildModel(numClasses, params);

  try {
    const history = await model.fit(X, y, {
      epochs: BASE_CONFIG.EPOCHS,
      batchSize: params.batch_size,
      validationSplit: BASE_CONFIG.VALIDATION_SPLIT,
      callbacks: [
        createEarlyStoppingCallback(BASE_CONFIG.EARLY_STOPPING_PATIENCE)
      ],
      verbose: 0  // Silencioso
    });

    const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1];
    const finalValLoss = history.history.val_loss[history.history.val_loss.length - 1];
    const finalTrainAcc = history.history.acc[history.history.acc.length - 1];

    console.log('\nResultados:');
    console.log(`  ✅ Val Accuracy: ${(finalValAcc * 100).toFixed(2)}%`);
    console.log(`  ✅ Val Loss: ${finalValLoss.toFixed(4)}`);
    console.log(`  ✅ Train Accuracy: ${(finalTrainAcc * 100).toFixed(2)}%`);
    console.log(`  ✅ Epochs completados: ${history.history.loss.length}`);

    // Limpiar
    model.dispose();
    tf.disposeVariables();

    return {
      params,
      val_accuracy: finalValAcc,
      val_loss: finalValLoss,
      train_accuracy: finalTrainAcc,
      epochs: history.history.loss.length,
      history: {
        loss: history.history.loss,
        acc: history.history.acc,
        val_loss: history.history.val_loss,
        val_acc: history.history.val_acc
      }
    };
  } catch (error) {
    console.error(`  ❌ Error en trial: ${error.message}`);
    model.dispose();
    return null;
  }
}

// ========================================
// BÚSQUEDA ALEATORIA
// ========================================

function randomChoice(array) {
  return array[Math.floor(Math.random() * array.length)];
}

function sampleParams() {
  return {
    lstm_units_1: randomChoice(SEARCH_SPACE.lstm_units_1),
    lstm_units_2: randomChoice(SEARCH_SPACE.lstm_units_2),
    dense_units: randomChoice(SEARCH_SPACE.dense_units),
    dropout: randomChoice(SEARCH_SPACE.dropout),
    l2_reg: randomChoice(SEARCH_SPACE.l2_reg),
    learning_rate: randomChoice(SEARCH_SPACE.learning_rate),
    batch_size: randomChoice(SEARCH_SPACE.batch_size),
  };
}

// ========================================
// MAIN
// ========================================

async function main() {
  console.log('🔍 Búsqueda de Hiperparámetros en Node.js\n');
  console.log('Backend:', backend.toUpperCase());
  console.log(`Configuración:`);
  console.log(`  - MODEL_FRAMES: ${BASE_CONFIG.MODEL_FRAMES}`);
  console.log(`  - EPOCHS por trial: ${BASE_CONFIG.EPOCHS}`);
  console.log(`  - Early stopping patience: ${BASE_CONFIG.EARLY_STOPPING_PATIENCE}\n`);

  // Obtener número de trials
  const args = process.argv.slice(2);
  const trialsIndex = args.indexOf('--trials');
  const numTrials = trialsIndex !== -1 && args[trialsIndex + 1] 
    ? parseInt(args[trialsIndex + 1]) 
    : 20;

  console.log(`🎯 Ejecutando ${numTrials} trials\n`);

  // 1. Cargar datos
  let gestures;
  if (fs.existsSync(BASE_CONFIG.WORDS_JSON_PATH)) {
    const wordsData = JSON.parse(fs.readFileSync(BASE_CONFIG.WORDS_JSON_PATH, 'utf-8'));
    gestures = wordsData.word_ids;
  } else {
    gestures = ['hola-der', 'dias-gen', 'paz-der'];
  }

  console.log(`📚 Gestos: ${gestures.join(', ')}\n`);

  const { sequences, labels } = loadSequences(gestures);
  if (sequences.length === 0) {
    throw new Error('❌ No hay secuencias válidas.');
  }

  console.log(`✅ Cargadas ${sequences.length} secuencias\n`);

  const X = padSequences(sequences, BASE_CONFIG.MODEL_FRAMES);
  const y = tf.oneHot(tf.tensor1d(labels, 'int32'), gestures.length);

  // 2. Ejecutar búsqueda
  const results = [];
  const startTime = Date.now();

  for (let i = 0; i < numTrials; i++) {
    const params = sampleParams();
    const result = await evaluateParams(params, X, y, gestures.length, i + 1);
    
    if (result) {
      results.push(result);
    }

    // Mostrar mejor resultado hasta ahora
    if (results.length > 0) {
      const bestSoFar = results.reduce((best, curr) => 
        curr.val_accuracy > best.val_accuracy ? curr : best
      );
      console.log(`\n📊 Mejor hasta ahora: ${(bestSoFar.val_accuracy * 100).toFixed(2)}% (Trial ${results.indexOf(bestSoFar) + 1})`);
    }
  }

  const endTime = Date.now();
  const totalTime = ((endTime - startTime) / 1000 / 60).toFixed(2);

  // 3. Analizar resultados
  console.log(`\n${'='.repeat(60)}`);
  console.log('📊 RESULTADOS FINALES');
  console.log(`${'='.repeat(60)}`);

  const sortedResults = results.sort((a, b) => b.val_accuracy - a.val_accuracy);

  console.log(`\n🏆 Top 5 Mejores Configuraciones:\n`);
  sortedResults.slice(0, 5).forEach((result, idx) => {
    console.log(`${idx + 1}. Val Accuracy: ${(result.val_accuracy * 100).toFixed(2)}% | Val Loss: ${result.val_loss.toFixed(4)}`);
    console.log(`   Params: LSTM[${result.params.lstm_units_1}, ${result.params.lstm_units_2}] Dense[${result.params.dense_units}] Drop[${result.params.dropout}] L2[${result.params.l2_reg}] LR[${result.params.learning_rate}] BS[${result.params.batch_size}]\n`);
  });

  const bestResult = sortedResults[0];
  console.log(`\n🎯 MEJOR CONFIGURACIÓN:`);
  console.log(`   Val Accuracy: ${(bestResult.val_accuracy * 100).toFixed(2)}%`);
  console.log(`   Val Loss: ${bestResult.val_loss.toFixed(4)}`);
  console.log(`   Train Accuracy: ${(bestResult.train_accuracy * 100).toFixed(2)}%`);
  console.log(`\n   Parámetros:`);
  console.log(`   - LSTM Layer 1: ${bestResult.params.lstm_units_1} units`);
  console.log(`   - LSTM Layer 2: ${bestResult.params.lstm_units_2} units`);
  console.log(`   - Dense Layer: ${bestResult.params.dense_units} units`);
  console.log(`   - Dropout: ${bestResult.params.dropout}`);
  console.log(`   - L2 regularization: ${bestResult.params.l2_reg}`);
  console.log(`   - Learning rate: ${bestResult.params.learning_rate}`);
  console.log(`   - Batch size: ${bestResult.params.batch_size}`);

  // 4. Guardar resultados
  if (!fs.existsSync(BASE_CONFIG.RESULTS_DIR)) {
    fs.mkdirSync(BASE_CONFIG.RESULTS_DIR, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const resultsFile = path.join(BASE_CONFIG.RESULTS_DIR, `node_search_${timestamp}.json`);

  const saveData = {
    timestamp,
    backend,
    config: BASE_CONFIG,
    search_space: SEARCH_SPACE,
    num_trials: numTrials,
    total_time_minutes: parseFloat(totalTime),
    best_params: bestResult.params,
    best_val_accuracy: bestResult.val_accuracy,
    best_val_loss: bestResult.val_loss,
    all_results: sortedResults.map(r => ({
      params: r.params,
      val_accuracy: r.val_accuracy,
      val_loss: r.val_loss,
      train_accuracy: r.train_accuracy,
      epochs: r.epochs
    }))
  };

  fs.writeFileSync(resultsFile, JSON.stringify(saveData, null, 2));
  console.log(`\n💾 Resultados guardados en: ${resultsFile}`);

  console.log(`\n⏱️  Tiempo total: ${totalTime} minutos`);
  console.log(`\n✅ Búsqueda completada!`);

  // Limpiar
  X.dispose();
  y.dispose();
  sequences.forEach(seq => seq.dispose());
}

// ========================================
// EJECUCIÓN
// ========================================

if (require.main === module) {
  main().catch(error => {
    console.error('❌ Error durante la búsqueda:', error);
    process.exit(1);
  });
}

module.exports = { buildModel, sampleParams, evaluateParams };

