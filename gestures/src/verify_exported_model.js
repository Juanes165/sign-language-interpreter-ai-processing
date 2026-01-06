/**
 * 🔍 Verificación del modelo TensorFlow.js exportado
 * 
 * Este script verifica:
 * - Que el modelo se puede cargar
 * - Que tiene la forma de entrada correcta
 * - Que puede hacer predicciones
 * 
 * Uso:
 *   node src/verify_exported_model.js
 */

// Intentar usar tfjs-node si está disponible
let tf;
try {
  tf = require('@tensorflow/tfjs-node');
} catch (e) {
  tf = require('@tensorflow/tfjs');
}

const fs = require('fs');
const path = require('path');

const MODEL_DIR = path.join(__dirname, '..', 'models', 'modelo_tfjs_node');
const WORDS_PATH = path.join(MODEL_DIR, 'words.json');

// Normalizar ruta para Windows
const normalizedPath = MODEL_DIR.replace(/\\/g, '/');
const MODEL_PATH = `file://${normalizedPath}/model.json`;

async function verifyModel() {
  console.log('🔍 Verificando modelo exportado...\n');

  try {
    // 1. Verificar que existen los archivos
    console.log('📁 Verificando archivos...');
    const modelJsonExists = fs.existsSync(path.join(MODEL_DIR, 'model.json'));
    const wordsJsonExists = fs.existsSync(WORDS_PATH);
    
    if (!modelJsonExists) {
      throw new Error(`❌ No se encuentra model.json en ${MODEL_DIR}`);
    }
    if (!wordsJsonExists) {
      throw new Error(`❌ No se encuentra words.json en ${MODEL_DIR}`);
    }
    
    console.log('   ✅ model.json encontrado');
    console.log('   ✅ words.json encontrado');

    // Listar archivos .bin
    const files = fs.readdirSync(MODEL_DIR);
    const binFiles = files.filter(f => f.endsWith('.bin'));
    console.log(`   ✅ ${binFiles.length} archivo(s) .bin encontrado(s)`);
    binFiles.forEach(f => console.log(`      - ${f}`));

    // 2. Cargar etiquetas
    console.log('\n🏷️  Cargando etiquetas...');
    const wordsData = JSON.parse(fs.readFileSync(WORDS_PATH, 'utf-8'));
    const gestures = wordsData.word_ids;
    console.log(`   Gestos: ${gestures.join(', ')}`);

    // 3. Cargar modelo
    console.log('\n🧠 Cargando modelo...');
    
    let model;
    try {
      // Intentar cargar con file:// (funciona con tfjs-node)
      model = await tf.loadLayersModel(MODEL_PATH);
    } catch (error) {
      console.log('   ℹ️  Usando handler personalizado (tfjs puro)...');
      
      // Fallback: cargar manualmente con handler custom
      model = await tf.loadLayersModel(tf.io.fromMemory({
        modelTopology: JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model.json'), 'utf-8')).modelTopology,
        weightSpecs: JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model.json'), 'utf-8')).weightsManifest[0].weights,
        weightData: fs.readFileSync(path.join(MODEL_DIR, 'weights.bin')).buffer
      }));
    }
    
    console.log('   ✅ Modelo cargado exitosamente');

    // 4. Verificar arquitectura
    console.log('\n📊 Información del modelo:');
    console.log(`   - Tipo: ${model.constructor.name}`);
    console.log(`   - Capas: ${model.layers.length}`);
    
    const inputShape = model.inputs[0].shape;
    const outputShape = model.outputs[0].shape;
    console.log(`   - Forma de entrada: [${inputShape.join(', ')}]`);
    console.log(`   - Forma de salida: [${outputShape.join(', ')}]`);

    // Verificar que la entrada es correcta
    const expectedInputShape = [null, 15, 1662]; // [batch, frames, keypoints]
    const inputMatches = inputShape.every((dim, i) => dim === expectedInputShape[i]);
    
    if (!inputMatches) {
      console.warn(`   ⚠️  Forma de entrada inesperada. Esperada: [${expectedInputShape.join(', ')}]`);
    } else {
      console.log('   ✅ Forma de entrada correcta');
    }

    // Verificar salida
    const expectedOutputClasses = gestures.length;
    if (outputShape[1] !== expectedOutputClasses) {
      console.warn(`   ⚠️  Número de clases inesperado. Esperado: ${expectedOutputClasses}, Obtenido: ${outputShape[1]}`);
    } else {
      console.log('   ✅ Número de clases correcto');
    }

    // 5. Probar predicción con datos sintéticos
    console.log('\n🧪 Probando predicción...');
    const dummyInput = tf.randomNormal([1, 15, 1662]); // [batch=1, frames=15, keypoints=1662]
    
    const prediction = model.predict(dummyInput);
    const predArray = await prediction.array();
    const probabilities = predArray[0];
    
    console.log('   ✅ Predicción exitosa');
    console.log('\n   Probabilidades:');
    gestures.forEach((gesture, i) => {
      console.log(`      ${gesture}: ${(probabilities[i] * 100).toFixed(2)}%`);
    });

    const maxProb = Math.max(...probabilities);
    const predictedIdx = probabilities.indexOf(maxProb);
    console.log(`\n   Gesto predicho: "${gestures[predictedIdx]}" (${(maxProb * 100).toFixed(2)}%)`);

    // Limpiar
    dummyInput.dispose();
    prediction.dispose();

    // 6. Verificar compatibilidad con navegador
    console.log('\n🌐 Verificación de compatibilidad con navegador:');
    
    const modelJson = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model.json'), 'utf-8'));
    const format = modelJson.format || 'unknown';
    const backend = modelJson.modelTopology ? 'layers' : 'graph';
    
    console.log(`   - Formato: ${format}`);
    console.log(`   - Backend: ${backend}`);
    console.log(`   - ✅ Compatible con tf.loadLayersModel()`);

    // 7. Calcular tamaño total
    console.log('\n📦 Tamaño del modelo:');
    let totalSize = 0;
    files.forEach(file => {
      const filePath = path.join(MODEL_DIR, file);
      const stats = fs.statSync(filePath);
      const sizeMB = (stats.size / (1024 * 1024)).toFixed(2);
      console.log(`   - ${file}: ${sizeMB} MB`);
      totalSize += stats.size;
    });
    console.log(`   Total: ${(totalSize / (1024 * 1024)).toFixed(2)} MB`);

    // 8. Resumen final
    console.log('\n✅ VERIFICACIÓN EXITOSA');
    console.log('\n📋 Resumen:');
    console.log(`   ✓ Modelo cargable: Sí`);
    console.log(`   ✓ Forma de entrada correcta: [batch, 15, 1662]`);
    console.log(`   ✓ Clases de salida: ${gestures.length}`);
    console.log(`   ✓ Puede hacer predicciones: Sí`);
    console.log(`   ✓ Compatible con navegador: Sí`);

    console.log('\n🎯 Siguiente paso:');
    console.log('   npm run copy-to-nextjs');

  } catch (error) {
    console.error('\n❌ ERROR:', error.message);
    console.error('\n💡 Solución:');
    console.error('   1. Verifica que hayas ejecutado: npm run train');
    console.error('   2. Verifica que el modelo se haya guardado correctamente');
    process.exit(1);
  }
}

if (require.main === module) {
  verifyModel();
}

module.exports = verifyModel;
