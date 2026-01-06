/**
 * 🧹 Script para limpiar caché y verificar estado de memoria
 * 
 * Uso:
 *   node src/clear_cache.js
 */

let tf;
try {
  tf = require('@tensorflow/tfjs-node');
  console.log('✅ Usando @tensorflow/tfjs-node\n');
} catch (e) {
  tf = require('@tensorflow/tfjs');
  console.log('✅ Usando @tensorflow/tfjs\n');
}

const fs = require('fs');
const path = require('path');

console.log('🧹 LIMPIEZA DE CACHÉ Y VERIFICACIÓN DE MEMORIA\n');
console.log('='.repeat(60));

// 1. Información de memoria de TensorFlow.js
console.log('\n📊 Estado de memoria de TensorFlow.js:');
const memInfo = tf.memory();
console.log(`   - Tensores en memoria: ${memInfo.numTensors}`);
console.log(`   - Bytes en memoria: ${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB`);
console.log(`   - Bytes en GPU: ${(memInfo.numBytesInGPU / 1024 / 1024).toFixed(2)} MB`);
console.log(`   - Datos no liberados: ${memInfo.unreliable ? '⚠️  Sí' : '✅ No'}`);

// 2. Información de archivos de modelo
console.log('\n📁 Archivos de modelo:');
const modelDir = path.join(__dirname, '..', 'models', 'modelo_tfjs_node');

if (fs.existsSync(modelDir)) {
  const files = fs.readdirSync(modelDir);
  let totalSize = 0;
  
  files.forEach(file => {
    const filePath = path.join(modelDir, file);
    const stats = fs.statSync(filePath);
    const sizeMB = (stats.size / 1024 / 1024).toFixed(2);
    totalSize += stats.size;
    console.log(`   - ${file}: ${sizeMB} MB`);
  });
  
  console.log(`   📦 Tamaño total: ${(totalSize / 1024 / 1024).toFixed(2)} MB`);
  
  // Opción para eliminar modelos antiguos
  console.log('\n❓ ¿Deseas eliminar los archivos de modelo? (Ctrl+C para cancelar)');
  console.log('   Esperando 5 segundos...');
  
  setTimeout(() => {
    console.log('\n🗑️  No se eliminaron archivos. Para eliminar manualmente:');
    console.log(`   rm -rf ${modelDir}/*`);
  }, 5000);
} else {
  console.log('   ✅ No hay archivos de modelo guardados');
}

// 3. Limpiar tensores en memoria
console.log('\n🧹 Limpiando tensores en memoria...');
try {
  tf.disposeVariables();
  console.log('   ✅ Variables de TensorFlow.js limpiadas');
} catch (e) {
  console.log('   ⚠️  No hay variables para limpiar');
}

// 4. Estado final
setTimeout(() => {
  const finalMemInfo = tf.memory();
  console.log('\n📊 Estado final de memoria:');
  console.log(`   - Tensores en memoria: ${finalMemInfo.numTensors}`);
  console.log(`   - Bytes en memoria: ${(finalMemInfo.numBytes / 1024 / 1024).toFixed(2)} MB`);
  
  if (finalMemInfo.numTensors > 0) {
    console.log('\n⚠️  Advertencia: Aún hay tensores en memoria');
    console.log('   Esto es normal si Node.js no ha liberado completamente los recursos');
  } else {
    console.log('\n✅ Memoria limpia correctamente');
  }
  
  console.log('\n' + '='.repeat(60));
  console.log('🎯 Recomendaciones:');
  console.log('   1. Reiniciar Node.js limpia completamente la memoria');
  console.log('   2. Usa tf.tidy() para operaciones complejas');
  console.log('   3. Llama .dispose() en todos los tensores manuales');
  console.log('   4. Monitorea con tf.memory() durante el desarrollo');
  console.log('='.repeat(60) + '\n');
}, 5500);

