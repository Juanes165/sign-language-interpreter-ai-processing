/**
 * 📥 Descarga contribuciones desde Google Drive
 * 
 * Este script:
 * 1. Se conecta a Google Drive usando Service Account
 * 2. Descarga todos los archivos JSON de la carpeta de contribuciones
 * 3. Guarda localmente en assets/web_contributions/
 * 
 * Uso:
 *   node scripts/download_from_drive.js
 */

const { google } = require('googleapis');
const fs = require('fs');
const fsp = require('fs').promises; // fs para promises
const path = require('path');

// Intentar cargar .env si existe
try {
  require('dotenv').config();
} catch (e) {
  // .env no es obligatorio si usamos valores por defecto
}

/**
 * Descarga todas las contribuciones desde Google Drive
 */
async function downloadContributions() {
  try {
    console.log('🔐 Autenticando con Google Drive...');

    // Autenticación con Service Account
    // Leer credenciales desde el archivo en la raíz del proyecto
    const credentialsPath = path.join(__dirname, '..', 'unavoz-bb3744af7f68.json');
    
    if (!fs.existsSync(credentialsPath)) {
      throw new Error(`⚠️ Archivo de credenciales no encontrado: ${credentialsPath}`);
    }

    const credentials = JSON.parse(await fsp.readFile(credentialsPath, 'utf-8'));
    console.log('🔑 Usando credenciales de:', credentials.client_email);

    const auth = new google.auth.GoogleAuth({
      credentials: credentials,
      scopes: ['https://www.googleapis.com/auth/drive.readonly'],
    });

    const drive = google.drive({ version: 'v3', auth });
    
    // ID de carpeta de Drive (hardcodeado)
    const folderId = process.env.GOOGLE_DRIVE_FOLDER_ID || '1zkP5QPXCZU1nM2hL11r6VIzK0053yNtb';

    if (!folderId) {
      throw new Error('⚠️ GOOGLE_DRIVE_FOLDER_ID no está configurado');
    }

    console.log(`📂 Buscando archivos en carpeta: ${folderId}\n`);

    // ⭐ Listar TODOS los archivos usando paginación
    let allFiles = [];
    let pageToken = null;
    
    do {
      const response = await drive.files.list({
        q: `'${folderId}' in parents and mimeType='application/json' and trashed=false`,
        fields: 'nextPageToken, files(id, name, createdTime, size)',
        orderBy: 'createdTime desc',
        pageSize: 1000, // Máximo permitido por Google Drive API
        pageToken: pageToken,
        supportsAllDrives: true, // ⭐ Permite usar carpetas compartidas
        includeItemsFromAllDrives: true
      });

      allFiles = allFiles.concat(response.data.files || []);
      pageToken = response.data.nextPageToken;
      
      if (pageToken) {
        console.log(`📄 Cargando más archivos... (${allFiles.length} encontrados hasta ahora)`);
      }
    } while (pageToken);

    const files = allFiles;
    
    if (files.length === 0) {
      console.log('⚠️ No se encontraron archivos en la carpeta');
      return null;
    }

    console.log(`✅ Encontrados ${files.length} archivos en total\n`);

    // Crear directorio local
    const outputDir = path.join(__dirname, '..', 'assets', 'web_contributions');
    await fsp.mkdir(outputDir, { recursive: true });

    let successCount = 0;
    let errorCount = 0;
    const totalFiles = files.length;

    console.log('📥 Iniciando descarga...\n');

    // Descargar cada archivo
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const fileData = await drive.files.get({
          fileId: file.id,
          alt: 'media',
          supportsAllDrives: true // ⭐ Permite descargar de carpetas compartidas
        }, { responseType: 'json' });

        const localPath = path.join(outputDir, file.name);
        await fsp.writeFile(localPath, JSON.stringify(fileData.data, null, 2));
        
        successCount++;
        const progress = ((successCount / totalFiles) * 100).toFixed(1);
        console.log(`✅ [${successCount}/${totalFiles}] (${progress}%) ${file.name} - ${(file.size / 1024).toFixed(1)} KB`);
      } catch (error) {
        errorCount++;
        console.error(`❌ [${i + 1}/${totalFiles}] Error en ${file.name}:`, error.message);
      }
    }

    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`✅ Descarga completa:`);
    console.log(`   • ${successCount} archivos exitosos`);
    console.log(`   • ${errorCount} errores`);
    console.log(`   • Ubicación: ${outputDir}`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);

    return outputDir;
  } catch (error) {
    console.error('❌ Error en download_from_drive:', error.message);
    
    if (error.message.includes('ENOENT')) {
      console.error('\n⚠️ Asegúrate de tener el archivo de credenciales en:');
      console.error('   ./credentials/service-account.json\n');
    }
    
    throw error;
  }
}

module.exports = { downloadContributions };

// Ejecutar si se llama directamente
if (require.main === module) {
  downloadContributions()
    .then(dir => {
      if (dir) {
        console.log('🎯 Siguiente paso: node scripts/convert_web_to_npy.js');
      }
    })
    .catch(error => {
      console.error('\n❌ Error fatal:', error.message);
      process.exit(1);
    });
}
