@echo off
REM ===================================================
REM Script de instalación y entrenamiento automático
REM ===================================================

echo.
echo ================================================
echo  LSTM Gesture Recognition - Instalacion y Entrenamiento
echo ================================================
echo.

REM Verificar que Node.js está instalado
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js no esta instalado.
    echo Por favor instala Node.js desde: https://nodejs.org/
    pause
    exit /b 1
)

echo [INFO] Node.js detectado:
node --version
npm --version
echo.

REM Verificar que estamos en el directorio correcto
if not exist "package.json" (
    echo [ERROR] No se encuentra package.json
    echo Asegurate de ejecutar este script desde el directorio gesto_releasev1
    pause
    exit /b 1
)

REM Instalar dependencias
echo ================================================
echo  Paso 1: Instalando dependencias de Node.js...
echo ================================================
echo.
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo la instalacion de dependencias
    pause
    exit /b 1
)
echo.
echo [OK] Dependencias instaladas correctamente
echo.

REM Verificar que existen los archivos de keypoints
echo ================================================
echo  Paso 2: Verificando datos de entrenamiento...
echo ================================================
echo.

if not exist "assets\data\keypoints\" (
    echo [ERROR] No existe el directorio de keypoints
    echo Ejecuta primero el script de Python para extraer keypoints
    pause
    exit /b 1
)

set KEYPOINT_COUNT=0
for %%f in (assets\data\keypoints\*.npy) do set /a KEYPOINT_COUNT+=1

if %KEYPOINT_COUNT% LSS 1 (
    echo [ERROR] No se encontraron archivos .npy en assets\data\keypoints\
    echo Ejecuta primero: python src/extract_keypoints.py
    pause
    exit /b 1
)

echo [OK] Encontrados %KEYPOINT_COUNT% archivos de keypoints
echo.

REM Entrenar el modelo
echo ================================================
echo  Paso 3: Entrenando modelo LSTM...
echo ================================================
echo.
echo Este proceso puede tardar 5-15 minutos...
echo.
call npm run train
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo el entrenamiento del modelo
    pause
    exit /b 1
)
echo.
echo [OK] Modelo entrenado correctamente
echo.

REM Verificar el modelo
echo ================================================
echo  Paso 4: Verificando modelo exportado...
echo ================================================
echo.
call npm run verify
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] La verificacion fallo, pero el modelo puede estar OK
    echo Revisa los logs arriba
)
echo.

REM Copiar a Next.js
echo ================================================
echo  Paso 5: Copiando modelo a Next.js...
echo ================================================
echo.
call npm run copy-to-nextjs
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] No se pudo copiar automaticamente
    echo Copia manualmente los archivos de:
    echo   models\modelo_tfjs_node\
    echo a:
    echo   ..\sign-language-interpreter-frontend\public\models\lstm_gestos\
)
echo.

REM Resumen final
echo ================================================
echo  COMPLETADO EXITOSAMENTE
echo ================================================
echo.
echo [OK] El modelo LSTM ha sido entrenado y exportado
echo.
echo Archivos generados:
echo   - models\modelo_tfjs_node\model.json
echo   - models\modelo_tfjs_node\group1-shard*.bin
echo   - models\modelo_tfjs_node\words.json
echo.
echo Proximos pasos:
echo   1. Reinicia el servidor de Next.js si esta corriendo
echo   2. Navega a http://localhost:3000/gestures
echo   3. Prueba el reconocimiento de gestos
echo.
echo Para re-entrenar en el futuro:
echo   npm run full-pipeline
echo.
pause
