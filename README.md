# Intérprete de Lengua de Señas (Procesamiento con IA)

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un intérprete de lengua de señas colombiana impulsado por inteligencia artificial. Utiliza modelos de aprendizaje automático para reconocer e interpretar gestos de lengua de señas, proporcionando un puente de comunicación entre usuarios de lengua de señas y personas que no lo utilizan. El sistema procesa datos de gestos, entrena modelos y ofrece herramientas para el reconocimiento y análisis en tiempo real.

## Características
- **Reconocimiento de Gestos**: Utiliza modelos preentrenados para identificar y clasificar gestos de lengua de señas.
- **Ajuste de Hiperparámetros**: Incluye scripts para optimizar el rendimiento del modelo mediante búsqueda de hiperparámetros.
- **Procesamiento de Datos**: Herramientas para extraer y normalizar puntos clave de los datos de gestos.
- **Entrenamiento de Modelos**: Soporta el entrenamiento de modelos LSTM para el reconocimiento de gestos.
- **Reconocimiento en Tiempo Real**: Scripts para capturar y reconocer gestos en tiempo real.

## Estructura de Carpetas
- **gestures/**: Contiene scripts, datos y modelos relacionados con el reconocimiento de gestos (señas completas).
- **signs/**: Incluye modelos, datos y scripts para el procesamiento de letras del alfabeto de la LSC.
- **ipynb/**: Notebooks de Jupyter con experimentación y análisis.

## Primeros Pasos
1. Clona el repositorio.
2. Instala las dependencias necesarias utilizando los archivos `requirements.txt` proporcionados.
3. Sigue las instrucciones en los archivos `README_TRAINING.md` y otros documentos para entrenar y usar los modelos.

## Requisitos
- Python 3.10
- TensorFlow
- NumPy
- Mediapipe
- Dependencias adicionales listadas en `requirements.txt` y `requirements_v2.txt`.

## Uso
- Ejecuta `setup_and_train.bat` o `setup_and_train.sh` para configurar el entorno y entrenar los modelos.
- Utiliza los scripts en la carpeta `src/` para tareas específicas como procesamiento de datos, entrenamiento y reconocimiento.

## Contribuidores
Este proyecto se desarrolla como parte de una tesis para avanzar en las aplicaciones de IA en la interpretación del lengua de señas.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

