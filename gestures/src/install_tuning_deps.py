"""
Instalación de dependencias necesarias para la búsqueda de hiperparámetros.
"""

import subprocess
import sys

def install_package(package):
    """Instala un paquete usando pip"""
    print(f"📦 Instalando {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    packages = [
        "optuna",  # Para optimización bayesiana
        "scikit-learn",  # Ya lo tienes, pero por si acaso
        "plotly",  # Para visualizaciones de Optuna
        "kaleido",  # Para exportar gráficos de Optuna
    ]
    
    print("🚀 Instalando dependencias para búsqueda de hiperparámetros...\n")
    
    for package in packages:
        try:
            install_package(package)
            print(f"✅ {package} instalado correctamente\n")
        except Exception as e:
            print(f"❌ Error instalando {package}: {e}\n")
    
    print("✅ ¡Instalación completada!")

if __name__ == '__main__':
    main()
