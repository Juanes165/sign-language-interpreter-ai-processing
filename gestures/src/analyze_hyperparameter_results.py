"""
Visualiza y compara resultados de diferentes búsquedas de hiperparámetros.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from app_constants import MODEL_DIR

def load_all_results():
    """Carga todos los archivos de resultados"""
    results_dir = Path(MODEL_DIR) / "hyperparameter_search"
    
    if not results_dir.exists():
        print("❌ No se encontró carpeta de resultados")
        return []
    
    results = []
    for file in results_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error leyendo {file.name}: {e}")
    
    return results


def print_summary(results):
    """Imprime resumen de todos los resultados"""
    print("="*80)
    print("📊 RESUMEN DE BÚSQUEDAS DE HIPERPARÁMETROS")
    print("="*80)
    
    if not results:
        print("No hay resultados disponibles")
        return
    
    # Ordenar por accuracy
    results_sorted = sorted(results, key=lambda x: x.get('best_accuracy', 0), reverse=True)
    
    print(f"\n{'#':<3} {'Archivo':<40} {'Método':<15} {'Accuracy':<10}")
    print("-"*80)
    
    for i, r in enumerate(results_sorted, 1):
        filename = r.get('filename', 'N/A')[:38]
        method = r.get('method', 'N/A')
        accuracy = r.get('best_accuracy', 0)
        print(f"{i:<3} {filename:<40} {method:<15} {accuracy:<10.4f}")
    
    print("\n" + "="*80)
    print(f"🏆 MEJOR RESULTADO: {results_sorted[0]['filename']}")
    print(f"   Accuracy: {results_sorted[0]['best_accuracy']:.4f}")
    print("="*80)


def compare_parameters(results):
    """Compara parámetros entre diferentes búsquedas"""
    if len(results) < 2:
        print("\n⚠️  Se necesitan al menos 2 resultados para comparar")
        return
    
    print("\n" + "="*80)
    print("🔍 COMPARACIÓN DE PARÁMETROS")
    print("="*80)
    
    # Obtener los 3 mejores
    top_3 = sorted(results, key=lambda x: x.get('best_accuracy', 0), reverse=True)[:3]
    
    # Parámetros a comparar
    param_keys = [
        'lstm_units_1', 'lstm_units_2', 'dense_units', 
        'dropout_rate', 'learning_rate', 'batch_size',
        'use_bidirectional', 'num_lstm_layers'
    ]
    
    print(f"\n{'Parámetro':<20}", end='')
    for i, r in enumerate(top_3, 1):
        acc = r.get('best_accuracy', 0)
        print(f"Top {i} ({acc:.3f})".ljust(20), end='')
    print()
    print("-"*80)
    
    for param in param_keys:
        print(f"{param:<20}", end='')
        for r in top_3:
            value = r.get('best_params', {}).get(param, 'N/A')
            if isinstance(value, float):
                print(f"{value:<20.4f}", end='')
            else:
                print(f"{str(value):<20}", end='')
        print()


def plot_parameter_importance():
    """Intenta cargar y mostrar importancia de parámetros de Optuna"""
    try:
        import pickle
        import optuna.visualization as vis
        
        results_dir = Path(MODEL_DIR) / "hyperparameter_search"
        study_files = list(results_dir.glob("optuna_study_*.pkl"))
        
        if not study_files:
            print("\n⚠️  No se encontraron estudios de Optuna")
            return
        
        # Usar el más reciente
        study_file = max(study_files, key=lambda p: p.stat().st_mtime)
        
        with open(study_file, 'rb') as f:
            study = pickle.load(f)
        
        print(f"\n📊 Generando visualizaciones desde: {study_file.name}")
        
        # Importancia de parámetros
        fig = vis.plot_param_importances(study)
        output = results_dir / "param_importance_latest.html"
        fig.write_html(output)
        print(f"✅ Importancia de parámetros: {output}")
        
        # Historial de optimización
        fig = vis.plot_optimization_history(study)
        output = results_dir / "optimization_history_latest.html"
        fig.write_html(output)
        print(f"✅ Historial de optimización: {output}")
        
        # Coordenadas paralelas
        fig = vis.plot_parallel_coordinate(study)
        output = results_dir / "parallel_coordinate_latest.html"
        fig.write_html(output)
        print(f"✅ Coordenadas paralelas: {output}")
        
        print("\n💡 Abre estos archivos HTML en tu navegador para ver las visualizaciones")
        
    except ImportError:
        print("\n⚠️  Instala optuna y plotly para generar visualizaciones:")
        print("   pip install optuna plotly")
    except Exception as e:
        print(f"\n⚠️  Error generando visualizaciones: {e}")


def analyze_convergence(results):
    """Analiza si la búsqueda convergió"""
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE CONVERGENCIA")
    print("="*80)
    
    for r in results:
        filename = r.get('filename', 'N/A')
        
        if 'all_results' in r and r['all_results']:
            accuracies = [x['val_accuracy'] for x in r['all_results']]
            
            # Calcular estadísticas
            best = max(accuracies)
            mean = sum(accuracies) / len(accuracies)
            
            # Los últimos 10 trials
            if len(accuracies) >= 10:
                recent_mean = sum(accuracies[-10:]) / 10
                improvement = recent_mean - mean
                
                print(f"\n📁 {filename}")
                print(f"   Mejor: {best:.4f}")
                print(f"   Promedio: {mean:.4f}")
                print(f"   Promedio últimos 10: {recent_mean:.4f}")
                
                if abs(improvement) < 0.01:
                    print("   ✅ La búsqueda parece haber convergido")
                else:
                    print(f"   ⚠️  Aún hay mejora potencial ({improvement:+.4f})")
                    print("      Considera ejecutar más trials")


def recommend_next_steps(results):
    """Recomienda próximos pasos basados en los resultados"""
    print("\n" + "="*80)
    print("💡 RECOMENDACIONES")
    print("="*80)
    
    if not results:
        print("No hay resultados para analizar")
        return
    
    best = max(results, key=lambda x: x.get('best_accuracy', 0))
    best_acc = best.get('best_accuracy', 0)
    best_params = best.get('best_params', {})
    
    print(f"\n🎯 Tu mejor modelo tiene accuracy: {best_acc:.4f}")
    
    if best_acc < 0.7:
        print("\n⚠️  El accuracy es bajo (<70%). Considera:")
        print("   1. Verificar que los datos estén correctos")
        print("   2. Aumentar el número de secuencias de entrenamiento")
        print("   3. Revisar la calidad de los keypoints")
        print("   4. Probar con más trials de optimización")
    
    elif best_acc < 0.85:
        print("\n📊 El accuracy es aceptable (70-85%). Para mejorar:")
        print("   1. Ejecutar más trials de optimización")
        print("   2. Agregar más datos de entrenamiento")
        print("   3. Probar data augmentation")
        print("   4. Considerar arquitecturas más complejas")
    
    else:
        print("\n🎉 ¡Excelente accuracy (>85%)!")
        print("   1. Valida con validación cruzada:")
        print("      python hyperparameter_tuning.py  # Opción 4")
        print("   2. Prueba el modelo en datos reales")
        print("   3. Si es consistente, despliega a producción")
    
    # Análisis de parámetros específicos
    if best_params.get('dropout_rate', 0) > 0.5:
        print("\n💡 Alto dropout detectado - posible overfitting en datos de entrenamiento")
    
    if best_params.get('learning_rate', 0) > 0.005:
        print("\n💡 Learning rate alto - el modelo puede converger rápido pero no óptimamente")
    
    if best_params.get('use_bidirectional'):
        print("\n✨ LSTM bidireccional funciona bien - considera mantenerlo")


def main():
    print("="*80)
    print("📊 ANÁLISIS DE RESULTADOS DE BÚSQUEDA DE HIPERPARÁMETROS")
    print("="*80)
    
    results = load_all_results()
    
    if not results:
        print("\n❌ No se encontraron resultados")
        print("\n💡 Ejecuta primero:")
        print("   python hyperparameter_tuning.py")
        print("   o")
        print("   python quick_hyperparameter_search.py")
        return
    
    print(f"\n✅ Se encontraron {len(results)} resultado(s)\n")
    
    # Menú interactivo
    while True:
        print("\n" + "="*80)
        print("¿Qué quieres hacer?")
        print("="*80)
        print("1. Ver resumen de todos los resultados")
        print("2. Comparar parámetros de los mejores modelos")
        print("3. Generar visualizaciones de Optuna")
        print("4. Analizar convergencia de la búsqueda")
        print("5. Ver recomendaciones")
        print("6. Todo lo anterior")
        print("0. Salir")
        
        choice = input("\nElige una opción (0-6): ").strip()
        
        if choice == '1':
            print_summary(results)
        elif choice == '2':
            compare_parameters(results)
        elif choice == '3':
            plot_parameter_importance()
        elif choice == '4':
            analyze_convergence(results)
        elif choice == '5':
            recommend_next_steps(results)
        elif choice == '6':
            print_summary(results)
            compare_parameters(results)
            plot_parameter_importance()
            analyze_convergence(results)
            recommend_next_steps(results)
        elif choice == '0':
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")


if __name__ == '__main__':
    main()
