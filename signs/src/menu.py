from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
import threading
import subprocess
import os


class MainApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Título
        layout.add_widget(Label(text='Sistema de Reconocimiento de Señas', font_size=20))

        # Botones
        btn_capture_images = Button(text='Capturar Imágenes')
        btn_capture_images.bind(on_press=self.run_capture_images)
        layout.add_widget(btn_capture_images)

        btn_dataset = Button(text='Crear DataSet')
        btn_dataset.bind(on_press=self.run_dataset)
        layout.add_widget(btn_dataset)

        btn_train_model = Button(text='Entrenar Modelo')
        btn_train_model.bind(on_press=self.run_train_model)
        layout.add_widget(btn_train_model)

        btn_task_model = Button(text='Crear/Actualizar Modelo .task (MediaPipe)')
        btn_task_model.bind(on_press=self.run_task_model)
        layout.add_widget(btn_task_model)

        btn_recognition = Button(text='Reconocer Gestos en Tiempo Real')
        btn_recognition.bind(on_press=self.run_recognition)
        layout.add_widget(btn_recognition)

        btn_exit = Button(text='Salir')
        btn_exit.bind(on_press=self.stop)
        layout.add_widget(btn_exit)

        # Aviso si no existe el .task
        self.check_model_exists()

        return layout

    def check_model_exists(self):
        model_path = os.path.join('..', 'models', 'gesture_recognizer.task')
        if not os.path.exists(model_path):
            self.show_message('Aviso: El modelo .task no existe. Cree o actualice el modelo desde el menú.')

    def run_capture_images(self, instance):
        self.run_script_with_popup('collector.py', 'Ejecutando captura de imágenes...', 'Imágenes capturadas.')

    def run_dataset(self, instance):
        self.run_script_with_popup('datasetCreator.py', 'Creando DataSet...', 'DataSet Creado.')

    def run_train_model(self, instance):
        self.run_script_with_popup('trainer.py', 'Entrenando Modelo...', 'Proceso de entrenamiento completado.')

    def run_task_model(self, instance):
        self.run_script_with_popup('create_task_model.py', 'Creando/actualizando modelo .task...', 'Modelo .task listo en carpeta models/.')

    def run_recognition(self, instance):
        threading.Thread(target=lambda: subprocess.run(['python', 'clasifierCamera.py'])).start()
        self.show_message('Reconocimiento de gestos iniciado en otra ventana.')

    def run_script_with_popup(self, script_name, start_message, end_message):
        popup = Popup(title='Procesando', content=Label(text=start_message), size_hint=(None, None), size=(400, 200))
        popup.open()

        def run_script():
            subprocess.run(['python', script_name], cwd=os.getcwd())
            popup.dismiss()
            self.show_message(end_message)

        threading.Thread(target=run_script).start()

    def show_message(self, message):
        popup = Popup(title='Información', content=Label(text=message), size_hint=(None, None), size=(400, 200))
        popup.open()


if __name__ == '__main__':
    MainApp().run()
