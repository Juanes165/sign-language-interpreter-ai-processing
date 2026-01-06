from gtts import gTTS
import os
import pygame
from time import sleep
from app_constants import SPEECH_LANGUAGE, AUDIO_OUTPUT_FILE
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_audio_from_text(text):
    if not text:
        logging.error("No text to generate audio")
        return
    tts = gTTS(text=text, lang=SPEECH_LANGUAGE)
    tts.save(AUDIO_OUTPUT_FILE)
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_OUTPUT_FILE)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        sleep(1)
    pygame.mixer.quit()
    pygame.quit()
    os.remove(AUDIO_OUTPUT_FILE)

if __name__ == "__main__":
    sample_text = "Hola, cómo estás?"
    generate_audio_from_text(sample_text)
