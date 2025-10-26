import pyttsx3
import threading
import queue
from gtts import gTTS
import os
import tempfile
import time
import pygame
import uuid

class AudioFeedback:
    def __init__(self):
        """Initialize the AudioFeedback class"""
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.temp_dir = tempfile.gettempdir()
        pygame.mixer.init()
        self.speech_thread.start()
        
    def _create_engine(self):
        """Create a new TTS engine instance for English"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 1.0)  # Volume level
        return engine
    
    def _create_arabic_audio_file(self, text):
        """Create a temporary audio file for Arabic text"""
        try:
            # Generate unique filename
            temp_file = os.path.join(self.temp_dir, f'arabic_speech_{uuid.uuid4().hex[:8]}.mp3')
            tts = gTTS(text=text, lang='ar')
            tts.save(temp_file)
            return temp_file
        except Exception as e:
            print(f"Error creating Arabic audio: {str(e)}")
            return None

    def _play_arabic_audio(self, file_path):
        """Play Arabic audio using pygame"""
        try:
            if file_path and os.path.exists(file_path):
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.music.unload()
                # Clean up the temporary file
                try:
                    os.remove(file_path)
                except:
                    pass
        except Exception as e:
            print(f"Error playing Arabic audio: {str(e)}")

    def _speech_worker(self):
        """Worker thread that processes speech requests"""
        while True:
            try:
                text, is_arabic = self.speech_queue.get()
                if text is None:  # Shutdown signal
                    break

                if is_arabic:
                    # Handle Arabic text
                    audio_file = self._create_arabic_audio_file(text)
                    if audio_file:
                        self._play_arabic_audio(audio_file)
                else:
                    # Handle English text
                    engine = self._create_engine()
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                
                # Add a small pause between languages
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Speech error: {str(e)}")
            finally:
                self.speech_queue.task_done()
    
    def speak(self, text_en, text_ar):
        """Speak the text in both English and Arabic"""
        try:
            # Queue English text (is_arabic=False)
            self.speech_queue.put((text_en, False))
            
            # Queue Arabic text (is_arabic=True)
            self.speech_queue.put((text_ar, True))
            
        except Exception as e:
            print(f"Error queuing speech: {str(e)}")
            
    def play_alert(self):
        """Play an emergency alert sound"""
        try:
            # Create a simple beep sound using pygame
            sample_rate = 44100
            duration = 0.5  # seconds
            frequency = 880  # Hz
            
            # Generate the beep sound
            num_samples = int(duration * sample_rate)
            sound_array = pygame.sndarray.make_sound(
                pygame.mixer.Sound(
                    buffer=bytes(
                        int(32767.0 * pygame.math.sin(2.0 * 3.14159 * frequency * t / sample_rate))
                        for t in range(num_samples)
                    )
                )
            )
            
            # Play the alert sound
            sound_array.play()
            pygame.time.wait(int(duration * 1000))  # Wait for the sound to finish
            
            # Play it again for emphasis
            sound_array.play()
            pygame.time.wait(int(duration * 1000))
            
        except Exception as e:
            print(f"Error playing alert: {str(e)}")
            
    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            self.speech_queue.put((None, False))  # Signal thread to stop
            if self.speech_thread.is_alive():
                self.speech_thread.join(timeout=1)
            pygame.mixer.quit()
        except:
            pass 