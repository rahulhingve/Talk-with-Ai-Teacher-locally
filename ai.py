import whisper
import ollama
import pyttsx3
import pyaudio
import wave
import numpy as np
from datetime import datetime
import keyboard
import time

# Initialize Whisper model (using base model for better performance on i7)
def initialize_whisper():
    return whisper.load_model("base")

# Initialize TTS engine
def initialize_tts():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    # Default to the first available voice
    engine.setProperty('voice', voices[0].id)
    
    # Try to find an Indian English voice if available
    for voice in voices:
        try:
            # Some voices might not have proper language tags
            # so we'll just check the name
            if "india" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                print(f"Selected voice: {voice.name}")
                break
        except Exception as e:
            continue
    
    engine.setProperty('rate', 150)
    return engine

# Record audio function
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("Press and hold SPACE to speak...")
    frames = []
    
    while True:
        if keyboard.is_pressed('space'):
            print("Recording... Release SPACE to stop.")
            while keyboard.is_pressed('space'):
                data = stream.read(CHUNK)
                frames.append(data)
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
    return audio_data, RATE

def main():
    print("Initializing systems...")
    whisper_model = initialize_whisper()
    tts_engine = initialize_tts()
    
    print("Ready for conversation! Press and hold SPACE to speak, ESC to exit.")
    
    while True:
        if keyboard.is_pressed('esc'):
            break
            
        # Record audio
        audio_data, sample_rate = record_audio()
        
        # Convert speech to text
        result = whisper_model.transcribe(audio_data)
        user_text = result["text"]
        print(f"You said: {user_text}")
        
        # Get response from Ollama
        response = ollama.chat(model='mistral', messages=[
            {
                'role': 'system',
                'content': 'You are an English teacher. Help the student improve their English by correcting mistakes and suggesting better ways to express themselves. Keep responses natural and conversational.'
            },
            {
                'role': 'user',
                'content': user_text
            }
        ])
        
        ai_response = response['message']['content']
        print(f"AI: {ai_response}")
        
        # Convert response to speech
        tts_engine.say(ai_response)
        tts_engine.runAndWait()
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
