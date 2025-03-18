import whisper
import ollama
import numpy as np
import pyaudio
import wave
import keyboard
import time
from TTS.api import TTS

# Initialize Whisper model (using base model for better performance on i7)
def initialize_whisper():
    return whisper.load_model("base")

# Initialize TTS engine
def initialize_tts():
    # Using a verified model that's known to work
#  Then try one of these alternative models in initialize_tts():
# "tts_models/en/ljspeech/fast_pitch"
# "tts_models/en/ljspeech/glow-tts"
# "tts_models/en/jenny/jenny"
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
              progress_bar=False)
    return tts

# Print available models for debugging (uncomment if needed)
def list_available_models():
    print("Available TTS models:")
    print(TTS.list_models())

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
    # Uncomment next line to see available models
    # list_available_models()
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
        # You are an English teacher. Help the student improve their English by correcting mistakes and suggesting better ways to express themselves.
        response = ollama.chat(model='mistral', messages=[
            {
                'role': 'system',
                'content': ' Keep responses natural and conversational. provide response  in few sentences. 1-3 max'
            },
            {
                'role': 'user',
                'content': user_text
            }
        ])
        
        ai_response = response['message']['content']
        print(f"AI: {ai_response}")
        
        # Convert response to speech using Coqui TTS
        tts_engine.tts_to_file(text=ai_response, 
                              file_path="response.wav")
        
        # Play the generated audio
        import soundfile as sf
        import sounddevice as sd
        audio_data, samplerate = sf.read("response.wav")
        sd.play(audio_data, samplerate)
        sd.wait()
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
