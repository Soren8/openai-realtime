import os
from dotenv import load_dotenv
import openai
import sounddevice as sd
import numpy as np
import wave
import threading
import queue

class RealtimeVoiceClient:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'int16'
        
    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        self.recording_thread.join()
        
    def _record_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())
            
        with sd.InputStream(samplerate=self.sample_rate,
                          channels=self.channels,
                          dtype=self.dtype,
                          callback=callback):
            while self.is_recording:
                sd.sleep(100)
                
    def process_audio(self):
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
            
        if audio_data:
            audio_data = np.concatenate(audio_data)
            return self._send_to_openai(audio_data)
        return None
        
    def _send_to_openai(self, audio_data):
        try:
            # Convert numpy array to WAV format
            with wave.open('temp.wav', 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
                
            with open('temp.wav', 'rb') as audio_file:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    voice="4o-mini-advanced"
                )
            os.remove('temp.wav')
            return response['text']
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please create a .env file with OPENAI_API_KEY")
        
    client = RealtimeVoiceClient(api_key)
    
    print("Starting recording... (Press Enter to stop)")
    client.start_recording()
    input()  # Wait for Enter key
    client.stop_recording()
    
    print("Processing audio...")
    result = client.process_audio()
    if result:
        print("\nTranscription Result:")
        print(result)
    else:
        print("No transcription available")
