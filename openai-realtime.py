import os
import json
import pyaudio
import numpy as np
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import aiohttp
import asyncio

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

class RealtimeVoiceClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pc = RTCPeerConnection()
        self.data_channel = None
        self.audio_stream = None
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.audio_track = None
        self.audio_queue = asyncio.Queue()

    def _handle_message(self, message):
        # Handle incoming messages
        print("Received:", message)

    async def start_audio_streams(self):
        # Initialize input stream (microphone)
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        # Initialize output stream (speaker)
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
        )

        # Start audio processing tasks
        asyncio.create_task(self.process_input_audio())
        asyncio.create_task(self.process_output_audio())

    async def process_input_audio(self):
        while True:
            # Read audio from microphone
            data = self.input_stream.read(CHUNK, exception_on_overflow=False)
            if self.audio_track:
                await self.audio_track.send_audio(data)
            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    async def process_output_audio(self):
        while True:
            # Play audio from queue
            if not self.audio_queue.empty():
                audio_data = await self.audio_queue.get()
                self.output_stream.write(audio_data)
            await asyncio.sleep(0.01)  # Small delay to prevent CPU overload

    async def play_audio(self, audio_data):
        # Add audio data to the output queue
        await self.audio_queue.put(audio_data)
        
    async def send_message(self, text):
        if not self.data_channel:
            raise RuntimeError("Data channel not initialized")
            
        message = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": text
            }
        }
        self.data_channel.send(json.dumps(message))
    async def connect(self):
        # Create ephemeral token
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini-realtime-preview-2024-12-17",
                    "voice": "verse"
                }
            ) as resp:
                token_data = await resp.json()
                ephemeral_key = token_data['client_secret']['value']

        # Initialize audio streams
        await self.start_audio_streams()

        # Add audio track
        self.audio_track = AudioTrack(self)
        self.pc.addTrack(self.audio_track)

        # Set up data channel
        self.data_channel = self.pc.createDataChannel("oai-events")
        self.data_channel.on("message", self._handle_message)

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17",
                headers={
                    "Authorization": f"Bearer {ephemeral_key}",
                    "Content-Type": "application/sdp"
                },
                data=offer.sdp
            ) as resp:
                answer_sdp = await resp.text()
                answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
                await self.pc.setRemoteDescription(answer)

class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, client):
        super().__init__()
        self.client = client
        self.audio_buffer = asyncio.Queue()

    async def recv(self):
        # Get audio frames from the buffer and play them
        frame = await self.audio_buffer.get()
        await self.client.play_audio(frame)
        return frame

    async def send_audio(self, audio_data):
        # Add audio frames to the buffer
        await self.audio_buffer.put(audio_data)


if __name__ == "__main__":
    async def main():
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please create a .env file with OPENAI_API_KEY")
            
        client = RealtimeVoiceClient(api_key)
        try:
            await client.connect()
            
            # Keep the connection alive while processing audio
            while True:
                await asyncio.sleep(1)
        finally:
            # Clean up audio resources
            if client.input_stream:
                client.input_stream.stop_stream()
                client.input_stream.close()
            if client.output_stream:
                client.output_stream.stop_stream()
                client.output_stream.close()
            client.audio.terminate()
            
    asyncio.run(main())
