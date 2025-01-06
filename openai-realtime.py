import os
import json
import logging
import pyaudio
import numpy as np
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import aiohttp
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

class RealtimeVoiceClient:
    def __init__(self, api_key):
        logger.debug("Initializing RealtimeVoiceClient")
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
        logger.debug("Starting audio streams")
        try:
            self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
            )
            logger.debug("Input stream opened successfully")

            self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK
            )
            logger.debug("Output stream opened successfully")

            asyncio.create_task(self.process_input_audio())
            asyncio.create_task(self.process_output_audio())
            logger.debug("Audio processing tasks started")
        except Exception as e:
            logger.error(f"Error starting audio streams: {e}")
            raise

    async def process_input_audio(self):
        logger.debug("Starting input audio processing")
        while True:
            try:
                data = self.input_stream.read(CHUNK, exception_on_overflow=False)
                if self.audio_track:
                    logger.debug(f"Sending {len(data)} bytes of audio data")
                    await self.audio_track.send_audio(data)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in input audio processing: {e}")
                break

    async def process_output_audio(self):
        logger.debug("Starting output audio processing")
        while True:
            try:
                if not self.audio_queue.empty():
                    audio_data = await self.audio_queue.get()
                    logger.debug(f"Playing {len(audio_data)} bytes of audio data")
                    self.output_stream.write(audio_data)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in output audio processing: {e}")
                break

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
        logger.debug("AudioTrack initialized")

    async def recv(self):
        try:
            frame = await self.audio_buffer.get()
            logger.debug(f"Received {len(frame)} bytes of audio data")
            await self.client.play_audio(frame)
            return frame
        except Exception as e:
            logger.error(f"Error in recv: {e}")
            raise

    async def send_audio(self, audio_data):
        try:
            await self.audio_buffer.put(audio_data)
            logger.debug(f"Sent {len(audio_data)} bytes to audio buffer")
        except Exception as e:
            logger.error(f"Error in send_audio: {e}")
            raise


if __name__ == "__main__":
    async def main():
        try:
            logger.info("Starting application")
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in .env file")
                raise ValueError("Please create a .env file with OPENAI_API_KEY")
            
            client = RealtimeVoiceClient(api_key)
            try:
                await client.connect()
                
                # Keep the connection alive while processing audio
                while True:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Application error: {e}")
                raise
            finally:
                logger.info("Cleaning up resources")
                if client.input_stream:
                    client.input_stream.stop_stream()
                    client.input_stream.close()
                if client.output_stream:
                    client.output_stream.stop_stream()
                    client.output_stream.close()
                client.audio.terminate()
                logger.info("Resources cleaned up successfully")
            
        asyncio.run(main())
