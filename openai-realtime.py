import os
import json
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription
import aiohttp
import asyncio

class RealtimeVoiceClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pc = RTCPeerConnection()
        self.data_channel = None
        self.audio_stream = None
        
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

        # Set up WebRTC connection
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

        # Set up data channel
        self.data_channel = self.pc.createDataChannel("oai-events")
        self.data_channel.on("message", self._handle_message)

        # Set up audio
        self.audio_stream = await self._setup_audio()

    async def _setup_audio(self):
        # Implement audio stream setup
        pass

    def _handle_message(self, message):
        # Handle incoming messages
        print("Received:", message)

    async def send_message(self, text):
        message = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": text
            }
        }
        self.data_channel.send(json.dumps(message))

if __name__ == "__main__":
    async def main():
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please create a .env file with OPENAI_API_KEY")
            
        client = RealtimeVoiceClient(api_key)
        await client.connect()
        
        while True:
            text = input("Enter message: ")
            await client.send_message(text)
            
    asyncio.run(main())
