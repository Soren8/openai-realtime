# Created file: updated_realtime_client.py

import os
import json
import logging
import asyncio
import aiohttp
import numpy as np
import base64
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

import sounddevice as sd
from pydub import AudioSegment

# ──────────────────────────────────────────────────────────────────────────
# Constants from OpenAI’s reference
# ──────────────────────────────────────────────────────────────────────────
CHUNK_LENGTH_S = 0.05   # 50 ms
SAMPLE_RATE = 24000     # 24kHz
CHANNELS = 1
FORMAT_NP = np.int16    # NumPy dtype for 16-bit signed
FORMAT_SD = 'int16'     # For sounddevice
BLOCKSIZE = int(CHUNK_LENGTH_S * SAMPLE_RATE)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Ring-buffer-based audio output class (callback style)
# Similar to AudioPlayerAsync in audio_util.py
# ──────────────────────────────────────────────────────────────────────────
class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.playing = False
        self._frame_count = 0

        # Use sounddevice's callback-based output
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FORMAT_SD,
            blocksize=BLOCKSIZE,
            callback=self.callback
        )

    def callback(self, outdata, frames, time, status):
        # This function is called by sounddevice to provide data to the sound card buffer.
        if status:
            logger.warning(f"Output stream status: {status}")

        data = np.empty(0, dtype=FORMAT_NP)

        # Pull from the ring buffer
        while len(data) < frames and len(self.queue) > 0:
            next_chunk = self.queue.pop(0)
            needed = frames - len(data)
            data = np.concatenate((data, next_chunk[:needed]))
            if len(next_chunk) > needed:
                self.queue.insert(0, next_chunk[needed:])

        # If fewer samples than needed, pad with zeros
        if len(data) < frames:
            data = np.concatenate((data, np.zeros(frames - len(data), dtype=FORMAT_NP)))

        outdata[:] = data.reshape(-1, CHANNELS)
        self._frame_count += frames

    def start(self):
        if not self.playing:
            self.stream.start()
            self.playing = True

    def stop(self):
        if self.playing:
            self.stream.stop()
            self.playing = False
            self.queue = []

    def terminate(self):
        self.stream.close()

    def add_data(self, pcm_data: bytes):
        """Add raw PCM16 data (mono, 24 kHz) to the queue."""
        np_data = np.frombuffer(pcm_data, dtype=FORMAT_NP)
        self.queue.append(np_data)
        if not self.playing:
            self.start()


# ──────────────────────────────────────────────────────────────────────────
# Minimal track for receiving remote audio frames
# We'll resample with pydub if the frame rate doesn't match
# ──────────────────────────────────────────────────────────────────────────
class RemoteAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, player: AudioPlayerAsync):
        super().__init__()
        self.player = player

    async def recv(self):
        frame = await super().recv()
        return frame


# ──────────────────────────────────────────────────────────────────────────
# Main RealtimeVoiceClient with aiortc
# ──────────────────────────────────────────────────────────────────────────
class RealtimeVoiceClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pc = RTCPeerConnection()
        self.data_channel = None

        # Our ring-buffer-based output
        self.audio_player = AudioPlayerAsync()

        # For local mic input with sounddevice
        self._input_stream = None
        self._sending_audio = False

    async def connect(self):
        """Establish the WebRTC connection with the OpenAI Realtime endpoint."""
        try:
            # 1) Acquire ephemeral token from the sessions endpoint
            ephemeral_key = await self._get_ephemeral_token()

            # 2) Setup local input for capturing mic at 24 kHz
            self._start_input_stream_sounddevice()

            # 3) Configure data channel & remote track handling
            self._setup_webrtc()

            # 4) Create an offer, send to OpenAI, receive their answer
            await self._negotiate(ephemeral_key)

            # 5) Wait for data channel to open
            if not await self._wait_for_data_channel():
                raise RuntimeError("Data channel failed to initialize")

        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            if self.pc:
                await self.pc.close()
            raise

    async def _negotiate(self, ephemeral_key):
        # Create a local offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # POST our SDP offer to the OpenAI endpoint
        url = "https://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17"
        headers = {
            "Authorization": f"Bearer {ephemeral_key}",
            "Content-Type": "application/sdp"
        }
        async with aiohttp.ClientSession() as session:
            resp = await session.post(url, headers=headers, data=offer.sdp)
            if resp.status not in (200, 201):
                text = await resp.text()
                raise RuntimeError(f"Negotiation Error: {resp.status}, {text}")

            answer_sdp = await resp.text()
            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

    async def _get_ephemeral_token(self) -> str:
        """Obtain ephemeral token for Realtime from OpenAI's /sessions endpoint."""
        url = "https://api.openai.com/v1/realtime/sessions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini-realtime-preview-2024-12-17",
            "voice": "verse"
        }
        async with aiohttp.ClientSession() as session:
            resp = await session.post(url, headers=headers, json=data)
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Failed to get token: {resp.status} {error_text}")

            token_data = await resp.json()
            ephemeral_key = token_data['client_secret']['value']
        return ephemeral_key

    def _setup_webrtc(self):
        """Setup datachannel and remote track callbacks on the RTCPeerConnection."""
        # Create a data channel
        self.data_channel = self.pc.createDataChannel("oai-events")

        @self.data_channel.on("open")
        def on_open():
            logger.info("Data channel opened")
            config_message = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"]
                }
            }
            self.data_channel.send(json.dumps(config_message))

        @self.data_channel.on("message")
        def on_message(msg):
            self._handle_message(msg)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info("Received remote audio track")
                remote_track = RemoteAudioTrack(self.audio_player)
                # process incoming frames asynchronously
                asyncio.create_task(self._process_remote_track(track, remote_track))

    async def _process_remote_track(self, track, remote_track):
        """Receive AudioFrame from the aiortc track and feed to the ring buffer."""
        while True:
            try:
                frame = await track.recv()
                # Convert to PCM16 24 kHz with pydub
                pcm_data = self._resample_to_24k(frame)
                # enqueue into the ring buffer for playback
                self.audio_player.add_data(pcm_data)
            except Exception as e:
                logger.error(f"Error receiving remote track: {e}")
                break

    def _resample_to_24k(self, frame) -> bytes:
        """
        Convert aiortc AudioFrame => PCM16 @ 24 kHz, mono,
        using pydub AudioSegment to ensure correct sample rate.
        """
        audio_data = frame.to_ndarray().astype(np.int16)
        raw_bytes = audio_data.tobytes()

        # If your frames can be multi-channel, you'd need the actual channel count:
        #   channels = frame.layout.channels
        # But for example simplicity, we assume 1 channel here.
        segment = AudioSegment(
            raw_bytes,
            sample_width=2,       # 16-bit
            frame_rate=frame.sample_rate,
            channels=1
        )
        # Now force 24k, mono, 16-bit
        resampled = segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        return resampled.raw_data

    def _handle_message(self, msg):
        try:
            event = json.loads(msg)
            event_type = event.get('type')
            if event_type == 'response.audio_transcript.delta':
                logger.info("Transcript: " + event.get('delta', {}).get('text', ''))
            elif event_type == 'response.created':
                logger.info("Response started")
            elif event_type == 'response.done':
                logger.info("Response completed")
            elif event_type == 'error':
                logger.error(f"API Error: {event.get('message', '')}")
            else:
                logger.debug(f"Received unknown event: {event_type}")
        except json.JSONDecodeError:
            logger.error(f"Error parsing message: {msg}")
        except Exception as e:
            logger.error(f"Error in _handle_message: {e}")

    async def _wait_for_data_channel(self, timeout=5):
        """Wait for our data channel to open."""
        start_time = asyncio.get_event_loop().time()
        while True:
            if self.data_channel and self.data_channel.readyState == "open":
                return True
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            await asyncio.sleep(0.1)

    def _start_input_stream_sounddevice(self):
        """
        Capture mic input at 24 kHz, sending frames to the server via data channel.
        This approximates the approach in audio_util.py's send_audio_worker_sounddevice.
        """
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Input stream status: {status}")

            if self._sending_audio:
                # Encode audio data in base64 for transport
                audio_b64 = base64.b64encode(indata.tobytes()).decode('utf-8')
                # Then send to your data channel
                if self.data_channel and self.data_channel.readyState == "open":
                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }
                    self.data_channel.send(json.dumps(message))

        self._input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=FORMAT_SD,
            blocksize=BLOCKSIZE,
            callback=callback
        )
        self._input_stream.start()
        self._sending_audio = True

    async def close(self):
        """Shut down the streams and RTCPeerConnection cleanly."""
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
        self.audio_player.stop()
        self.audio_player.terminate()
        if self.pc:
            await self.pc.close()


# ──────────────────────────────────────────────────────────────────────────
# Example "main" entrypoint
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def main():
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        client = RealtimeVoiceClient(api_key)
        try:
            await client.connect()
            # Keep running
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            await client.close()

    asyncio.run(main())