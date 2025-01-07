#!/usr/bin/env python3

import os
import asyncio
import base64
import logging
import numpy as np

import sounddevice as sd
from pydub import AudioSegment

# Ensure you have installed the latest openai library (and the [realtime] extra):
#   pip install --upgrade openai[realtime]
import openai
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session import Session

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Constants to match the approach from openai/audio_util.py
# ---------------------------------------------------------
SAMPLE_RATE = 24000    # 24 kHz
CHANNELS = 1           # mono
CHUNK_DURATION_S = 0.05  # 50 ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_S)

# ---------------------------------------------------------
# A callback-based audio playback class (ring buffer)
# adapted from the OpenAI examples
# ---------------------------------------------------------
class AudioPlayerAsync:
    def __init__(self):
        self.queue = []
        self.playing = False
        self._frame_count = 0

        # sounddevice OutputStream
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=self._callback
        )

    def _callback(self, outdata, frames, time, status):
        if status:
            logger.warning(f"Playback stream status: {status}")

        data = np.empty(0, dtype=np.int16)

        # Dequeue from ring buffer
        while len(data) < frames and len(self.queue) > 0:
            chunk = self.queue.pop(0)
            needed = frames - len(data)
            data = np.concatenate((data, chunk[:needed]))
            if len(chunk) > needed:
                # leftover chunk
                self.queue.insert(0, chunk[needed:])

        # Pad with zeros if lacking enough samples
        if len(data) < frames:
            data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))

        outdata[:] = data.reshape(-1, CHANNELS)
        self._frame_count += frames

    def add_data(self, pcm_data: bytes):
        """
        Add raw 16-bit mono PCM data to the ring buffer.
        """
        arr = np.frombuffer(pcm_data, dtype=np.int16)
        self.queue.append(arr)
        if not self.playing:
            self.start()

    def reset_frame_count(self):
        self._frame_count = 0

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

# ---------------------------------------------------------
# Primary class for managing a Realtime connection
#  - No manual SDP creation
#  - Uses openai’s library for audio transcript & playback
# ---------------------------------------------------------
class RefactoredRealtimeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Connection and session info
        self.connection: AsyncRealtimeConnection | None = None
        self.session: Session | None = None

        self.audio_player = AudioPlayerAsync()  # ring-buffer playback
        self._should_stream_mic = False         # gate to turn mic streaming on/off
        self.transcripts = {}                   # store transcripts by item_id
        self.last_audio_item_id = None

        # Create an AsyncOpenAI client with your API key
        self.client = AsyncOpenAI(api_key=api_key)

    async def connect(self):
        """
        Connect to the OpenAI Realtime API using built-in Realtime methods.
        """
        try:
            logger.info("Connecting to Realtime endpoint...")
            async with self.client.beta.realtime.connect(
                model="gpt-4o-mini-realtime-preview-2024-12-17"
            ) as conn:
                self.connection = conn
                logger.info("Realtime connection established!")

                # Optionally configure session (e.g., server-based VAD)
                await conn.session.update(session={"turn_detection": {"type": "server_vad"}})

                # Launch tasks:
                event_task = asyncio.create_task(self._listen_for_events(conn))
                mic_task = asyncio.create_task(self._send_mic_frames(conn))

                # Keep them running until done
                await asyncio.gather(event_task, mic_task)

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

        finally:
            logger.info("Connection cleanup.")
            self.audio_player.stop()
            self.audio_player.terminate()

    async def _listen_for_events(self, conn: AsyncRealtimeConnection):
        """ Read events from the Realtime connection. """
        async for event in conn:
            event_type = event.type

            if event_type == "session.created":
                self.session = event.session
                logger.info(f"Session created. ID={self.session.id}")
                continue

            if event_type == "session.updated":
                self.session = event.session
                logger.debug("Session updated.")
                continue

            if event_type == "response.audio.delta":
                # Audio chunk from the model
                if event.item_id != self.last_audio_item_id:
                    self.audio_player.reset_frame_count()
                    self.last_audio_item_id = event.item_id

                bytes_data = base64.b64decode(event.delta)
                self.audio_player.add_data(bytes_data)
                continue

            if event_type == "response.audio_transcript.delta":
                so_far = self.transcripts.get(event.item_id, "")
                self.transcripts[event.item_id] = so_far + event.delta
                logger.info(f"Transcript: {self.transcripts[event.item_id]}")
                continue

            if event_type == "error":
                logger.error(f"Realtime API Error: {event.message}")
                continue

            logger.debug(f"Unhandled event from connection: {event}")

    async def _send_mic_frames(self, conn: AsyncRealtimeConnection):
        """
        Continuously read microphone frames from sounddevice,
        push them into connection.input_audio_buffer if streaming is enabled.
        """
        mic_stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16"
        )
        mic_stream.start()

        # We'll read in 20ms increments
        read_size = int(SAMPLE_RATE * 0.02)

        try:
            logger.info("Microphone streaming started. (Press Ctrl+C to stop program).")
            while True:
                if mic_stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                # Only send frames if _should_stream_mic is enabled
                if not self._should_stream_mic:
                    await asyncio.sleep(0.1)
                    continue

                data_np, _ = mic_stream.read(read_size)
                data_bytes = data_np.tobytes()
                audio_b64 = base64.b64encode(data_bytes).decode('utf-8')
                await conn.input_audio_buffer.append(audio=audio_b64)

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Mic streaming error: {e}")
        finally:
            logger.debug("Stopping mic input stream.")
            mic_stream.stop()
            mic_stream.close()

    def start_mic(self):
        """Enable streaming mic audio to the model."""
        self._should_stream_mic = True

    def stop_mic(self):
        """Disable streaming mic audio to the model."""
        self._should_stream_mic = False

# ---------------------------------------------------------
# Example main usage
# ---------------------------------------------------------
async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    client = RefactoredRealtimeClient(api_key=api_key)

    # Start the connection in a separate task
    conn_task = asyncio.create_task(client.connect())

    # Let it establish the connection
    await asyncio.sleep(2)
    client.start_mic()  # begin streaming mic audio

    # Let audio flow for 5 seconds
    await asyncio.sleep(5)
    client.stop_mic()

    # Keep the connection alive. Press Ctrl+C to abort.
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        conn_task.cancel()
        await conn_task

if __name__ == "__main__":
    asyncio.run(main())