import os
import json
import logging
import pyaudio
import numpy as np
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import aiohttp
import asyncio
from av import AudioFrame

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add these lines to suppress aiortc's verbose RTP logging
aiortc_logger = logging.getLogger('aiortc')
aiortc_logger.setLevel(logging.INFO)
rtp_logger = logging.getLogger('aiortc.rtcrtpsender')
rtp_logger.setLevel(logging.WARNING)
rtp_logger = logging.getLogger('aiortc.rtcrtpreceiver')
rtp_logger.setLevel(logging.WARNING)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # Default rate, will be adjusted based on device support
CHUNK = 960   # Default for 48kHz, will be adjusted based on actual rate

def get_supported_sample_rates(audio):
    """Test a wide range of sample rates to find supported ones"""
    try:
        # Get default output device info
        default_output = audio.get_default_output_device_info()
        device_index = default_output['index']
        device_name = default_output['name']
        
        logger.debug(f"Testing output device: {device_name} (index {device_index})")
        
        # Expanded list of sample rates to test
        test_rates = [
            8000, 11025, 12000, 16000, 22050, 
            24000, 32000, 44100, 48000, 
            88200, 96000, 176400, 192000
        ]
        
        supported_rates = []
        
        # Test each sample rate
        for rate in test_rates:
            try:
                # Try opening a stream with this rate
                stream = audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=rate,
                    output=True,
                    frames_per_buffer=1024,
                    output_device_index=device_index
                )
                stream.close()
                supported_rates.append(rate)
                logger.debug(f"Sample rate {rate} Hz - Supported")
            except Exception as e:
                logger.debug(f"Sample rate {rate} Hz - Not supported ({str(e)})")
                
        logger.info(f"Supported sample rates: {supported_rates}")
        return supported_rates
        
    except Exception as e:
        logger.error(f"Error testing sample rates: {e}")
        # Fallback to common rates if testing fails
        return [44100, 48000, 16000]

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
        self.remote_audio_track = None

    async def wait_for_data_channel(self, timeout=5):
        """Wait for data channel to be ready with timeout"""
        start_time = asyncio.get_event_loop().time()
        while True:
            if self.data_channel and self.data_channel.readyState == "open":
                logger.info("Data channel is ready")
                return True
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.error(f"Data channel not ready after {timeout} seconds")
                return False
                
            await asyncio.sleep(0.1)

    def _handle_message(self, message):
        try:
            event = json.loads(message)
            event_type = event.get('type')
            
            if event_type == 'response.audio_transcript.delta':
                logger.info(f"Transcript: {event.get('delta', {}).get('text', '')}")
            elif event_type == 'response.created':
                logger.info("Response started")
            elif event_type == 'response.done':
                logger.info("Response completed")
            elif event_type == 'error':
                logger.error(f"API Error: {event.get('message', '')}")
            else:
                logger.debug(f"Received event: {event_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def start_audio_streams(self):
        # Declare globals at the start of the method
        global RATE, CHUNK
        
        # Default to 44100 Hz
        RATE = 44100
        CHUNK = int(RATE * 0.02)  # 20ms frame size
        
        logger.info(f"Using sample rate: {RATE} Hz with chunk size: {CHUNK}")
        
        try:
            output_info = self.audio.get_default_output_device_info()
            logger.debug(f"Using output device: {output_info['name']}")
            
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK,
                output_device_index=output_info['index']
            )
            
            self.input_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            asyncio.create_task(self.process_input_audio())
            
        except Exception as e:
            logger.error(f"Error starting audio streams: {e}")
            raise

    async def process_input_audio(self):
        logger.debug("Starting input audio processing")
        while True:
            try:
                data = self.input_stream.read(CHUNK, exception_on_overflow=False)
                # Check if we're getting non-silent audio input
                audio_array = np.frombuffer(data, dtype=np.int16)
                max_input_amplitude = np.max(np.abs(audio_array))
                if max_input_amplitude > 100:  # Threshold to avoid logging silence
                    logger.debug(f"Input audio max amplitude: {max_input_amplitude}")
                
                if self.audio_track:
                    await self.audio_track.send_audio(data)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in input audio processing: {e}")
                break

    async def handle_remote_track(self, track):
        logger.debug(f"Remote audio track received with format: {track.kind}, {track.readyState}")
        self.remote_audio_track = track
        
        # Initialize sample rate conversion variables
        last_reported_rate = None
        conversion_factor = 1.0
        
        while True:
            try:
                frame = await track.recv()
                
                # Log frame details if sample rate changes
                if frame.sample_rate != last_reported_rate:
                    logger.info(f"Received frame with sample rate: {frame.sample_rate} Hz")
                    last_reported_rate = frame.sample_rate
                    conversion_factor = RATE / frame.sample_rate
                    logger.info(f"Sample rate conversion factor: {conversion_factor}")
                
                # Convert the frame to raw audio data
                audio_data = frame.to_ndarray()
                
                # If conversion is needed
                if conversion_factor != 1.0:
                    original_samples = audio_data.shape[0]
                    target_samples = int(original_samples * conversion_factor)
                    
                    # Resample using numpy's interpolation
                    x_old = np.linspace(0, 1, original_samples)
                    x_new = np.linspace(0, 1, target_samples)
                    audio_data = np.interp(x_new, x_old, audio_data)
                    
                    logger.debug(f"Resampled audio: {original_samples} -> {target_samples} samples")
                
                # Convert to int16 and ensure proper byte order
                audio_data = audio_data.astype(np.int16)
                
                # Convert to bytes
                raw_audio = audio_data.tobytes()
                
                await self.play_audio(raw_audio)
                    
            except Exception as e:
                logger.error(f"Error handling remote track: {e}")
                logger.exception("Full traceback:")
                break


    async def play_audio(self, audio_data):
        """Add audio data to the output queue"""
        try:
            if len(audio_data) > 0:
                logger.debug(f"Writing {len(audio_data)} bytes to output stream")
                self.output_stream.write(audio_data)
            else:
                logger.warning("Received empty audio data")
        except Exception as e:
            logger.error(f"Error in play_audio: {e}")
            logger.exception("Full traceback:")
            raise


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
        """Create ephemeral token and establish connection"""
        try:
            # Get ephemeral token
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
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Failed to get token: {error_text}")
                        raise RuntimeError(f"Failed to get token: {resp.status}")
                    token_data = await resp.json()
                    ephemeral_key = token_data['client_secret']['value']

            # Initialize audio streams
            await self.start_audio_streams()

            # Add audio track
            self.audio_track = AudioTrack(self)
            self.pc.addTrack(self.audio_track)

            # Set up track handler
            @self.pc.on("track")
            def on_track(track):
                logger.debug(f"Track received: {track.kind}")
                if track.kind == "audio":
                    asyncio.create_task(self.handle_remote_track(track))

            # Set up data channel with event handlers
            self.data_channel = self.pc.createDataChannel("oai-events")
            
            @self.data_channel.on("open")
            def on_open():
                logger.info("Data channel opened")
                # Send initial configuration
                config_message = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"]
                    }
                }
                self.data_channel.send(json.dumps(config_message))
                logger.info("Sent initial configuration message")
                
            @self.data_channel.on("close")
            def on_close():
                logger.warning("Data channel closed")
                
            @self.data_channel.on("error")
            def on_error(error):
                logger.error(f"Data channel error: {error}")

            self.data_channel.on("message", self._handle_message)

            # Create and set local description
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            # Send offer and get answer
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17",
                    headers={
                        "Authorization": f"Bearer {ephemeral_key}",
                        "Content-Type": "application/sdp"
                    },
                    data=offer.sdp
                ) as resp:
                    # Handle both 200 and 201 status codes
                    if resp.status not in (200, 201):
                        error_text = await resp.text()
                        logger.error(f"Failed to get answer: {error_text}")
                        raise RuntimeError(f"Failed to get answer: {resp.status}")
                    answer_sdp = await resp.text()
                    answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
                    await self.pc.setRemoteDescription(answer)

            # Wait for data channel to be ready
            if not await self.wait_for_data_channel():
                raise RuntimeError("Data channel failed to initialize")

        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            # Clean up on failure
            if self.pc:
                await self.pc.close()
            raise

class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, client):
        super().__init__()
        self.client = client
        self.audio_buffer = asyncio.Queue()
        logger.debug("AudioTrack initialized")
        self.sample_rate = RATE  # Now 48000
        self.channels = CHANNELS
        self.samples_per_channel = CHUNK  # Now 960
        self.timestamp = 0  # Initialize timestamp counter

    async def recv(self):
        try:
            frame = await self.audio_buffer.get()
            return frame
        except Exception as e:
            logger.error(f"Error in recv: {e}")
            raise

    async def send_audio(self, audio_data):
        try:
            # Convert raw bytes to AudioFrame
            audio_frame = AudioFrame(format='s16', layout='mono', samples=self.samples_per_channel)
            audio_frame.planes[0].update(audio_data)
            audio_frame.sample_rate = self.sample_rate
            audio_frame.time_base = '1/{}'.format(self.sample_rate)
            
            # Log frame details before sending
            logger.debug(f"Sending audio frame: format={audio_frame.format.name}, "
                        f"samples={audio_frame.samples}, "
                        f"sample_rate={audio_frame.sample_rate}")
            
            # Set the timestamp and increment
            audio_frame.pts = self.timestamp
            self.timestamp += self.samples_per_channel
            
            await self.audio_buffer.put(audio_frame)
        except Exception as e:
            logger.error(f"Error in send_audio: {e}")
            raise


if __name__ == "__main__":
    async def main():
        client = None
        try:
            logger.info("Starting application")
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in .env file")
                raise ValueError("Please create a .env file with OPENAI_API_KEY")
            
            client = RealtimeVoiceClient(api_key)
            await client.connect()
            
            # Keep the connection alive while processing audio
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Application error: {e}")
            # Add more detailed error information
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response content: {await e.response.text()}")
            raise
        finally:
            if client:
                logger.info("Cleaning up resources")
                if client.input_stream:
                    client.input_stream.stop_stream()
                    client.input_stream.close()
                if client.output_stream:
                    client.output_stream.stop_stream()
                    client.output_stream.close()
                if client.audio:
                    client.audio.terminate()
                if client.pc:
                    await client.pc.close()
                logger.info("Resources cleaned up successfully")

    asyncio.run(main())
