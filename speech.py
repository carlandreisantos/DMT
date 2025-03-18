import queue
import re
import sys
import time
from google.cloud import speech
import pyaudio
import os

transcribed_text = ''
# Set the path to your Google Cloud service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Dual-Modal-Translator\dual-modal-translator-cb789111ae3d.json"

# Audio recording parameters
STREAMING_LIMIT = 1800000  
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time() -> int:
    return int(round(time.time() * 1000))

class ResumableMicrophoneStream:
    def __init__(self, rate: int, chunk_size: int) -> None:
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )

    def __enter__(self) -> object:
        self.closed = False
        return self

    def __exit__(self, type: object, value: object, traceback: object) -> object:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data: object, *args: object, **kwargs: object) -> object:
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self) -> object:
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses, stream, callback=None):
    transcription = ""
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        transcription += transcript + "\n"

        if callback:
            callback(transcript)  # Send real-time updates via the callback

        if result.is_final:
            sys.stdout.write("\033[K")
            sys.stdout.write(transcript + "\n")
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
    return transcription


def reset_transcription():
    """Resets the transcription output."""
    global transcription_output
    transcription_output = ""  # Clear the global transcription output


def start_speech_recognition(callback=None) -> str:
    """Start speech recognition and return the transcribed text."""
    global transcription_output  # Ensure we're modifying the global variable
    transcription_output = ""  # Reset transcription output at the start

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="fil-PH",
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    
    # Reset transcription each time a new stream starts
    with mic_manager as stream:
        while not stream.closed:
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Capture transcription output and invoke callback
            transcription_output = listen_print_loop(responses, stream, callback)

    return transcription_output


# If you want to run this as a standalone script:
if __name__ == "__main__":
    print(start_speech_recognition())