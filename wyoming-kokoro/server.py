import os
import io
import math
import wave
import asyncio
from aiohttp import ClientSession

from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped

KOKORO_URL = os.getenv("KOKORO_URL", "http://kokoro:8880/v1/audio/speech")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
HOST = os.getenv("WYOMING_HOST", "0.0.0.0")
PORT = int(os.getenv("WYOMING_PORT", "10200"))
SAMPLES_PER_CHUNK = int(os.getenv("SAMPLES_PER_CHUNK", "2048"))

print(f"wyoming-kokoro starting on {HOST}:{PORT}, voice={DEFAULT_VOICE}", flush=True)

KOKORO_VOICES = [
  "af_alloy","af_aoede","af_bella","af_heart","af_jadzia","af_jessica","af_kore","af_nicole","af_nova","af_river",
  "af_sarah","af_sky","af_v0","af_v0bella","af_v0irulan","af_v0nicole","af_v0sarah","af_v0sky",
  "am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael","am_onyx","am_puck","am_santa","am_v0adam",
  "am_v0gurney","am_v0michael","bf_alice","bf_emma","bf_lily","bf_v0emma","bf_v0isabella","bm_daniel","bm_fable",
  "bm_george","bm_lewis","bm_v0george","bm_v0lewis","ef_dora","em_alex","em_santa","ff_siwis","hf_alpha","hf_beta",
  "hm_omega","hm_psi","if_sara","im_nicola","jf_alpha","jf_gongitsune","jf_nezumi","jf_tebukuro","jm_kumo",
  "pf_dora","pm_alex","pm_santa","zf_xiaobei","zf_xiaoni","zf_xiaoxiao","zf_xiaoyi","zm_yunjian","zm_yunxi",
  "zm_yunxia","zm_yunyang"
]

def make_info_event() -> Event:
    return Event(
        type="info",
        data={
            "tts": [
                {
                    "name": "kokoro",
                    "description": "Kokoro TTS via HTTP bridge",
                    "attribution": {"name": "Kokoro", "url": "https://huggingface.co/hexgrad/Kokoro-82M"},
                    "installed": True,
                    "voices": [
                        {
                            "name": v,
                            "description": v,
                            "attribution": {"name": "Kokoro", "url": "https://huggingface.co/hexgrad/Kokoro-82M"},
                            "installed": True,
                            "languages": ["en"],  # you can refine later
                        }
                        for v in KOKORO_VOICES
                    ],
                }
            ]
        },
    )

async def kokoro_wav_bytes(session: ClientSession, text: str, voice: str) -> bytes:
    payload = {"model": "kokoro", "input": text, "voice": voice}
    async with session.post(KOKORO_URL, json=payload) as resp:
        resp.raise_for_status()
        return await resp.read()

def wav_to_pcm_chunks(wav_bytes: bytes, samples_per_chunk: int):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

    bytes_per_sample = width * channels
    bytes_per_chunk = bytes_per_sample * samples_per_chunk
    num_chunks = int(math.ceil(len(frames) / bytes_per_chunk)) if bytes_per_chunk else 0
    chunks = []
    for i in range(num_chunks):
        chunk = frames[i * bytes_per_chunk : (i + 1) * bytes_per_chunk]
        if chunk:
            chunks.append(chunk)
    return rate, width, channels, chunks

class KokoroTtsHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming = False
        self._stream_text_parts = []
        self._stream_voice = DEFAULT_VOICE

    async def _send_audio(self, session: ClientSession, text: str, voice: str):
        wav_bytes = await kokoro_wav_bytes(session, text=text, voice=voice)
        rate, width, channels, chunks = wav_to_pcm_chunks(wav_bytes, SAMPLES_PER_CHUNK)

        await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        for c in chunks:
            await self.write_event(AudioChunk(audio=c, rate=rate, width=width, channels=channels).event())
        await self.write_event(AudioStop().event())

    async def handle_event(self, event: Event) -> bool:
        print("event:", event.type, flush=True)

        if event.type == "describe":
            await self.write_event(make_info_event())
            return True

        async with ClientSession() as session:
            if Synthesize.is_type(event.type):
                synth = Synthesize.from_event(event)
                voice = DEFAULT_VOICE
                if synth.voice and getattr(synth.voice, "name", None):
                    voice = synth.voice.name
                await self._send_audio(session, text=synth.text, voice=voice)
                return True

            if SynthesizeStart.is_type(event.type):
                st = SynthesizeStart.from_event(event)
                self._streaming = True
                self._stream_text_parts = []
                self._stream_voice = DEFAULT_VOICE
                if st.voice and getattr(st.voice, "name", None):
                    self._stream_voice = st.voice.name
                return True

            if self._streaming and SynthesizeChunk.is_type(event.type):
                ch = SynthesizeChunk.from_event(event)
                self._stream_text_parts.append(ch.text)
                return True

            if self._streaming and SynthesizeStop.is_type(event.type):
                self._streaming = False
                full_text = "".join(self._stream_text_parts).strip()
                if full_text:
                    await self._send_audio(session, text=full_text, voice=self._stream_voice)
                await self.write_event(SynthesizeStopped().event())
                return True

        return True

async def main():
    server = AsyncTcpServer(host=HOST, port=PORT)
    await server.run(KokoroTtsHandler)

if __name__ == "__main__":
    asyncio.run(main())
