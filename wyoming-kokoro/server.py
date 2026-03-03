import os, asyncio
from aiohttp import ClientSession
from wyoming.server import AsyncServer
from wyoming.event import Event

KOKORO_URL = os.getenv("KOKORO_URL")
VOICE = os.getenv("KOKORO_VOICE", "af")

async def tts(text):
    async with ClientSession() as s:
        async with s.post(KOKORO_URL, json={
            "model": "kokoro",
            "input": text,
            "voice": VOICE
        }) as r:
            r.raise_for_status()
            return await r.read()

async def handler(reader, writer):
    info = Event(type="info", data={
        "name": "kokoro",
        "version": "1.0",
        "tts": True
    })
    await info.write(writer)

    while not reader.at_eof():
        event = await Event.read(reader)
        if event and event.type.startswith("tts"):
            text = (event.data or {}).get("text", "")
            audio = await tts(text)
            await Event(
                type="tts.audio",
                data={"format": "wav"},
                payload=audio
            ).write(writer)

async def main():
    await AsyncServer(host="0.0.0.0", port=10200, handler=handler).run()

asyncio.run(main())
