"""Pipecat WebSocket transport with bounded device-buffer send-ahead."""

import asyncio
import time

from pipecat.transports.websocket.server import (
    WebsocketServerOutputTransport,
    WebsocketServerTransport,
)


class BufferedWebsocketServerOutputTransport(WebsocketServerOutputTransport):
    """Keep a bounded PCM cushion at a playback-buffered WebSocket client."""

    def __init__(self, *args, audio_send_ahead_ms: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_send_ahead_s = max(0, audio_send_ahead_ms) / 1000.0

    async def _write_audio_sleep(self):
        """Pace at real time only after the bounded send-ahead credit is full.

        Pipecat's stock scheduler resets its clock as soon as it falls behind,
        which prevents a device jitter buffer from recovering after a brief
        event-loop or network stall. This scheduler preserves the media clock
        long enough to catch up, while capping the lead so a long answer cannot
        outrun the Voice PE's finite PSRAM ring.
        """
        now = time.monotonic()
        # Start a fresh media clock after an interruption, startup, or a gap
        # longer than the cushion. That bounds both initial and catch-up bursts.
        if (
            self._next_send_time <= 0
            or self._next_send_time < now - self._audio_send_ahead_s
        ):
            self._next_send_time = now

        self._next_send_time += self._send_interval
        sleep_duration = max(
            0,
            self._next_send_time - now - self._audio_send_ahead_s,
        )
        # sleep(0) remains a cooperative event-loop yield during catch-up.
        await asyncio.sleep(sleep_duration)


class BufferedWebsocketServerTransport(WebsocketServerTransport):
    """Create the stock Pipecat server with bounded-ahead audio output."""

    def __init__(self, *args, audio_send_ahead_ms: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_send_ahead_ms = max(0, int(audio_send_ahead_ms))

    def output(self) -> WebsocketServerOutputTransport:
        # Pipecat 0.0.97 has no public output-class injection hook, so mirror
        # its tiny lazy factory and change only the concrete output class.
        if not self._output:
            self._output = BufferedWebsocketServerOutputTransport(
                self,
                self._params,
                name=self._output_name,
                audio_send_ahead_ms=self._audio_send_ahead_ms,
            )
        return self._output
