import sys
import types
import unittest
from unittest.mock import patch


# Production supplies Pipecat. Keep this focused subclass test runnable in the
# lightweight repository test environment, where Pipecat is intentionally not
# installed.
try:
    from pipecat.transports.websocket.server import WebsocketServerParams
except ModuleNotFoundError:
    using_pipecat_stubs = True
    stub_module_names = (
        "pipecat.transports.websocket.server",
        "pipecat.transports.websocket",
        "pipecat.transports",
        "pipecat",
    )
    server_module = types.ModuleType("pipecat.transports.websocket.server")

    class WebsocketServerParams:
        pass

    class WebsocketServerOutputTransport:
        def __init__(self, transport, params, **kwargs):
            self._transport = transport
            self._params = params
            self._name = kwargs.get("name")
            self._send_interval = 0.04
            self._next_send_time = 0

    class WebsocketServerTransport:
        def __init__(self, params, output_name=None, **kwargs):
            self._params = params
            self._output_name = output_name
            self._output = None

    server_module.WebsocketServerParams = WebsocketServerParams
    server_module.WebsocketServerOutputTransport = WebsocketServerOutputTransport
    server_module.WebsocketServerTransport = WebsocketServerTransport

    sys.modules.setdefault("pipecat", types.ModuleType("pipecat"))
    sys.modules.setdefault("pipecat.transports", types.ModuleType("pipecat.transports"))
    sys.modules.setdefault(
        "pipecat.transports.websocket", types.ModuleType("pipecat.transports.websocket")
    )
    sys.modules["pipecat.transports.websocket.server"] = server_module
else:
    using_pipecat_stubs = False

from app.buffered_websocket_transport import (
    BufferedWebsocketServerOutputTransport,
    BufferedWebsocketServerTransport,
)

# Do not leak the lightweight stand-ins into subsequently collected tests,
# which install their own, broader Pipecat fakes.
if using_pipecat_stubs:
    for module_name in stub_module_names:
        sys.modules.pop(module_name, None)


class BufferedOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_burst_is_bounded_then_paced(self):
        delays = []

        async def capture_sleep(delay):
            delays.append(delay)

        output = object.__new__(BufferedWebsocketServerOutputTransport)
        output._audio_send_ahead_s = 0.12
        output._send_interval = 0.04
        output._next_send_time = 0

        with (
            patch("app.buffered_websocket_transport.time.monotonic", return_value=100.0),
            patch("app.buffered_websocket_transport.asyncio.sleep", capture_sleep),
        ):
            for _ in range(4):
                await output._write_audio_sleep()

        self.assertAlmostEqual(delays[0], 0.0)
        self.assertAlmostEqual(delays[1], 0.0)
        self.assertAlmostEqual(delays[2], 0.0)
        self.assertAlmostEqual(delays[3], 0.04)

    async def test_brief_stall_spends_credit_to_refill_device_buffer(self):
        delays = []

        async def capture_sleep(delay):
            delays.append(delay)

        output = object.__new__(BufferedWebsocketServerOutputTransport)
        output._audio_send_ahead_s = 3.0
        output._send_interval = 0.04
        # The media clock was three seconds ahead before a 700 ms stall.
        output._next_send_time = 103.0

        with (
            patch("app.buffered_websocket_transport.time.monotonic", return_value=100.7),
            patch("app.buffered_websocket_transport.asyncio.sleep", capture_sleep),
        ):
            await output._write_audio_sleep()

        self.assertEqual(delays, [0])
        self.assertAlmostEqual(output._next_send_time, 103.04)

    async def test_long_gap_resets_clock_and_keeps_burst_bounded(self):
        delays = []

        async def capture_sleep(delay):
            delays.append(delay)

        output = object.__new__(BufferedWebsocketServerOutputTransport)
        output._audio_send_ahead_s = 3.0
        output._send_interval = 0.04
        output._next_send_time = 90.0

        with (
            patch("app.buffered_websocket_transport.time.monotonic", return_value=100.0),
            patch("app.buffered_websocket_transport.asyncio.sleep", capture_sleep),
        ):
            await output._write_audio_sleep()

        self.assertEqual(delays, [0])
        self.assertAlmostEqual(output._next_send_time, 100.04)


class BufferedServerTests(unittest.TestCase):
    def test_factory_builds_one_bounded_output_and_reuses_it(self):
        params = WebsocketServerParams()
        transport = BufferedWebsocketServerTransport(
            params=params,
            output_name="voice-pe-output",
            audio_send_ahead_ms=3000,
        )

        first = transport.output()
        second = transport.output()

        self.assertIs(first, second)
        self.assertIsInstance(first, BufferedWebsocketServerOutputTransport)
        self.assertIs(first._transport, transport)
        self.assertIs(first._params, params)
        self.assertEqual(first._audio_send_ahead_s, 3.0)


if __name__ == "__main__":
    unittest.main()
