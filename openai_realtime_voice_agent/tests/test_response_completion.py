import asyncio
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum


# Production supplies Pipecat.  These tests exercise our response-lifecycle
# state machine without requiring the full audio/runtime dependency locally.
try:
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection
except ModuleNotFoundError:
    frames_module = types.ModuleType("pipecat.frames.frames")

    class Frame:
        pass

    class ControlFrame(Frame):
        pass

    class UserStartedSpeakingFrame(Frame):
        pass

    class UserStoppedSpeakingFrame(Frame):
        pass

    class BotStartedSpeakingFrame(Frame):
        pass

    class BotStoppedSpeakingFrame(Frame):
        pass

    frames_module.Frame = Frame
    frames_module.ControlFrame = ControlFrame
    frames_module.UserStartedSpeakingFrame = UserStartedSpeakingFrame
    frames_module.UserStoppedSpeakingFrame = UserStoppedSpeakingFrame
    frames_module.BotStartedSpeakingFrame = BotStartedSpeakingFrame
    frames_module.BotStoppedSpeakingFrame = BotStoppedSpeakingFrame

    processor_module = types.ModuleType("pipecat.processors.frame_processor")

    class FrameDirection(Enum):
        DOWNSTREAM = "downstream"
        UPSTREAM = "upstream"

    class FrameProcessor:
        def __init__(self, **kwargs):
            self.forwarded_frames = []

        async def process_frame(self, frame, direction):
            return None

        async def push_frame(self, frame, direction):
            self.forwarded_frames.append((frame, direction))

    processor_module.FrameDirection = FrameDirection
    processor_module.FrameProcessor = FrameProcessor

    pipecat_module = types.ModuleType("pipecat")
    pipecat_frames_module = types.ModuleType("pipecat.frames")
    pipecat_processors_module = types.ModuleType("pipecat.processors")
    sys.modules["pipecat"] = pipecat_module
    sys.modules["pipecat.frames"] = pipecat_frames_module
    sys.modules["pipecat.frames.frames"] = frames_module
    sys.modules["pipecat.processors"] = pipecat_processors_module
    sys.modules["pipecat.processors.frame_processor"] = processor_module


from app.phase_emitter import PhaseEmitter, ResponseCompletionObserver, TURN_LIVENESS
from app.response_lifecycle import (
    FinalResponseDoneFrame,
    is_final_non_tool_response,
    output_drain_padding_audio,
)


@dataclass
class _ResponseItem:
    type: str


@dataclass
class _Response:
    status: str
    output: list[_ResponseItem]


class ResponseClassificationTests(unittest.TestCase):
    def test_output_drain_padding_is_40_ms_pcm16_mono_silence(self):
        audio = output_drain_padding_audio()

        self.assertEqual(len(audio), 24_000 * 40 // 1000 * 2)
        self.assertEqual(set(audio), {0})

    def test_only_terminal_non_tool_responses_are_final(self):
        cases = (
            (
                "completed message",
                _Response("completed", [_ResponseItem("message")]),
                True,
            ),
            (
                "incomplete message",
                _Response("incomplete", [_ResponseItem("message")]),
                True,
            ),
            (
                "completed function call",
                _Response("completed", [_ResponseItem("function_call")]),
                False,
            ),
            (
                "incomplete function call",
                _Response("incomplete", [_ResponseItem("function_call")]),
                False,
            ),
            (
                "failed message",
                _Response("failed", [_ResponseItem("message")]),
                False,
            ),
            (
                "cancelled message",
                _Response("cancelled", [_ResponseItem("message")]),
                False,
            ),
        )

        for label, response, expected in cases:
            with self.subTest(label=label):
                self.assertIs(is_final_non_tool_response(response), expected)

    def test_mapping_responses_are_classified_too(self):
        self.assertTrue(
            is_final_non_tool_response(
                {"status": "completed", "output": [{"type": "message"}]}
            )
        )
        self.assertFalse(
            is_final_non_tool_response(
                {"status": "completed", "output": [{"type": "function_call"}]}
            )
        )
        self.assertTrue(
            is_final_non_tool_response(
                {"status": "incomplete", "output": [{"type": "message"}]}
            )
        )


class ResponseCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        TURN_LIVENESS.in_flight = 0
        TURN_LIVENESS.last_activity = 0.0
        self.phases = []

        async def send_phase(value):
            self.phases.append(value)

        self.send_phase = send_phase
        self.emitter = PhaseEmitter(send_phase=send_phase)
        self.observer = ResponseCompletionObserver(self.emitter)

    async def asyncTearDown(self):
        self.emitter._cancel_pending_idle()
        self.emitter._cancel_watchdog()

    async def _send_to_emitter(self, frame):
        await self.emitter.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _drain_through_observer(self, frame):
        await self.observer.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_final_marker_does_not_idle_until_transport_observer(self):
        await self._send_to_emitter(BotStartedSpeakingFrame())
        marker = FinalResponseDoneFrame(response_id="response-1")

        await self._send_to_emitter(marker)
        self.assertEqual(self.phases, ["replying"])

        await self._drain_through_observer(marker)
        self.assertEqual(self.phases, ["replying", "idle"])

    async def test_completion_from_thinking_emits_idle(self):
        await self._send_to_emitter(UserStoppedSpeakingFrame())
        marker = FinalResponseDoneFrame(response_id="response-from-thinking")

        await self._send_to_emitter(marker)
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, ["thinking", "idle"])

    async def test_completion_from_none_is_ignored(self):
        marker = FinalResponseDoneFrame(response_id="response-without-phase")

        await self._send_to_emitter(marker)
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, [])

    async def test_completion_from_listening_is_ignored(self):
        await self._send_to_emitter(UserStartedSpeakingFrame())
        marker = FinalResponseDoneFrame(response_id="response-while-listening")

        await self._send_to_emitter(marker)
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, ["listening"])

    async def test_completion_from_idle_is_idempotent(self):
        await self.emitter.force_idle()
        marker = FinalResponseDoneFrame(response_id="response-after-idle")

        await self._send_to_emitter(marker)
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, ["idle"])
        self.assertIsNone(self.emitter._pending_final_response_id)

    async def test_wake_invalidates_marker_waiting_behind_transport(self):
        marker = FinalResponseDoneFrame(response_id="response-before-wake")
        await self._send_to_emitter(marker)

        self.emitter.note_wake()
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, [])

    async def test_user_start_invalidates_marker_waiting_behind_transport(self):
        marker = FinalResponseDoneFrame(response_id="response-before-user")
        await self._send_to_emitter(marker)

        await self._send_to_emitter(UserStartedSpeakingFrame())
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, ["listening"])

    async def test_duplicate_completion_is_idempotent(self):
        await self._send_to_emitter(BotStartedSpeakingFrame())
        marker = FinalResponseDoneFrame(response_id="response-once")
        await self._send_to_emitter(marker)

        await self._drain_through_observer(marker)
        await self._drain_through_observer(marker)

        self.assertEqual(self.phases, ["replying", "idle"])

    async def test_bot_stopped_does_not_idle_immediately(self):
        await self._send_to_emitter(BotStartedSpeakingFrame())
        await self._send_to_emitter(BotStoppedSpeakingFrame())

        self.assertEqual(self.phases, ["replying"])

    async def test_bot_stopped_uses_legacy_fallback_after_debounce(self):
        emitter = PhaseEmitter(send_phase=self.send_phase, idle_debounce_s=0.01)
        try:
            await emitter.process_frame(
                BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await emitter.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            self.assertEqual(self.phases, ["replying"])

            await asyncio.sleep(0.03)
            self.assertEqual(self.phases, ["replying", "idle"])
        finally:
            emitter._cancel_pending_idle()
            emitter._cancel_watchdog()

    async def test_late_bot_stop_cannot_close_a_new_thinking_turn(self):
        emitter = PhaseEmitter(send_phase=self.send_phase, idle_debounce_s=0.01)
        try:
            await emitter.process_frame(
                BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await emitter.process_frame(
                UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await emitter.process_frame(
                UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )

            # This belongs to the old reply but arrives after the new user's
            # speaking frames. It must not arm the legacy silence fallback.
            await emitter.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await asyncio.sleep(0.03)

            self.assertEqual(self.phases, ["replying", "listening", "thinking"])
            self.assertIsNone(emitter._idle_task)
        finally:
            emitter._cancel_pending_idle()
            emitter._cancel_watchdog()

    async def test_fallback_rechecks_generation_before_emitting_idle(self):
        emitter = PhaseEmitter(send_phase=self.send_phase, idle_debounce_s=0.01)
        try:
            await emitter.process_frame(
                BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await emitter.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            emitter._idle_generation += 1  # simulate invalidation racing wakeup

            await asyncio.sleep(0.03)

            self.assertEqual(self.phases, ["replying"])
        finally:
            emitter._cancel_pending_idle()
            emitter._cancel_watchdog()

    async def test_transport_barrier_cancels_legacy_fallback(self):
        emitter = PhaseEmitter(send_phase=self.send_phase, idle_debounce_s=0.02)
        observer = ResponseCompletionObserver(emitter)
        marker = FinalResponseDoneFrame(response_id="response-before-fallback")
        try:
            await emitter.process_frame(
                BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await emitter.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
            await emitter.process_frame(marker, FrameDirection.DOWNSTREAM)
            await observer.process_frame(marker, FrameDirection.DOWNSTREAM)

            await asyncio.sleep(0.04)
            self.assertEqual(self.phases, ["replying", "idle"])
        finally:
            emitter._cancel_pending_idle()
            emitter._cancel_watchdog()


if __name__ == "__main__":
    unittest.main()
