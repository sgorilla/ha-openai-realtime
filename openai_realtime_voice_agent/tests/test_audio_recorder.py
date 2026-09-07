import ast
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from app.audio_recorder import AudioRecorder


DEVICE_SAMPLE_RATE = 16_000
TEST_TONE_HZ = 440


def _pcm16_sine(*, frequency: int, sample_rate: int, duration: float) -> bytes:
    sample_count = round(sample_rate * duration)
    samples = (
        round(12_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(sample_count)
    )
    return b"".join(struct.pack("<h", sample) for sample in samples)


def _positive_crossing_frequency(samples: tuple[int, ...], duration: float) -> float:
    crossings = sum(
        previous <= 0 < current
        for previous, current in zip(samples, samples[1:])
    )
    return crossings / duration


class AudioRecorderFormatTests(unittest.TestCase):
    def test_device_recorder_tap_precedes_input_resampler(self):
        """Lock the raw-input tap to the device side of the rate boundary.

        Importing websocket_handler requires the full Pipecat runtime, which is
        deliberately absent from lightweight checkout tests. Inspecting the
        small build_pipeline function still gives this regression a direct test:
        the old implementation constructed InputResampler before it appended the
        input recorder; the fixed implementation does the opposite.
        """

        handler_path = Path(__file__).parents[1] / "app" / "websocket_handler.py"
        module = ast.parse(handler_path.read_text(encoding="utf-8"))
        build_pipeline = next(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "build_pipeline"
        )

        recorder_append_lines = []
        resampler_lines = []
        for node in ast.walk(build_pipeline):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "input_recorder"
            ):
                recorder_append_lines.append(node.lineno)
            if isinstance(node.func, ast.Name) and node.func.id == "InputResampler":
                resampler_lines.append(node.lineno)

        self.assertEqual(len(recorder_append_lines), 1)
        self.assertEqual(len(resampler_lines), 1)
        self.assertLess(recorder_append_lines[0], resampler_lines[0])

    def test_native_device_capture_has_true_duration_rate_and_pitch(self):
        """A one-second 16 kHz device frame must remain one-second, 440 Hz audio.

        This is the end-to-end invariant broken when the recorder was placed
        after the 16->24 kHz InputResampler but continued to write a 16 kHz WAV
        header: playback became 1.5 seconds long and the pitch fell to ~293 Hz.
        """

        with tempfile.TemporaryDirectory() as output_dir:
            recorder = AudioRecorder(
                output_dir=output_dir,
                input_sample_rate=DEVICE_SAMPLE_RATE,
                output_sample_rate=24_000,
            )
            recorder.start_recording("format-test")
            recorder.record_input_audio(
                _pcm16_sine(
                    frequency=TEST_TONE_HZ,
                    sample_rate=DEVICE_SAMPLE_RATE,
                    duration=1.0,
                )
            )
            recorder.stop_recording()

            input_path = next(Path(output_dir).glob("input_*.wav"))
            with wave.open(str(input_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), DEVICE_SAMPLE_RATE)
                self.assertEqual(wav_file.getnframes(), DEVICE_SAMPLE_RATE)
                duration = wav_file.getnframes() / wav_file.getframerate()
                samples = struct.unpack(
                    f"<{wav_file.getnframes()}h",
                    wav_file.readframes(wav_file.getnframes()),
                )

            self.assertAlmostEqual(duration, 1.0, places=6)
            self.assertAlmostEqual(
                _positive_crossing_frequency(samples, duration),
                TEST_TONE_HZ,
                delta=1.0,
            )


if __name__ == "__main__":
    unittest.main()
