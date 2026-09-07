"""Response-lifecycle signals shared by the Realtime service and pipeline.

OpenAI's ``response.done`` means model generation has finished.  It does not
mean that Pipecat has finished writing the already-generated audio to the
device.  ``FinalResponseDoneFrame`` is therefore an ordered barrier: the
Realtime service emits it after the final response, and Pipecat's output
transport queues it behind every preceding audio frame.  A processor placed
after the output transport can then safely announce the turn boundary.
"""

from dataclasses import dataclass
from typing import Any, Iterable

from pipecat.frames.frames import ControlFrame


# Pipecat 0.0.97's websocket output transport emits 40 ms PCM chunks.  A full
# chunk of trailing zeroes guarantees that any sub-chunk of real reply audio is
# flushed before the ordered completion marker reaches that transport.
OUTPUT_DRAIN_PADDING_MS = 40


@dataclass
class FinalResponseDoneFrame(ControlFrame):
    """Ordered marker for a terminal, non-tool Realtime response."""

    response_id: str


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK model or a plain mapping."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _output_items(response: Any) -> Iterable[Any]:
    return _field(response, "output", None) or ()


def is_final_non_tool_response(response: Any) -> bool:
    """Return whether ``response.done`` represents the final spoken answer.

    Tool-call responses are intermediate even when their status is
    ``completed``.  Pipecat executes the tool and requests another response;
    only that later, non-tool response may end the conversational turn.
    Failed and cancelled responses remain owned by the existing
    recovery/interrupt paths and must not open a follow-up microphone window.
    """

    # ``incomplete`` is also terminal: the model can hit max-output-tokens or a
    # content boundary after already producing valid audio. Waiting for another
    # response in that case would strand the device in replying/thinking.
    if _field(response, "status") not in {"completed", "incomplete"}:
        return False
    return not any(_field(item, "type") == "function_call" for item in _output_items(response))


def response_id(response: Any) -> str:
    """Return the API response id, with a stable fallback for old SDK models."""

    value = _field(response, "id", "")
    return str(value) if value else f"local-response-{id(response)}"


def output_drain_padding_audio(
    sample_rate: int = 24_000,
    num_channels: int = 1,
) -> bytes:
    """Return one transport chunk of PCM16 silence for the drain barrier."""

    samples = sample_rate * OUTPUT_DRAIN_PADDING_MS // 1000 * num_channels
    return b"\x00" * (samples * 2)
