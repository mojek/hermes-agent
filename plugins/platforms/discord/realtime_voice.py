"""OpenAI Realtime transport used by the bundled Discord voice adapter."""

from __future__ import annotations

import base64
import asyncio
from collections import deque
from dataclasses import dataclass
import json
import logging
import struct
import threading
import time
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You are only the realtime voice UI for TARS. For every substantive user request, call "
    "delegate_to_tars exactly once with the user's complete request. Never answer substantive "
    "requests yourself, never invent a result, and emit no audio before the tool returns. After "
    "the tool returns, speak its output exactly."
)
_DELEGATE_TOOL = {
    "type": "function",
    "name": "delegate_to_tars",
    "description": "Send the user's substantive request to the existing TARS conversation.",
    "parameters": {
        "type": "object",
        "properties": {"input": {"type": "string", "description": "The complete user request."}},
        "required": ["input"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class LiveVoiceConfig:
    enabled: bool = False
    model: str = "gpt-realtime-2.1"
    voice: str = "cedar"
    max_session_seconds: int = 3600
    idle_seconds: int = 300


@dataclass
class _InputTurn:
    speaker_id: int
    generation: int
    transcript: asyncio.Future[str]


class DiscordToRealtimePCM:
    """Stateful 48 kHz stereo PCM16 to 24 kHz mono PCM16 converter."""

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._pending_frame: tuple[int, int] | None = None

    def feed(self, chunk: bytes) -> bytes:
        self._bytes.extend(chunk)
        complete = len(self._bytes) - (len(self._bytes) % 4)
        if not complete:
            return b""
        data = bytes(self._bytes[:complete])
        del self._bytes[:complete]
        out = bytearray()
        for left, right in struct.iter_unpack("<hh", data):
            if self._pending_frame is None:
                self._pending_frame = (left, right)
                continue
            prior_left, prior_right = self._pending_frame
            out.extend(struct.pack("<h", (prior_left + prior_right + left + right) // 4))
            self._pending_frame = None
        return bytes(out)


class RealtimeToDiscordPCM:
    """Stateful 24 kHz mono PCM16 to 48 kHz stereo PCM16 converter."""

    def __init__(self) -> None:
        self._bytes = bytearray()

    def feed(self, chunk: bytes) -> bytes:
        self._bytes.extend(chunk)
        complete = len(self._bytes) - (len(self._bytes) % 2)
        if not complete:
            return b""
        data = bytes(self._bytes[:complete])
        del self._bytes[:complete]
        out = bytearray()
        for (sample,) in struct.iter_unpack("<h", data):
            out.extend(struct.pack("<hhhh", sample, sample, sample, sample))
        return bytes(out)


class RealtimeVoiceClient:
    """Small state machine for one Discord guild's GA Realtime WebSocket."""

    def __init__(
        self, *, config: LiveVoiceConfig, mixer: Any,
        delegate: Callable[[int, str], Awaitable[str]], websocket: Any = None,
        api_key: Optional[str] = None, on_closed: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.websocket = websocket
        self.mixer = mixer
        self.delegate = delegate
        self._api_key = api_key
        self.on_closed = on_closed
        self._to_realtime = DiscordToRealtimePCM()
        self._to_discord = RealtimeToDiscordPCM()
        self._active_response_id: Optional[str] = None
        self._cancelled_response_ids: set[str] = set()
        self._allowed_audio_response_ids: dict[str, int] = {}
        self._handled_call_ids: set[str] = set()
        self._turns: dict[str, _InputTurn] = {}
        self._pending_turn_ids = deque()
        self._response_turn_ids: dict[str, str] = {}
        self._generation = 0
        self._delegate_lock = asyncio.Lock()
        self._delegation_tasks: set[asyncio.Task] = set()
        self._delegation_generations: dict[asyncio.Task, int] = {}
        self._closed = False
        self._ready = False
        self._closed_event = asyncio.Event()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        self._tasks: list[asyncio.Task] = []
        self._close_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._speaker_lock = threading.Lock()
        self._active_speaker: Optional[int] = None
        self._speaker_last_at = 0.0
        self.started_at = 0.0
        self.last_activity_at = 0.0
        self.connection_attempts = 0

    @property
    def ready(self) -> bool:
        return self._ready and not self._closed

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_speaker(self) -> Optional[int]:
        with self._speaker_lock:
            return self._active_speaker

    @property
    def tasks(self) -> tuple[asyncio.Task, ...]:
        return tuple(self._tasks) + tuple(self._delegation_tasks)

    def session_update_event(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.config.model,
                "output_modalities": ["text"],
                "instructions": _INSTRUCTIONS,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "gpt-4o-mini-transcribe"},
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": self.config.voice,
                    },
                },
                "tools": [_DELEGATE_TOOL],
                "tool_choice": "required",
            },
        }

    async def _send(self, event: dict[str, Any]) -> None:
        if not self._closed:
            await self.websocket.send(json.dumps(event))

    @staticmethod
    def _parse_server_event(payload: Any) -> dict[str, Any]:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        return value if isinstance(value, dict) else {}

    async def start(self) -> None:
        """Connect once and return only after the server confirms ``session.updated``."""
        self.connection_attempts += 1
        self._loop = asyncio.get_running_loop()
        self.started_at = self.last_activity_at = time.monotonic()
        try:
            if self.websocket is None:
                from websockets.asyncio.client import connect

                url = f"wss://api.openai.com/v1/realtime?model={quote(self.config.model)}"
                self.websocket = await connect(
                    url, additional_headers={"Authorization": f"Bearer {self._api_key}"},
                )
                self._api_key = None
            created = self._parse_server_event(await asyncio.wait_for(self.websocket.recv(), 15.0))
            if created.get("type") != "session.created":
                raise RuntimeError("Realtime server did not create a session")
            await self._send(self.session_update_event())
            while True:
                event = self._parse_server_event(await asyncio.wait_for(self.websocket.recv(), 15.0))
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise RuntimeError("Realtime server rejected session configuration")
                await self.handle_server_event(event)
            self._ready = True
            self._tasks = [
                asyncio.create_task(self._read_loop()),
                asyncio.create_task(self._audio_send_loop()),
                asyncio.create_task(self._watchdog_loop()),
            ]
        except BaseException:
            await self.close()
            raise

    def feed_discord_pcm(self, user_id: int, pcm: bytes, *, now: Optional[float] = None) -> bool:
        """Accept only one mapped speaker at a time and enqueue converted PCM off the socket thread."""
        if self._closed or not self._loop or not pcm:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        with self._speaker_lock:
            if self._active_speaker != user_id:
                if self._active_speaker is not None and timestamp - self._speaker_last_at < 1.5:
                    return False
                self._active_speaker = user_id
                self._to_realtime = DiscordToRealtimePCM()
            self._speaker_last_at = timestamp
            converted = self._to_realtime.feed(pcm)
        self.last_activity_at = timestamp
        if converted:
            def _enqueue() -> None:
                if not self._closed and not self._audio_queue.full():
                    self._audio_queue.put_nowait(converted)
            self._loop.call_soon_threadsafe(_enqueue)
        return True

    async def _audio_send_loop(self) -> None:
        try:
            while True:
                pcm = await self._audio_queue.get()
                await self._send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("OpenAI Realtime audio transport closed (%s)", type(exc).__name__)
            await self.close()

    async def _read_loop(self) -> None:
        try:
            while True:
                event = self._parse_server_event(await self.websocket.recv())
                self.last_activity_at = time.monotonic()
                await self.handle_server_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("OpenAI Realtime network connection closed (%s)", type(exc).__name__)
        finally:
            await self.close()

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.5)
                await self.check_timeouts()
        except asyncio.CancelledError:
            raise

    async def check_timeouts(self, *, now: Optional[float] = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        if (timestamp - self.started_at >= self.config.max_session_seconds
                or timestamp - self.last_activity_at >= self.config.idle_seconds):
            await self.close()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
        if self._close_task is not None and self._close_task is not asyncio.current_task():
            await asyncio.shield(self._close_task)

    async def wait_for_delegations(self) -> None:
        """Wait for the currently scheduled TARS calls (primarily useful for clean shutdown/tests)."""
        if self._delegation_tasks:
            await asyncio.gather(*tuple(self._delegation_tasks), return_exceptions=True)

    async def close(self) -> None:
        """Join one bounded cleanup task so every caller observes local closure."""
        if self._close_task is None:
            self._closed = True
            self._ready = False
            self._close_task = asyncio.create_task(self._close_impl(asyncio.current_task()))
        elif self._close_task is asyncio.current_task():
            return
        elif asyncio.current_task() in (*self._tasks, *self._delegation_tasks):
            await asyncio.shield(self._closed_event.wait())
            return
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await asyncio.shield(self._close_task)
            raise

    async def _close_impl(self, initiator: Optional[asyncio.Task]) -> None:
        current = asyncio.current_task()
        try:
            pending = [
                task for task in (*self._tasks, *self._delegation_tasks)
                if task is not current and task is not initiator and not task.done()
            ]
            for task in pending:
                task.cancel()
            self.mixer.clear_speech()
            if self.websocket is not None:
                try:
                    await asyncio.wait_for(self.websocket.close(), timeout=5.0)
                except Exception:
                    pass
            if self.on_closed is not None:
                try:
                    result = self.on_closed()
                    if asyncio.iscoroutine(result):
                        async with asyncio.timeout(15.0):
                            await result
                except Exception:
                    logger.warning("Discord live voice close callback failed")
        finally:
            self._closed_event.set()
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Discord live voice tasks did not stop before timeout")

    async def _delegate_and_respond(
        self, turn: _InputTurn, call_id: str,
    ) -> None:
        try:
            transcript = await asyncio.wait_for(asyncio.shield(turn.transcript), timeout=15.0)
            if not transcript.strip():
                return
            async with self._delegate_lock:
                if self._closed or turn.generation != self._generation:
                    return
                final_text = await self.delegate(turn.speaker_id, transcript.strip())
                if self._closed or turn.generation != self._generation:
                    return
                await self._send({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": final_text,
                    },
                })
                await self._send({
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "tool_choice": "none",
                        "instructions": "Speak the preceding function output exactly, with no additions.",
                        "metadata": {"hermes_live_generation": str(turn.generation)},
                    },
                })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Discord live voice delegation failed (%s)", type(exc).__name__)

    def _new_input_turn(self, item_id: str) -> None:
        self._generation += 1
        speaker_id = self.active_speaker
        if not item_id or speaker_id is None:
            return
        transcript = asyncio.get_running_loop().create_future()
        self._turns[item_id] = _InputTurn(speaker_id, self._generation, transcript)
        self._pending_turn_ids.append(item_id)

    def _complete_transcript(self, item_id: str, transcript: Any) -> None:
        turn = self._turns.get(item_id)
        if turn is None or turn.transcript.done() or not isinstance(transcript, str):
            return
        turn.transcript.set_result(transcript)

    async def handle_server_event(self, event: dict[str, Any]) -> None:
        """Consume one server event without retaining or logging its payload."""
        event_type = event.get("type")
        if event_type == "response.created":
            response_id = (event.get("response") or {}).get("id")
            if response_id:
                response_id = str(response_id)
                self._active_response_id = response_id
                metadata = (event.get("response") or {}).get("metadata") or {}
                try:
                    audio_generation = int(metadata.get("hermes_live_generation", -1))
                except (TypeError, ValueError):
                    audio_generation = -1
                if audio_generation == self._generation:
                    self._allowed_audio_response_ids[response_id] = audio_generation
                elif self._pending_turn_ids:
                    self._response_turn_ids[response_id] = self._pending_turn_ids.popleft()
            return
        if event_type == "input_audio_buffer.speech_started":
            self._new_input_turn(str(event.get("item_id") or ""))
            for task, generation in tuple(self._delegation_generations.items()):
                if generation < self._generation and not task.done():
                    task.cancel()
            self.mixer.clear_speech()
            self._to_discord = RealtimeToDiscordPCM()
            if self._active_response_id:
                response_id = self._active_response_id
                self._cancelled_response_ids.add(response_id)
                self._allowed_audio_response_ids.pop(response_id, None)
                self._active_response_id = None
                await self._send({"type": "response.cancel", "response_id": response_id})
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            self._complete_transcript(
                str(event.get("item_id") or ""), event.get("transcript"),
            )
            return
        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            response_id = str(event.get("response_id") or self._active_response_id or "")
            generation = self._allowed_audio_response_ids.get(response_id)
            if (not response_id or response_id in self._cancelled_response_ids
                    or generation != self._generation):
                return
            try:
                pcm = base64.b64decode(event.get("delta") or "", validate=True)
            except (ValueError, TypeError):
                return
            converted = self._to_discord.feed(pcm)
            if converted:
                self.mixer.queue_speech(converted, fade_in_ms=0)
            return
        if event_type == "response.done":
            response = event.get("response") or {}
            response_id = str(response.get("id") or self._active_response_id or "")
            self._allowed_audio_response_ids.pop(response_id, None)
            turn_id = self._response_turn_ids.pop(response_id, "")
            turn = self._turns.pop(turn_id, None)
            if response_id in self._cancelled_response_ids:
                self._cancelled_response_ids.discard(response_id)
                return
            self._active_response_id = None
            for item in response.get("output") or []:
                if item.get("type") != "function_call" or item.get("name") != "delegate_to_tars":
                    continue
                if turn is None:
                    continue
                call_id = str(item.get("call_id") or "")
                if not call_id or call_id in self._handled_call_ids:
                    continue
                self._handled_call_ids.add(call_id)
                task = asyncio.create_task(self._delegate_and_respond(turn, call_id))
                self._delegation_tasks.add(task)
                self._delegation_generations[task] = turn.generation

                def _forget_delegation(done_task: asyncio.Task) -> None:
                    self._delegation_tasks.discard(done_task)
                    self._delegation_generations.pop(done_task, None)

                task.add_done_callback(_forget_delegation)
                break
        elif event_type == "error":
            logger.warning("OpenAI Realtime session reported an error")


def load_live_voice_config() -> LiveVoiceConfig:
    """Resolve ``discord.live_voice`` for the currently bound profile."""
    from hermes_cli.config_effective import load_user_config_effective

    raw = ((load_user_config_effective().get("discord") or {}).get("live_voice") or {})
    if not isinstance(raw, dict):
        raw = {}

    def _seconds(name: str, default: int) -> int:
        try:
            return max(1, int(raw.get(name, default)))
        except (TypeError, ValueError):
            return default

    return LiveVoiceConfig(
        enabled=raw.get("enabled") is True,
        model=str(raw.get("model") or LiveVoiceConfig.model),
        voice=str(raw.get("voice") or LiveVoiceConfig.voice),
        max_session_seconds=_seconds("max_session_seconds", LiveVoiceConfig.max_session_seconds),
        idle_seconds=_seconds("idle_seconds", LiveVoiceConfig.idle_seconds),
    )
