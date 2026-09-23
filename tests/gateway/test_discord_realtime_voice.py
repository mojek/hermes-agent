"""Behavioral tests for Discord's GPT Realtime live voice bridge."""

import asyncio
import base64
import contextvars
import json
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.config import load_config


def test_live_voice_defaults_and_profile_runtime_config(tmp_path, monkeypatch):
    from plugins.platforms.discord.realtime_voice import load_live_voice_config

    hermes_home = tmp_path / "profile-a"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  live_voice:\n"
        "    enabled: true\n"
        "    model: profile-model\n"
        "    voice: profile-voice\n"
        "    max_session_seconds: 321\n"
        "    idle_seconds: 45\n",
        encoding="utf-8",
    )

    defaults = load_config()["discord"]["live_voice"]
    runtime = load_live_voice_config()

    assert defaults == {
        "enabled": True,
        "model": "profile-model",
        "voice": "profile-voice",
        "max_session_seconds": 321,
        "idle_seconds": 45,
    }
    assert runtime.enabled is True
    assert runtime.model == "profile-model"
    assert runtime.voice == "profile-voice"
    assert runtime.max_session_seconds == 321
    assert runtime.idle_seconds == 45


def test_live_voice_registered_defaults_are_fail_closed():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from plugins.platforms.discord.realtime_voice import LiveVoiceConfig

    registered = DEFAULT_CONFIG["discord"]["live_voice"]
    defaults = LiveVoiceConfig()
    assert registered["enabled"] is False
    assert registered["model"] == defaults.model
    assert registered["voice"] == defaults.voice
    assert registered["max_session_seconds"] == defaults.max_session_seconds
    assert registered["idle_seconds"] == defaults.idle_seconds


def test_pcm_conversion_is_exact_across_arbitrary_chunk_boundaries():
    from plugins.platforms.discord.realtime_voice import (
        DiscordToRealtimePCM,
        RealtimeToDiscordPCM,
    )

    discord_pcm = struct.pack(
        "<hhhhhhhh", 1000, 3000, -2000, 0, 32767, 32767, -32768, -32768,
    )
    down = DiscordToRealtimePCM()
    converted = b"".join(
        down.feed(discord_pcm[start:end])
        for start, end in ((0, 1), (1, 6), (6, 9), (9, 15), (15, 16))
    )
    assert converted == struct.pack("<hh", 500, -1)

    up = RealtimeToDiscordPCM()
    realtime_pcm = struct.pack("<hh", 1234, -2345)
    expanded = b"".join(
        up.feed(realtime_pcm[start:end])
        for start, end in ((0, 1), (1, 3), (3, 4))
    )
    assert expanded == struct.pack(
        "<hhhhhhhh", 1234, 1234, 1234, 1234, -2345, -2345, -2345, -2345,
    )


class _FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.close = AsyncMock()

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class _LifecycleWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.incoming = asyncio.Queue()

    async def recv(self):
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)


@pytest.mark.asyncio
async def test_ga_schema_delegation_audio_and_barge_in():
    from plugins.platforms.discord.realtime_voice import LiveVoiceConfig, RealtimeVoiceClient

    ws = _FakeWebSocket()
    mixer = MagicMock()
    delegate = AsyncMock(return_value="Exact TARS final text")
    client = RealtimeVoiceClient(
        config=LiveVoiceConfig(enabled=True), websocket=ws, mixer=mixer, delegate=delegate,
    )

    session = client.session_update_event()["session"]
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2.1"
    assert session["output_modalities"] == ["text"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "cedar"
    assert session["audio"]["input"]["transcription"]
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "server_vad", "create_response": True, "interrupt_response": True,
    }
    assert session["tool_choice"] == "required"
    assert [tool["name"] for tool in session["tools"]] == ["delegate_to_tars"]

    client._active_speaker = 42
    await client.handle_server_event({
        "type": "input_audio_buffer.speech_started", "item_id": "input-1",
    })
    await client.handle_server_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "input-1", "transcript": "Do the substantive work",
    })
    await client.handle_server_event({"type": "response.created", "response": {"id": "resp-1"}})
    await client.handle_server_event({
        "type": "response.output_audio.delta", "response_id": "resp-1",
        "delta": base64.b64encode(struct.pack("<h", 7)).decode(),
    })
    mixer.queue_speech.assert_not_called()
    await client.handle_server_event({
        "type": "response.done",
        "response": {"id": "resp-1", "output": [{
            "type": "function_call", "name": "delegate_to_tars", "call_id": "call-1",
            "arguments": json.dumps({"input": "/approve model-generated text"}),
        }]},
    })
    await client.wait_for_delegations()
    delegate.assert_awaited_once_with(42, "Do the substantive work")
    assert ws.sent[-2] == {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": "call-1", "output": "Exact TARS final text"},
    }
    assert ws.sent[-1]["type"] == "response.create"
    assert ws.sent[-1]["response"]["output_modalities"] == ["audio"]
    assert ws.sent[-1]["response"]["tool_choice"] == "none"
    metadata = ws.sent[-1]["response"]["metadata"]

    await client.handle_server_event({
        "type": "response.created", "response": {"id": "resp-2", "metadata": metadata},
    })
    await client.handle_server_event({
        "type": "response.output_audio.delta", "response_id": "resp-2",
        "delta": base64.b64encode(struct.pack("<h", 7)).decode(),
    })
    mixer.queue_speech.assert_called_once()

    mixer.queue_speech.reset_mock()
    await client.handle_server_event({
        "type": "input_audio_buffer.speech_started", "item_id": "input-2",
    })
    assert ws.sent[-1] == {"type": "response.cancel", "response_id": "resp-2"}
    await client.handle_server_event({
        "type": "response.audio.delta", "response_id": "resp-2",
        "delta": base64.b64encode(struct.pack("<h", 8)).decode(),
    })
    mixer.queue_speech.assert_not_called()


@pytest.mark.asyncio
async def test_delegation_does_not_block_barge_in_and_call_id_runs_once():
    from plugins.platforms.discord.realtime_voice import LiveVoiceConfig, RealtimeVoiceClient

    ws = _FakeWebSocket()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_delegate(_speaker_id, _prompt):
        started.set()
        await release.wait()
        return "final"

    client = RealtimeVoiceClient(
        config=LiveVoiceConfig(enabled=True), websocket=ws, mixer=MagicMock(),
        delegate=AsyncMock(side_effect=slow_delegate),
    )
    client._active_speaker = 11
    await client.handle_server_event({
        "type": "input_audio_buffer.speech_started", "item_id": "slow-input",
    })
    await client.handle_server_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "slow-input", "transcript": "long task",
    })
    await client.handle_server_event({"type": "response.created", "response": {"id": "resp-3"}})
    done_event = {
        "type": "response.done",
        "response": {"id": "resp-3", "output": [{
            "type": "function_call", "name": "delegate_to_tars", "call_id": "call-once",
            "arguments": json.dumps({"input": "long task"}),
        }]},
    }
    dispatch = asyncio.create_task(client.handle_server_event(done_event))
    await started.wait()

    assert dispatch.done()
    await client.handle_server_event({
        "type": "input_audio_buffer.speech_started", "item_id": "interrupting-input",
    })

    release.set()
    await dispatch
    await client.wait_for_delegations()
    await client.handle_server_event(done_event)
    client.delegate.assert_awaited_once_with(11, "long task")
    assert not any(event["type"] == "conversation.item.create" for event in ws.sent)


@pytest.mark.asyncio
async def test_ready_streaming_speaker_policy_timeouts_and_close_once():
    from plugins.platforms.discord.realtime_voice import LiveVoiceConfig, RealtimeVoiceClient

    ws = _LifecycleWebSocket()
    await ws.incoming.put({"type": "session.created"})
    await ws.incoming.put({"type": "session.updated"})
    closed = AsyncMock()
    client = RealtimeVoiceClient(
        config=LiveVoiceConfig(enabled=True, max_session_seconds=30, idle_seconds=10),
        websocket=ws, mixer=MagicMock(), delegate=AsyncMock(), on_closed=closed,
    )

    await client.start()
    assert client.ready is True
    assert ws.sent[0]["type"] == "session.update"
    assert client.feed_discord_pcm(11, struct.pack("<hhhh", 10, 10, 20, 20), now=1.0) is True
    assert client.feed_discord_pcm(22, struct.pack("<hhhh", 30, 30, 40, 40), now=1.1) is False
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert ws.sent[1] == {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(struct.pack("<h", 15)).decode(),
    }

    await client.check_timeouts(now=client.started_at + 31)
    assert client.closed is True
    ws.close.assert_awaited_once()
    closed.assert_awaited_once()
    await client.close()
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_network_close_has_no_retry_or_orphan_tasks():
    from plugins.platforms.discord.realtime_voice import LiveVoiceConfig, RealtimeVoiceClient

    ws = _LifecycleWebSocket()
    await ws.incoming.put({"type": "session.created"})
    await ws.incoming.put({"type": "session.updated"})
    closed = AsyncMock()
    client = RealtimeVoiceClient(
        config=LiveVoiceConfig(enabled=True), websocket=ws, mixer=MagicMock(),
        delegate=AsyncMock(), on_closed=closed,
    )
    await client.start()
    await ws.incoming.put(ConnectionError("Authorization: Bearer secret"))
    await client.wait_closed()

    assert client.closed is True
    assert client.connection_attempts == 1
    assert all(task.done() for task in client.tasks)
    closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_adapter_live_start_fails_closed_and_rejects_duplicate(monkeypatch):
    from agent import secret_scope
    from gateway.config import Platform
    from plugins.platforms.discord import realtime_voice
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._live_voice_sessions = {}
    adapter._live_voice_locks = {}
    adapter._voice_mixers = {77: MagicMock()}
    adapter._voice_clients = {}
    adapter._voice_receivers = {}
    adapter._cancel_voice_timeout = MagicMock()
    adapter.join_voice_channel = AsyncMock(return_value=True)
    adapter.leave_voice_channel = AsyncMock()
    channel = MagicMock()
    channel.guild.id = 77
    delegate = AsyncMock()

    monkeypatch.setattr(
        realtime_voice, "load_live_voice_config",
        lambda: realtime_voice.LiveVoiceConfig(enabled=False),
    )
    assert "disabled" in (await adapter.start_live_voice(
        channel, text_channel_id=99, source={"chat_id": "99"}, delegate=delegate,
    )).lower()
    adapter.join_voice_channel.assert_not_awaited()

    monkeypatch.setattr(
        realtime_voice, "load_live_voice_config",
        lambda: realtime_voice.LiveVoiceConfig(enabled=True),
    )
    monkeypatch.setattr(secret_scope, "get_secret", lambda name, default=None: default)
    assert "openai_api_key" in (await adapter.start_live_voice(
        channel, text_channel_id=99, source={"chat_id": "99"}, delegate=delegate,
    )).lower()
    adapter.join_voice_channel.assert_not_awaited()

    monkeypatch.setattr(secret_scope, "get_secret", lambda name, default=None: "scoped-key")
    assert "input" in (await adapter.start_live_voice(
        channel, text_channel_id=99, source={"chat_id": "99"}, delegate=delegate,
    )).lower()
    adapter.join_voice_channel.assert_awaited_once()
    adapter.leave_voice_channel.assert_awaited_once_with(77)
    adapter.join_voice_channel.reset_mock()
    adapter.leave_voice_channel.reset_mock()
    adapter._cancel_voice_timeout.reset_mock()
    receiver = MagicMock()
    receiver._running = True
    adapter._voice_receivers[77] = receiver

    fake_client = MagicMock()
    fake_client.closed = False
    fake_client.active_speaker = 42
    fake_client.start = AsyncMock()
    client_factory = MagicMock(return_value=fake_client)
    monkeypatch.setattr(realtime_voice, "RealtimeVoiceClient", client_factory)
    assert "ready" in (await adapter.start_live_voice(
        channel, text_channel_id=99, source={"chat_id": "99"}, delegate=delegate,
    )).lower()
    assert "already" in (await adapter.start_live_voice(
        channel, text_channel_id=99, source={"chat_id": "99"}, delegate=delegate,
    )).lower()
    adapter.join_voice_channel.assert_awaited_once()
    fake_client.start.assert_awaited_once()
    adapter._cancel_voice_timeout.assert_called_once_with(77)
    assert await client_factory.call_args.kwargs["delegate"](42, "from voice") == delegate.return_value
    delegate.assert_awaited_once_with(42, "from voice")


@pytest.mark.asyncio
async def test_live_pcm_authorization_is_profile_scoped_and_rechecked(monkeypatch):
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord.adapter import DiscordAdapter

    profile = contextvars.ContextVar("test_live_profile", default="default")
    profile.set("profile-b")
    clock = [0.0]
    allowed = [True]
    monkeypatch.setattr(discord_adapter.time, "monotonic", lambda: clock[0])
    adapter = object.__new__(DiscordAdapter)
    adapter._client = MagicMock()
    adapter._client.get_guild.return_value = MagicMock()
    adapter._is_allowed_user = MagicMock(
        side_effect=lambda *_args, **_kwargs: profile.get() == "profile-b" and allowed[0],
    )
    client = MagicMock()
    callback = adapter._live_pcm_callback(77, client)

    profile.set("default")
    callback(42, b"pcm")
    await asyncio.sleep(0)

    client.feed_discord_pcm.assert_called_once_with(42, b"pcm")

    allowed[0] = False
    clock[0] = 1.1
    callback(42, b"revoked")
    await asyncio.sleep(0)

    assert adapter._is_allowed_user.call_count == 2
    client.feed_discord_pcm.assert_called_once_with(42, b"pcm")


@pytest.mark.asyncio
async def test_close_callback_can_reenter_close_without_deadlock():
    from plugins.platforms.discord.realtime_voice import LiveVoiceConfig, RealtimeVoiceClient

    holder = {}

    async def reentrant_close():
        await holder["client"].close()

    client = RealtimeVoiceClient(
        config=LiveVoiceConfig(enabled=True), websocket=_FakeWebSocket(), mixer=MagicMock(),
        delegate=AsyncMock(), on_closed=reentrant_close,
    )
    holder["client"] = client

    await asyncio.wait_for(client.close(), timeout=2)
    assert client.closed is True
    await client.wait_closed()


@pytest.mark.asyncio
async def test_existing_voice_connection_rebinds_live_text_session(monkeypatch):
    from plugins.platforms.discord import adapter as discord_adapter
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
    adapter = object.__new__(DiscordAdapter)
    adapter._client = MagicMock()
    adapter._voice_locks = {}
    adapter._voice_text_channels = {77: 1}
    adapter._voice_sources = {77: {"chat_id": "1"}}
    adapter._reset_voice_timeout = MagicMock()
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    voice_client.channel.id = 88
    adapter._voice_clients = {77: voice_client}
    channel = MagicMock()
    channel.id = 88
    channel.guild.id = 77

    assert await adapter.join_voice_channel(
        channel, text_channel_id=99, source={"chat_id": "99"},
    ) is True
    assert adapter._voice_text_channels[77] == 99
    assert adapter._voice_sources[77] == {"chat_id": "99"}


@pytest.mark.asyncio
async def test_response_wait_uses_normal_final_delivery_contract():
    from types import SimpleNamespace

    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource

    class _Adapter(BasePlatformAdapter):
        async def connect(self, **kwargs):
            return True

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, **kwargs):
            return SendResult(success=True, message_id="final-1")

        async def get_chat_info(self, chat_id):
            return None

    adapter = _Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter.set_message_handler(AsyncMock(return_value="Exact TARS final text"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="final-1"))
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    adapter._auto_tts_enabled_chats.add("99")
    adapter._synthesize_auto_tts = AsyncMock(return_value=([], None))
    event = MessageEvent(
        text="Do the work",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=Platform.DISCORD, chat_id="99", user_id="11", chat_type="group",
        ),
        metadata={"suppress_auto_tts": True},
    )

    result = await asyncio.wait_for(adapter.handle_message_and_wait(event), timeout=2)

    assert result == "Exact TARS final text"
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "Exact TARS final text"
    adapter._synthesize_auto_tts.assert_not_awaited()

    cancelled = MessageEvent(text="old", message_type=MessageType.VOICE, source=event.source)
    cancelled._gateway_response_future = asyncio.get_running_loop().create_future()
    cancelled._gateway_response_future.cancel()
    successor = MessageEvent(text="new", message_type=MessageType.VOICE, source=event.source)
    adapter._process_message_background = AsyncMock()
    runner = SimpleNamespace(
        _promote_queued_event=MagicMock(return_value=successor),
    )
    setattr(adapter, "gateway_runner", runner)

    adapter._spawn_drain_task(cancelled, "discord:99")
    await asyncio.sleep(0)

    runner._promote_queued_event.assert_called_once_with("discord:99", adapter, None)
    adapter._process_message_background.assert_awaited_once_with(successor, "discord:99")


@pytest.mark.asyncio
async def test_leave_closes_live_client_before_discord_disconnect():
    from gateway.config import Platform
    from plugins.platforms.discord.adapter import DiscordAdapter

    order = []
    live_client = MagicMock()
    live_client.close = AsyncMock(side_effect=lambda: order.append("live-close"))
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    voice_client.is_playing.return_value = False
    voice_client.disconnect = AsyncMock(side_effect=lambda: order.append("discord-disconnect"))
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._voice_locks = {}
    adapter._live_voice_sessions = {77: live_client}
    adapter._closing_live_guilds = set()
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {77: MagicMock()}
    adapter._voice_clients = {77: voice_client}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {77: 99}
    adapter._voice_sources = {77: {"chat_id": "99"}}
    adapter._client = MagicMock()

    await adapter.leave_voice_channel(77)

    assert order == ["live-close", "discord-disconnect"]
    assert 77 not in adapter._live_voice_sessions
    assert 77 not in adapter._voice_clients


@pytest.mark.asyncio
async def test_live_leave_discards_legacy_receiver_flush():
    from gateway.config import Platform
    from plugins.platforms.discord.adapter import DiscordAdapter

    live_client = MagicMock()
    live_client.close = AsyncMock()
    receiver = MagicMock()
    receiver.flush_pending.return_value = [(42, b"already streamed")]
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    voice_client.is_playing.return_value = False
    voice_client.disconnect = AsyncMock()
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._voice_locks = {}
    adapter._live_voice_sessions = {77: live_client}
    adapter._closing_live_guilds = set()
    adapter._voice_receivers = {77: receiver}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_clients = {77: voice_client}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {77: 99}
    adapter._voice_sources = {77: {"chat_id": "99"}}
    adapter._client = MagicMock()
    adapter._is_allowed_user = MagicMock(return_value=True)
    adapter._process_voice_input = AsyncMock()

    await adapter.leave_voice_channel(77)

    adapter._process_voice_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_disconnect_closes_live_session_once():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = MagicMock(spec=DiscordAdapter)
    adapter.leave_voice_channel = AsyncMock()

    await DiscordAdapter._handle_voice_disconnect(adapter, 77)

    adapter.leave_voice_channel.assert_awaited_once_with(77)


@pytest.mark.asyncio
async def test_gateway_shutdown_closes_live_session_without_voice_client():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = MagicMock(spec=DiscordAdapter)
    adapter._voice_clients = {}
    adapter._live_voice_sessions = {77: MagicMock()}
    adapter._cancel_liveness_task = AsyncMock()
    adapter.leave_voice_channel = AsyncMock()
    adapter._cancel_bot_task = AsyncMock()
    adapter._client = None
    adapter._post_connect_task = None
    adapter._missed_message_backfill_task = None
    adapter._ready_event = MagicMock()
    adapter._release_platform_lock = MagicMock()

    await DiscordAdapter.disconnect(adapter)

    adapter.leave_voice_channel.assert_awaited_once_with(77)


@pytest.mark.asyncio
async def test_live_session_does_not_also_run_legacy_stt(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    receiver = MagicMock()
    receiver._running = True
    receiver.check_silence.return_value = [(42, b"completed utterance")]
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_receivers = {77: receiver}
    adapter._voice_clients = {}
    adapter._live_voice_sessions = {77: MagicMock(closed=False)}
    adapter._client = MagicMock()
    adapter._client.get_guild.return_value = MagicMock()
    adapter._is_allowed_user = MagicMock(return_value=True)
    adapter._reset_voice_timeout = MagicMock()
    adapter._process_voice_input = AsyncMock()

    async def stop_after_tick(_delay):
        receiver._running = False

    monkeypatch.setattr(asyncio, "sleep", stop_after_tick)

    await adapter._voice_listen_loop(77)

    adapter._process_voice_input.assert_not_awaited()
