import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _event(platform=Platform.TELEGRAM, *, chat_type="dm", chat_id="67890"):
    return MessageEvent(
        text="/retitle",
        source=SessionSource(
            platform=platform,
            user_id="12345",
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id="42" if platform == Platform.TELEGRAM else None,
        ),
    )


def _runner(db, history):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=db)
    runner._async_session_store = SimpleNamespace(
        _store=MagicMock(),
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1", session_key="current-key")
        ),
        load_transcript=AsyncMock(return_value=history),
    )
    runner.session_store = runner._async_session_store._store
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=lambda function, *args: function(*args)
    )
    runner._rename_telegram_topic_for_session_title = AsyncMock(return_value=True)
    runner._rename_current_telegram_topic = AsyncMock(return_value=True)
    return runner


@pytest.mark.asyncio
async def test_retitle_persists_then_renames_telegram_once_with_exact_stored_title():
    db = MagicMock()
    history = [{"role": "user", "content": "Fix Telegram topic titles"}]
    runner = _runner(db, history)
    event = _event()

    with patch(
        "agent.session_retitle.retitle_session", return_value="Persisted exact title"
    ) as retitle:
        reply = await runner._handle_retitle_command(event)

    retitle.assert_called_once_with(db, "session-1", history)
    runner._rename_current_telegram_topic.assert_awaited_once_with(
        event.source, "Persisted exact title"
    )
    assert "Persisted exact title" in reply


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_type", "chat_id"),
    [("dm", "67890"), ("group", "-10067890")],
)
async def test_retitle_renames_current_unmanaged_telegram_topic_with_delivery_adapter(
    chat_type, chat_id,
):
    class Adapter:
        def __init__(self):
            self.rename_dm_topic = AsyncMock()

    db = MagicMock()
    db.is_telegram_topic_mode_enabled.return_value = False
    runner = _runner(db, [{"role": "user", "content": "Rename this topic"}])
    del runner._rename_telegram_topic_for_session_title
    del runner._rename_current_telegram_topic
    primary_adapter = Adapter()
    delivery_adapter = Adapter()
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {"routed": {Platform.TELEGRAM: delivery_adapter}}
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={"disable_topic_auto_rename": True},
            )
        }
    )
    event = _event(chat_type=chat_type, chat_id=chat_id)
    event.source.profile = "routed"
    event.source._transport_adapter_ref = weakref.ref(delivery_adapter)

    with patch("agent.session_retitle.retitle_session", return_value="Explicit topic title"):
        reply = await runner._handle_retitle_command(event)

    delivery_adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id=chat_id,
        thread_id="42",
        name="Explicit topic title",
    )
    primary_adapter.rename_dm_topic.assert_not_awaited()
    assert reply == "Session retitled to: Explicit topic title"


@pytest.mark.asyncio
async def test_retitle_never_renames_non_telegram_and_reports_known_telegram_failure():
    history = [{"role": "user", "content": "Fix gateway titles"}]
    discord_runner = _runner(MagicMock(), history)

    with patch("agent.session_retitle.retitle_session", return_value="Gateway titles"):
        reply = await discord_runner._handle_retitle_command(_event(Platform.DISCORD))

    assert "Gateway titles" in reply
    discord_runner._rename_current_telegram_topic.assert_not_awaited()

    telegram_runner = _runner(MagicMock(), history)
    telegram_runner._rename_current_telegram_topic.return_value = False
    with patch("agent.session_retitle.retitle_session", return_value="Stored despite rename"):
        partial = await telegram_runner._handle_retitle_command(_event())

    telegram_runner._rename_current_telegram_topic.assert_awaited_once()
    assert "Stored despite rename" in partial
    assert "topic" in partial.lower()
    assert "could not" in partial.lower()

    telegram_runner._rename_current_telegram_topic.side_effect = RuntimeError("Telegram unavailable")
    with patch("agent.session_retitle.retitle_session", return_value="Still stored"):
        partial = await telegram_runner._handle_retitle_command(_event())

    assert "Still stored" in partial
    assert "could not" in partial.lower()


def test_retitle_is_a_central_idle_gateway_command():
    from gateway.run import GatewayRunner
    from hermes_cli.commands import resolve_command

    command = resolve_command("retitle")
    assert command is not None
    assert command.gateway_only is True
    assert command.busy_policy == "reject"
    assert "retitle" in GatewayRunner._IDLE_COMMANDS
