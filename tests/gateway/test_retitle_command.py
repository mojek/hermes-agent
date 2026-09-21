from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _event(platform=Platform.TELEGRAM):
    return MessageEvent(
        text="/retitle",
        source=SessionSource(
            platform=platform,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
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
    runner._rename_telegram_topic_for_session_title.assert_awaited_once_with(
        event.source, "session-1", "Persisted exact title"
    )
    assert "Persisted exact title" in reply


@pytest.mark.asyncio
async def test_retitle_never_renames_non_telegram_and_reports_known_telegram_failure():
    history = [{"role": "user", "content": "Fix gateway titles"}]
    discord_runner = _runner(MagicMock(), history)

    with patch("agent.session_retitle.retitle_session", return_value="Gateway titles"):
        reply = await discord_runner._handle_retitle_command(_event(Platform.DISCORD))

    assert "Gateway titles" in reply
    discord_runner._rename_telegram_topic_for_session_title.assert_not_awaited()

    telegram_runner = _runner(MagicMock(), history)
    telegram_runner._rename_telegram_topic_for_session_title.return_value = False
    with patch("agent.session_retitle.retitle_session", return_value="Stored despite rename"):
        partial = await telegram_runner._handle_retitle_command(_event())

    telegram_runner._rename_telegram_topic_for_session_title.assert_awaited_once()
    assert "Stored despite rename" in partial
    assert "topic" in partial.lower()
    assert "could not" in partial.lower()

    telegram_runner._rename_telegram_topic_for_session_title.side_effect = RuntimeError("Telegram unavailable")
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
