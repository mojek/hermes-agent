"""The ``chat -q`` echo prints the user's query as written, under Hermes's own styled label.

The non-quiet one-shot path interpolated the query into a Rich markup string, so a query holding a
closing tag (``[/nirvana, /noski]``) crashed the run with ``MarkupError`` before the turn started,
and bracketed text was silently restyled or swallowed. Regression for MOJ-168.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.color import Color
from rich.console import Console
from rich.style import Style
from rich.text import Text

import cli
from agent.i18n import t

QUERIES = [
    "[/nirvana, /noski]",
    "[bold]not bold[/bold]",
    "[red]unclosed",
    r"\[escaped] stays escaped",
    "see [link=https://example.invalid]docs[/link]",
    "fix the *bug* in __init__.py and `run` it",
    "# heading\n- [ ] todo [x] done",
    "ship it :thumbs_up:",
]


def _run_non_quiet(monkeypatch, query):
    """Drive the real non-quiet ``-q`` tail with a real Rich console; ``chat`` only records."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)

    out = io.StringIO()
    console = Console(file=out, force_terminal=True, color_system="truecolor", width=200)
    chats = []

    def chat(message, images=None):
        chats.append(message)
        return "done"

    stub = SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=console,
        _show_security_advisories=lambda: None,
        chat=chat,
        _print_exit_summary=lambda **k: None,
        _last_turn_result={"final_response": "done", "completed": True},
    )
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(stub, query, None, False, True)
    return exc.value.code, Text.from_ansi(out.getvalue()), chats


def _style_at(text: Text, index: int) -> Style:
    style = Style()
    for span in text.spans:
        if span.start <= index < span.end:
            style += span.style if isinstance(span.style, Style) else Style.parse(span.style)
    return style


@pytest.mark.parametrize("query", QUERIES)
def test_query_echo_is_literal_and_chat_gets_the_original(monkeypatch, query):
    code, rendered, chats = _run_non_quiet(monkeypatch, query)

    label = t("cli.single_query.query_label")
    assert rendered.plain.rstrip("\n") == f"{label} {query}"
    assert chats == [query]
    assert code == 0

    label_style = _style_at(rendered, 0)
    assert label_style.bold and label_style.color is not None
    assert label_style.color.number == Color.parse("blue").number
    query_style = _style_at(rendered, len(label) + 1)
    assert not query_style.bold and query_style.color is None
