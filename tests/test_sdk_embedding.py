import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot import Nanobot, Tool, ToolPolicy
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionPolicy


class LabelTool(Tool):
    description = "Record a test label."
    parameters = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }

    def __init__(self, name):
        self._name = name
        self.labels = []

    @property
    def name(self):
        return self._name

    async def execute(self, **kwargs):
        self.labels.append(kwargs["label"])
        return kwargs["label"]


@pytest.fixture
def embedded(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"agents": {"defaults": {
        "workspace": str(tmp_path / "workspace"), "model": "sdk-test",
    }}}), encoding="utf-8")
    provider = MagicMock()
    provider.get_default_model.return_value = "sdk-test"
    provider.generation = GenerationSettings()
    provider.estimate_prompt_tokens.return_value = (30, "mock")
    provider.aclose = AsyncMock()
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda *_args: provider)
    return config, provider


def scripted(provider, replies):
    response = AsyncMock(side_effect=replies)
    provider.chat_stream_with_retry = provider.chat_with_retry = response
    return response


def tool_call(tool, call_id):
    return LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id=call_id, name=tool.name, arguments={"label": call_id}),
    ])


@pytest.mark.asyncio
async def test_embedding_registers_tools_and_resumes_a_session(embedded):
    config, provider = embedded
    primary, extra = LabelTool("primary_label"), LabelTool("extra_label")
    response = scripted(provider, [
        tool_call(primary, "label-1"), LLMResponse(content="First reply"),
        LLMResponse(content="Direct reply"),
        tool_call(extra, "label-2"), LLMResponse(content="Second label"),
        LLMResponse(content="Back to defaults"),
    ])
    instructions = "Keep replies brief and use labels only on request."
    policy = ToolPolicy(defaults=(primary.name,), add=(extra.name,), remove=(extra.name,))
    async with Nanobot.from_config(
        config, instructions=instructions, tools=(primary, extra), tool_policy=policy,
    ) as bot:
        run = await bot.run_streamed("Record the primary label", session_key="sdk:example")
        events = [event async for event in run.stream_events()]
        assert events[0].type == "run.started" and events[-1].type == "run.completed"
        assert [(event.type, event.tool_call_id) for event in events if event.type.startswith("tool.")] == [
            ("tool.started", "label-1"), ("tool.completed", "label-1"),
        ]
        assert (await run.wait()).status == run.status == "completed"

        direct = await bot.run(
            "Continue without tools", session_key="sdk:example", tool_policy=ToolPolicy(defaults=()),
        )
        assert direct.content == "Direct reply"
        await bot.run(
            "Use the extra label", session_key="sdk:example",
            tool_policy=ToolPolicy(add=(extra.name,), remove=(primary.name,)),
        )
        await bot.run("Use the role defaults again", session_key="sdk:example")

        snapshot = await bot.sessions.get("sdk:example")
        assert snapshot is not None
        assert any(msg["content"] == "First reply" for msg in snapshot.messages)

    assert primary.labels == ["label-1"] and extra.labels == ["label-2"]
    requests = [call.kwargs for call in response.await_args_list]
    assert [[t["function"]["name"] for t in (r.get("tools") or [])] for r in requests] == [
        [primary.name], [primary.name], [], [extra.name], [extra.name], [primary.name],
    ]
    assert any(row["content"] == "First reply" for row in requests[2]["messages"])
    for request in requests:
        system = request["messages"][0]
        assert system["role"] == "system" and system["content"].count(instructions) == 1


@pytest.mark.parametrize("session_denial", [False, True])
@pytest.mark.asyncio
async def test_tool_selection_and_session_permissions_both_restrict_execution(
    embedded, monkeypatch, session_denial,
):
    config, provider = embedded
    tool = LabelTool("host_label")
    response = scripted(provider, [tool_call(tool, "excluded"), LLMResponse(content="Continued")])
    connect = AsyncMock()
    monkeypatch.setattr("nanobot.agent.tools.mcp.MCPProvider.connect", connect)
    async with Nanobot.from_config(config, tools=(tool,)) as bot:
        key = "sdk:restricted"
        if session_denial:
            await bot.sessions.ingest(key, [{"role": "user", "content": "A prior turn"}])
            await bot._loop.sessions.state.set_policy(key, SessionPolicy(disabled_tools=frozenset({tool.name})))
        policy = ToolPolicy(defaults=(tool.name,), remove=() if session_denial else (tool.name,))
        events = [event async for event in bot.stream("Answer briefly", session_key=key, tool_policy=policy)]
    assert not tool.labels
    assert all(not call.kwargs.get("tools") for call in response.await_args_list)
    assert any(event.type == "tool.failed" and event.tool_call_id == "excluded" for event in events)
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_explicit_mcp_tool_is_selected_after_discovery(embedded, monkeypatch):
    config, provider = embedded
    tool = LabelTool("mcp_demo_label")
    response = scripted(provider, [tool_call(tool, "mcp-call"), LLMResponse(content="Ready")])

    async def discover(mcp):
        mcp._registry.register(tool)

    monkeypatch.setattr("nanobot.agent.tools.mcp.MCPProvider.connect", discover)
    async with Nanobot.from_config(config, tool_policy=ToolPolicy(defaults=(tool.name,))) as bot:
        result = await bot.run("Record the MCP label")
    assert result.tools_used == [tool.name] and tool.labels == ["mcp-call"]
    assert [schema["function"]["name"] for schema in response.await_args_list[0].kwargs["tools"]] == [tool.name]


@pytest.mark.asyncio
async def test_external_tool_names_must_be_registered_and_cannot_replace_a_builtin(embedded):
    config, provider = embedded
    response = scripted(provider, [LLMResponse(content="Should not run")])
    with pytest.raises(ValueError, match="already registered"):
        Nanobot.from_config(config, tools=(LabelTool("read_file"),))
    async with Nanobot.from_config(config) as bot:
        with pytest.raises(ValueError, match="is not registered"):
            await bot.run("Hello", tool_policy=ToolPolicy(defaults=(), add=("missing_label",)))
        with pytest.raises(TypeError, match=r"ToolPolicy\.remove"):
            await bot.run("Hello", tool_policy=ToolPolicy(remove="read_file"))
    response.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_result_reports_model_failure_without_changing_the_event(embedded):
    config, provider = embedded
    scripted(provider, [LLMResponse(content="Mock provider failed", finish_reason="error")])
    async with Nanobot.from_config(config) as bot:
        run = await bot.run_streamed("Check result")
        events = [event async for event in run.stream_events()]
        assert events[-1].type == "run.completed"
        assert events[-1].result is not None and events[-1].result.status == "failed"
        assert (await run.wait()).status == run.status == "failed"


@pytest.mark.asyncio
async def test_cancelling_a_selected_run_leaves_the_other_session_running(embedded):
    config, provider = embedded
    entered = (asyncio.Event(), asyncio.Event())
    release_second = asyncio.Event()
    tool = LabelTool("host_label")
    advertised = []

    async def reply(*_args, tools=None, **_kwargs):
        index = len(advertised)
        advertised.append([tool["function"]["name"] for tool in (tools or [])])
        if index < 2:
            entered[index].set()
        if index == 0:
            await asyncio.Future()
        if index == 1:
            await release_second.wait()
        return LLMResponse(content="Ready")

    scripted(provider, reply)
    async with Nanobot.from_config(config, tools=(tool,), tool_policy=ToolPolicy(defaults=(tool.name,))) as bot:
        async with asyncio.timeout(5):
            first = await bot.run_streamed("Slow run", session_key="sdk:first")
            second = None
            try:
                await entered[0].wait()
                second = await bot.run_streamed(
                    "Independent run", session_key="sdk:second", tool_policy=ToolPolicy(defaults=()),
                )
                await entered[1].wait()
                assert first.status is None and second.status is None
                await first.cancel()
                assert first.status == "cancelled" and second.status is None
                with pytest.raises(asyncio.CancelledError):
                    await first.wait()
                release_second.set()
                assert (await second.wait()).status == "completed"
                assert (await bot.run("Try again", session_key="sdk:first")).content == "Ready"
            finally:
                await first.aclose()
                if second is not None:
                    await second.aclose()
    assert advertised == [[tool.name], [], [tool.name]]


@pytest.mark.asyncio
async def test_a_turn_deadline_surfaces_a_timeout_and_releases_the_session(embedded):
    config, provider = embedded
    stopped = asyncio.Event()

    async def blocked(*_args, **_kwargs):
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    scripted(provider, blocked)
    async with Nanobot.from_config(config) as bot, asyncio.timeout(5):
        run = await bot.run_streamed("Allow a short turn", timeout_s=0.25, session_key="sdk:deadline")
        events = [event async for event in run.stream_events()]
        assert events[-1].type == "run.failed" and events[-1].metadata["exception_type"] == "TimeoutError"
        with pytest.raises(TimeoutError):
            await run.wait()
        assert run.status == "failed" and stopped.is_set()
        scripted(provider, [LLMResponse(content="Ready")])
        assert (await bot.run("Continue", session_key="sdk:deadline")).content == "Ready"


@pytest.mark.asyncio
async def test_standalone_example_delegates_without_blocking_chat(embedded, monkeypatch, capsys, tmp_path):
    from examples.sdk_embedding import chat

    config, provider = embedded
    requests = ["Is this chat ready?", "Count these words", "Start a task", "Check task 1", "Stop task 1", ""]
    replies = iter(requests)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
    worker_started, worker_stopped = asyncio.Event(), asyncio.Event()
    calls = {
        "Count these words": ("word_count", {"text": "hello sdk example"}),
        "Start a task": ("background_task", {"action": "start", "request": "Mock work"}),
        "Check task 1": ("background_task", {"action": "status", "task_id": "1"}),
        "Stop task 1": ("background_task", {"action": "cancel", "task_id": "1"}),
    }

    async def reply(*_args, messages, tools=None, **_kwargs):
        if not tools:
            worker_started.set()
            try:
                await asyncio.Future()
            finally:
                worker_stopped.set()
        last = messages[-1]
        text = last["content"]
        if last["role"] == "tool":
            if "Task 1 queued." in text:
                await worker_started.wait()
            return LLMResponse(content=text)
        for instruction, (name, args) in calls.items():
            if instruction in text:
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id=name + str(requests.index(instruction)), name=name, arguments=args,
                )])
        return LLMResponse(content="Yes")

    scripted(provider, reply)
    async with asyncio.timeout(5):
        await chat(config, tmp_path / "example-workspace")
    output = capsys.readouterr().out
    assert "Agent [completed]: Yes" in output and "Agent [completed]: 3" in output
    assert all(text in output for text in ("Task 1 queued.", "Task 1 is running.", "Task 1 cancelled."))
    assert worker_started.is_set() and worker_stopped.is_set()
