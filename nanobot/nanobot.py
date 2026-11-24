from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any

from nanobot.agent.hook import AgentHook, SDKCaptureHook
from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.image_generation import register_image_generation_tool
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config
from nanobot.providers.base import LLMUsage
from nanobot.sdk.clients import MemoryClient, RuntimeClient, SessionClient
from nanobot.sdk.runtime import (
    build_process_direct_kwargs,
    ensure_single_model_selector,
    scoped_tool_registry,
    select_tool_names,
)
from nanobot.sdk.streaming import RunStream, SDKStreamEmitter, SDKStreamingHook
from nanobot.sdk.types import (
    STREAM_EVENT_REASONING_COMPLETED,
    STREAM_EVENT_REASONING_DELTA,
    STREAM_EVENT_RUN_COMPLETED,
    STREAM_EVENT_RUN_FAILED,
    STREAM_EVENT_RUN_STARTED,
    STREAM_EVENT_TEXT_COMPLETED,
    STREAM_EVENT_TEXT_DELTA,
    STREAM_EVENT_TOOL_COMPLETED,
    STREAM_EVENT_TOOL_FAILED,
    STREAM_EVENT_TOOL_STARTED,
    STREAM_EVENT_TYPES,
    RunResult,
    SessionInfo,
    SessionSnapshot,
    StreamEvent,
    StreamEventType,
    ToolPolicy,
    result_from_response,
)
from nanobot.utils.llm_runtime import LLMRuntime

__all__ = [
    "Nanobot",
    "LLMUsage",
    "RunResult",
    "RunStream",
    "SessionInfo",
    "SessionSnapshot",
    "STREAM_EVENT_REASONING_COMPLETED",
    "STREAM_EVENT_REASONING_DELTA",
    "STREAM_EVENT_RUN_COMPLETED",
    "STREAM_EVENT_RUN_FAILED",
    "STREAM_EVENT_RUN_STARTED",
    "STREAM_EVENT_TEXT_COMPLETED",
    "STREAM_EVENT_TEXT_DELTA",
    "STREAM_EVENT_TOOL_COMPLETED",
    "STREAM_EVENT_TOOL_FAILED",
    "STREAM_EVENT_TOOL_STARTED",
    "STREAM_EVENT_TYPES",
    "StreamEvent",
    "StreamEventType",
]


class Nanobot:
    """Programmatic facade for running the nanobot agent.

    Usage::

        bot = Nanobot.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)
    """

    def __init__(
        self,
        loop: AgentLoop,
        *,
        config: Config | None = None,
        mcp_provider: MCPProvider | None = None,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        self._loop = loop
        self._config = config
        self._mcp_provider = mcp_provider
        self._tool_policy = tool_policy
        self.sessions = SessionClient(loop)
        self.memory = MemoryClient(loop)
        self.runtime = RuntimeClient(loop)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
        model: str | None = None,
        model_preset: str | None = None,
        instructions: str | None = None,
        tools: Iterable[Tool] = (),
        tool_policy: ToolPolicy | None = None,
    ) -> Nanobot:
        """Create a Nanobot instance from a config file.

        Args:
            config_path: Path to ``config.json``. Defaults to the active instance's config.
            workspace: Override the workspace directory from config.
            model: Override the instance default model.
            model_preset: Override the instance default model preset.
            instructions: Trusted host instructions added to the system prompt.
            tools: Trusted tool implementations to register for this instance.
            tool_policy: Default selection from the configured and registered tools.
        """
        from nanobot.config.loader import load_config, resolve_config_env_vars

        ensure_single_model_selector(model=model, model_preset=model_preset)
        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(
            load_config(resolved),
            config_path=resolved,
        )
        if workspace is not None:
            config.agents.defaults.workspace = str(
                Path(workspace).expanduser().resolve()
            )
        if model is not None:
            config.agents.defaults.model_preset = None
            config.agents.defaults.model = model
            config.agents.defaults.provider = "auto"
        elif model_preset is not None:
            config.agents.defaults.model_preset = model_preset

        registry = ToolRegistry()
        register_image_generation_tool(config, registry, workspace=config.workspace_path)
        mcp_provider = MCPProvider.from_config(config, registry)
        from nanobot.goal.assembly import install_goal

        loop = AgentLoop.from_config(
            config,
            hook_factories=[create_file_edit_activity_hook],
            tool_registry=registry,
            builtin_assembly=install_goal,
            instructions=instructions,
        )
        for tool in tools:
            name = tool.name
            if not name or name.startswith("mcp_"):
                raise ValueError(f"Invalid host tool name {name!r}: use a non-empty name outside mcp_")
            if registry.has(name):
                raise ValueError(f"Tool {name!r} is already registered")
            registry.register(tool)
        return cls(loop, config=config, mcp_provider=mcp_provider, tool_policy=tool_policy)

    def _resolve_override(self, model: str | None, model_preset: str | None) -> LLMRuntime | None:
        return self._loop.runtime_resolver.resolve_override(
            model=model, model_preset=model_preset, config=self._config,
        )

    async def _execute_turn(
        self,
        message: str,
        kwargs: dict[str, Any],
        runtime: LLMRuntime | None,
        hooks: list[AgentHook],
        tool_policy: ToolPolicy | None,
        timeout_s: float | None,
    ) -> RunResult:
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be finite and greater than zero")
        capture = SDKCaptureHook()
        per_run_hooks = [capture, *hooks]
        if runtime is not None:
            kwargs["runtime"] = runtime
        policies = tuple(policy for policy in (self._tool_policy, tool_policy) if policy is not None)
        names = (
            select_tool_names((), *policies)
            if any(policy.defaults is not None for policy in policies) else None
        )
        async with asyncio.timeout(timeout_s):
            if self._mcp_provider is not None and (
                names is None or any(name.startswith("mcp_") for name in names)
            ):
                await self._mcp_provider.connect()
            if policies:
                registry = self._loop.tools
                if names is None:
                    names = select_tool_names(registry.tool_names, *policies)
                kwargs["tools"] = scoped_tool_registry(registry, names)
            response = await self._loop.process_direct(message, **kwargs, hooks=per_run_hooks)
        return result_from_response(response, capture)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        attributes: Mapping[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
        tool_policy: ToolPolicy | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            channel: Logical channel label for runtime context.
            chat_id: Logical chat identifier for runtime context.
            sender_id: Logical sender identifier for runtime context.
            media: Optional local media paths attached to the message.
            ephemeral: If true, do not persist the turn or compact session history.
            attributes: Optional caller-owned request data exposed to context
                providers and turn-hook factories. Attributes are kept separate
                from nanobot's trusted internal message metadata.
            hooks: Optional lifecycle hooks for this run.
            model: Override the model for this run only.
            model_preset: Override the model preset for this run only.
            tool_policy: Override the inherited tool selection for this run.
            timeout_s: Optional wall-clock deadline in seconds for the turn.
        """
        runtime = self._resolve_override(model, model_preset)
        kwargs = build_process_direct_kwargs(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            media=media,
            ephemeral=ephemeral,
            attributes=attributes,
        )
        return await self._execute_turn(message, kwargs, runtime, hooks or [], tool_policy, timeout_s)

    async def run_streamed(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        attributes: Mapping[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
        tool_policy: ToolPolicy | None = None,
        timeout_s: float | None = None,
    ) -> RunStream:
        """Start a streamed run and return a handle for events and final result."""
        override_runtime = self._resolve_override(model, model_preset)
        queue: asyncio.Queue[StreamEvent | object] = asyncio.Queue(maxsize=256)
        emitter = SDKStreamEmitter(queue)
        stream_hooks = [SDKStreamingHook(emitter), *(hooks or [])]
        run_started = False

        async def _emit_run_started(runtime: LLMRuntime | None = None) -> None:
            nonlocal run_started
            if run_started:
                return
            if runtime is None:
                runtime = override_runtime
            metadata: dict[str, Any] = {
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "sender_id": sender_id,
            }
            if runtime is not None:
                metadata.update({
                    "model": runtime.model,
                    "model_preset": runtime.model_preset,
                })
            await emitter.emit(StreamEvent(
                type=STREAM_EVENT_RUN_STARTED,
                metadata=metadata,
            ))
            run_started = True

        async def _on_stream(delta: str) -> None:
            await emitter.text_delta(delta)

        async def _on_stream_end(*_args: Any, resuming: bool = False, **_kwargs: Any) -> None:
            await emitter.text_completed(resuming=resuming)

        async def _run() -> RunResult:
            kwargs = build_process_direct_kwargs(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                sender_id=sender_id,
                media=media,
                ephemeral=ephemeral,
                attributes=attributes,
                on_stream=_on_stream,
                on_stream_end=_on_stream_end,
            )
            kwargs["on_runtime_admitted"] = _emit_run_started
            try:
                result = await self._execute_turn(
                    message, kwargs, override_runtime, stream_hooks, tool_policy, timeout_s,
                )
                await _emit_run_started()
                await emitter.text_completed(resuming=False, force=False)
                await emitter.emit(StreamEvent(
                    type=STREAM_EVENT_RUN_COMPLETED,
                    content=result.content,
                    result=result,
                    usage=result.usage,
                    metadata=dict(result.metadata),
                ))
                return result
            except Exception as exc:
                await _emit_run_started()
                await emitter.emit(StreamEvent(
                    type=STREAM_EVENT_RUN_FAILED,
                    error=str(exc),
                    metadata={"exception_type": type(exc).__name__},
                ))
                raise
            finally:
                await emitter.close()

        task = asyncio.create_task(_run())
        return RunStream(task, queue)

    async def stream(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        attributes: Mapping[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
        tool_policy: ToolPolicy | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events for one agent turn."""
        run = await self.run_streamed(
            message,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            media=media,
            ephemeral=ephemeral,
            attributes=attributes,
            hooks=hooks,
            model=model,
            model_preset=model_preset,
            tool_policy=tool_policy,
            timeout_s=timeout_s,
        )
        try:
            async for event in run.stream_events():
                yield event
            await run.wait()
        finally:
            if not run.done:
                await run.aclose()

    async def aclose(self) -> None:
        """Release resources held by this instance."""
        try:
            await self._loop.aclose()
        finally:
            if self._mcp_provider is not None:
                await self._mcp_provider.aclose()

    async def __aenter__(self) -> Nanobot:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
