from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.sdk.types import ToolPolicy
from nanobot.session.manager import SessionPolicy


def select_tool_names(defaults: Iterable[str], *policies: ToolPolicy) -> set[str]:
    names = set(defaults)
    for policy in policies:
        if policy.defaults is not None:
            names = set(policy.defaults)
        names.update(policy.add)
        names.difference_update(policy.remove)
    return names


def scoped_tool_registry(registry: ToolRegistry, names: set[str]) -> ToolRegistry:
    scoped = ToolRegistry()
    for name in sorted(names):
        tool = registry.get(name)
        if tool is None:
            raise ValueError(f"Tool {name!r} is not registered for this SDK instance")
        scoped.register(tool)
    return scoped


def ensure_single_model_selector(
    *,
    model: str | None,
    model_preset: str | None,
) -> None:
    if model is not None and model_preset is not None:
        raise ValueError("model and model_preset are mutually exclusive")


def build_process_direct_kwargs(
    *,
    session_key: str,
    channel: str,
    chat_id: str,
    sender_id: str,
    media: list[str] | None,
    ephemeral: bool,
    attributes: Mapping[str, Any] | None = None,
    on_stream: Any | None = None,
    on_stream_end: Any | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"session_key": session_key}
    if channel != "cli":
        kwargs["channel"] = channel
    if chat_id != "direct":
        kwargs["chat_id"] = chat_id
    if sender_id != "user":
        kwargs["sender_id"] = sender_id
    if media is not None:
        kwargs["media"] = media
    if ephemeral:
        kwargs["ephemeral"] = True
        kwargs["_run_extra_hooks_for_ephemeral"] = True
        kwargs["_session_policy"] = SessionPolicy(persist=False, log_content=False)
    if attributes is not None:
        kwargs["attributes"] = dict(attributes)
    if on_stream is not None:
        kwargs["on_stream"] = on_stream
    if on_stream_end is not None:
        kwargs["on_stream_end"] = on_stream_end
    return kwargs
