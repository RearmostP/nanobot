"""Embed nanobot with two host tools and independent background sessions."""

import argparse
import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from nanobot import Nanobot, RunResult, Tool, ToolPolicy


class WordCount(Tool):
    @property
    def name(self) -> str:
        return "word_count"

    @property
    def description(self) -> str:
        return "Count the words in a short piece of text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return str(len(kwargs["text"].split()))


class BackgroundTasks(Tool):
    def __init__(self, worker: Callable[[str, str], Coroutine[Any, Any, RunResult]]) -> None:
        self._worker = worker
        self._tasks: dict[str, asyncio.Task[RunResult]] = {}

    @property
    def name(self) -> str:
        return "background_task"

    @property
    def description(self) -> str:
        return "Start background work, check a task by ID, or cancel it."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "status", "cancel"]},
                "request": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if kwargs["action"] == "start":
            request = kwargs.get("request", "").strip()
            if not request:
                return self.error("Provide a request for the new task.")
            task_id = str(len(self._tasks) + 1)
            self._tasks[task_id] = asyncio.create_task(self._worker(request, f"embed:task:{task_id}"))
            return f"Task {task_id} queued."

        task_id = kwargs.get("task_id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self.error(f"No task with ID {task_id!r}.")
        if kwargs["action"] == "cancel" and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if task.cancelled():
            return f"Task {task_id} cancelled."
        if not task.done():
            return f"Task {task_id} is running."
        if error := task.exception():
            return f"Task {task_id} failed: {type(error).__name__}."
        result = task.result()
        return f"Task {task_id} {result.status}: {result.content}"

    async def aclose(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)


async def chat(config: Path, workspace: Path) -> None:
    async def worker(request: str, session_key: str) -> RunResult:
        return await bot.run(
            request, session_key=session_key, tool_policy=ToolPolicy(defaults=()), timeout_s=60,
        )

    counter, tasks = WordCount(), BackgroundTasks(worker)
    async with Nanobot.from_config(
        config,
        workspace=workspace,
        instructions=(
            "Answer short questions directly. Use word_count when asked to count words. "
            "When background_task is available, use it for explicit requests to run, "
            "check or stop background work, then report what the tool actually returned."
        ),
        tools=(counter, tasks),
        tool_policy=ToolPolicy(defaults=(counter.name, tasks.name)),
    ) as bot:
        try:
            while request := await asyncio.to_thread(input, "You (empty to quit): "):
                run = await bot.run_streamed(request, session_key="embed:chat", timeout_s=30)
                async for event in run.stream_events():
                    if event.type in {"tool.completed", "tool.failed"}:
                        print(f"  [{event.type}] {event.name} ({event.tool_call_id})")
                result = await run.wait()
                print(f"Agent [{result.status}]: {result.content}")
        except EOFError:
            pass
        finally:
            await tasks.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Config for a separate SDK instance")
    parser.add_argument("workspace", type=Path, help="Workspace for the example")
    options = parser.parse_args()
    asyncio.run(chat(options.config, options.workspace))
