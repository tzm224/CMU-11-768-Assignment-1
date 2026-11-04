"""The Part 1 coding agent: fix a software issue and submit a git patch.
第 1 部分的编程智能体：修复软件问题并提交 git 补丁。
"""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch.
    修复软件问题并提交 git 补丁的智能体。
    """

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        # TODO(1.1.b)：构建系统提示词和 task_prompt，使其可供
        # `Agent.build_prompt` 方法使用。
        self.system_prompt: str = (
            "You are an autonomous coding agent working inside a disposable sandbox.\n"
            "Use the `execute` tool to run shell commands: reproduce the problem, locate its cause, make a minimal fix, and verify it. Prefer small output; read large files with `head`, `tail`, or `sed -n` ranges.\n"
            "<system_information>\n"
            f"{json.dumps({'machine': environment.machine, 'release': environment.release, 'system': environment.system, 'version': environment.version}, indent=2)}\n"
            "</system_information>\n"
        )

        self.task_prompt: str = (
            f"Work in {getattr(environment, 'cwd', '/testbed')}. Fix the issue described "
            f"below so the project's tests pass.\n\n{self.task}"
        )

        if self.skills:
            skills_catalog = "\n\n".join(
                skill["metadata"] for skill in self.skills.values()
            )
            self.system_prompt += (
                "\n\nAvailable skills. Use `invoke_skill` with a skill name to load "
                "its full instructions when needed:\n"
                f"{skills_catalog}"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute coding, submission, and skill calls and return their observations.
        执行编程、提交和技能调用，并返回对应的观察结果。
        """
        observations: list[dict[str, str]] = []

        for tool_call in tool_calls:
            call_id = str(tool_call.get("id", ""))
            function = tool_call.get("function")
            if not isinstance(function, dict):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "<error>malformed tool call</error>",
                    }
                )
                continue

            call_name = function.get("name", "")
            raw_arguments = function.get("arguments", "")
            try:
                if not isinstance(raw_arguments, str):
                    raise ValueError("arguments must be a JSON string")
                args = json.loads(raw_arguments)
                if not isinstance(args, dict):
                    raise ValueError("arguments must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"<error>malformed arguments: {exc}</error>",
                    }
                )
                continue

            if call_name == EXECUTE_TOOL["function"]["name"]:
                command = args.get("command")
                if not isinstance(command, (str, list)) or (
                    isinstance(command, list)
                    and not all(isinstance(item, str) for item in command)
                ):
                    content = (
                        "<error>execute requires a 'command' string or argv list"
                        "</error>"
                    )
                else:
                    options = {
                        key: args[key]
                        for key in ("shell", "cwd", "timeout", "env")
                        if key in args
                    }
                    result = self.env.execute(command=command, **options)
                    content = format_tool_output(result)

            elif call_name == SEND_MESSAGE_TOOL["function"]["name"]:
                summary = args.get("summary")
                if not isinstance(summary, str):
                    content = "<error>send_message requires a string 'summary'</error>"
                else:
                    content = f"<message>{summary}</message>"
                    self.finished = True

            elif call_name == "invoke_skill" and self.skills:
                skill_name = args.get("name")
                if not isinstance(skill_name, str):
                    content = "<error>invoke_skill requires a string 'name'</error>"
                elif skill_name not in self.skills:
                    content = f"<error>unknown skill: {skill_name}</error>"
                else:
                    content = self.skills[skill_name]["content"]

            else:
                content = f"<error>unknown tool: {call_name}</error>"

            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )

        return observations