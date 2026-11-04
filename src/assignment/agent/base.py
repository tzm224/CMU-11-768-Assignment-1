"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.

两个智能体共享的、与领域无关的 ReAct 循环。

第 1 部分在这里完成通用循环；本包中的两个子类只提供各自的工具和工具执行器。
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
# TODO(第 2 部分)：编写指令，让模型为软件智能体生成简洁的工作记忆。提示词应
# 保留具体进展、失败情况、测试结果、约束和下一步，而不是复制原始输出。
COMPACTION_SYSTEM_PROMPT = (
    "Rewrite the task and action history above as a concise working memory that "
    "a fresh run could continue from, given only this summary and the most "
    "recent step still shown. Be factual and concrete. Keep, where present: the "
    "objective and constraints; files read or edited and the commands that "
    "matter; changes made and their concrete results; approaches that failed "
    "and why; tests run and their outcomes; open blockers; and the next action. "
    "Do not copy commands or tool output verbatim; condense them to the facts "
    "needed later. Omit reasoning that no longer matters. Return only the "
    "summary text."
)


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget.
    当智能体耗尽模型调用额度时抛出。
    """


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation.
    将终端结果格式化为紧凑且带标签的模型观察结果。
    """

    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            # 为省略提示留出空间，使格式化后的值本身（而不仅是保留的源文本片段）
            # 也低于长度上限。
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer.
    在不使用提供商专用分词器的情况下估算提示词 token 数。
    """

    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools.
    具有可插拔工具的 ReAct 智能体基类。
    """

    def __init__(
        self,
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
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        # None 阈值会关闭压缩。此时另外两个设置描述的是永远不会发生的压缩，
        # 因此使用默认值，避免将 None 留给后续代码处理而出错。
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        # 每个智能体提供自己的初始消息：长期有效的指令，以及启动运行的任务说明。
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # TODO(1.1.a): Add machinery to maintain agent state as it takes actions
        # and observes the results.
        # TODO(1.1.a)：添加相应机制，在智能体执行操作并观察结果时维护智能体状态。
        self.messages: List[dict[str, Any]] = []


    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent.
        加载向此智能体提供的技能目录。
        """

        # TODO(1.4): Validate ``skills_path``, discover one ``SKILL.md``
        # per child directory, parse its YAML frontmatter (what's between the
        # `---` tags at the head of the file), and return a mapping
        # keyed by the frontmatter ``name``. Each value must contain a concise
        # ``metadata`` string for the model's skill catalog and the complete
        # ``content`` of the skill file for ``invoke_skill``. Reject duplicate
        # names and malformed or missing frontmatter with a clear
        # ``ValueError``.
        # TODO(1.4)：验证 ``skills_path``，在每个子目录中发现一个 ``SKILL.md``，
        # 解析其 YAML frontmatter（文件开头两个 `---` 标记之间的内容），并返回
        # 以 frontmatter 中的 ``name`` 为键的映射。每个值必须包含用于模型技能
        # 目录的简洁 ``metadata`` 字符串，以及供 ``invoke_skill`` 使用的技能文件
        # 完整 ``content``。对于重复名称、格式错误或缺失的 frontmatter，抛出
        # 明确的 ``ValueError``。
        if not skills_path.is_dir():
            raise ValueError(f"skills_path {skills_path} is not a directory")

        skills: dict[str, dict[str, str]] = {}
        for child in sorted(skills_path.iterdir()):
            if not child.is_dir():
                continue

            md_file = [p for p in child.iterdir() if p.name == "SKILL.md"]
            if len(md_file) != 1:
                raise ValueError(
                    f"Expected exactly one SKILL.md in {child}, found {len(md_file)}"
                )
            
            text = md_file[0].read_text(encoding="utf-8")
            if not text.startswith("---"):
                raise ValueError(f"SKILL.md in {child} is missing frontmatter")

            parts = text.split("---", 2)
            if len(parts) < 3:
                raise ValueError(f"SKILL.md in {child} is missing frontmatter")
            
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"SKILL.md in {child} has malformed frontmatter") from exc

            if (
                not isinstance(meta, dict)
                or not meta.get("name")
                or not meta.get("description")
            ):
                raise ValueError(f"SKILL.md in {child} has malformed frontmatter")

            name = str(meta["name"])

            if name in skills:
                raise ValueError(f"Duplicate skill name {name} in {child}")

            skills[name] = {
                "metadata": "\n".join(f"{k}: {v}" for k, v in meta.items()),
                "content": text,
            }

        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it.
        发送一次启用工具的 Chat Completions 请求并将其规范化。
        """

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response.
        返回语言模型响应中的相关部分。
        """

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        # TODO(1.1.a): Construct a sequence of messages that form the language
        # model prompt. This should include standing instructions, task
        # specification, prior interaction including observations, reasoning,
        # and actions from previous turns. Note that this method should be
        # domain-agnostic and construct the prompt in a way that would apply
        # to any of the inheriting domain-specific agents.
        # TODO(1.1.a)：构建构成语言模型提示词的消息序列。其中应包括长期有效的
        # 指令、任务说明、先前的交互（包括观察、推理和前几轮的操作）。注意，
        # 该方法应与领域无关，其构建提示词的方式应适用于任何继承它的领域专用
        # 智能体。

        # You want to be careful about which attributes of the class you modify
        # here as they may also be handled by the subclasses.
        # 请谨慎选择在此处修改哪些类属性，因为子类可能也会处理这些属性。

        messages_head: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ]

        return messages_head + self.messages


    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage.
        估算下一个提示词的 token 数，并根据提供商最近报告的用量进行校准。
        """

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)
        if not self.api_prompts or not self.api_responses:
            return rough_current

        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all.
        此智能体是否启用上下文压缩。
        """

        return self.compact_threshold_tokens is not None

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits.
        用模型生成的工作记忆替换提示词中的部分内容，改变 `build_prompt` 的输出。
        """

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.
        # TODO(2.1)：提示模型压缩上下文。系统提示词应要求生成简洁、事实性的
        # 工作记忆，并保留目标、约束、文件、命令、编辑、具体结果、失败的尝试、
        # 测试、阻塞因素和下一步操作。只总结较旧的前缀；原样保留原始系统/任务
        # 消息，并至少保留最近一条完整的助手操作及其所有关联的工具观察。生成的
        # 摘要应改变 `build_prompt` 的输出，并缩短提示词长度。

        assistant_idx = [i for i, m in enumerate(self.messages) if m.get("role") == "assistant"]
        n = len(assistant_idx)
        k = self.compaction_keep_recent_steps
        boundary = assistant_idx[max(0, n - k)] if n > 0 else len(self.messages)

        old_prefix = self.messages[:boundary]
        recent_tail = self.messages[boundary:]
        
        compaction_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
        ] + old_prefix + [
            {"role": "user", "content": COMPACTION_SYSTEM_PROMPT},
        ]

        ### Do not modify this section ###
        ### 请勿修改此部分 ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        # 使用 `compaction_response` 更新 `build_prompt` 的输出，但不要修改
        # 该对象本身。让方法原样返回它。

        summary = compaction_response.choices[0].message.content or ""

        self.messages = [
            {"role": "user", "content": f"<working_memory>{summary}</working_memory>"},
        ] + recent_tail

        ### Do not modify this section ###
        ### 请勿修改此部分 ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached.
        达到阈值时，在请求下一个操作之前压缩上下文。
        """

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        # 上下文还太短，暂时无需压缩
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        # 步数（每个助手轮次对应一步）还不足以触发压缩
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal.
        运行 ReAct 步骤，并始终保存交互轨迹和停止 Modal。
        """

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.
            # TODO(1.2)：运行 ReAct 循环。协调以下流程：提示语言模型生成推理和
            # 操作，提取模型产生的工具调用，执行工具调用，从而获得智能体下一步
            # 所需的观察结果。确保通过设置 `Agent.finished` 来识别智能体何时
            # 完成任务。如果智能体超过 `step_limit`，则抛出 `StepLimitError`。

            while not self.finished:

            # TODO(2.2) Call `maybe_compact_context()` before each new action
            # request in your shared loop. It already estimates active tokens
            # and handles the threshold, and tracks compaction events for
            # logging.
            # TODO(2.2)：在共享循环中，每次请求新操作之前调用
            # `maybe_compact_context()`。该方法已经会估算活跃 token 数、处理
            # 阈值，并记录压缩事件以供日志使用。

                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(
                        f"Agent exceeded step limit of {self.step_limit}"
                    )

                self.maybe_compact_context()

                message: dict[str, Any] = self.query_language_model()
                self.messages.append(message)

                tool_calls: list[dict[str, Any]] = message.get("tool_calls", [])
                if tool_calls:
                    observations = self.execute_tool_calls(tool_calls)
                    self.messages.extend(observations)
                else:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "If you are finished, finish via the appropriate completion tool; otherwise take your next action with a tool call if necessary."
                        }
                    )
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            # 此代码块是提供的基础设施。请勿修改：即使运行失败，也必须保存
            # 交互轨迹。
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations.
        执行领域专用调用并返回关联的工具观察结果。
        """

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        # 你无需在此处实现任何内容。此方法是领域专用的，由相关子类实现。
        raise NotImplementedError
