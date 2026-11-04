"""The Part 3 chess agent: reuse the shared loop with one new domain tool.
第 3 部分的国际象棋智能体：复用共享循环并添加一个新的领域工具。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
)
from assignment.agent.chess_tools import (
    _game_state,
    _invoke_skill,
    _play_move,
    CHESS_PORT,
    _run_python,
    _simulate_move,
)
from assignment.agent.tools import (
    INVOKE_SKILL_TOOL,
    PLAY_MOVE_TOOL,
    RUN_PYTHON_TOOL,
    SIMULATE_MOVE_TOOL,
)
from assignment.prompts import (
    CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE,
    CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE,
    PROGRAMMATIC_CHESS_PROMPT,
)
from assignment.env import Environment


def format_chess_state(
    state: dict[str, Any], *, include_legal_moves: bool = True
) -> str:
    """Turn chess API JSON into a compact observation an LLM can act on.
    将国际象棋 API 的 JSON 转换为紧凑的观察结果，供 LLM 据此操作。
    """

    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")

    recent_history = state.get("history", [])[-8:]
    history = (
        " ".join(
            f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history
        )
        or "(none)"
    )
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    board_text = "\n".join(board_lines)

    observation = (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n"
        f"{board_text}\n"
        f"recent_history: {history}\n"
    )
    if include_legal_moves:
        observation += f"legal_moves: {legal_moves}\n"
    return observation + "</chess_state>"


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot.
    执白棋与服务端确定性的黑方机器人对弈的智能体。
    """

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 200,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        http_client: Any | None = None,
        reset_game: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None,
        programmatic_tools: bool = False,
        python_sandbox_port: int = CHESS_PORT,
        include_legal_moves: bool = True,
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

        self.tools.append(PLAY_MOVE_TOOL)
        if programmatic_tools:
            self.tools.extend([SIMULATE_MOVE_TOOL, RUN_PYTHON_TOOL])

        # run_python always executes in the sandbox, on the port the chess
        # server is listening on there.
        # run_python 始终在沙箱中执行，并使用国际象棋服务端在沙箱内监听的端口。
        self.python_sandbox_port = python_sandbox_port
        self.include_legal_moves = include_legal_moves

        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError(
                    "ChessAgent needs an environment with server_url or an http_client."
                )
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client

        initial_state = _game_state(self.chess_client, reset=reset_game)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        prompt_template = (
            CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE
            if include_legal_moves
            else CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE
        )
        self.system_prompt = prompt_template.render()
        if programmatic_tools:
            self.system_prompt += "\n\n" + PROGRAMMATIC_CHESS_PROMPT
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )
        opening_instruction = (
            "Choose one move from legal_moves and call play_move."
            if include_legal_moves
            else "Infer a legal move from the board and FEN, then call play_move."
        )
        self.task_prompt = (
            f"Play this game as White. {opening_instruction}\n\n"
            f"{self.format_state(initial_state)}"
        )

    def format_state(self, state: dict[str, Any]) -> str:
        """Format a state according to this run's observation condition.
        根据本次运行的观察条件格式化状态。
        """

        return format_chess_state(state, include_legal_moves=self.include_legal_moves)

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute model-generated ``play_move`` calls against the chess API.
        针对国际象棋 API 执行模型生成的 ``play_move`` 调用。
        """

        # TODO(Part 3.1):
        # TODO(第 3.1 部分)：
        # 1. Dispatch on the function name, and ignore a tool this agent did
        #    not register.
        # 1. 根据函数名进行调度，并忽略此智能体未注册的工具。
        # 2. Hand the raw arguments to the matching chess_tools helper, with
        #    self.chess_client as its first argument. Each helper takes the
        #    client explicitly so the same code can run inside the sandbox.
        # 2. 将原始参数交给匹配的 chess_tools 辅助函数，并以 self.chess_client
        #    作为第一个参数。每个辅助函数都显式接收客户端，因此同一份代码可在
        #    沙箱内运行。
        # 3. Format a played move with self.format_state, then update
        #    last_state and finished.
        # 3. 使用 self.format_state 格式化已执行的走法，然后更新 last_state
        #    和 finished。
        # 4. Link every observation to its call with tool_call_id.
        # 4. 使用 tool_call_id 将每个观察结果关联到其调用。
        # 5. Turn malformed, unknown, rejected, or extra parallel calls into
        #    recoverable <chess_error> observations instead of crashing.
        # 5. 将格式错误、未知、被拒绝或额外的并行调用转换为可恢复的
        #    <chess_error> 观察结果，而不是崩溃。

        # TODO(Part 3.3-4): add cases for simulate_move and run_python, with
        # linked observations and recoverable errors, just like the old tool.
        # TODO(第 3.3—4 部分)：为 simulate_move 和 run_python 添加处理分支，
        # 像原有工具一样提供关联的观察结果和可恢复错误。

        observations = []
        moved = False
        for call in tool_calls:
            call_id = call.get("id")
            name = call.get("function", {}).get("name")
            arguments = call.get("function", {}).get("arguments", "{}")

            try:
                if name == "play_move":
                    if moved:
                        content = (
                            "<chess_error>a move was already played this turn; "
                            "wait for the returned board</chess_error>"
                        )
        
                    else:
                        result = _play_move(self.chess_client, arguments)

                        if result.startswith("<chess_error>"):
                            content = result
                        
                        else:
                            state = json.loads(result)
                            self.last_state = state
                            self.finished = bool(state.get("game_over"))
                            moved = True
                            content = self.format_state(state)

                elif name == "simulate_move":
                    content = _simulate_move(self.chess_client, arguments)

                elif name == "run_python":
                    content = _run_python(self.env, self.python_sandbox_port, arguments)

                    if not content.startswith("<chess_error>"):
                        live = _game_state(self.chess_client, reset=False)
                        self.last_state = live
                        self.finished = bool(live.get("game_over"))
                        content = content + "\n" + self.format_state(live)

                elif name == "invoke_skill":
                    content = _invoke_skill(self.skills, arguments)
                
                else:
                    content = (
                        f"<chess_error>tool {name} is not registered for this agent</chess_error>"
                    )
            except Exception as exc:
                content = f"<chess_error>{exc}</chess_error>"
            
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        return observations