"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.

与注册它们的智能体解耦的国际象棋工具实现。

这里的每个函数都显式接收 HTTP 客户端，而不是从智能体读取，因此同一份代码既可在
智能体进程中运行，也可在其所通信服务端旁的沙箱中运行。
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _load_object(arguments: str) -> dict[str, Any]:
    if not isinstance(arguments, str):
        raise ValueError("Tool arguments must be a JSON string.")
    payload = json.loads(arguments)
    if not isinstance(payload, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return payload



def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response.
    发起一次国际象棋 API 请求并验证其 JSON 响应。
    """

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.

    新工具：检查 FEN 或模拟一步棋，而不改变实际棋局。

    接收一次工具调用的原始 JSON 参数并返回要传回的观察结果，使错误参数或服务端
    错误以可恢复的 ``<chess_error>`` 形式到达模型，而不是终止运行。
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    # TODO(第 3.3.b 部分)：解析参数，使用 fen 和可选 move 调用提供的
    # /api/simulate 端点，并返回其 JSON。捕获工具抛出的所有错误，并在
    # `<chess_error></chess_error>` 之间返回错误消息，交由智能体处理。需覆盖
    # JSON 参数格式错误、参数不是对象、fen 缺失或不是字符串、move 不是字符串、
    # 服务端拒绝局面或走法，以及传输失败。
    try:
        args = _load_object(arguments)
        fen = args.get("fen")

        if not isinstance(fen, str) or not fen:
            raise ValueError("Missing or invalid 'fen' argument.")

        move = args.get("move")
        if move is not None and (not isinstance(move, str) or not move):
            raise ValueError("Invalid 'move' argument.")
        
        body = {"fen": fen}
        if move is not None:
            body["move"] = move
        
        payload = _request_state(client, "POST", "/api/simulate", json=body)
    
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"

    return json.dumps(payload)

def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.

    现有工具：作为白方走一步棋并返回结果状态。

    接收一次工具调用的原始 JSON 参数。返回新状态；如果无法走子，则返回
    `<chess_error>` 观察结果。
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.
    # TODO(3.1.b)：解析参数，并向 /api/move 发送 POST 请求
    # {"move": <uci move>}。返回其 JSON 对象。捕获工具抛出的所有错误，并在
    # `<chess_error></chess_error>` 之间返回错误消息，交由智能体处理。需覆盖
    # JSON 参数格式错误、参数不是对象、fen 缺失或不是字符串、move 不是字符串、
    # 服务端拒绝局面或走法，以及传输失败。
    try:
        args = _load_object(arguments)
        move = args.get("move")

        if not isinstance(move, str) or not move:
            raise ValueError("Missing or invalid 'move' argument.")

        payload = _request_state(client, "POST", "/api/move", json={"move": move})
    
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"

    return json.dumps(payload)

def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.

    新工具：运行 Python，并可访问已注册的现有工具。

    代码片段在沙箱中运行，沙箱已经包含工具实现和国际象棋服务端，因此模型编写的
    代码绝不会在智能体进程中执行。
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    # TODO(3.4)：解析参数，在沙箱中运行代码，并按名称提供已注册的工具。
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    # `/opt/assignment/sandbox_python.py` 是 `env` 沙箱上的一个脚本，可以访问
    # 与本文件中相同的工具定义。用它运行模型生成的、作为 run_python 工具参数
    # 传入的代码。该脚本接受两个位置参数——`port` 和经过 base64 编码的代码
    # 字符串（避免引号问题）。实现此工具调用。
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    # 脚本会打印一个 JSON 对象，其中包含运行代码产生的 `stdout`、`stderr` 和
    # `error`——原样返回该字符串。
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    # 非零 returncode 表示沙箱本身失败，而不是模型代码失败。将
    # `exception_info` 或 `stderr` 作为 <chess_error> 报告。
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    # 如果出现类型不匹配或解析失败等问题，返回
    # <chess_error>{message}</chess_error>。
    try:
        args = _load_object(arguments)
        code = args.get("code")

        if not isinstance(code, str) or not code:
            raise ValueError("Missing or invalid 'code' argument.")

        encode = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        result = env.execute(
            command = [
                "python",
                "/opt/assignment/sandbox_python.py",
                str(port),
                encode,
            ],
            shell = False,
        )
    
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"
    
    if result.get("returncode") != 0:
        message = (
            result.get("exception_info")
            or result.get("stderr")
            or "Sandbox failed with unknown error."
        )
        return f"<chess_error>{message}</chess_error>"
    
    return result.get("stdout", "") or result.get("output") or ""

def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation.
    现有工具：将一个技能的指令加载到对话中。
    """
    # TODO(3.5): parse the arguments and return the named skill's content.
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    # TODO(3.5)：解析参数并返回指定技能的内容。如果出现类型不匹配或解析失败
    # 等问题，返回 <chess_error>{message}</chess_error>。
    try:
        args = _load_object(arguments)
        name = args.get("name")

        if not isinstance(name, str) or not name:
            raise ValueError("Missing or invalid 'name' argument.")

        skill = skills.get(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found.")

    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"
    
    return skill["content"]


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position.
    读取实时棋局，或开始新棋局并读取初始局面。
    """

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
