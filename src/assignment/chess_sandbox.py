"""Run the chess server in a Modal sandbox and play moves against it.

This is the part 2 surface. It does not know anything about fixing the bug: it
takes an optional patch (the fix produced in part 1), applies it to the
testbed, serves the app, and exposes the game over HTTP.

在 Modal 沙箱中运行国际象棋服务端并与其对弈。

这是第 2 部分的操作界面。它不负责修复缺陷：接收可选补丁（第 1 部分生成的
修复），将其应用到测试环境，启动应用，并通过 HTTP 提供棋局。
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import httpx

from assignment.agent.chess_tools import CHESS_PORT
from assignment.task import Task
from assignment.env import Environment
from assignment.utils.image import build_testbed_image

TESTBED = "/testbed"
SERVER_LOG = "/tmp/chess-server.log"
DEFAULT_TASK = Path(__file__).resolve().parents[2] / "tasks" / "chess-terminal-move"

class IllegalMove(Exception):
    """The server rejected a move as illegal or malformed.
    服务端因走法非法或格式错误而拒绝了该走法。
    """

# The agent's own tool implementations, so a model-written snippet can run in
# here against the local server instead of in the agent process.
# 智能体自己的工具实现，使模型编写的代码片段可在这里针对本地服务端运行，
# 而不是在智能体进程中运行。
SANDBOX_FILES = (
    Path(__file__).with_name("chess_server.py"),
    Path(__file__).parent / "agent" / "chess_tools.py",
    Path(__file__).with_name("sandbox_python.py"),
)

def _with_assignment_files(image):
    """Copy the files the sandbox runs into /opt/assignment.
    将沙箱运行所需的文件复制到 /opt/assignment。
    """

    for source in SANDBOX_FILES:
        image = image.add_local_file(
            str(source),
            f"/opt/assignment/{source.name}",
            copy=True,  # SWE-ReX adds its runtime build layer afterwards.
            # SWE-ReX 随后会添加其运行时构建层。
        )
    return image

class ChessSandbox(Environment):
    """A chess server hosted in an isolated Modal sandbox.

    Built from the same testbed image the evaluation harness uses, so the code
    under test and the code being played are identical. Without a patch the
    sandbox runs the unfixed engine, which still plays normally right up until
    a game-ending move.

    托管在隔离 Modal 沙箱中的国际象棋服务端。

    使用与评估运行框架相同的测试环境镜像构建，因此被测代码与对弈代码完全一致。
    没有补丁时，沙箱运行未修复的引擎；在导致棋局结束的走法之前，它仍能正常对弈。
    """

    def __init__(
        self,
        task: str | Path | Task | None = None,
        patch: str | Path | None = None,
        port: int = CHESS_PORT,
        strict: bool = True,
        startup_timeout: float = 300,
        runtime_timeout: float = 600,
        deployment_timeout: float = 1800,
        server_timeout: float = 30,
    ):
        """Launch the sandbox and block until the server answers.

        Args:
            task: The task whose testbed to serve, as a directory or a loaded
                `Task`. Defaults to the chess task.
            patch: A unified diff applied to the testbed before the server
                starts, normally the fix from part 1.
            port: Port to serve on, forwarded through an encrypted tunnel.
            strict: Refuse to build when the local checkout does not match the
                task's base commit.
            startup_timeout: Seconds to wait for the SWE-ReX runtime.
            runtime_timeout: Seconds a single command may run.
            deployment_timeout: Seconds the sandbox may stay alive.
            server_timeout: Seconds to wait for `/health` to succeed.

        启动沙箱并阻塞，直到服务端响应。

        参数：
            task：要运行其测试环境的任务，可以是目录或已加载的 `Task`。
                默认为国际象棋任务。
            patch：服务端启动前应用到测试环境的 unified diff，通常是第 1 部分
                的修复。
            port：服务监听端口，通过加密隧道转发。
            strict：本地检出与任务的基准提交不一致时拒绝构建。
            startup_timeout：等待 SWE-ReX 运行时的秒数。
            runtime_timeout：单条命令可运行的秒数。
            deployment_timeout：沙箱可保持运行的秒数。
            server_timeout：等待 `/health` 成功的秒数。
        """
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid port: {port}")

        self.task = task if isinstance(task, Task) else Task.load(task or DEFAULT_TASK)
        self.port = port
        self.server_url = ""
        self._client: httpx.Client | None = None

        super().__init__(
            image=_with_assignment_files(build_testbed_image(self.task, strict=strict)),
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            modal_sandbox_kwargs={"encrypted_ports": [port]},
        )

        try:
            if patch is not None:
                self._apply_patch(patch)
            self.server_url = self.tunnel_url(port).rstrip("/")
            self._start_server(server_timeout)
            self._client = httpx.Client(base_url=self.server_url, timeout=30)
        except Exception:
            self.stop()
            raise

    def _apply_patch(self, patch: str | Path) -> None:
        """Apply the part 1 fix to the testbed before serving it.
        在运行测试环境之前，将第 1 部分的修复应用到其中。
        """
        # Accept either a path to a diff or the diff itself; a diff always has
        # newlines, a path never usefully does.
        # 接受 diff 的路径或 diff 本身；diff 总是包含换行，而路径通常不应包含。
        if isinstance(patch, Path):
            text = patch.read_text()
        else:
            text = patch if "\n" in patch else Path(patch).read_text()
        encoded = base64.b64encode(text.encode()).decode()

        written = self.execute(f"echo {encoded} | base64 -d > /tmp/fix.diff")
        if written["returncode"] != 0:
            raise RuntimeError(f"Could not upload the patch: {written['stderr'].strip()}")

        applied = self.execute("git apply -v /tmp/fix.diff", cwd=TESTBED)
        if applied["returncode"] != 0:
            raise RuntimeError(f"Patch did not apply:\n{applied['stderr'].strip()}")

    def _start_server(self, timeout: float) -> None:
        command = (
            "nohup python /opt/assignment/chess_server.py "
            f"--host 0.0.0.0 --port {self.port} "
            f"> {SERVER_LOG} 2>&1 < /dev/null & echo $!"
        )
        result = self.execute(command, cwd=TESTBED)
        if result["returncode"] != 0:
            raise RuntimeError(f"Could not start the chess server: {result['stderr'].strip()}")

        deadline = time.monotonic() + timeout
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{self.server_url}/health", timeout=3) as response:
                    payload = json.load(response)
                if response.status == 200 and payload == {"status": "ok"}:
                    return
                last_error = f"unexpected health response: {payload!r}"
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            time.sleep(0.25)

        logs = self.execute(f"tail -n 80 {SERVER_LOG}")
        log_text = (logs["stdout"] or logs["stderr"]).strip()
        raise RuntimeError(
            f"Chess server did not become ready within {timeout:g}s ({last_error}).\n{log_text}"
        )

    # -- Playing ------------------------------------------------------------
    # -- 对弈 ------------------------------------------------------------
    # A chess agent drives the game through these. They speak the same HTTP API
    # the browser frontend uses.
    # 国际象棋智能体通过这些方法驱动棋局。它们使用与浏览器前端相同的 HTTP API。

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("The sandbox is not running.")
        return self._client

    def state(self) -> dict:
        """Return the current board state, legal moves, history, and result.
        返回当前棋盘状态、合法走法、历史和结果。
        """
        response = self.client.get("/api/state")
        response.raise_for_status()
        return response.json()

    def play(self, move: str) -> dict:
        """Play one move as White and return the resulting state.

        Args:
            move: A move in UCI notation, such as `e2e4`.

        Returns:
            The new state, including `human_move` and the opponent's
            `engine_move`. `engine_move` is None once the game has ended.

        Raises:
            IllegalMove: The move was rejected. An agent should treat this as a
                recoverable mistake and choose another move.

        作为白方走一步棋并返回结果状态。

        参数：
            move：UCI 记法的走法，例如 `e2e4`。

        返回：
            新状态，包括 `human_move` 和对手的 `engine_move`。棋局结束后，
            `engine_move` 为 None。

        抛出：
            IllegalMove：走法被拒绝。智能体应将其视为可恢复的错误并另选走法。
        """
        response = self.client.post("/api/move", json={"move": move})
        if response.status_code == 400:
            raise IllegalMove(response.json().get("detail", f"{move} was rejected"))
        response.raise_for_status()
        return response.json()

    def reset(self) -> dict:
        """Start a new game and return the opening state.
        开始新棋局并返回初始状态。
        """
        response = self.client.post("/api/reset")
        response.raise_for_status()
        return response.json()

    def stop(self, timeout: float = 10) -> None:
        """Close the HTTP client, then tear the sandbox down.
        关闭 HTTP 客户端，然后拆除沙箱。
        """
        if self._client is not None:
            self._client.close()
            self._client = None
        super().stop(timeout=timeout)

def main() -> None:
    """Launch a chess sandbox and keep it alive until the user exits.
    启动国际象棋沙箱并保持运行，直到用户退出。
    """
    import argparse

    parser = argparse.ArgumentParser(description="Serve the chess app from a Modal sandbox")
    parser.add_argument("--patch", type=Path, help="Fix to apply before serving, e.g. the part 1 output")
    parser.add_argument("--task", type=Path, help="Task directory to serve the testbed of")
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=1800,
        help="maximum lifetime of the Modal sandbox in seconds",
    )
    args = parser.parse_args()

    with ChessSandbox(
        task=args.task,
        patch=args.patch,
        deployment_timeout=args.sandbox_timeout,
    ) as sandbox:
        print(f"Chess sandbox ready: {sandbox.server_url}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping chess sandbox…", flush=True)

if __name__ == "__main__":
    main()
