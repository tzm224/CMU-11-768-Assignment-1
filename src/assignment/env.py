import asyncio
import posixpath
import time
from pathlib import PurePath
from typing import Any

import modal
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command
from swerex.runtime.remote import RemoteRuntime


def _tls_port_configuration(
    modal_sandbox_kwargs: dict[str, Any], runtime_port: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Put the SWE-ReX control port on the TLS tunnel exactly once.
    将 SWE-ReX 控制端口恰好一次加入 TLS 隧道。
    """

    kwargs = dict(modal_sandbox_kwargs)
    encrypted = list(kwargs.pop("encrypted_ports", []))
    unencrypted = [
        port for port in kwargs.pop("unencrypted_ports", []) if port != runtime_port
    ]
    encrypted = list(dict.fromkeys([runtime_port, *encrypted]))
    return encrypted, unencrypted, kwargs


class AssignmentModalDeployment(ModalDeployment):
    """Modal deployment with a correctly matched TLS tunnel and URL.

    SWE-ReX 1.4.0 opens its control port as unencrypted TCP but initializes the
    HTTP client with ``Tunnel.url``, which is Modal's HTTPS endpoint. Keep the
    upstream runtime and authentication protocol, but expose that control port
    through ``encrypted_ports`` so its URL and transport agree.

    具有正确匹配的 TLS 隧道和 URL 的 Modal 部署。

    SWE-ReX 1.4.0 将控制端口作为未加密 TCP 开放，却使用 ``Tunnel.url``
    （Modal 的 HTTPS 端点）初始化 HTTP 客户端。保留上游运行时和身份验证协议，
    但通过 ``encrypted_ports`` 暴露该控制端口，使其 URL 与传输方式一致。
    """

    async def start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started; ignoring duplicate start().")
            return

        self.logger.info("Starting Modal sandbox")
        self._hooks.on_custom_step("Starting Modal sandbox")
        started = time.monotonic()
        token = self._get_token()
        encrypted, unencrypted, extra_kwargs = _tls_port_configuration(
            self._modal_kwargs, self._port
        )
        try:
            self._sandbox = await modal.Sandbox.create.aio(
                "/usr/bin/env",
                "bash",
                "-c",
                self._start_swerex_cmd(token),
                image=self._image,
                timeout=int(self._deployment_timeout),
                encrypted_ports=encrypted,
                unencrypted_ports=unencrypted,
                app=self._app,
                **extra_kwargs,
            )
            tunnels = await self._sandbox.tunnels.aio()
            tunnel = tunnels[self._port]
            elapsed = time.monotonic() - started
            log_url = await self.get_modal_log_url()
            self.logger.info("Sandbox %s created in %.2fs", self._sandbox.object_id, elapsed)
            self.logger.info("Modal logs: %s", log_url)
            await asyncio.sleep(1)
            self.logger.info("Starting SWE-ReX runtime at %s", tunnel.url)
            self._hooks.on_custom_step("Starting runtime")
            self._runtime = RemoteRuntime(
                host=tunnel.url,
                timeout=self._runtime_timeout,
                auth_token=token,
                logger=self.logger,
            )
            remaining = max(0, self._startup_timeout - elapsed)
            await self._wait_until_alive(timeout=remaining)
        except Exception as exc:
            details = [f"{type(exc).__name__}: {exc}"]
            if self._sandbox is not None:
                details.append(f"sandbox_id={self._sandbox.object_id}")
                try:
                    details.append(f"logs={await self.get_modal_log_url()}")
                except Exception:
                    pass
                try:
                    await self._sandbox.terminate.aio()
                except Exception:
                    pass
            self._runtime = None
            self._sandbox = None
            raise RuntimeError(
                "Modal sandbox started, but the SWE-ReX control runtime did not "
                "become reachable. " + " | ".join(details)
            ) from exc

class Environment:
    """
    Executes bash commands in a Modal sandbox via SWE-ReX.
    通过 SWE-ReX 在 Modal 沙箱中执行 bash 命令。
    """

    # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py
    # 注意（来源）：https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py

    def __init__(
        self,
        image: "str | PurePath | modal.Image" = "python:3.12",
        cwd: str = "/",
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        install_pipx: bool = True,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        conda_env: str | None = None,
    ):
        """Launch a Modal sandbox and block until its runtime answers.

        Args:
            image: A prebuilt `modal.Image`, a Dockerhub or ECR image name, or a
                path to a Dockerfile. Build a task testbed with
                `assignment.utils.image.build_testbed_image`, which is the only
                way to pass a credential to a private clone.
            cwd: Working directory for commands that do not specify one.
            startup_timeout: Seconds to wait for the SWE-ReX runtime to come up.
            runtime_timeout: Seconds a single command may run before timing out.
            deployment_timeout: Seconds the sandbox may stay alive before Modal
                reclaims it.
            install_pipx: Install pipx in the image, needed to bootstrap the
                SWE-ReX server when it is not already present.
            modal_sandbox_kwargs: Additional keyword arguments forwarded to
                ``modal.Sandbox.create``. This is used for capabilities such as
                encrypted port forwarding.
            conda_env: Name of a conda environment to put on PATH for every
                command. SWE-bench images install the repository under test into
                an environment named ``testbed`` but never activate it, so
                without this ``python`` is conda's base environment, where the
                repository and its dependencies are not installed.

        启动 Modal 沙箱并阻塞，直到其运行时响应。

        参数：
            image：预构建的 `modal.Image`、Docker Hub 或 ECR 镜像名称，或
                Dockerfile 路径。使用 `assignment.utils.image.build_testbed_image`
                构建任务测试环境，这是向私有克隆传递凭据的唯一方式。
            cwd：未指定工作目录的命令所使用的目录。
            startup_timeout：等待 SWE-ReX 运行时启动的秒数。
            runtime_timeout：单条命令超时前可运行的秒数。
            deployment_timeout：Modal 回收沙箱前，沙箱可保持运行的秒数。
            install_pipx：在镜像中安装 pipx；当 SWE-ReX 服务端尚未存在时，需要
                用它来引导该服务端。
            modal_sandbox_kwargs：转发给 ``modal.Sandbox.create`` 的额外关键字
                参数，用于加密端口转发等功能。
            conda_env：要为每条命令加入 PATH 的 conda 环境名称。SWE-bench 镜像
                将被测仓库安装到名为 ``testbed`` 的环境中但从不激活它，因此如果
                不设置此项，``python`` 会指向 conda 的基础环境，其中未安装仓库
                及其依赖。
        """
        self.cwd = cwd
        # Merged into every command's environment; a per-call `env` wins.
        # 合并到每条命令的环境中；单次调用提供的 `env` 优先。
        self.env_defaults: dict[str, str] = {}
        self.deployment = AssignmentModalDeployment(
            image=image,
            startup_timeout=startup_timeout,
            runtime_timeout=runtime_timeout,
            deployment_timeout=deployment_timeout,
            install_pipx=install_pipx,
            modal_sandbox_kwargs=modal_sandbox_kwargs,
        )

        async def _start():
            await self.deployment.start()
            await self.deployment.is_alive()

        asyncio.run(_start())

        if conda_env:
            self.activate_conda_env(conda_env)

        # Read the platform from inside the container, not from platform.uname(),
        # which would describe the machine running this code instead.
        # 从容器内部读取平台信息，而不是使用 platform.uname()；后者描述的是
        # 运行此代码的机器。
        self.system, self.release, self.version, self.machine = self.execute(
            "uname -s; uname -r; uname -v; uname -m"
        )["output"].splitlines()

    def is_alive(self) -> bool:
        """Whether the sandbox is still running.

        Modal reclaims a sandbox once `deployment_timeout` elapses, and the
        deployment keeps its handle afterwards, so this asks the sandbox itself
        rather than trusting the handle's existence.

        沙箱是否仍在运行。

        `deployment_timeout` 到期后 Modal 会回收沙箱，而部署对象仍保留其句柄，
        因此这里直接询问沙箱本身，而不是仅凭句柄存在来判断。
        """
        sandbox = self.deployment._sandbox
        return sandbox is not None and sandbox.poll() is None

    def activate_conda_env(self, name: str, root: str = "/opt/miniconda3") -> str:
        """Put a conda environment's bin directory first on PATH for all commands.

        Activating properly needs a login shell, which these commands do not
        get. Prepending the environment's bin directory has the same effect for
        ``python``, ``pip``, and anything else installed there.

        Args:
            name: The environment name, for example ``testbed``.
            root: Where conda is installed in the image.

        Returns:
            The PATH now used for every command.

        Raises:
            FileNotFoundError: If the environment is not in the image, rather
                than silently leaving the wrong interpreter on PATH.

        对所有命令，将 conda 环境的 bin 目录置于 PATH 最前面。

        正确激活环境需要登录 shell，而这些命令不会获得登录 shell。将环境的 bin
        目录前置，对 ``python``、``pip`` 及其中安装的其他程序具有相同效果。

        参数：
            name：环境名称，例如 ``testbed``。
            root：conda 在镜像中的安装位置。

        返回：
            现在每条命令使用的 PATH。

        抛出：
            FileNotFoundError：环境中不存在该 conda 环境时抛出，避免在不提示的
            情况下将错误的解释器留在 PATH 中。
        """
        binary_dir = posixpath.join(root, "envs", name, "bin")
        if self.execute(f"test -d {binary_dir}", cwd="/")["returncode"] != 0:
            raise FileNotFoundError(f"No conda environment at {binary_dir}")

        current = self.execute("printenv PATH", cwd="/")["output"].strip()
        self.env_defaults["PATH"] = f"{binary_dir}:{current}"
        return self.env_defaults["PATH"]

    def execute(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        # Command defaults to shell=False, which requires `command` to be an argv
        # list; default to shell=True so a plain command string works.
        # Command 默认 shell=False，这要求 `command` 是 argv 列表；这里默认
        # shell=True，使普通命令字符串也能工作。
        shell: bool | None = True,
        # check=False so a failing command returns its exit code instead of
        # raising and discarding the output the agent needs to see.
        # check=False 使失败的命令返回退出码，而不是抛出异常并丢弃智能体需要
        # 查看的输出。
        check: bool = False,
    ) -> dict:
        """Run a command in the sandbox and return its result.

        Args:
            command: A shell string, or an argv list if `shell` is False.
            timeout: Seconds before the command is killed. None means no timeout.
            cwd: Working directory to run the command in. Defaults to the
                environment's `cwd`.
            env: Environment variables to set for the command.
            shell: Run the command through a shell, as with `subprocess.run`.
                None is treated as True, since a model calling this as a tool may
                send null for an omitted argument.
            check: Raise on a non-zero exit code instead of reporting it.

        Returns:
            A dict with `output`, `returncode`, and `exception_info`. `output`
            holds stdout and stderr interleaved, as they would appear in a
            terminal. `exception_info` is empty when the command ran, whatever
            its exit code. If the call itself fails (the sandbox died, the
            command timed out), the exception is caught rather than raised:
            `returncode` is -1, `exception_info` describes it, and an `extra` key
            carries the exception type.

        在沙箱中运行命令并返回结果。

        参数：
            command：shell 字符串；如果 `shell` 为 False，则为 argv 列表。
            timeout：命令被终止前的秒数。None 表示无超时。
            cwd：运行命令的工作目录，默认为环境的 `cwd`。
            env：为命令设置的环境变量。
            shell：像 `subprocess.run` 一样通过 shell 运行命令。None 视为 True，
                因为通过工具调用此方法的模型可能会为省略的参数发送 null。
            check：在退出码非零时抛出异常，而不是报告退出码。

        返回：
            包含 `output`、`returncode` 和 `exception_info` 的字典。`output`
            按终端中的显示方式交错保存 stdout 和 stderr。只要命令已运行，无论
            退出码为何，`exception_info` 都为空。如果调用本身失败（沙箱终止、
            命令超时），异常会被捕获而不是抛出：`returncode` 为 -1，
            `exception_info` 描述异常，`extra` 键保存异常类型。
        """
        # Command.shell is a strict bool, so None would fail validation.
        # Command.shell 是严格的布尔值，因此 None 会无法通过验证。
        shell = True if shell is None else shell

        arguments = {
            "command": command,
            "timeout": timeout,
            "env": {**self.env_defaults, **(env or {})} or None,
            "shell": shell,
            "check": check,
            "merge_output_streams": False,
        }
        if cwd is not None:
            arguments["cwd"] = cwd
        # A cwd from the caller wins; otherwise the environment's default applies.
        # 调用方提供的 cwd 优先；否则使用环境的默认值。
        arguments.setdefault("cwd", self.cwd)

        try:
            result = asyncio.run(self.deployment.runtime.execute(Command(**arguments)))
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                # The coding agent consumes one observation string, while the
                # task harness needs the streams separately for diagnostics.
                # 编程智能体使用一个观察字符串，而任务运行框架需要分开的输出流
                # 进行诊断。
                "output": result.stdout,
                "returncode": result.exit_code,
                "exception_info": "",
            }
            if result.stderr:
                output["output"] += result.stderr
        except Exception as e:
            # A command can fail for reasons worth reporting to the model, but a
            # sandbox that is gone is terminal: every later command fails the
            # same way, so raise rather than let the caller keep going.
            # 命令可能因值得报告给模型的原因而失败，但沙箱消失是终止性故障：
            # 后续每条命令都会以同样方式失败，因此应抛出异常，而不是让调用方
            # 继续运行。
            if not self.is_alive():
                raise RuntimeError(
                    "The sandbox is no longer running, so no further commands can "
                    f"be executed. Last error: {e}"
                ) from e

            # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/extra/swerex_modal.py#L82-L87
            # 注意（来源）：https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/extra/swerex_modal.py#L82-L87
            # Same keys as the success path, so callers never have to branch on
            # which one they got.
            # 使用与成功路径相同的键，因此调用方无需根据获得的结果类型分别处理。
            output = {
                "stdout": "",
                "stderr": str(e) if str(e) else "",
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        return output

    def stop(self, timeout: float = 10):
        """Shut down the runtime and terminate the sandbox.

        Args:
            timeout: Seconds allowed for each of the shutdown and the
                termination steps.

        关闭运行时并终止沙箱。

        参数：
            timeout：关闭和终止两个步骤各自允许的秒数。
        """

        async def _stop():
            # ModalDeployment.stop() has an inverted poll() check and only
            # terminates sandboxes that have *already* exited, so a live sandbox
            # leaks until deployment_timeout. Terminate it explicitly. Grab the
            # reference first: stop() clears _sandbox.
            # ModalDeployment.stop() 的 poll() 检查写反了，只会终止*已经*退出的
            # 沙箱，因此仍在运行的沙箱会泄漏到 deployment_timeout。这里显式终止
            # 沙箱。先获取引用，因为 stop() 会清除 _sandbox。
            sandbox = self.deployment._sandbox
            try:
                await asyncio.wait_for(self.deployment.stop(), timeout=timeout)
            finally:
                if sandbox is not None:
                    await asyncio.wait_for(sandbox.terminate.aio(), timeout=timeout)

        asyncio.run(_stop())

    def tunnel_url(self, port: int) -> str:
        """Return the public URL for a port forwarded when the sandbox started.
        返回沙箱启动时转发端口的公开 URL。
        """

        async def _tunnel_url() -> str:
            tunnels = await self.deployment.sandbox.tunnels.aio()
            try:
                return tunnels[port].url
            except KeyError as exc:
                forwarded = ", ".join(str(item) for item in sorted(tunnels)) or "none"
                raise ValueError(
                    f"Port {port} was not forwarded. Available forwarded ports: {forwarded}."
                ) from exc

        return asyncio.run(_tunnel_url())

    def __enter__(self):
        """Enter a `with` block. The sandbox is already running by this point.
        进入 `with` 代码块。此时沙箱已在运行。
        """
        return self

    def __exit__(self, *exc):
        """Terminate the sandbox on leaving a `with` block, including on error.
        离开 `with` 代码块时终止沙箱，包括发生错误的情况。
        """
        self.stop()
