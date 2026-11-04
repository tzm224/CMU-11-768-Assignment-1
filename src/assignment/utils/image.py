"""Build a task's testbed image on Modal from a local checkout.

The repository under test is copied in from `task.source` — normally the
`chess_app/` submodule — rather than cloned inside the build. That keeps the
build offline, works with a private repository without any credential, and
avoids a network round trip on every rebuild.

The cost is that the image reflects whatever is on disk, so `verify_source`
checks the `chess_app` source before every build to confirm it matches the
source the assignment intends people to fix. A testbed silently built from an
already-fixed working tree would make a broken agent look like it passed.

根据本地检出在 Modal 上构建任务的测试环境镜像。

被测仓库从 `task.source`（通常是 `chess_app/` 子模块）复制进来，而不是在构建
过程中克隆。这使构建保持离线，无需凭据即可使用私有仓库，并避免每次重新构建
都进行网络往返。

代价是镜像会反映磁盘上的任何内容，因此 `verify_source` 会在每次构建前检查
`chess_app` 源代码，确认它与作业要求修复的源代码一致。如果测试环境在未提示的
情况下根据已修复的工作树构建，就会使有缺陷的智能体看起来通过了测试。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import modal

from assignment.task import Task

logger = logging.getLogger(__name__)

# Local caches, credentials, and git metadata never belong in the testbed. The
# .git directory in particular is a gitlink file in a submodule checkout and
# would be meaningless inside the container; the build makes a fresh repository
# instead.
# 本地缓存、凭据和 git 元数据绝不应进入测试环境。尤其是在子模块检出中，.git
# 是一个 gitlink 文件，在容器内没有意义；构建过程会改为创建一个新的仓库。
IGNORED_NAMES = {".git", ".venv", ".pytest_cache", "__pycache__", ".env", ".DS_Store"}

class SourceMismatch(Exception):
    """The local checkout is not the commit the task is defined against.
    本地检出不是任务定义所基于的提交。
    """

def _git(source: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True)

def is_ignored(path: Path) -> bool:
    """Whether a file should be kept out of the build context.
    文件是否应排除在构建上下文之外。
    """
    return any(part in IGNORED_NAMES for part in path.parts) or path.suffix == ".pyc"

def verify_source(task: Task, strict: bool = True) -> None:
    """Check that the local checkout matches the task's base commit.

    Args:
        task: The task whose `source` to check.
        strict: Raise on a mismatch. When False, mismatches are logged as
            warnings and the build proceeds — useful when deliberately testing
            a modified checkout.

    Raises:
        SourceMismatch: The checkout is missing, on the wrong commit, or dirty.

    检查本地检出是否与任务的基准提交一致。

    参数：
        task：要检查其 `source` 的任务。
        strict：不匹配时抛出异常。为 False 时，将不匹配记录为警告并继续构建，
            适用于有意测试修改后检出的情况。

    抛出：
        SourceMismatch：检出缺失、位于错误提交或存在未提交改动。
    """

    def complain(message: str) -> None:
        if strict:
            raise SourceMismatch(message)
        logger.warning("%s (continuing: strict=False)", message)

    if not task.source.is_dir():
        raise SourceMismatch(
            f"{task.source} does not exist. If it is a submodule, run `git submodule update --init`."
        )

    head = _git(task.source, "rev-parse", "HEAD")
    if head.returncode != 0:
        complain(f"{task.source} is not a git checkout: {head.stderr.strip()}")
        return

    if head.stdout.strip() != task.base_commit:
        complain(
            f"{task.source} is at {head.stdout.strip()[:12]}, but {task.id} is defined against "
            f"{task.base_commit[:12]}. The testbed would not be the task's base commit."
        )

    dirty = _git(task.source, "status", "--porcelain")
    if dirty.stdout.strip():
        # Each line is a status code then the path. The code's width varies, so
        # split on whitespace rather than slicing at a fixed offset.
        # 每行依次是状态码和路径。状态码的宽度会变化，因此按空白拆分，而不是
        # 在固定偏移处切片。
        paths = [line.strip().split(None, 1)[-1] for line in dirty.stdout.strip().splitlines()[:5]]
        changed = ", ".join(paths)
        complain(
            f"{task.source} has uncommitted changes ({changed}). The testbed would contain them, "
            "which silently invalidates the evaluation if one of them is the fix."
        )

def build_testbed_image(task: Task, strict: bool = True, force_build: bool = False) -> modal.Image:
    """Build the image holding the repository under test at its base commit.

    Args:
        task: The task whose Dockerfile and source checkout to build from.
        strict: Refuse to build when the checkout does not match the task's
            base commit. See `verify_source`.
        force_build: Skip Modal's build cache.

    Returns:
        A Modal image with the repository installed at /testbed.

    构建包含基准提交处被测仓库的镜像。

    参数：
        task：使用其 Dockerfile 和源检出进行构建的任务。
        strict：检出与任务的基准提交不一致时拒绝构建。参见 `verify_source`。
        force_build：跳过 Modal 的构建缓存。

    返回：
        将仓库安装到 /testbed 的 Modal 镜像。
    """
    verify_source(task, strict=strict)

    logger.info("Building %s from %s at %s", task.id, task.source, task.base_commit[:12])
    image = modal.Image.from_dockerfile(
        str(task.dockerfile),
        context_dir=str(task.source),
        force_build=force_build,
        ignore=is_ignored,
    )

    # Pin last, deliberately. Modal appends its own dependency install after the
    # Dockerfile's commands on image builder versions <= 2024.10, and the 2023.12
    # requirements file pins fastapi==0.88.0, which pulls starlette down to 0.22
    # and breaks every test using TestClient. Chaining here puts this layer on
    # top of that one, so the task's versions are the ones that survive.
    # 有意最后固定版本。在镜像构建器版本 <= 2024.10 时，Modal 会在 Dockerfile
    # 命令之后追加自己的依赖安装，而 2023.12 的 requirements 文件固定了
    # fastapi==0.88.0，会将 starlette 降至 0.22，导致所有使用 TestClient 的
    # 测试失败。在这里链式调用会将此层置于那一层之上，使任务指定的版本最终生效。
    if task.pins:
        logger.info("Pinning %d packages on top of the built image", len(task.pins))
        image = image.pip_install(*task.pins)

    return image
