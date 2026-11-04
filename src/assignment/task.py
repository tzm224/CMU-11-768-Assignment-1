"""Public task definitions used to build an isolated testbed.

Shared by both halves of the assignment. Private tests and grading metadata are
deliberately not part of this object or its task directory.

用于构建隔离测试环境的公开任务定义。

由作业的两个部分共享。私有测试和评分元数据有意不包含在此对象或其任务目录中。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Task:
    """One evaluation task, loaded from a task directory.

    The directory holds a `task.json` alongside the public files it names: a
    Dockerfile that builds the testbed and the issue shown to the agent.

    The testbed is built from `source`, a local checkout of the repository —
    normally a submodule. `repo` and `base_commit` record where that checkout is
    supposed to come from, and are checked against it before every build.

    从任务目录加载的一个评估任务。

    目录中包含一个 `task.json`，以及其中指定的公开文件：用于构建测试环境的
    Dockerfile 和展示给智能体的问题。

    测试环境根据 `source`（仓库的本地检出，通常是子模块）构建。`repo` 和
    `base_commit` 记录该检出的预期来源，并在每次构建前进行核对。
    """

    id: str
    repo: str
    base_commit: str
    source: Path
    dockerfile: Path
    problem_statement: str
    pins: tuple[str, ...]
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> Task:
        """Read a task from its directory, or directly from its `task.json`.

        Args:
            path: The task directory, or the `task.json` inside it.

        Returns:
            The task, with its sibling files already read into memory.

        从任务目录读取任务，或直接读取其 `task.json`。

        参数：
            path：任务目录，或其中的 `task.json`。

        返回：
            任务对象，其同目录文件已读入内存。
        """
        path = Path(path).resolve()
        spec_path = path if path.is_file() else path / "task.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"No task.json found at {spec_path}")
        root = spec_path.parent

        spec = json.loads(spec_path.read_text())
        missing = {
            "id",
            "repo",
            "base_commit",
            "source",
            "dockerfile",
            "problem_statement",
        } - set(spec)
        if missing:
            raise ValueError(f"{spec_path} is missing required keys: {', '.join(sorted(missing))}")

        def sibling(key: str) -> Path:
            target = root / spec[key]
            if not target.is_file():
                raise FileNotFoundError(f"{spec_path} points at {key}={spec[key]}, which does not exist")
            return target

        # The source checkout is resolved but not required to exist yet: a fresh
        # clone has an empty submodule until `git submodule update --init`, and
        # the error for that should come from the build, which can say so.
        # 会解析源检出的路径，但此时不要求它已存在：新克隆的仓库在执行
        # `git submodule update --init` 前子模块为空；相应错误应由构建步骤给出，
        # 因为构建步骤能明确说明这一点。
        source = (root / spec["source"]).resolve()

        # Exact versions for the testbed. Optional: a task without them takes
        # whatever its Dockerfile resolves.
        # 测试环境使用的确切版本。此项可选：未提供这些版本的任务会使用其
        # Dockerfile 解析出的版本。
        pins = tuple(spec.get("pins", ()))

        return cls(
            id=spec["id"],
            repo=spec["repo"],
            base_commit=spec["base_commit"],
            source=source,
            dockerfile=sibling("dockerfile"),
            problem_statement=sibling("problem_statement").read_text(),
            pins=pins,
            root=root,
        )
