"""The SWE-bench instances this repository evaluates against.

Both are vendored under `tasks/swebench/`: the issue text, the test patch, the
graded test ids, and the published image name. Nothing is fetched at run time,
so evaluation works offline and cannot drift when the upstream dataset or its
serving API changes.

Adding an instance means adding a directory here, not reaching for a dataset.

本仓库用于评估的 SWE-bench 实例。

实例随仓库放在 `tasks/swebench/` 下，包括问题文本、测试补丁、参与评分的测试
ID 和已发布镜像名称。运行时不获取任何内容，因此评估可离线进行，也不会因上游
数据集或其服务 API 变化而漂移。

添加实例意味着在这里添加一个目录，而不是引入数据集。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TASKS = Path(__file__).resolve().parents[3] / "tasks" / "swebench"

@dataclass(frozen=True)
class Instance:
    """One benchmark instance, read from its `instance.json`.

    Attributes:
        instance_id: The benchmark's id, for example `django__django-15368`.
        repo: Upstream repository the issue belongs to.
        version: Repository version, which fixes how its tests are run.
        base_commit: The commit the published image is built at.
        image: The published evaluation image, already holding the repository
            with its dependencies installed.
        framework: Which runner the repository uses, `django` or `pytest`. It
            selects both the test command and the output parser.
        test_cmd: The command to run, with `{python}` where the interpreter goes.
        test_patch: The diff adding the graded tests.
        problem_statement: The issue text shown to the agent.
        fail_to_pass: Tests that must go from failing to passing.
        pass_to_pass: Tests that must keep passing.
        cwd: Working directory the tests run in.

    从实例的 `instance.json` 读取的一个基准测试实例。

    属性：
        instance_id：基准测试 ID，例如 `django__django-15368`。
        repo：问题所属的上游仓库。
        version：仓库版本，决定其测试的运行方式。
        base_commit：已发布镜像构建时使用的提交。
        image：已发布的评估镜像，其中已包含安装好依赖的仓库。
        framework：仓库使用的测试运行器，即 `django` 或 `pytest`，同时决定
            测试命令和输出解析器。
        test_cmd：要运行的命令，`{python}` 表示解释器所在位置。
        test_patch：添加评分测试的 diff。
        problem_statement：展示给智能体的问题文本。
        fail_to_pass：必须从失败变为通过的测试。
        pass_to_pass：必须保持通过的测试。
        cwd：运行测试的工作目录。
    """

    instance_id: str
    repo: str
    version: str
    base_commit: str
    image: str
    framework: str
    test_cmd: str
    test_patch: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    cwd: str = "/testbed"

    @property
    def task_id(self) -> str:
        """The id under the name `evaluate` uses, shared with `EvaluationSpec`.
        `evaluate` 使用的 ID，与 `EvaluationSpec` 共享同一名称。
        """
        return self.instance_id

    @classmethod
    def load(cls, path: str | Path) -> Instance:
        """Read an instance from its directory, or from its `instance.json`.
        从实例目录或其 `instance.json` 读取实例。
        """
        path = Path(path).resolve()
        spec_path = path if path.is_file() else path / "instance.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"No instance.json found at {spec_path}")

        root = spec_path.parent
        spec = json.loads(spec_path.read_text())
        return cls(
            instance_id=spec["instance_id"],
            repo=spec["repo"],
            version=spec["version"],
            base_commit=spec["base_commit"],
            image=spec["image"],
            framework=spec["framework"],
            test_cmd=spec["test_cmd"],
            test_patch=(root / spec["test_patch"]).read_text(),
            problem_statement=(root / spec["problem_statement"]).read_text(),
            fail_to_pass=tuple(spec["fail_to_pass"]),
            pass_to_pass=tuple(spec["pass_to_pass"]),
            cwd=spec.get("cwd", "/testbed"),
        )

def published_images() -> dict[str, str]:
    """Task id to published image, for every instance that ships one.

    `evaluate` consults this before building: an instance listed here already
    has an image on Docker Hub, so nothing needs to be built from source.

    对每个附带已发布镜像的实例，返回任务 ID 到镜像的映射。

    `evaluate` 在构建前查询此映射：列在这里的实例已经在 Docker Hub 上有镜像，
    因此无需从源代码构建。
    """
    return {name: load(name).image for name in available()}

def available() -> list[str]:
    """The instance ids this repository can evaluate, in sorted order.
    本仓库可评估的实例 ID，按排序顺序返回。
    """
    return sorted(path.name for path in TASKS.iterdir() if (path / "instance.json").is_file())

def load(instance_id: str) -> Instance:
    """Load a vendored instance by id.

    Raises:
        KeyError: If the id is not one of the vendored instances, listing what is.

    按 ID 加载随仓库提供的实例。

    抛出：
        KeyError：如果该 ID 不属于随仓库提供的实例，错误中会列出可用实例。
    """
    directory = TASKS / instance_id
    if not (directory / "instance.json").is_file():
        raise KeyError(f"{instance_id!r} is not vendored. Available: {', '.join(available())}")
    return Instance.load(directory)
