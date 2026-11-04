"""Run a candidate patch against a task's testbed and grade the result.
在任务的测试环境中运行候选补丁并对结果评分。
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from assignment.task import Task
from assignment.env import Environment
from assignment.eval.instances import Instance, published_images
from assignment.utils.image import build_testbed_image

logger = logging.getLogger(__name__)

TESTBED = "/testbed"

# `pytest -rA` ends its run with a short summary of every test, one line each:
#     PASSED tests/test_chess_server.py::test_reset_endpoint_starts_a_new_game
#     FAILED tests/test_chess_server.py::test_checkmating_move - assert 500 == 200
# `pytest -rA` 结束运行时会输出每个测试的简短摘要，每项一行：
#     PASSED tests/test_chess_server.py::test_reset_endpoint_starts_a_new_game
#     FAILED tests/test_chess_server.py::test_checkmating_move - assert 500 == 200
_SUMMARY_LINE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE)

class TestStatus(str, Enum):
    __test__ = False  # Not a pytest test class, despite the name.
    # 尽管名称相似，但这不是 pytest 测试类。

    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    XFAIL = "XFAIL"
    XPASS = "XPASS"
    MISSING = "MISSING"
    """The test never reported a result, usually because collection failed.
    测试从未报告结果，通常是因为收集失败。
    """

@dataclass(frozen=True)
class EvaluationSpec:
    """Tests and grading metadata for one task.
    一个任务的测试和评分元数据。
    """

    task_id: str
    test_patch: str
    test_cmd: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    root: Path
    framework: str = "pytest"
    cwd: str = TESTBED

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationSpec":
        """Load ``evaluation.json`` and its sibling test patch.
        加载 ``evaluation.json`` 及其同目录的测试补丁。
        """

        path = Path(path).resolve()
        spec_path = path if path.is_file() else path / "evaluation.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"No evaluation.json found at {spec_path}")
        root = spec_path.parent
        spec = json.loads(spec_path.read_text())
        missing = {"task_id", "test_patch", "test_cmd", "fail_to_pass"} - set(spec)
        if missing:
            raise ValueError(
                f"{spec_path} is missing required keys: {', '.join(sorted(missing))}"
            )
        test_patch_path = root / spec["test_patch"]
        if not test_patch_path.is_file():
            raise FileNotFoundError(
                f"{spec_path} points at test_patch={spec['test_patch']}, which does not exist"
            )
        return cls(
            task_id=spec["task_id"],
            test_patch=test_patch_path.read_text(),
            test_cmd=spec["test_cmd"],
            fail_to_pass=tuple(spec["fail_to_pass"]),
            pass_to_pass=tuple(spec.get("pass_to_pass", ())),
            root=root,
            framework=spec.get("framework", "pytest"),
            cwd=spec.get("cwd", TESTBED),
        )

@dataclass
class Report:
    """The outcome of one evaluation run.
    一次评估运行的结果。
    """

    task_id: str
    patch_applied: bool
    results: dict[str, TestStatus] = field(default_factory=dict)
    error: str = ""
    """Set when the run could not produce results at all.
    当运行完全无法产生结果时设置。
    """

    fail_to_pass: dict[str, TestStatus] = field(default_factory=dict)
    pass_to_pass: dict[str, TestStatus] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        """True when every graded test passed, which is what "fixed" means.
        当所有评分测试都通过时为 True，这就是“已修复”的含义。
        """
        if self.error or not self.fail_to_pass:
            return False
        graded = {**self.fail_to_pass, **self.pass_to_pass}
        return all(status is TestStatus.PASSED for status in graded.values())

    def summary(self) -> str:
        lines = [f"task: {self.task_id}"]
        if self.error:
            lines.append(f"ERROR: {self.error}")
            return "\n".join(lines)

        for label, group in (("FAIL_TO_PASS", self.fail_to_pass), ("PASS_TO_PASS", self.pass_to_pass)):
            passed = sum(1 for status in group.values() if status is TestStatus.PASSED)
            lines.append(f"\n{label} ({passed}/{len(group)})")
            for test, status in group.items():
                mark = "PASS" if status is TestStatus.PASSED else status.value
                lines.append(f"  [{mark:>7}] {test}")

        lines.append(f"\nRESOLVED: {'yes' if self.resolved else 'no'}")
        return "\n".join(lines)

# SWE-bench images install the repository under test into a conda environment
# named `testbed`. The image's default `python` is conda's base environment,
# which has neither pytest nor the repository, so the interpreter has to be
# named explicitly rather than relying on PATH.
# SWE-bench 镜像将被测仓库安装到名为 `testbed` 的 conda 环境中。镜像默认的
# `python` 是 conda 的基础环境，其中既没有 pytest 也没有该仓库，因此必须显式
# 指定解释器，而不能依赖 PATH。
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"

# Django does not use pytest: its runner prints one line per test, as
# `name (dotted.Case) ... ok`, with the status at the end. FAIL and ERROR are
# also re-reported further down as `FAIL: name`.
# Django 不使用 pytest：其运行器为每个测试打印一行，格式如
# `name (dotted.Case) ... ok`，状态位于行尾。FAIL 和 ERROR 还会在后文以
# `FAIL: name` 的形式再次报告。
_DJANGO_LINE = re.compile(r"^(.+?) \.\.\. (ok|OK|FAIL|ERROR|skipped.*)$", re.MULTILINE)
_DJANGO_HEADER = re.compile(r"^(FAIL|ERROR): (\S+(?: \(\S+\))?)$", re.MULTILINE)

def find_testbed_python(env: Environment) -> str:
    """The interpreter that has the repository under test installed.

    Args:
        env: A running sandbox.

    Returns:
        `TESTBED_PYTHON` when the conda environment is present, otherwise
        whatever `python` resolves to on PATH.

    安装了被测仓库的解释器。

    参数：
        env：正在运行的沙箱。

    返回：
        conda 环境存在时返回 `TESTBED_PYTHON`，否则返回 PATH 上 `python`
        解析到的解释器。
    """
    if env.execute(f"test -x {TESTBED_PYTHON}")["returncode"] == 0:
        return TESTBED_PYTHON

    logger.warning("%s not found; falling back to python on PATH", TESTBED_PYTHON)
    return "python"

def parse_django_report(output: str) -> dict[str, TestStatus]:
    """Extract per-test statuses from Django's `runtests.py --verbosity 2` output.

    Args:
        output: Combined stdout and stderr of the test command.

    Returns:
        A mapping of test id to status, keyed as Django prints it, for example
        `test_f_expression (queries.test_bulk_update.BulkUpdateTests)`.

    从 Django 的 `runtests.py --verbosity 2` 输出中提取每个测试的状态。

    参数：
        output：测试命令合并后的 stdout 和 stderr。

    返回：
        测试 ID 到状态的映射，键与 Django 打印的格式一致，例如
        `test_f_expression (queries.test_bulk_update.BulkUpdateTests)`。
    """
    statuses = {
        "ok": TestStatus.PASSED,
        "OK": TestStatus.PASSED,
        "FAIL": TestStatus.FAILED,
        "ERROR": TestStatus.ERROR,
    }
    results: dict[str, TestStatus] = {}

    for test, outcome in _DJANGO_LINE.findall(output):
        if outcome.startswith("skipped"):
            results[test.strip()] = TestStatus.SKIPPED
        else:
            results[test.strip()] = statuses[outcome]

    # A test whose line was interrupted by other output never matched above, but
    # Django repeats every failure in a `FAIL: <test>` header further down.
    # 如果测试所在行被其他输出打断，上面的匹配会遗漏它；但 Django 会在后文的
    # `FAIL: <test>` 标题中重复报告每个失败。
    for outcome, test in _DJANGO_HEADER.findall(output):
        results[test.strip()] = statuses[outcome]

    return results

def parse_report(output: str, framework: str) -> dict[str, TestStatus]:
    """Parse test output with the parser for that framework.
    使用对应框架的解析器解析测试输出。
    """
    if framework == "django":
        return parse_django_report(output)
    return parse_pytest_report(output)

def parse_pytest_report(output: str) -> dict[str, TestStatus]:
    """Extract per-test statuses from `pytest -rA` output.

    Args:
        output: Combined stdout and stderr of the test command.

    Returns:
        A mapping of test id to status, keyed exactly as pytest prints it.

    从 `pytest -rA` 输出中提取每个测试的状态。

    参数：
        output：测试命令合并后的 stdout 和 stderr。

    返回：
        测试 ID 到状态的映射，键与 pytest 打印的格式完全一致。
    """
    return {test: TestStatus(status) for status, test in _SUMMARY_LINE.findall(output)}

def _write_remote_file(env: Environment, content: str, remote_path: str) -> None:
    """Put a local string on the sandbox filesystem without quoting hazards.
    将本地字符串写入沙箱文件系统，避免引号转义问题。
    """
    encoded = base64.b64encode(content.encode()).decode()
    result = env.execute(f"echo {encoded} | base64 -d > {remote_path}")
    if result["returncode"] != 0:
        raise RuntimeError(f"Could not write {remote_path}: {result['stderr'].strip()}")

def _apply_patch(env: Environment, patch: str, remote_path: str) -> tuple[bool, str]:
    """Apply a unified diff inside the testbed.

    Returns:
        Whether it applied, and the error output when it did not.

    在测试环境内应用 unified diff。

    返回：
        是否应用成功；失败时同时返回错误输出。
    """
    _write_remote_file(env, patch, remote_path)
    # -p1 matches `git diff` output; fall back to `patch` for diffs git rejects
    # (fuzzy context, missing index lines) since agents produce those routinely.
    # -p1 与 `git diff` 输出匹配；对于 git 拒绝的 diff（模糊上下文、缺少索引行），
    # 回退到 `patch`，因为智能体经常生成这类 diff。
    result = env.execute(f"git apply -v {remote_path}", cwd=TESTBED)
    if result["returncode"] == 0:
        return True, ""

    logger.debug("git apply failed, retrying with patch(1): %s", result["stderr"].strip())
    fallback = env.execute(f"patch --batch --fuzz=5 -p1 -i {remote_path}", cwd=TESTBED)
    if fallback["returncode"] == 0:
        return True, ""
    return False, (result["stderr"] or fallback["stdout"] or fallback["stderr"]).strip()

def _grade(
    env: Environment,
    report: Report,
    test_patch: str,
    test_cmd: str,
    fail_to_pass: tuple[str, ...],
    pass_to_pass: tuple[str, ...],
    patch: str | None,
    timeout: float,
    framework: str = "pytest",
) -> Report:
    """Apply the patches in a running testbed, run the tests, and fill in `report`.

    Shared by both entry points, so a locally built testbed and a prebuilt
    SWE-bench image are graded by exactly the same rules.

    The order matters and mirrors SWE-bench: the candidate patch goes on first,
    against a repository that has never seen the graded test, and the test patch
    is applied on top only afterwards. A candidate cannot satisfy the test by
    editing it.

    在正在运行的测试环境中应用补丁、运行测试并填写 `report`。

    两个入口共用此函数，因此本地构建的测试环境和预构建的 SWE-bench 镜像采用
    完全相同的评分规则。

    顺序很重要，并与 SWE-bench 一致：先将候选补丁应用到从未见过评分测试的仓库，
    然后才在其上应用测试补丁。候选补丁不能通过编辑测试来使测试通过。
    """
    if patch is not None:
        applied, error = _apply_patch(env, patch, "/tmp/candidate.diff")
        report.patch_applied = applied
        if not applied:
            report.error = f"Candidate patch did not apply:\n{error}"
            return report
        logger.info("Applied candidate patch")

    applied, error = _apply_patch(env, test_patch, "/tmp/test.diff")
    if not applied:
        report.error = (
            f"Test patch did not apply:\n{error}\n"
            "The candidate patch probably touched a file the test patch also edits."
        )
        return report
    logger.info("Applied test patch")

    result = env.execute(test_cmd, cwd=TESTBED, timeout=timeout)
    output = f"{result['stdout']}\n{result['stderr']}"
    report.results = parse_report(output, framework)

    if not report.results:
        report.error = (
            f"Test command produced no results (exit {result['returncode']}):\n"
            f"{output.strip()[-2000:]}"
        )
        return report

    report.fail_to_pass = {
        test: report.results.get(test, TestStatus.MISSING) for test in fail_to_pass
    }
    report.pass_to_pass = {
        test: report.results.get(test, TestStatus.MISSING) for test in pass_to_pass
    }
    return report

def resolve_image(spec, task: Task | None, strict: bool):
    """The image to grade in: published when one exists, otherwise built.

    Args:
        spec: The evaluation being run, consulted for its `task_id`.
        task: The task to build from, required when nothing is published.
        strict: Passed to the build; refuses a checkout off the base commit.

    Returns:
        A published image name, or a `modal.Image` built from the task.

    Raises:
        ValueError: If no image is published and no task was given to build.

    用于评分的镜像：有已发布镜像时使用它，否则进行构建。

    参数：
        spec：正在运行的评估，查询其 `task_id`。
        task：用于构建的任务；没有已发布镜像时必须提供。
        strict：传递给构建过程；拒绝不在基准提交上的检出。

    返回：
        已发布的镜像名称，或根据任务构建的 `modal.Image`。

    抛出：
        ValueError：没有已发布镜像，也未提供用于构建的任务。
    """
    published = published_images().get(spec.task_id)
    if published:
        logger.info("Using published image for %s: %s", spec.task_id, published)
        return published

    if task is None:
        raise ValueError(
            f"{spec.task_id!r} has no published image, so a task is needed to build one."
        )
    return build_testbed_image(task, strict=strict)

def evaluate(
    spec,
    patch: str | None = None,
    task: Task | None = None,
    timeout: float = 600,
    strict: bool = True,
) -> Report:
    """Build or fetch the testbed, apply a candidate patch, and grade it.

    Handles both kinds of evaluation. A vendored SWE-bench instance names an
    image that already exists; a local task is built from its Dockerfile, with
    its checkout verified against the base commit first.

    The order matters and mirrors SWE-bench: the candidate patch goes on first,
    against a repository that has never seen the graded test, and the test patch
    is applied on top only afterwards. A candidate cannot satisfy the test by
    editing it.

    Args:
        spec: What to run and how to score it: an `EvaluationSpec` or an
            `Instance`. Both carry `task_id`, `test_patch`, `test_cmd`,
            `fail_to_pass`, `pass_to_pass`, `framework`, and `cwd`. A `test_cmd`
            containing `{python}` gets the testbed interpreter substituted in;
            one without is run as written.
        patch: A unified diff to apply before testing. None runs the unmodified
            base commit, the baseline every agent starts from.
        task: The task to build an image from. Only needed when the spec has no
            published image.
        timeout: Seconds allowed for the test command.
        strict: Refuse to build when the local checkout does not match the
            task's base commit, which would silently invalidate the result.

    Returns:
        A report. Failures to apply a patch or to run the tests are recorded in
        the report rather than raised.

    Raises:
        ValueError: If `task` is given but describes a different task than the
            spec, or if no image is available and none can be built.

    构建或获取测试环境，应用候选补丁并评分。

    处理两类评估。随仓库提供的 SWE-bench 实例指定一个已存在的镜像；本地任务
    根据其 Dockerfile 构建，并先核对检出是否与基准提交一致。

    顺序很重要，并与 SWE-bench 一致：先将候选补丁应用到从未见过评分测试的仓库，
    然后才在其上应用测试补丁。候选补丁不能通过编辑测试来使测试通过。

    参数：
        spec：运行内容和评分方式，为 `EvaluationSpec` 或 `Instance`。两者都
            包含 `task_id`、`test_patch`、`test_cmd`、`fail_to_pass`、
            `pass_to_pass`、`framework` 和 `cwd`。包含 `{python}` 的
            `test_cmd` 会替换为测试环境解释器；不包含时按原样运行。
        patch：测试前应用的 unified diff。None 表示运行未修改的基准提交，
            即每个智能体的起始基线。
        task：用于构建镜像的任务。仅在 spec 没有已发布镜像时需要。
        timeout：测试命令允许的秒数。
        strict：本地检出与任务的基准提交不一致时拒绝构建，否则会在不提示的
            情况下使结果无效。

    返回：
        报告。补丁应用或测试运行失败会记录在报告中，而不是抛出异常。

    抛出：
        ValueError：提供了 `task`，但它与 spec 描述的任务不同；或没有可用镜像
        且无法构建。
    """
    if task is not None and spec.task_id != task.id:
        raise ValueError(f"Evaluation is for {spec.task_id!r}, not task {task.id!r}.")

    report = Report(task_id=spec.task_id, patch_applied=patch is None)
    image = resolve_image(spec, task, strict)

    with Environment(image=image, cwd=spec.cwd, runtime_timeout=timeout) as env:
        # Only substituted when the command asks for it, so a spec that names
        # its own interpreter keeps it.
        # 仅在命令要求时替换，因此指定了自己解释器的 spec 会保留该解释器。
        test_cmd = spec.test_cmd
        if "{python}" in test_cmd:
            test_cmd = test_cmd.replace("{python}", find_testbed_python(env))

        return _grade(
            env,
            report,
            test_patch=spec.test_patch,
            test_cmd=test_cmd,
            fail_to_pass=spec.fail_to_pass,
            pass_to_pass=spec.pass_to_pass,
            patch=patch,
            timeout=timeout,
            framework=spec.framework,
        )
