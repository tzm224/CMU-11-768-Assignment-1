# CMU 11-768 作业 1——参考实现

[English](README.md) | [中文](README_CN.md)

本仓库包含我完成的 CMU 11-768 作业 1“构建智能体运行框架”的实现和实验产物，供学习参考。它不是可以独立使用的原始作业仓库。

**学术诚信：** 使用本仓库前，请先阅读课程规定。不要将本实现、补丁或交互轨迹作为你自己的作业提交。建议先独立完成作业，再用本仓库比较设计方案和调试方法。

## 实现内容

- 与领域无关的 ReAct 循环，包含操作/观察历史和步数限制。
- OpenAI Chat Completions 函数工具定义，以及通过调用 ID 关联的工具观察。
- 编程智能体工具：沙箱命令、技能调用和补丁提交。
- 通过 YAML frontmatter 元数据实现技能渐进式披露。
- 由模型生成的工作记忆压缩，同时保留最近的完整工具调用步骤。
- 国际象棋智能体工具：实时走子、无状态模拟、沙箱 Python 和技能。
- 在全新 Modal 测试环境中生成并验收补丁及交互轨迹。

作业说明：[English](ASSIGNMENT.md) | [中文](ASSIGNMENT_CN.md)。

原始作业脚手架位于 [cmu-agents/assignment-1](https://github.com/cmu-agents/assignment-1)。开发时以脚手架提交 `67498d8` 为参考版本。

## 仓库导航

### 源代码

主要实现位于 [`src/assignment/agent/`](src/assignment/agent/)：

- `base.py`：共享智能体状态、ReAct 生命周期、技能和上下文压缩。
- `code_agent.py`：编程工具注册、调度和提交行为。
- `chess_agent.py`：国际象棋工具注册、调度和实时状态更新。
- `chess_tools.py`：国际象棋 API 辅助函数和沙箱 Python 执行。
- `tools.py`：OpenAI 函数工具规范。

其他作业运行代码位于 [`src/assignment/`](src/assignment/)，提示词模板位于 [`src/assignment/prompts.py`](src/assignment/prompts.py)。

### 实验产物

[`artifacts/`](artifacts/) 包含补丁、交互轨迹、棋局结果和实验报告。建议先阅读下面两份报告，而不是直接打开大型 JSON 轨迹：

- [Token 用量分析](artifacts/token-usage-analysis.md)
- [观察条件 A/B 实验](artifacts/observation-experiment.md)

[`assignment-1-submission.zip`](assignment-1-submission.zip) 是课程提交用压缩包，包含要求的源码和现有提交产物。仓库还额外保留了完整上下文基线轨迹和程序化技能实验结果，便于核查报告。

[AI_USAGE.md](AI_USAGE.md) 记录了本作业中的 AI 使用情况。

## 实验结果

- 公开离线测试：**15 项通过**；4 项 Modal 标记测试被排除。
- 第一部分国际象棋补丁：**RESOLVED**——FAIL_TO_PASS 1/1，PASS_TO_PASS 9/9。
- SWE-bench 压缩补丁：**RESOLVED**——FAIL_TO_PASS 1/1，PASS_TO_PASS 29/29。
- SWE-bench 完整上下文基线：**RESOLVED**——FAIL_TO_PASS 1/1，PASS_TO_PASS 29/29。
- 压缩轨迹：包含 **9 次压缩事件**，每次都降低了估算的活跃上下文。
- 上述两次运行的模型总用量：**压缩 130,738 token**，**基线 136,932 token**，已计入额外压缩请求的开销。
- 基础国际象棋运行：`game_over: true`，**白方获胜**，共有 44 次模型响应。
- DeepSeek 观察实验：两组均达到 `game_over: true`；无合法走法组在 31 次调用中出现 1 次非法走法，含合法走法组在 31 次调用中出现 0 次。
- 程序化技能轨迹：先调用 `invoke_skill`，随后 Python 代码使用 `simulate_move` 搜索，并通过 `play_move` 提交走法。

### 尚未完成的实验条件

两组 GPT-OSS 条件**没有运行**。可用的 DeepSeek 端点不提供 `openai/gpt-oss-120b`，同时没有可用的 GPT-OSS 兼容端点。我没有把 DeepSeek 输出改名或伪装成 GPT-OSS。观察实验报告中也记录了这一限制。

## 配合原始作业仓库使用

本参考仓库刻意不包含课程测试、任务定义、`chess_app` 子模块或凭据。要复现实验环境，请先获取原始作业仓库，再覆盖参考实现源码：

```bash
git clone https://github.com/cmu-agents/assignment-1.git
cd assignment-1
git checkout 67498d8

git clone https://github.com/tzm224/cmu-11768-assignment1.git reference
cp -r reference/src/assignment/. src/assignment/
mkdir -p artifacts
cp -r reference/artifacts/. artifacts/
```

安装并验证环境：

```bash
make setup
uv run modal setup
cp .env.example .env
# 在本地编辑 .env，填写你自己的接口地址、API Key 和模型。
make doctor
make test
```

绝不要提交 `.env` 或 API Key。本仓库的 DeepSeek 实验使用官方 `https://api.deepseek.com` 端点和 `deepseek-flash` 模型。使用该端点时，需要显式覆盖原始 Makefile 中的课程网关模型名，例如：

```bash
make run-code-agent MODEL=deepseek-flash
make run-chess-agent MODEL=deepseek-flash
```

运行压缩版 SWE-bench 实验：

```bash
COMPACT_THRESHOLD=6000 \
SWEBENCH_PATCH=artifacts/django__django-15368.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368 MODEL=deepseek-flash
```

模型和 Modal 运行会产生费用。请先运行 `make doctor`，并检查仍在运行的 Modal 容器。

## 安全与上游材料

本仓库不包含 `.env` 或 API 凭据。交互轨迹保留了模型交互和沙箱输出以供审计；修改并重新分发这些文件前，请再次检查其中的信息。

上游脚手架代码、作业文字及相关材料仍受其原作者条款约束。本仓库不为上游课程材料提供新的许可证。重用或再分发前，请检查原始仓库和课程政策。
