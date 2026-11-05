# 作业 1：构建智能体运行框架

智能体运行框架（agent harness）提供了一套接口，使语言模型（生成概率性字符串的模型）能够观察环境并在其中采取行动。构建智能体运行框架最常用的框架之一是 [ReAct](https://arxiv.org/abs/2210.03629)，它将环境观察、推理/思维链以及智能体操作所对应的文本交错放入语言模型的提示词中。在本作业中，你将基于 ReAct 框架构建一个运行框架。

在第 1 部分，你将实现基础 [`Agent`](./src/assignment/agent/base.py) 的 ReAct 循环；该基础智能体会被实例化为 [`CodeAgent`](./src/assignment/agent/code_agent.py)，在终端中操作以解决软件问题。你的 ReAct 循环将是抽象的，并可应用于多个领域，让你了解智能体运行框架的通用结构。你将实现相关机制，使编程智能体能够观察终端环境并在其中操作，同时运用可复用的智能体技能来完成任务。随后，你将让这个编程智能体修复国际象棋应用中的一个问题。

在第 2 部分，你将研究编程智能体在解决更复杂的软件问题时如何使用上下文。你将实现一种简单的上下文压缩方法，管理智能体对语言模型上下文窗口的占用量。

在第 3 部分，你将把已经构建的基础 `Agent` 实例化为 `ChessAgent`，让它在由 `CodeAgent` 修复的应用上下国际象棋。你将实现该智能体的工具规范，使语言模型能够与基于规则的机器人对弈应用中的棋局（你可以观战！）。你还将探索程序化工具调用，它把程序的能力引入智能体操作，并允许智能体通过执行 Python 代码实施更复杂的策略。

这三个部分之间的关系如下：

```text
                     SWE-Bench 问题
                           |
                           v
有缺陷的国际象棋应用 -> CodeAgent -> fix.patch -> 修复后的国际象棋服务端
                                                     ^
                                                     |
                                              ChessAgent 工具
```

## 环境设置

安装 [uv](https://docs.astral.sh/uv/)，然后运行：
```bash
make setup
```
这将使用 `uv` 建立 Python 环境，并下载和安装相关依赖。如果固定版本的 `chess_app` 源代码（供智能体处理的任务使用）缺失或不在正确的提交上，此步骤会失败。

本作业将使用 [Modal](https://modal.com/)，一个用于运行代码的云平台。该平台可让你在云端机器上安全运行智能体生成的代码。智能体将运行在远程 Modal 沙箱环境中，并执行实现智能体操作的代码。首先，注册一个 Modal 账号（如果还没有），并按照收到的说明获取 Modal 计算额度。然后运行：
```bash
uv run modal setup
```
登录并在你的环境中设置 Modal。


设置好 Modal 后，运行：
```bash
cp .env.example .env
```
创建保存环境密钥的文件，其中主要是 LLM API 信息。如果你选修了本课程，将收到如何获取 LLM 提供商额度的说明。按照说明生成 API 密钥，并确保正确配置服务的基础 URL。

```dotenv
OPENAI_BASE_URL=<OpenAI-compatible base URL>
OPENAI_API_KEY=...
OPENAI_MODEL=deepseek/deepseek-v4-flash-0731
OPENAI_MAX_RETRIES=5
```
如上所示，默认使用 DeepSeek-V4-Flash 模型。在要求使用其他模型时，你应当改用相应模型。如果额度允许，你也可以尝试同一提供商的其他模型，但本作业预期使用上述默认模型完成。

凡会消耗 API 额度（来自 Modal 或 LLM 提供商）的活动，我们都称为_计费_活动。整个作业中你将运行多次计费评估。建议你在相关控制面板上监控用量，确保合理使用 API 额度。

在每次计费运行之前，请验证子模块、Modal 身份验证和模型端点，且不要启动沙箱或生成 token：
```bash
make doctor
```

_（可选）_ 如果你想测试所有 Modal 组件是否按预期运行，请运行：
```bash
make test-modal
make test-chess-modal
```

一般来说，你可以使用：
```bash
modal container list
```
检查是否有正在计费的 Modal 沙箱在运行。如果环境异常关闭，沙箱可能仍在运行并消耗额度。你可以使用 `modal container stop <container ID>` 停止未正确终止的沙箱。

## 公开测试

`make test` 运行速度快、可离线执行且不计费。初始代码会故意无法通过与学生的 TODO 相关的测试。请以下面的里程碑作为指引；如果增加澄清性测试，pytest 的确切测试数量可能会变化。

通过公开测试并不能证明实现完全正确。私有测试还涵盖失败时的清理、重复或格式错误的技能、并行国际象棋调用、传输错误、产物一致性、补丁重放以及真实的 Modal 集成。

## 规则

- 不要修改 `tests/`、`tasks/`、`chess_app/`。
- 不要更改提供的日志记录/清理逻辑，也不要在子类中重复实现共享的 ReAct 循环。
- 不要为了让智能体任务通过测试而硬编码预期解法。你的智能体应当针对给定任务生成走法或补丁。
- **绝不要暴露、记录或提交 API 密钥等凭据。** 请谨慎处理 `.env` 文件的内容。
- 在作业过程中，你将使用多个智能体以及这些智能体的多个版本。任何时候都只能使用该智能体预期配备的工具集。

## 第 1 部分：构建编程智能体并修复国际象棋应用

共享的 `Agent` 必须实现常规的 ReAct 循环：构建请求、获取助手操作、执行其工具调用、添加关联的观察结果，然后重复此过程，直到任务完成。

### 1. 构建提示词

语言模型的提示词是在交互的每一步用于查询语言模型的一系列消息。消息是字典对象，至少包含 `role` 和 `content` 键，也可以包含其他信息。消息的 `content` 可以是字符串，也可以是包含推理 token 的结构化对象；在本作业未涉及的情况下，还可以包含图像。消息的 `role` 可以取若干种值之一。

`system` 角色用于提供长期有效的指令，包括领域相关指令、环境信息、需要遵守的规则、可跨任务使用的通用策略等。`user` 角色提供任务信息[^1]，并在需要时提供解决任务的附加指南。`assistant` 角色消息表示 LLM 的生成结果，`tool` 角色消息表示工具调用的结果，也就是对环境的观察。你的任务是组织这些消息的顺序。按照惯例，最前面必须恰好有一条 `system` 消息，随后在任何助手消息之前有一条 `user` 消息。由于 LLM 每次生成一个响应，`assistant` 消息后应跟随 `tool` 消息（展示工具调用结果）或 `user` 消息。你可以在 [OpenAI API 参考](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)和 vLLM 文档中的这个[示例](https://docs.vllm.ai/en/v0.7.2/getting_started/examples/openai_chat_completion_client_with_tools.html)里找到这些消息结构的相关信息。

你需要实现 `Agent.build_prompt` 方法。给定系统提示词、可作为智能体类属性访问的任务提示词，以及你可能添加的其他状态记录，该方法应当准备语言模型的输入。如你所见，我们提供的 `Agent.query_language_model` 方法会直接使用该方法的输出查询语言模型 API。你的实现需要为这次查询提供正确的输入。`Agent.query_language_model` 方法会向语言模型 API 提供工具和推理参数，并返回包含所有相关信息的结构。它还维护步数，即 API 被查询的次数。API 会负责把给定提示词和工具渲染成一个 token 序列，这部分不需要你处理。它还提供了用于记录请求以便评分的附加机制，你不应修改这部分。

当提供商发生临时故障时，智能体的 LLM 客户端最多重试 `OPENAI_MAX_RETRIES` 次。

> **TODO(1.1.a)**
> 添加相应机制，在智能体执行操作并观察结果时维护智能体状态。构建构成语言模型提示词的消息序列。其中应包括长期有效的指令、任务说明、先前的交互（包括观察、推理和前几轮的操作）。注意，该方法应与领域无关，其构建提示词的方式应适用于任何继承它的领域专用智能体。

接下来，你应当为 `CodeAgent` 构建系统提示词和任务提示词。系统消息_必须_原样包含下面这个块，并使用环境提供的值：
```text
<system_information>
{
  "machine": <machine>,
  "release": <release>,
  "system": <system>,
  "version": <version>
}
</system_information>
```

> **TODO(1.1.b)**
> 为 `CodeAgent` 构建系统提示词和 task_prompt。它们应当可供 `Agent.build_prompt` 方法使用。

### 2. 运行 ReAct 循环

实现 `Agent.run` 的主体。

> **TODO(1.2)**
> 运行 ReAct 循环。协调以下流程：提示语言模型生成推理和操作，提取模型产生的工具调用，执行工具调用，从而获得智能体下一步所需的观察结果。确保通过设置 `Agent.finished` 来识别智能体何时完成任务。如果智能体超过 `step_limit`，则抛出 `StepLimitError`。

### 3. 执行编程工具

实现 `CodeAgent.execute_tool_calls`。编程智能体支持两种工具：`execute` 和 `send_message`。你可以在[这里](./src/assignment/agent/tools.py)找到这些工具的定义。你需要实现执行这些工具的机制，并准备其输出以展示给语言模型。这些输出也是形式为 `{"role": str, "tool_call_id": str, "content": str}` 的消息。每条消息都是执行其中一种工具后得到的观察结果，并使用名为 `tool` 的特殊角色。[^2] 你应当使用 `Environment.execute` 方法，确保代码在正确的 Modal 沙箱中执行。可选参数可以保留当前默认值，但应允许智能体在工具调用中覆盖它们。

> **TODO(1.3)**
> 让智能体可以使用 `execute` 和 `send_message` 工具。解析每次调用，执行已识别的工具，并为每次调用返回一条消息（一次智能体响应中可能有多个工具调用！）。格式错误的 JSON 和未知工具必须转换为可恢复的观察结果并传回智能体，而不是抛出异常。

### 4. 加载技能

你可能认为现在已经可以让智能体处理软件问题了，但还不行！`CodeAgent` 还需要的最后一项知识是如何提交解法。我们使用的具体协议是：智能体解决问题后，将解法保存到名为 `patch.txt` 的文件中。评估时会从环境中提取这个文件。

为了教会智能体这个协议，我们将使用技能。技能是通过专业化、可复用的工作流扩展智能体能力的一种方式。[Agent Skills 协议](https://agentskills.io/home)定义了技能应遵循的标准，使其能够由多个智能体框架支持。`tasks/code-skills/submit-task` 定义了一个遵循该协议提交解法的基础技能。虽然该协议还有更高级的功能可供探索，但在本作业中你只需使用最简技能。

你需要实现的核心功能是_渐进式披露_。为了让智能体能够使用许多可能很复杂的技能，技能信息会按逐步增加的详细程度披露给智能体，由智能体获取所需的信息。本作业中，你将实现一种简单的渐进式披露：在_系统提示词_中向智能体提供所有可用技能的描述。你还将让智能体可以使用 `invoke_skill` 工具，智能体可以通过技能名称调用它，以查看完整技能。这样，当智能体查看完整的 `submit-task` 技能时，就会知道如何提交修复以供评分。我们会检查系统提示词是否提及技能名称和描述。当智能体没有可用技能时，提示词的任何部分都不应提及 `patch.txt`，也不应向智能体提供提交说明。

技能文件可在[本地](./tasks/code-skills/)找到。如果智能体在运行时使用技能，`Agent.skills_path` 会指向此位置（否则设为 `None`）。注意，本作业中智能体只会使用一个技能（`./tasks/code-skills/` 下的一个子目录），但原则上智能体可以使用许多技能，这时渐进式披露会更加重要和有用。

> **TODO(1.4)**
> 验证 ``skills_path``，在每个子目录中发现一个 ``SKILL.md``，解析其 YAML frontmatter（文件开头两个 `---` 标记之间的内容），并返回以 frontmatter 中的 ``name`` 为键的映射。每个值必须包含用于模型技能目录的简洁 ``metadata`` 字符串，以及供 ``invoke_skill`` 使用的技能文件完整 ``content``。对于重复名称、格式错误或缺失的 frontmatter，抛出明确的 ``ValueError``。如果智能体有可用技能，则在提示词中向其提供这些技能的描述/元数据。

例如，`SKILL.md` 的 `content` 为：
```
---
name: hello-world
description: Write "hello, world" to the terminal
---

echo "hello, world"
```
而 `metadata` 为：
```
name: hello-world
description: Write "hello, world" to the terminal
```

### 5. 运行并检查修复

现在，让智能体修复一个软件问题。

```bash
make run-code-agent
make check-part1
```

默认模型为 `deepseek/deepseek-v4-flash-0731`。智能体接收 `tasks/chess-terminal-move/problem_statement.md`，在 `/testbed` 内工作，并且必须复现、修复和验证故障。一次运行会生成：

- `artifacts/fix.patch`
- `artifacts/part1-trajectory.json`

`make check-part1` 会把生成的补丁应用到全新的测试环境，并运行公开回归测试和国际象棋应用测试套件。不要直接编辑 `chess_app/` 中的目标。

完成第 1 部分后，你的解法应通过与提示词构建、截断、步数限制、格式错误/未知工具以及补丁提交相关的测试。

## 第 2 部分：实现上下文压缩

较长的 ReAct 交互记录会提高成本，并最终挤占上下文中的有用信息。随着智能体处理的任务越来越复杂、时间跨度越来越长，它们最终也会触及语言模型上下文窗口的上限。为了让智能体高效处理耗时更长的任务，我们希望只保留先前操作和观察中的相关信息。这可以通过将上下文压缩为工作记忆来实现。现在，你将在共享的 `Agent` 中实现由模型生成的工作记忆；不要使用提供商专用的压缩端点。

> **TODO(2.1)**
> 实现 `Agent.compact_context`。提示模型压缩上下文。压缩所用的系统提示词应要求生成简洁、事实性的工作记忆，并保留目标、约束、文件、命令、编辑、具体结果、失败的尝试、测试、阻塞因素和下一步操作。只总结较旧的前缀；原样保留原始系统/任务消息，并至少保留最近一条完整的助手操作及其所有关联的工具观察。生成的摘要应改变 `build_prompt` 的输出，并缩短提示词长度。

**不要**修改智能体类的 `api_prompt` 和 `api_responses` 属性。它们用于记录状态和评估。

> **TODO(2.2)**
> 在共享循环中，每次请求新操作之前调用 `maybe_compact_context()`。该方法已经会估算活跃 token 数、处理阈值，并记录压缩事件以供日志使用。

使用 6,000 token 的阈值运行随仓库提供的 `django__django-15368` 任务：

```bash
COMPACT_THRESHOLD=6000 \
SWEBENCH_PATCH=artifacts/django__django-15368.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368
make check-swebench INSTANCE=django__django-15368
```

设置 `COMPACT_THRESHOLD=0` 可省略压缩标志，运行完整上下文基线。保留两次运行时，请使用不同的输出文件名，例如：

```bash
COMPACT_THRESHOLD=0 \
SWEBENCH_PATCH=artifacts/django__django-15368-baseline.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-baseline-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368
```

提交的压缩运行必须至少触发一次压缩，显著减少活跃上下文，并生成能通过 `check-swebench` 的补丁。生成过程具有随机性，因此不要求它使用的 ReAct 步数少于每一次基线采样。

生成压缩和不压缩的交互轨迹后，比较这些轨迹的 token 用量。在 `artifacts/token-usage-analysis.md` 中展示你的观察结果，并简要解释观察到的趋势。压缩与不压缩两种条件在上下文使用方面各有什么取舍？

## 第 3 部分：构建 `ChessAgent`

你的智能体已经在第 1 部分修复了国际象棋应用的问题，现在可以构建一个会下国际象棋的智能体。`ChessAgent` 复用第 1、2 部分完全相同的循环。它执白棋，服务端的确定性机器人执黑棋。白棋每走出一步合法着法，服务端都会自动回应。

### 1. 实现 `play_move`

首先定义一个新工具，让智能体能够对应用中正在进行的棋局走子。请遵循 [OpenAI 函数调用指南](https://developers.openai.com/api/docs/guides/function-calling)来定义工具。

> **TODO(3.1.a)**
> 定义名为 ``play_move`` 的 OpenAI 函数工具规范。它必须只接受一个必需的字符串参数 ``move``，说明走法使用 [UCI 记法](https://en.wikipedia.org/wiki/Universal_Chess_Interface)（例如 `e2e4` 和 `e7e8q`），并拒绝额外参数。

然后，你将实现运行该工具的机制。[^3]


> **TODO(3.1.b)**
> 在 `chess_tools.py` 中实现 `_play_move`。解析参数，并向 `/api/move` 发送 POST 请求 `{"move": <uci move>}`。返回序列化后的 JSON 对象。捕获工具抛出的所有错误，并在 `<chess_error></chess_error>` 之间返回错误消息，交由智能体处理。需覆盖 JSON 参数格式错误、参数不是对象、fen 缺失或不是字符串、move 不是字符串、服务端拒绝局面或走法，以及传输失败。

你可以查看 `chess_app`，了解 API 的工作方式。

使用 `format_state` 方法格式化成功返回的状态，更新 `last_state`，并根据 `game_over` 设置 `finished`。使用原始 `tool_call_id` 关联观察结果。格式错误的 JSON、未知工具、非法走法和网络错误都应转换为 `<chess_error>...</chess_error>` 观察结果。由于实时局面会在走子后变化，一组并行调用中最多执行一步走法，并以可恢复的方式拒绝其余调用。

初始观察以及成功走子后的观察已经展示棋盘、黑方回应和下一组合法走法，因此不需要单独的棋盘读取工具。

运行：

```bash
make run-chess-agent
```

这会应用你的 `fix.patch`、启动修复后的服务端，并保存：

- `artifacts/part3-trajectory.json`
- `artifacts/game-result.json`

打印出的 HTTPS URL 提供棋盘和 API。智能体对弈时，棋盘会轮询实时状态；棋盘上的按钮会刷新状态，但不会重置棋局。`CHESS_TIMEOUT=1800` 控制沙箱的存活时间。

### 2. 运行观察 A/B 实验

为了观察工具接口对智能体行为的影响，你将针对两个模型比较仅含棋盘的观察与同时包含棋盘和合法走法的观察。运行器无需修改源代码即可配置这些条件，并使用不同的文件名：

```bash
make run-obs-deepseek-no-legal
make run-obs-deepseek-legal
make run-obs-gpt-oss-no-legal
make run-obs-gpt-oss-legal
```

对四次运行分别记录 `play_move` 总调用次数、因非法走法而被拒绝的调用次数、无效走法比率，以及是否达到 `game_over: true`。在 `artifacts/observation-experiment.md` 中写一段简短比较。评分依据是实验和证据，而不是特定结果或是否赢棋。

### 3. 添加 `simulate_move`

为了让智能体能够规划更复杂的策略，你将赋予它模拟走法的能力。模拟让智能体能够模拟走一步棋的效果，但不会真正改变实际棋盘的状态。你将定义并注册 `SIMULATE_MOVE_TOOL`、实现 `_simulate_move`，并将其加入现有调度器。

> **TODO(3.3.a)**
> 仿照 `play_move` 工具定义 `simulate_move` 工具。

`simulate_move(fen, move=None)` 调用 `POST /api/simulate`。

仅向 `simulate_move` 传入完整的六字段 [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)，会返回该局面及其合法走法。传入 FEN 和 UCI 走法，则会返回任一方恰好走一步之后的局面。返回 JSON，以便 Python 使用 `fen`、`squares`、`turn`、`legal_moves` 和终局结果字段。

> **TODO(3.3.b)**
> 在 `chess_tools.py` 中实现 `_simulate_move`。解析参数，使用 FEN 和可选走法调用提供的 `/api/simulate` 端点，并返回序列化后的 JSON。捕获工具抛出的所有错误，并在 `<chess_error></chess_error>` 之间返回错误消息，交由智能体处理。需覆盖 JSON 参数格式错误、参数不是对象、fen 缺失或不是字符串、move 不是字符串、服务端拒绝局面或走法，以及传输失败。

你应当使用 `ChessAgent.chess_client` 发送 POST 请求。

### 4. 添加 `run_python`

具备模拟棋盘走法的能力后，智能体现在可以进行更复杂的规划。执行代码的能力将让智能体更可靠地执行它制定的计划，你将通过程序化工具调用启用这一能力。定义并注册 `RUN_PYTHON_TOOL`、实现 `_run_python`，并将其加入调度器。模型提供 `run_python(code)`，代码片段可以像普通同步 Python 函数一样调用 `simulate_move` 和 `play_move`。

> **TODO(3.4)**
> 解析参数，在沙箱中运行代码，并按名称提供已注册的工具。`/opt/assignment/sandbox_python.py` 是 `env` 沙箱上的一个脚本，可以访问与本文件中相同的工具定义。用它运行模型生成的、作为 run_python 工具参数传入的代码。该脚本接受两个位置参数——`port` 和经过 base64 编码的代码字符串（避免引号问题）。实现此工具调用。
> 脚本会打印一个 JSON 对象，其中包含运行代码产生的 `stdout`、`stderr` 和 `error`——原样返回该字符串。非零 returncode 表示沙箱本身失败，而不是模型代码失败。将 `exception_info` 或 `stderr` 作为 <chess_error> 报告。
> 如果出现类型不匹配或解析失败等问题，返回 <chess_error>{message}</chess_error>。

模型编写的代码绝不能在本地智能体进程中运行。`_run_python` 通过 `env.execute` 将经过 base64 编码的代码发送给提供的沙箱运行器：

```text
python /opt/assignment/sandbox_python.py <port> <base64-code>
```

返回运行器包含 `stdout`、`stderr` 和 `error` 的 JSON 字符串。非零沙箱命令状态转换为 `<chess_error>`。代码片段抛出的 Python 异常属于运行器成功调用的情况，应放在其 `error` 字段中。每个代码片段执行后，重新读取实时棋盘，更新 `last_state` 和 `finished`，并将格式化后的状态附加到观察结果中；否则模型可能会重复执行代码片段已经提交的走法。

使用以下命令启用这些工具：

```bash
uv run assignment-play-chess --programmatic-tools
```

实现后，你的智能体应能生成一段选择走法的代码，并在棋局中执行该走法。你可能仍会发现，国际象棋智能体没有利用可用工具走出好棋，而是经常直接使用 `play_move` 操作。为了给它提供更明确的结构和策略，我们再次使用前面探索过的技能思路。

### 5. 加载并使用国际象棋技能

虽然运行 Python 代码的能力让智能体可以选择执行复杂计划，但智能体未必倾向于这样做。为了给智能体一个可执行的具体策略，我们在 `tasks/chess-skills/select-move` 中提供了另一个技能。为了让模型使用该技能，你将赋予 `ChessAgent` 一部分 `CodeAgent` 的技能使用能力。由于工具执行机制略有不同，你必须在 `_invoke_skill(skills, arguments)` 中为 `ChessAgent` 重新实现 `invoke_skill` 工具的处理逻辑，仅在成功加载技能时注册 `INVOKE_SKILL_TOOL`，并返回指定技能的完整内容。

> **TODO(3.5)**
> 解析参数并返回指定技能的内容。如果出现类型不匹配或解析失败等问题，返回 <chess_error>{message}</chess_error>。

```bash
uv run assignment-play-chess \
  --programmatic-tools \
  --skills-path tasks/chess-skills \
  --trajectory artifacts/part3-python-skill-trajectory.json
```

交互轨迹必须显示：先调用 `invoke_skill`，随后 `run_python` 的代码调用 `simulate_move` 进行搜索，并调用一次 `play_move` 提交走法。读取技能后每回合只直接调用一次 `play_move`，不能证明这些工具协同工作。棋局不必结束，也不必获胜。

## 评分

本作业总分为 **100 分**。每一行独立评分；一次具有随机性的模型运行失败，不会抹去无关实现部分的得分。

| 部分 | 评分标准 | 分值 | 证据 |
|---|---|---|---|
| 1 | 提示词构建以及有效的操作/观察历史 | 6 | 私有单元测试 |
| 1 | ReAct 生命周期、纯文本恢复、步数限制、清理、交互轨迹 | 6 | 私有单元测试 |
| 1 | 编程工具调度、可恢复错误、补丁提交 | 6 | 私有单元测试 |
| 1 | 国际象棋补丁可应用并通过私有/回归测试 | 8 | 在全新测试环境中重放补丁 |
| 1 | 技能发现和 `invoke_skill` 行为 | 4 | 私有测试 |
| 2 | 压缩触发和模型生成的摘要 | 6 | 私有测试和交互轨迹 |
| 2 | 原始指令和完整的近期工具步骤保持有效 | 6 | 私有测试和交互轨迹 |
| 2 | 可审计的压缩显著减少活跃上下文 | 4 | 压缩事件和用量 |
| 2 | 完整上下文与压缩的 token 用量分析报告 | 4 | 报告 |
| 2 | SWE-bench 补丁通过 FAIL_TO_PASS 和 PASS_TO_PASS | 8 | 补丁重放 |
| 3 | `play_move` 规范和注册 | 4 | 私有测试 |
| 3 | `play_move` 状态更新和可恢复错误 | 6 | 私有测试 |
| 3 | 基础国际象棋交互轨迹达到终局状态 | 4 | 交互轨迹和结果 |
| 3 | 完成四次运行的观察 A/B 实验 | 4 | 四条交互轨迹、四份结果 |
| 3 | 观察 A/B 实验报告 | 4 | 报告 |
| 3 | `simulate_move` 无状态行为和错误处理 | 6 | 私有单元/集成测试 |
| 3 | `run_python` 沙箱执行、状态刷新、错误处理 | 6 | 使用沙箱替代实现的私有测试 |
| 3 | 技能交互轨迹结合技能、程序化搜索和实时走子 | 6 | 交互轨迹重放 |
| — | 完整、可解析且符合规则的提交 | 2 | 压缩包验证 |
| | **总分** | **100** | |

评分器会重放补丁和提交的交互轨迹，不会发起新的 LLM 调用。缺失或不一致的证据只会影响对应行的得分。本仓库不包含教师测试和参考补丁。

## 提交

提交一个 ZIP 压缩包，其中包含你在 `src/assignment/agent/` 下修改的文件以及以下产物：

```text
artifacts/fix.patch
artifacts/part1-trajectory.json
artifacts/django__django-15368.patch
artifacts/django__django-15368-trajectory.json
artifacts/token-usage-analysis.md
artifacts/part3-trajectory.json
artifacts/game-result.json
artifacts/part3-no-legal-moves-deepseek.json
artifacts/part3-no-legal-moves-deepseek-result.json
artifacts/part3-legal-moves-deepseek.json
artifacts/part3-legal-moves-deepseek-result.json
artifacts/part3-no-legal-moves-gpt-oss.json
artifacts/part3-no-legal-moves-gpt-oss-result.json
artifacts/part3-legal-moves-gpt-oss.json
artifacts/part3-legal-moves-gpt-oss-result.json
artifacts/observation-experiment.md
artifacts/part3-python-skill-trajectory.json
```

仅当你修改了 `src/assignment/prompts.py` 时才需将其包含在内。**不要**提交凭据、`.env`、任务文件、测试、子模块内容或教师文件。你的 ZIP 应包含 `src/` 和 `artifacts` 子目录，以及一个 `AI_USAGE.md` 文件。`AI_USAGE.md` 应详细说明你在本作业中对任何 AI 技术的使用。列出你使用的所有工具，并清楚描述每种工具的使用方式。如果你未使用任何 AI 辅助，请在此文件中声明。我们不会为提交的 `AI_USAGE.md` 评分，但会通过测验检查你对所提交代码的理解（详情将另行公布）。

[^1]: 在像这样的离线评估环境中，通常通过将任务组织成用户请求来指定任务。更多示例请参阅这些[文档](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/chatgpt?tabs=python-key%2Cdotnet-secure%2Cjavascript-secure&pivots=programming-language-python)。

[^2]: 提供商重复使用 `call_0` ID 是有效的：将每条工具观察与同一次助手操作中的调用匹配，不要假设 ID 在整个交互轨迹中全局唯一。

[^3]: `_play_move` 和其他工具被隔离在 `chess_tools.py` 中，以便能够在远程沙箱中执行这些工具。请在此代码结构内工作，正确使用 Modal 沙箱执行国际象棋走法。
