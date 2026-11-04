# CMU 11-768 Assignment 1 — Reference Implementation

This repository contains my completed implementation and experiment artifacts for CMU 11-768 Assignment 1: building an agent harness. It is published as a learning reference, not as a standalone starter repository.

**Academic integrity:** Read your course's rules before using this material. Do not submit this implementation, its patches, or its trajectories as your own work. I recommend attempting the assignment first and using this repository only to compare designs and debugging approaches afterward.

## What was implemented

- A domain-independent ReAct loop with action/observation history and step limits.
- OpenAI Chat Completions function-tool definitions and linked tool observations.
- Coding-agent tools for sandbox commands, skill invocation, and patch submission.
- Progressive skill disclosure through YAML-frontmatter metadata.
- Model-generated working-memory compaction with preserved recent tool steps.
- Chess-agent tools for live moves, stateless simulation, sandboxed Python, and skills.
- Generated patches and trajectories evaluated in fresh Modal testbeds.

The original assignment scaffold is available at [cmu-agents/assignment-1](https://github.com/cmu-agents/assignment-1). Development used scaffold commit `67498d8` as a reference point.

## Repository guide

### Source code

The main implementation is under [`src/assignment/agent/`](src/assignment/agent/):

- `base.py`: shared agent state, ReAct lifecycle, skills, and context compaction.
- `code_agent.py`: coding-tool registration, dispatch, and submission behavior.
- `chess_agent.py`: chess-tool registration, dispatch, and live-state updates.
- `chess_tools.py`: chess API helpers and sandboxed Python execution.
- `tools.py`: OpenAI function-tool schemas.

Supporting assignment runtime code is under [`src/assignment/`](src/assignment/). Prompt templates are in [`src/assignment/prompts.py`](src/assignment/prompts.py).

### Experiment artifacts

[`artifacts/`](artifacts/) contains patches, trajectories, game results, and reports. Start with these two reports rather than reading the large JSON trajectories:

- [Token usage analysis](artifacts/token-usage-analysis.md)
- [Observation experiment](artifacts/observation-experiment.md)

[`assignment-1-submission.zip`](assignment-1-submission.zip) is the course-oriented archive containing the required source files and available submission artifacts. The repository also retains the full-context baseline trajectory and the programmatic-skill result as additional evidence.

[AI_USAGE.md](AI_USAGE.md) describes how AI assistance was used.

## Results

- Public offline tests: **15 passed**; 4 Modal-marked tests deselected.
- Part 1 chess patch: **RESOLVED** — FAIL_TO_PASS 1/1 and PASS_TO_PASS 9/9.
- Compacted SWE-bench patch: **RESOLVED** — FAIL_TO_PASS 1/1 and PASS_TO_PASS 29/29.
- Full-context SWE-bench baseline: **RESOLVED** — FAIL_TO_PASS 1/1 and PASS_TO_PASS 29/29.
- Compaction trajectory: **9 compaction events**, each reducing estimated active context.
- Combined model usage in those runs: **130,738 tokens with compaction**, **136,932 tokens baseline**, including compaction-request overhead.
- Basic chess run: `game_over: true`, **White wins**, with 44 model responses.
- DeepSeek observation runs: both reached `game_over: true`; the no-legal-moves condition had 1 illegal move in 31 calls, while the legal-moves condition had 0 in 31 calls.
- Programmatic skill trajectory: `invoke_skill` followed by Python snippets that searched with `simulate_move` and committed with `play_move`.

### Incomplete experiment conditions

The two GPT-OSS conditions were **not run**. The available DeepSeek endpoint did not provide `openai/gpt-oss-120b`, and no compatible GPT-OSS endpoint was available. DeepSeek output was not relabeled as GPT-OSS. This limitation is also recorded in the observation report.

## Using this code with the starter repository

This reference repository intentionally does not include course tests, task definitions, the `chess_app` submodule, or credentials. To reproduce the setup, first obtain the starter repository and then overlay the reference source:

```bash
git clone https://github.com/cmu-agents/assignment-1.git
cd assignment-1
git checkout 67498d8

git clone https://github.com/tzm224/cmu-11768-assignment1.git reference
cp -r reference/src/assignment/. src/assignment/
mkdir -p artifacts
cp -r reference/artifacts/. artifacts/
```

Install and validate the environment:

```bash
make setup
uv run modal setup
cp .env.example .env
# Edit .env locally and provide your own endpoint, API key, and model.
make doctor
make test
```

Never commit your `.env` or API keys. The DeepSeek experiments in this repository used the official `https://api.deepseek.com` endpoint with `deepseek-flash`. When using that endpoint, override the starter Makefile's course-gateway model name explicitly, for example:

```bash
make run-code-agent MODEL=deepseek-flash
make run-chess-agent MODEL=deepseek-flash
```

For the compacted SWE-bench run:

```bash
COMPACT_THRESHOLD=6000 \
SWEBENCH_PATCH=artifacts/django__django-15368.patch \
SWEBENCH_TRAJECTORY=artifacts/django__django-15368-trajectory.json \
make run-swebench-agent INSTANCE=django__django-15368 MODEL=deepseek-flash
```

Model and Modal runs are billable. Check `make doctor` first and monitor active Modal containers.

## Security and upstream material

No `.env` file or API credential is included. Trajectories retain model interactions and sandbox-generated output for auditability, so review them before redistributing any modified version.

Upstream scaffold code, task wording, and related material remain subject to their original owners' terms. This repository does not grant a new license for upstream course material. Check the original repository and course policies before reuse or redistribution.
