# AI Usage

The following AI technologies were used for this assignment.

## Kimi Code CLI

Kimi Code CLI was used as an interactive programming assistant. It helped with the following work:

- Set up the `uv` environment and select a compatible Python version.
- Explain the OpenAI Chat Completions tool-call and tool-observation formats.
- Diagnose implementation and runtime failures in the agent loop, skill loading, context compaction, coding tools, and chess tools.
- Suggest and apply targeted code fixes, including fixes found while running the public tests.
- Run offline tests and patch-replay evaluations and interpret their results.
- Inspect generated trajectories for completeness, compaction events, tool usage, and experiment requirements.
- Organize and run missing experiments, analyze token-usage and observation-experiment data, and prepare the submission archive.
- Draft the experiment reports and this AI usage document based on the generated artifacts.

The assistant was also given a local API credential for configuration. Credentials were excluded from the submission archive and are not recorded here.

## DeepSeek (`deepseek-flash`)

DeepSeek's OpenAI-compatible Chat Completions API was used as the language model for the assignment agents. It generated coding-agent actions and patches, working-memory summaries for context compaction, chess moves, and Python snippets that called `simulate_move` and `play_move`. The model was used through the provided ReAct harness; tool execution and patch or game-state updates were performed by the assignment code and Modal sandboxes rather than by the model itself.

## Infrastructure and non-AI tools

`uv`, `pytest`, Modal, SWE-ReX, the OpenAI Python SDK, and the supplied evaluation scripts were used for dependency management, testing, sandbox execution, API access, and patch replay. These tools were not treated as AI assistance except where they invoked the DeepSeek model described above.
