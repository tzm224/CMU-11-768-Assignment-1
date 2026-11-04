# Observation Experiment

The DeepSeek A/B runs used the same `deepseek-flash` model, chess patch, and step limit. The only intended observation difference was whether `legal_moves` was included. Call counts come from the submitted trajectories; terminal states come from their corresponding result files.

| Model | Observation | `play_move` calls | Illegal-move rejections | Invalid-move ratio | `game_over` |
|---|---|---:|---:|---:|---|
| DeepSeek Flash | Board and FEN only | 31 | 1 | 3.2% (1/31) | `true` — White wins |
| DeepSeek Flash | Board, FEN, and legal moves | 31 | 0 | 0.0% (0/31) | `true` — White wins |

The no-legal-moves run also encountered one transport error (`Server disconnected without sending a response`). It was returned as a recoverable `<chess_error>` but is not counted as an illegal-move rejection. Both runs completed their games; the no-legal-moves game ended after 57 recorded plies and the legal-moves game after 61 plies.

In this sample, exposing legal moves eliminated illegal model-generated moves while still allowing the agent to reach a terminal win. Without the list, the model proposed `c1d2` once when it was not legal. The similar total call counts and terminal outcomes suggest that the main observed benefit was move selection validity rather than simply more attempts. These are single stochastic runs, so they do not establish a general win-rate or token-efficiency advantage.

## GPT-OSS conditions

The two `openai/gpt-oss-120b` conditions were not run because the available DeepSeek endpoint does not provide that model and no compatible GPT-OSS endpoint or credential was available. DeepSeek results have not been relabeled or substituted for GPT-OSS. Consequently, the four GPT-OSS trajectory/result artifacts are absent, and the cross-model portion of the experiment remains incomplete.
