# Token Usage Analysis

Both runs solved `django__django-15368` with `deepseek-flash`. The compaction run used a 6,000-token threshold; the baseline omitted the compaction flag. Usage below comes from the provider's `usage` fields. Action requests and compaction requests are shown separately so that the cost of compaction is not hidden.

| Metric | Compaction | Full-context baseline |
|---|---:|---:|
| Action model calls | 21 | 13 |
| Compaction model calls | 9 | 0 |
| Action-request prompt tokens | 85,983 | 134,493 |
| Compaction-request prompt tokens | 34,647 | 0 |
| Action-response completion tokens | 4,747 | 2,439 |
| Compaction-response completion tokens | 5,361 | 0 |
| Combined total tokens | 130,738 | 136,932 |
| Average prompt tokens per action request | 4,094 | 10,346 |
| Maximum prompt tokens in an action request | 5,793 | 14,770 |
| Patch replay result | Resolved | Resolved |

The trajectory records nine compaction events. Their estimated active-context reductions range from 31.2% to 66.3%, with a mean reduction of 54.1%. For example, one event reduced the estimated active prompt from 6,011 to 2,036 tokens. The baseline's action-request prompt instead grew from 1,497 to 14,770 tokens as the raw history accumulated.

Including all compaction overhead, this compaction run used 6,194 fewer total tokens than the baseline, a reduction of about 4.5%. Compaction substantially reduced active context, but its nine additional requests consumed 40,008 tokens themselves. Consequently, lower active context does not translate one-for-one into lower total usage. The compaction run also took more action steps (21 versus 13); model runs are stochastic, so this single sample does not establish that compaction generally requires more steps.

The main tradeoff is therefore bounded, repeatedly summarized context and smaller action prompts versus extra summarization requests and completion tokens. In this sample, both patches passed the same replay evaluation, while compaction reduced combined usage modestly and reduced peak action-prompt size substantially.
