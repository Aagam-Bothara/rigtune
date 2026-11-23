# Qwen/Qwen3.5-9B, heldout set (500 problems), thinking off

| Method | Problems | Accuracy | 95% interval | GPU-s / problem | Tokens / problem | Cut off | Failed | Utilisation | Cached prompt | J / problem | Preempted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | 500 | 92.6% | 90.0%–94.6% | 0.49 | 1915 | 33 | 0 | 73% (low: wall time, not GPU time) | 0% | 194 | 0 |
| self-correction | 500 | 96.6% | 94.6%–97.9% | 1.03 | 2995 | 36 | 0 | 73% (low: wall time, not GPU time) | 0% | 405 | 0 |
| self-consistency-3 | 500 | 95.4% | 93.2%–96.9% | 1.37 | 5692 | 96 | 0 | 91% | 3% | 544 | 0 |
| self-consistency-5 | 500 | 95.6% | 93.4%–97.1% | 2.23 | 9424 | 168 | 0 | 95% | 3% | 885 | 0 |

Against direct answers, paired (descriptive; the criterion below is the test):

- self-consistency-5: +3.0% (+1.0% to +4.8%), 20 gained, 5 lost, McNemar p = 0.00408
- self-consistency-3: +2.8% (+0.8% to +4.6%), 20 gained, 6 lost, McNemar p = 0.00936
- self-correction: +4.0% (+2.0% to +6.0%), 22 gained, 2 lost, McNemar p = 3.59e-05

self-consistency-5: best of its answers by the reference 96.8% (an upper bound, not a method)

self-consistency-3: best of its answers by the reference 96.2% (an upper bound, not a method)

Self-correction, first answer to final: 16 repaired, 1 harmed, 16 still wrong, 467 kept.

Criterion not decided: utilisation below 80% in self-correction, direct.

