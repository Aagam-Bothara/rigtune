# Qwen/Qwen3-4B, dev set (98 problems), thinking off

| Method | Problems | Accuracy | 95% interval | GPU-s / problem | Tokens / problem | Cut off | Failed | Utilisation | Cached prompt | J / problem | Preempted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | 98 | 74.5% | 65.0%–82.1% | 3.10 | 1194 | 4 | 0 | 62% (low: wall time, not GPU time) | 0% | 223 | 0 |
| self-correction | 98 | 73.5% | 64.0%–81.2% | 5.61 | 1804 | 5 | 0 | 60% (low: wall time, not GPU time) | 0% | 403 | 0 |
| self-consistency-3 | 98 | 77.6% | 68.3%–84.7% | 6.53 | 3418 | 6 | 0 | 90% | 0% | 469 | 0 |
| self-consistency-5 | 98 | 76.5% | 67.2%–83.8% | 10.73 | 5771 | 16 | 0 | 95% | 0% | 772 | 0 |
| adaptive-8 | 98 | 76.5% | 67.2%–83.8% | 12.14 | 5777 | 23 | 0 | 85% | 0% | 873 | 0 |

Against direct answers, paired (descriptive; the criterion below is the test):

- self-consistency-5: +2.0% (-4.1% to +8.2%), 6 gained, 4 lost, McNemar p = 0.754
- self-consistency-3: +3.1% (-3.1% to +9.2%), 6 gained, 3 lost, McNemar p = 0.508
- adaptive-8: +2.0% (-3.1% to +8.2%), 5 gained, 3 lost, McNemar p = 0.727
- self-correction: -1.0% (-6.1% to +4.1%), 3 gained, 4 lost, McNemar p = 1

self-consistency-5: best of its answers by the reference 84.7% (an upper bound, not a method)

self-consistency-3: best of its answers by the reference 81.6% (an upper bound, not a method)

adaptive-8: best of its answers by the reference 81.6% (an upper bound, not a method)

Self-correction, first answer to final: 2 repaired, 1 harmed, 25 still wrong, 70 kept.

Criterion not decided: utilisation below 80% in self-correction, direct.
