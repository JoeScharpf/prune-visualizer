# Results on google/gemma-4-e4b-it — HiPrune family + naive

MME is perception+cognition. Average is mean relative to Vanilla across MMB, MME, POPE, SQA$^{IMG}$, TextVQA, VizWiz. † HiPrune++ effective keep is higher because text-guided tokens are additive. POPE is a balanced subset (not necessarily full POPE). Gemma uses dynamic visual length (~262 tokens avg). Mean TTFT is the equal-weight macro-average of client-observed TTFT across the six benches (serial streaming, max_tokens=1); speedup is vs Vanilla. Bold Mean TTFT = lowest in the requested-retention block (effective retention may differ).

| Method | Keep | MMB | MME | POPE | SQA | TextVQA | VizWiz | Average | Mean TTFT |
|--------|------|-----|-----|------|-----|---------|--------|---------|----------|
| **Vanilla** | 100% | 79.6 | 1871 | 86.9 | 78.4 | 68.4 | 58.1 | **100.0%** | **63 ms (1.00×)** |
| **Retain ~75%** | | | | | | | | | |
| HiPrune | 75% | 79.2 | 1827 | 86.4 | 78.9 | 66.9 | 58.2 | 99.2% | **81 ms (0.78×)** |
| HiPrune++ | 75%→82.6%† | 79.4 | 1867 | 86.4 | 78.4 | 67.2 | 58.6 | 99.7% | 83 ms (0.76×) |
| HyDART | 75% | 79.3 | 1836 | 85.9 | 78.0 | 66.3 | 58.3 | 98.9% | 99 ms (0.64×) |
| **Retain ~50%** | | | | | | | | | |
| HiPrune | 50% | 76.4 | 1719 | 85.4 | 79.0 | 60.3 | 57.7 | 95.7% | 79 ms (0.80×) |
| HiPrune++ | 50%→55.0%† | 77.1 | 1739 | 85.6 | 78.5 | 61.7 | 57.7 | 96.4% | 82 ms (0.77×) |
| HyDART | 50% | 77.8 | 1766 | 84.1 | 79.5 | 63.5 | 58.1 | 97.2% | 93 ms (0.68×) |
| Checkered | 50.0% (eff) | 77.9 | 1712 | 86.6 | 79.4 | 56.8 | 58.3 | 95.6% | **67 ms (0.94×)** |
| **Retain ~25%** | | | | | | | | | |
| HiPrune | 25% | 71.5 | 1394 | 84.1 | 78.5 | 49.5 | 55.3 | 88.1% | 80 ms (0.79×) |
| HiPrune++ | 25%→27.5%† | 72.8 | 1454 | 83.3 | 79.1 | 50.3 | 55.4 | 89.1% | 83 ms (0.77×) |
| HyDART | 25% | 74.5 | 1577 | 84.1 | 79.1 | 54.3 | 57.7 | 92.4% | 87 ms (0.73×) |
| NPrune | 26.3% (eff) | 73.9 | 1501 | 84.3 | 78.5 | 39.5 | 58.1 | 88.0% | **68 ms (0.93×)** |
| **Retain ~14%** | | | | | | | | | |
| HiPrune | 14% | 67.5 | 1245 | 77.3 | 76.4 | 39.5 | 53.8 | 81.3% | **80 ms (0.79×)** |
| HiPrune++ | 14%→15.4%† | 68.9 | 1264 | 76.5 | 76.6 | 40.3 | 54.5 | 82.1% | 81 ms (0.79×) |
| HyDART | 14% | 71.9 | 1321 | 82.3 | 76.9 | 45.7 | 56.1 | 86.2% | 84 ms (0.75×) |
