# Mean TTFT on llava-hf/llava-1.5-7b-hf — HiPrune family + naive

Client-observed TTFT (serial streaming, max_tokens=1), macro-average over POPE, MME, TextVQA, ScienceQA, MMBench, VizWiz (n=50 each). Speedup vs Vanilla. Bold = lowest Mean TTFT in the retention block. † HiPrune++ effective keep can exceed requested (additive text tokens). LLaVA-1.5 uses 576 visual tokens.

| Method | Keep | Mean TTFT |
|--------|------|-----------|
| **Vanilla** | 100% | **40 ms (1.00×)** |
| **Retain ~75%** | | |
| HiPrune | 75% | **42 ms (0.95×)** |
| HiPrune++ | 75%→82.5%† | 44 ms (0.91×) |
| HyDART | 75% | 84 ms (0.48×) |
| **Retain ~50%** | | |
| HiPrune | 50% | 41 ms (0.99×) |
| HiPrune++ | 50%→55.0%† | 42 ms (0.96×) |
| HyDART | 50% | 69 ms (0.59×) |
| Checkered | 50.0% (eff) | **38 ms (1.05×)** |
| **Retain ~25%** | | |
| HiPrune | 25% | 40 ms (1.00×) |
| HiPrune++ | 25%→27.4%† | 41 ms (0.97×) |
| HyDART | 25% | 54 ms (0.74×) |
| NPrune | stride2 (~25% eff) | **39 ms (1.05×)** |
| **Retain ~14%** | | |
| HiPrune | 14% | **40 ms (1.00×)** |
| HiPrune++ | 14%→15.5%† | 41 ms (0.98×) |
| HyDART | 14% | 48 ms (0.84×) |
