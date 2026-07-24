# LLaVA-1.5-7B latency (TTFT)

- Model: `llava-hf/llava-1.5-7b-hf`
- Protocol: serial client-observed TTFT, streaming, `max_tokens=1`, n=50 × 6 benches
- Visual tokens: 576 (LLaVA-1.5)
- Source sweep: `benchmarks/llava15_ttft/`
- Vanilla macro-mean TTFT: 40.3 ms

Note: this package is TTFT-only (no accuracy matrix).
