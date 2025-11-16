---
title: "Apriel-H1"
subtitle: "Why efficiency-optimized reasoning matters now"
author: "Torsten Scholak — Lead Research Scientist"
institute: "SLAM Lab — ServiceNow"
date: "November 2025"
publication:
  status: published
  date: Nov 5, 2025
image: apriel-h.png
---

## Efficiency = Capability

**Apriel matches frontier reasoning at 15B.**  
But full attention pays the quadratic tax → long ctx throughput is now the bottleneck.

**Speed creates capability:**

- **Agents keep full tickets/logs in memory** → fewer compactions
- **More tools per turn** at same latency
- **Deeper reasoning chains** with more steps → better accuracy
- **Larger RAG contexts** stay in-context
- **Higher req/s on existing fleet** → lower unit cost, better UX

**That's why we're building efficient hybrids.**

::: notes
60s

Sathwik showed Apriel matches much bigger frontier models at 15B parameters.

But it runs full attention - every token sees every token, cost scales quadratically.

That quadratic cost is now the ceiling: how long can a reasoning chain run? How many tools can an agent call? How much context fits before we truncate?

Efficiency determines what's possible inside a latency budget. That's the strategic shift.

The path is hybrid architectures - that's Apriel-H.
:::

## How Hybrids Work

|  | **Full Attention** | **Efficient (Linear/Sparse)** | **Hybrid** |
|---|---|---|---|
| **Complexity** | O(n²) | O(n) or sub-quadratic | Mixed |
| **KV cache** | Large, grows with n² | Small or none | Reduced ~50-75% |
| **Global fidelity** | Perfect | Limited | Preserved in key layers |
| **Throughput gain** | 1× | 2-10× (but quality risk) | 2-10× at minimal Δ |

**Pattern:** Keep ~20-30% full attention for global reasoning,  
replace rest with Mamba/linear/sparse mechanisms.

::: notes
45s

Here's the trade-off table.

Full attention: it's great but expensive.
Efficient mechanisms: fast but risky.
Hybrids: keep 20 to 30 percent of layers as full attention to preserve global patterns, replace the rest with efficient mechanisms.

Result: 2 to 10 times throughput with quality deltas you can ship.

And there's research that shows that hybrids can beat full attention in some cases, i.e. they might be pareto-optimal. The detail is in how you design them, what the ratio is, and how you distill them.
:::

## Hybrids Are Shipping at Scale

- **Apr**: [NVIDIA Nemotron-H-47B](https://arxiv.org/abs/2504.03624)  
  **9:1 Mamba-2:FA** hybrid, **≈3× faster** vs dense 70B at long ctx

- **May**: [Falcon-H1-34B](https://falcon-lm.github.io/blog/falcon-h1/)  
  Parallel Mamba-2 + FA hybrid, **4× prefill**, **8× decode** at long ctx

- **Jun**: [MiniMax-M1](https://arxiv.org/abs/2506.13585)  
  **7:1 Lightning:FA** hybrid, **≈3–4× faster decode** @100k tokens

- **Aug**: [Nemotron-Nano-9B-v2](https://arxiv.org/abs/2508.14444)  
  **7:1 Mamba-2:FA** hybrid, **up to 6× throughput** vs Qwen3-8B

- **Sep**: [Qwen3-Next-80B-A3B](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)  
  **3:1** Gated-DeltaNet:FA hybrid, **>10× throughput** vs Qwen3-32B @>32k

- **Sep**: [DeepSeek V3.2-Exp](https://api-docs.deepseek.com/news/news250929)  
  **MLA+DSA** sparse, 1:64 attended:total tokens @128k, **3× faster** at long ctx

- **Oct**: [Kimi-Linear-48B-A3B](https://arxiv.org/abs/2510.26692)  
  **3:1** KLA:FA hybrid, **75% KV↓**, **up to 6× decode** @1M

::: notes
50s

The industry is converging on hybrids.

April: Nemotron-H, 3× faster.

May: Falcon-H1, 4-8× gains.

June: MiniMax Lightning Attention, 3-4× at 100k tokens.

August-September: Nemotron-Nano v2 and Qwen3-Next in the 3:1 to 7:1 range, pushing 6-10× throughput.

Then DeepSeek and most recently Kimi with production-grade sparse and linear hybrids.

Hybrids and efficient attention are becoming the new normal for long context and agentic workloads.
:::

## Apriel-H1

### What You Get

![Apriel-H](/images/apriel-h.png){ width=250px }

Today we release **Apriel-H1**:

- **Hybrid reasoner** distilled from Apriel-15B
- **~2× throughput** in vLLM with **minimal quality deltas**
- **Runs today** in vLLM

::: notes
60s

So what happens when we apply this?
Apriel-H1 30 - thirty Mamba-2 layers, twenty full attention layers.

About 2× throughput in vLLM - meaning you can serve twice as many requests on the same hardware, or cut latency in half for the same load.
:::

### Proof It Works

| Metric | Apriel 15B | **Apriel-H1 30** | Δ |
|---------|-------------|--------------|---|
| **Throughput (vLLM)** | 1× | **~2×** | **+2×** |
| MATH500 | 90 | **92** | +2 |
| GSM8k | 97 | **95** | −2 |
| AIME'24 | 70 | **65** | −5 |
| GPQA-D | 59 | **55** | −4 |
| MBPP | 86 | **85** | −1 |
| MT-Bench | 8.30 | **8.58** | +0.28 |

::: notes
Here are the numbers.
2× throughput. Benchmark quality nearly flat - MATH500 actually goes up, GSM8k down 2 points, AIME down 5.
The hardest reasoning tasks dip slightly but well within recoverable range via tuning.

Our hybrids are production-grade and can ship today.
:::

### Evaluation Results

![Apriel-H1 Evaluation Results](/images/apriel-h-vs-apriel-15b-eval-thrput-comparison.png)

::: notes
Here's the side-by-side comparison showing the quality-throughput trade.
:::

## How Apriel-H1 Works

### Architecture — H1-30

- Start: Apriel-15B teacher (50 FA layers)
- Replace least-critical FA layers with **Mamba** (no KV cache, linear time)
- Keep **20 FA layers** to preserve global patterns

::: notes
20s

The architecture is straightforward: start with Apriel-15B - 50 full-attention layers.

We identify which layers are least critical for reasoning, replace them with Mamba blocks - no KV cache, linear time complexity.

Keep 20 full-attention layers to preserve the global reasoning patterns.

That's the H1-30 configuration.
:::

### Distillation — 3 steps

1. **Score layer importance** (LOO perf drop + MMR distill loss)
2. **Swap** low-importance FA → Mamba (MIL-style init from attention)
3. **Stage & gate**: H1-25 → H1-27 → **H1-30** (… H1-34/37/40) with reverse-KL; ship at best quality/throughput trade

```
Teacher (50L): [FA][FA][FA][FA][FA][FA][FA][FA][FA][FA] ...
H1-30:         [FA][FA][FA][M ][FA][M ][M ][M ][M ][FA] ...
                ^           ^           ^           ^
              "keep"     "convert"   "convert"    "keep"
```

::: notes
Three-step process:

First, we score every layer - measure performance drop when removed, measure distillation loss when replaced with Mamba. That tells us which layers the model actually needs.

Second, swap the low-importance layers to Mamba, initialize from the attention weights so we don't start from scratch.

Third, staged distillation - walk the ratio up gradually: H1-25, H1-27, H1-30, gate at each step with reverse-KL divergence. Ship where quality-throughput trade is best.
:::

### Eval Score vs Throughput

![Apriel-H1 Family Performance](/images/apriel-h1-eval-score-vs-throughput.png)

::: notes
Here's the trade-off curve.

Baseline Apriel-15B on the left. As we convert more layers to Mamba, throughput climbs while quality incurs small deltas.

H1-30 is the sweet spot we're shipping - 2× throughput, same general reasoning strength.

Beyond that, H1-40 pushes toward 3× with acceptable quality deltas.

With more compute, each of these can be tuned further to recover quality if needed.
:::

## What's Next

**Apriel-H2 roadmap:**

- **Advanced mixers**: Gated DeltaNet, Kimi Linear Attention
- **Higher efficient-to-full ratios**: Search-guided layer placement
- **Stacked optimizations**: Sliding window + quantization + sparse attention
- **Target**: 5-10× throughput while maintaining reasoning quality

**The path forward:**

- From-scratch hybrid training gives the best ceiling
- Distillation offers practical retrofitting for existing models
- Both approaches matter for different constraints

::: notes
45s

Looking ahead to H2:

We'll explore more advanced mixers like Gated DeltaNet and Kimi Linear Attention.

Use search-guided layer placement to push higher efficient-to-full attention ratios.

Stack multiple optimizations - sliding window, quantization, sparse attention - to compound gains.

Target: 5-10× throughput while maintaining reasoning quality.

The lesson: from-scratch hybrid training gives you the best possible result, but distillation offers a practical path when you can't retrain from scratch. Both approaches have their place.
:::

::: {data-background-color="#000"}

# Thank You

:::

::: {data-background-color="#000"}

## Apriel-H1: Efficient Reasoning Through Hybrid Architectures

**SLAM Lab — ServiceNow**

**Contact**: Torsten Scholak (<torsten.scholak@servicenow.com>)

**Team**: Oleksiy Ostapenko, Luke Kumar, Raymond Li, Denis Kocetkov, Joel Lamy-Poirier

:::
