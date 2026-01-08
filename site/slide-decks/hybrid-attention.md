---
title: "Hybrid Attention Architectures for Efficient LLMs"
subtitle: "Apriel-H1 (shipped) and Apriel-H2 (in progress)"
author: "Torsten Scholak — Lead Research Scientist"
institute: "Foundation Models Lab — ServiceNow AI Research"
date: "January 2026"
publication:
  status: draft
  date: Jan 2, 2026
---

## The Apriel story so far

### Frontier reasoning at 15B

<!--
TIMING ESTIMATES:
- Main stream (realistic): ~20 min
- With all extras:         ~30 min

Main stream = first slide under each ## heading (19 slides incl. roadmap)
Extra slides = additional ### slides for depth (19 slides)

Roadmap breakdown:
- Why hybrids:    ~5 min
- H1 shipped:     ~7 min
- H2 building:    ~5 min
- Field + Future: ~3 min
-->

**The question:** Does frontier reasoning require frontier compute?

**Our answer:** No.

- **Apriel Nemotron** (May '25) → in production (Now LLM)
- **Apriel-1.5** (Sep '25) → AA Index 52, midtraining beats brute-force scaling
- **Apriel-1.6** (Dec '25) → AA Index 57, matching Qwen3 235B at 1/15th the params

All three: **full attention**.

::: notes
30s

Thanks for having me. I'm Torsten, and I currently lead the Foundation Models Lab at ServiceNow AI Research.

Today I want to talk about our efficiency-optimized Apriel models, H1 and H2, that my team has been building.

Let's start with where we are. Over the past year, ServiceNow has built three major versions of Apriel. My team led the architectural design for all three, while ServiceNow platform research led the mid-training, post-training and deployment. Together we have shown that 15B-parameter models can match or exceed the reasoning capabilities of much larger models. Apriel 1.6 hits an Artificial Analysis index of 57 — that's Qwen3-235B territory at a fraction of the size.

But here's the thing: all three of these models use full attention. That's great for quality, but it's also a bottleneck for throughput and efficiency.
:::

## Efficiency creates capability

### Why we're here

**Apriel matches frontier reasoning at 15B** — but full attention pays the quadratic tax.
Throughput ceilings become capability ceilings.

[Speed Creates Capability](https://tscholak.github.io/posts/speed-creates-capability.html):

- **Agents**: more tool calls before memory must be compressed
- **Reasoning**: deeper chains → better accuracy
- **RAG**: larger contexts stay in-context
- **RL**: more rollouts per $ → denser feedback → better policies

> If your models think slowly, your roadmap does too.

**That's why we are optimizing Apriel for efficiency.**

::: notes
45s

So why does attention efficiency matter?

Full attention scales quadratically. That means throughput ceilings become capability ceilings.

If your agent can only make 10 tool calls before you have to compress memory, that's a hard limit on what it can do.

Same story for reasoning: longer chains mean better answers.

For RAG: more context stays in-context.

For RL: more rollouts per dollar means denser feedback and better policies and thus better models to ship.

That's why I think of efficiency as capability enablement, not just cost reduction.

This is why my team is optimizing Apriel for efficiency.
:::

### The fast–good–cheap simplex

![Speed creates capability simplex](/images/speed-creates-capability.svg){width=50%}

**Traditional view:** fast, cheap, good — pick two.

**The twist:** speed converts to quality.

- 3× faster sampling → 3× more RL rollouts → better models
- 3× faster inference → longer reasoning chains → better answers

::: notes
30s

This is the "why hybrids are strategic" visual.
Quality is not monotone in architecture once you include RL/test-time compute.

Why does the top veer left (toward Speed, away from Low Cost)?
Left = fast + expensive. Right = slow + cheap.
To maximize quality, you spend money (RL training, longer CoT) while keeping speed.
The alternative (right-leaning) would be: slash costs, accept higher latency.
We reject that. Speed is what you protect; money is what you spend to climb quality.
:::

## This talk

[Why hybrids] ──▶ [H1: shipped] ──▶ [H2: building] ──▶ [Field + Future]

```
 ~5 min            ~7 min             ~5 min             ~3 min
```

::: notes
30s

Here's the roadmap for this talk. There are four parts, I'm aiming for about 20 minutes total.

First, why hybrids matter — the roofline model, KV bandwidth problem, and what efficient mixers buy you. About 5 minutes.

Second, Apriel-H1 — what we actually shipped. The three-stage procedure, the results, and what we learned. About 7 minutes.

Third, Apriel-H2 — what we're building now. Supernets, beam search, the deployment story. About 5 minutes.

Finally, where the field is going and what's next. About 3 minutes.

There's additional depth on mixers, weight transfer, and search mechanics if questions go there.

Let's start with why hybrids matter.
:::

## Roofline model + ridge point for kernels

### Definitions

Hardware:  $P_{\text{peak}}$ [FLOP/s],  $B_{\text{HBM}}$ [byte/s]  
Kernel:    $W$ [FLOP],  $\mathrm{OI}=W/Q$ [FLOP/byte]

**Kernel time lower bound:**
$$
T \;\gtrsim\; \max\!\left(\frac{W}{P_{\text{peak}}},\ \frac{W}{\mathrm{OI}\,B_{\text{HBM}}}\right)
\;=\;
\frac{W}{P_{\text{peak}}}\cdot \max\!\left(1,\ \frac{\mathrm{OI}^\star}{\mathrm{OI}}\right)
$$

**Ridge point:**
$$
\mathrm{OI}^\star \;=\;\frac{P_{\text{peak}}}{B_{\text{HBM}}}
$$

- if $\mathrm{OI} \ll \mathrm{OI}^\star$ ⇒ **bandwidth-bound** ($T \approx Q/B_{\text{HBM}}$)
- if $\mathrm{OI} \gg \mathrm{OI}^\star$ ⇒ **compute-bound** ($T \approx W/P_{\text{peak}}$)

::: notes
60s

On modern GPUs, a kernel's runtime is lower-bounded by either compute throughput or memory bandwidth. The roofline model tells you which one.

Operational intensity — OI — is the key concept. It's FLOPs per byte: how much compute work you do for each byte you move from memory. High OI means compute-dense. Low OI means you're shuffling data more than crunching numbers.

The ridge point is where the two constraints are equal — it's the hardware's balance point. On H100/H200 with BF16, it's around 200 FLOP/byte. B200 pushes it to 280.

If your kernel's OI is below the ridge point, you're bandwidth-bound. The GPU's compute units are sitting idle waiting for data. Adding more FLOP/s won't help — you need to move less data or move it faster.

If your kernel's OI is above the ridge point, you're compute-bound. Memory is keeping up fine; the bottleneck is raw math throughput.

So the question becomes: where does attention sit on this curve? Is it bandwidth-bound or compute-bound?
:::

### Ridge point illustration

![Ridge point](/images/ridge-point.svg){width=100%}

::: notes
20s

How to read the time roofline plot:

- x-axis is OI.
- Two lower bounds on time:
  - compute floor: horizontal line.
  - bandwidth floor: falls like 1/OI.
- Ridge point is where they meet: OI*.
- Left of OI*: time is set by memory; right: time is set by compute.
:::

## KV-attention through the roofline lens

Causal SDPA (GQA implied; assume $d_v=d_h$): $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+\Lambda\right)V$

**Prefill vs decode per KV-attention layer:**

| Phase | Attention FLOPs | KV-cache HBM bytes |
|---|---|---|
| **Prefill** (prompt length $n$) | $\sim n^2\, h_q\, d_h$ | **KV writes** $\sim 2n\, h_{kv}\, d_h\, b$ |
| **Decode** (new token at $t$) | $\sim t\, h_q\, d_h$ | **KV reads** $\sim 2t\, h_{kv}\, d_h\, b$ |

**KV-read OI per decode step at length $t$:**
$$
\mathrm{OI}_{\text{KV-read}}(t)
\;\approx\;
\frac{t\, h_q\, d_h}{2t\, h_{kv}\, d_h\, b}
\;=\;
\frac{h_q}{2\,h_{kv}\,b}
$$

- $t$ cancels ⇒ KV-read OI is ~constant as context grows.
- other **HBM traffic per token** is ~$t$-independent, while KV reads grow $\propto t$ and **dominate** at long context.
- therefore, at large $t$, the *step-level* OI satisfies $\mathrm{OI}_{\text{step}}(t)\approx \mathrm{OI}_{\text{KV-read}}$.
- if $\mathrm{OI}_{\text{step}} < \mathrm{OI}^\star$, KV-attention decode is **bandwidth-bound**, and for large $t$:

$$
T_{\text{decode}}(t)\;\sim\;\frac{2t\,h_{kv}\,d_h\,b}{B_{\text{HBM}}}
$$

::: notes
90s

We have to split the attention story into prefill and decode because they stress the hardware in fundamentally different ways.

Prefill is the prompt pass.
We process all n prompt tokens in parallel, which gives high GPU occupancy.
Attention does quadratic work once, and we write the KV cache once for those n tokens.
This can be expensive, but it is a one-time cost.

Decode is different. We generate tokens sequentially.
At decode step t, the new token attends to all t previous tokens, and we have to reread the entire KV history at every step.
That KV traffic grows linearly with context length.

Both the compute and the KV traffic per decode step scale linearly with t, but their ratio stays constant. That cancellation on the slide is the key point.
The KV reread does not become more compute-dense as context grows.
If it is bandwidth-bound at small t, it stays bandwidth-bound.

What changes with context is absolute cost, not operational intensity.
Per-token KV bytes keep growing with t, while most other per-token work (MLPs, layer norms, etc.) is roughly constant.
As a result, decode latency looks like a constant term plus a linear term in t, and at long context the KV reread dominates and sets the slope.

So the problem is not too many FLOPs.
The problem is a bandwidth-bound KV reread whose cost grows linearly with context.

Everything that follows is about reducing or eliminating that growing bandwidth term.

Optional details / Q&A ammo (only if needed):

- **What exactly is T?**
T is the wall-clock time contribution of the attention component we're talking about (here: the KV reread + the associated dot-products / weighted sum) for one layer, one decode step. It's a lower bound; real kernels have overheads, imperfect overlap, and additional reads/writes.

- **What do the table entries include / omit?**
The FLOP counts are for the dominant attention math (QK^T and AV) up to constants; the byte counts are only the *KV-cache traffic* (read in decode, write in prefill). Real attention also reads Q, writes output, touches softmax state, etc., which typically makes OI *worse* than this simplified ratio—this ratio is already the "best case" intuition.

- **Why is the OI expression 'KV-read OI' and not full attention OI?**
Because this isolates the part that scales with context and is the bottleneck driver in long-context decode: rereading K and V. The full kernel adds more bytes (and some extra FLOPs), but it doesn't change the key cancellation: both FLOPs and KV bytes scale proportionally to t, so OI stays roughly constant with t.

- **Quick numerical sanity check (H100-ish).**
From NVIDIA's Hopper whitepaper: H100 SXM shows peak BF16 tensor throughput on the order of 1000 TFLOP/s and memory bandwidth on the order of 3000 GB/s (prelim specs).
That's a ridge point OI*on the order of ~1000 TF / 3000 GB ≈ 333 FLOP/byte.
Now take a plausible GQA config: h_q=64, h_kv=8, b=2 (BF16 KV):
OI_KV-read ≈ 64 / (2 × 8 × 2) = 2 FLOP/byte,
which is **orders of magnitude** below OI*. So the KV-read component is deep in bandwidth-bound territory on H100-class hardware.

- **Definitions (non-trivial symbols):**
  - n: prompt length (tokens) during prefill.
  - t: current context length at a particular decode step.
  - h_q: number of query heads.
  - h_kv: number of KV heads (GQA/MQA: h_kv ≤ h_q).
  - d_h: head dimension (assume d_v = d_h).
  - b: bytes per KV element (BF16: 2, FP8: 1, etc.).
  - P_peak: peak compute throughput (for the datatype / kernel regime you care about).
  - B_HBM: sustained HBM bandwidth.
  - OI: operational intensity = FLOPs per HBM byte.
  - OI*: ridge point = P_peak / B_HBM.

:::

## What to do about it: breaking the decode bytes law

For KV-attention decode, the dominant term at $t$ large is: $\mathrm{Bytes/token}(t)\;\approx\;2\,L_{\text{attn}}\,t\,h_{kv}\,d_h\,b$

**Goal:** reduce this term, change its $t$-dependence, or remove it.

| Lever | What it changes | Examples | Asymptotic effect |
|---|---|---|---|
| **Improve utilization** | constants only | FlashAttn, paging, chunked prefill | ❌ |
| **Reduce $b$** | bytes per KV element | FP8 / INT8 KV, compression | ❌ (still $\propto t$) |
| **Reduce $h_{kv}, d_h$** | KV tensor shape | stronger GQA/MQA, smaller head dim | ❌ |
| **Replace $t \to W$** | effective history length | SWA, local / sparse attention | ✅ (bounded by $W$) |
| **Reduce $L_{\text{attn}}$** | number of KV layers | hybrids with state-space mixers | ✅ (linear in fewer layers) |
| **Eliminate KV rereads entirely** | algorithmic class | linear attention, SSMs, Mamba | ✅ (no $\propto t$ term) |

**This work:** reduce $L_{\text{attn}}$ and KV rereads via hybrid architectures with SSMs.

::: notes
30s

This is the menu of options. The first few rows — quantization, stronger GQA — help with constants but don't change the linear growth in t. They shave bytes but don't change the law.

The rows that matter change the structure: bound the window so you only read the last W tokens, reduce how many layers carry KV state, or exit the KV paradigm entirely with linear attention or SSMs.

Hybrids are our focus: keep some full-attention layers for global fidelity, replace the rest with mixers that don't grow with context.

Optional (if asked): vLLM's chunked prefill batches compute-bound prefill with memory-bound decode to improve utilization — a nice proof that these are distinct regimes.
:::

## Efficient mixers: the bounded-state constraint

**Eliminating the $\propto t$ KV term means maintaining a fixed-size state $S_t$.**

A fixed state has $O(d n)$ (Mamba) or $O(d^2)$ (GDN/KDA) degrees of freedom,
but a length-$t$ token sequence has $O(t \log |V|)$ bits.

Thus, for large $t$, no update rule can preserve all information.
**Compression is unavoidable.**

**This is the core tradeoff:**
efficient mixers replace exact recall with *learned retention*.

::: notes
25s

We just showed that the linear-in-t decode cost comes from rereading a growing KV cache.

If you eliminate that term, you're committing to a fixed-size state.
Once the state is fixed, you cannot store unbounded history exactly.

So efficiency forces compression.
The only remaining question is how forgetting is structured and learned.
:::

## The universal structure

### Forget–Write–Read template

All efficient mixers follow the same template:

$$
S_t = \underbrace{F_t(S_{t-1})}_{\text{forget}} + \underbrace{W_t}_{\text{write}}, \qquad o_t = \mathrm{Read}(S_t)
$$

| | Forget | Write | Key difference |
|---|---|---|---|
| **Mamba-1** | uniform decay | gated input | fast, but can't overwrite |
| **GDN** | selective erasure | key–value pair | overwrites along key direction |
| **KDA** | selective erasure | key–value pair | + channel-wise gating |

**Same template, different forgetting strategies.**

::: notes
35s

All efficient mixers follow the same template. There are always 3 operations.

Forget: decide what to throw away from the current state.
Write: add new information.
Read: extract an output from the updated state.

Every mixer we'll discuss fits this pattern: Mamba-1, Gated Delta Net, and Kimi Delta Attention.

Mamba uses uniform decay: when it forgets, it forgets everything equally.
It cannot say "forget this specific thing but keep that."

The delta-rule variants — Gated Delta Net and Kimi Delta Attention — can selectively erase.
They clear the old value for a key before writing a new one.
:::

### Mamba-1: uniform decay

**State:** vector $s_t \in \mathbb{R}^{d}$ (per channel)

**Update:**
$$
s_t = \underbrace{\bar{A}_t s_{t-1}}_{\text{decay everything}} + \underbrace{\bar{B}_t x_t}_{\text{add input}}
$$

where $\bar{A}_t = \exp(\Delta_t A)$ and $\bar{B}_t = A^{-1} (\bar{A}_t - I) B_t$.

**In plain terms:**

- $A$: learned diagonal decay matrix
- $B_t$: input-dependent learned projection matrix
- $\Delta_t$: input-dependent step size (this is the "selectivity")
- Large $\Delta_t$ → forget more, attend to current input
- Small $\Delta_t$ → retain history, ignore current input

**The problem:** decay is uniform. No way to say "forget *this specific thing*."

::: notes
30s

Mamba fits the Forget-Write-Read template, but with a simpler structure.
The state is a vector, not a matrix. There are no key-value pairs.

Conceptually, Mamba is a gated leaky integrator.

The "selectivity" is the step size Δ_t.
Large step: forget more, attend to current input.
Small step: retain history, coast through.

The model learns when to pay attention.
But the forgetting is uniform, it can't forget a specific item but keep another.

That's the limitation delta-rule variants fix.
:::

### The delta rule: selective forgetting

**Mamba's limitation:** can't overwrite specific memories without decaying everything.

**The fix:** before writing $k_t v_t^\top$, erase the existing content along $k_t$.

$$
S_t = \alpha_t \underbrace{(I - \beta_t k_t k_t^\top) S_{t-1}}_{\text{erase along } k_t} + \beta_t k_t v_t^\top
$$

**Why $(I - \beta_t k_t k_t^\top)$?**

This is a projection that removes the component of $S_{t-1}$ pointing in the $k_t$ direction.

→ Old value for this key is cleared before new value is written.

→ Bounded state, no unbounded accumulation.

This is the **delta rule** from Hebbian learning (Widrow-Hoff, 1960).

::: notes
30s

The key innovation of delta-rule methods is selective forgetting.

Think of the state as a soft dictionary and each token as a key-value pair.
Writing adds an entry. Reading queries by similarity.

The problem: writing the same key twice accumulates values, and unchecked this would lead to unbounded growth in the state.
The delta rule addresses this by erasing the old value for that key before writing the new one.
And that makes it better than Mamba, which can only decay everything uniformly.
:::

### GDN and KDA: delta-rule variants

**Gated DeltaNet (GDN):**
$$
S_t = \alpha_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top
$$

**Kimi DeltaAttention (KDA):**
$$
S_t = (I - \beta_t k_t k_t^\top)\, \mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top
$$

| | Decay | Effect |
|---|---|---|
| **GDN** | scalar $\alpha_t$ | all channels decay equally |
| **KDA** | diagonal $\mathrm{Diag}(\alpha_t)$ | per-channel decay rates |

**Both:** matrix state, associative write, selective erasure.

**KDA's edge:** more expressive gating at minimal cost.

::: notes
30s

GDN and KDA are practical variants of the delta rule.

The "gated" part: decay and write strength are input-dependent.
The model learns when to retain more, when to forget more.

The difference: GDN has scalar decay — all channels equal.
KDA has per-channel decay — more expressive, at little more cost.
:::

### Unified view: the forgetting spectrum

$$
S_t = F_t(S_{t-1}) + W_t, \qquad o_t = S_t^\top q_t
$$

| Model | State | Forget $F_t(S)$ | Write $W_t$ |
|-------|-------|-----------------|-------------|
| **Mamba-1** | $s \in \mathbb{R}^{d}$ | $\bar{a}_t \odot s$ | $\bar{b}_t \odot x_t$ |
| **GDN** | $S \in \mathbb{R}^{d_k \times d_v}$ | $\alpha_t(I - \beta_t k_t k_t^\top) S$ | $\beta_t k_t v_t^\top$ |
| **KDA** | $S \in \mathbb{R}^{d_k \times d_v}$ | $(I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t) S$ | $\beta_t k_t v_t^\top$ |

**The spectrum:**

- Mamba: fast uniform decay, no associative structure
- GDN/KDA: selective key-based erasure, explicit key–value memory

All achieve $O(1)$ memory per token. All are lossy compressors.

::: notes
30s

I've put all three mixers together here to show how they differ.

They all follow the same template — that's the equation at the top.
The differences are in the Forget and State columns.

Mamba uses a vector state and decays everything uniformly. It's fast and simple, but there's no key-value structure.

GDN and KDA use a matrix state and erase selectively. Richer memory, but more compute.

What they share: all are O(1) per token, and all are lossy compressors.
:::

### MIL / DIL / KIL mapping table

When converting FA layers to another mixer, we want:

- faster convergence
- less catastrophic quality loss
- more stable distillation / post-training

**Idea:** map learned projections $(Q,K,V,O)$ into the target mixer's input/output/state interfaces, and randomly init only what has no analogue.

| Converter | Transfer from FA | Random init | What it preserves |
|---|---|---|---|
| **MIL** (Attn→Mamba) | $V\to x$, $K\to B$, $Q\to C$, $O\to out\_proj$ | conv, dt, $A$, $D$, gates | basic token mixing geometry |
| **DIL** (Attn→GDN) | Q/K/V/O into fused projections (respecting GQA) | delta/gates/conv specifics | head-group structure + output proj |
| **KIL** (Attn→KDA) | Q/K/V/O (tile if needed) | gate low-rank factors, beta, conv, $A$, dt | projection subspaces |

::: notes
30s

Here's a mapping table for our three layer converters: MIL, DIL, and KIL.
Each converter maps attention parameters into the target mixer, preserving different structures.
There is no one-to-one mapping for all parameters; some are randomly initialized.
It shows what each converter preserves from the original attention layer.
:::

## Why full attention is still needed

**What efficient mixers can't do (or not well enough):**

1. **Exact retrieval**: "What was the 7th word?" requires indexing, not blending
2. **Rare-key lookup**: Infrequent patterns get washed out by compression
3. **Long-range verbatim copying**: State is lossy

**Empirical symptom:** degraded in-context learning on retrieval-heavy tasks.

**The hybrid thesis:**

A few FA layers handle exact operations.

The rest: efficient mixers for throughput.

**Next question:** How few FA layers? Where do they go?

::: notes
60s

All efficient mixers are lossy compressors. Three things break:

You can't ask "what was the 7th word?" There's no way to index — everything is blended together.

Rare patterns fade because the state has limited capacity. Distinct memories blur.

Long-range copying fails because of exponential decay. Distant tokens just fade away.

Empirically, you see this as degraded in-context learning on retrieval-heavy tasks.

That's why nobody uses pure efficient mixers for serious LLMs.
Every production model keeps some full-attention layers for these exact operations.

That's the hybrid thesis: let efficient mixers handle the bulk, let attention handle the exceptions.
:::

## The placement problem

### We can't just swap attention layers whole-sale

Two coupled decisions:

1. **Ratio**: *how many* FA/SWA layers remain?
2. **Placement**: *which* layers remain FA/SWA?

| | From scratch | Conversion |
|---|---|---|
| **Placement** | Design pattern upfront | Discover progressively |
| **Circuits** | Model learns | Must respect existing |
| **Compute** | Pretraining scale | Fine-tuning scale |

**Why placement matters (in conversion):**

- attention layers form **inter-layer circuits**
- a layer can look unimportant alone, yet become **critical in combination**
- one-shot swaps break those circuits and destabilize training

**Apriel setting:**
we start from a strong pretrained base model, so we follow the **path of least resistance**:
replace the *easiest* layers first, then iterate.

::: notes
60s

So we've established that we need some full attention. The question is: how much and where?

If you're training from scratch, you can pick a pattern — say, attention every 8 layers — and the model learns to use it.

But we're converting a pretrained model. The circuits are already there.

So we can't just pick a pattern. We have to discover which layers are safe to replace.

That's why placement matters. Layers form circuits with each other. A layer might look unimportant alone, but be critical in combination.

Static importance scores lie — the ranking changes as you make replacements.

So we take the path of least resistance: replace the easiest layers first, then iterate.
:::

### Why one-shot conversion fails

A static importance score (e.g., remove one layer) assumes additivity.

In practice:

- layers interact nonlinearly (residual stream + normalization)
- low-importance layers can become **keystones** once other layers are replaced
- errors compound; training gradients shift; the ranking changes mid-flight

**Therefore:**
placement must be discovered **progressively**, not declared upfront.

::: notes
30s

Any static importance score assumes layers are independent. They're not.

Layers interact through the residual stream. A layer that looks unimportant alone can become a keystone once you've removed others.

So the ranking you compute upfront becomes wrong as you make changes. It gives you a good start, then it lies to you.

That's why placement must be discovered progressively.
:::

## Apriel-H1: progressive conversion

### Three-stage procedure

**Goal:** convert attention → Mamba while preserving quality.

```
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│        STAGE 1         │    │        STAGE 2         │    │        STAGE 3         │
│    Static scoring      │───▶│    Dynamic probing     │───▶│          SFT           │
│                        │    │                        │    │                        │
│  LOO importance        │    │  Probe each layer      │    │  Fine-tune on          │
│  -> replace 25         │    │  -> batch replace      │    │  reasoning data        │
│                        │    │                        │    │                        │
│  Output: H-25          │    │  H-25 -> H-30 -> H-40  │    │  Output: ship-ready    │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
```

Teacher throughout: **original Apriel-15B**.

::: notes
45s

Here's the procedure. Three stages.

Stage 1: use a static importance score to find the easiest layers to replace. We swap about half the attention layers in one go — 25 out of 50 — and distill from the original model.

Stage 2: the static score stops working. Layers that looked unimportant in isolation start mattering once you've removed others. So we probe each remaining layer, measure how easy it is to replace now, and swap the easiest ones in small batches. This gets us from 25 to 40 replaced layers.

Stage 3: distillation gets you close, but there's a last-mile problem. A final SFT pass stabilizes instruction-following and reasoning behavior.

The teacher throughout is the original Apriel-15B. We always distill from the same source.
:::

### Stage 1: static scoring + first big swap

**LOO importance:** for each attention layer $\ell$, replace with identity, measure eval drop on MMLU.

- small drop → easier to remove *in isolation*
- large drop → likely essential early on

**Action:** replace bottom 25 layers with MIL-initialized Mamba, distill end-to-end (reverse-KL, $T = 1.0$).

**Output:** H-25 checkpoint.

::: notes
30s

LOO is cheap and gives a reasonable starting set. Replace layer with identity — no mixing beyond residual path — and measure the drop.

We swap the 25 least important layers in one go, initialize with MIL, and distill from the original model using reverse-KL. That's mode-seeking: teacher is confident, student should commit rather than hedge.
:::

### Stage 2: dynamic probing (MMR)

**Why LOO breaks down past ~25:**

- "unimportant" layers collide: removing many at once creates a new failure mode
- importance is no longer local; it depends on what's already swapped

**MIL–Mamba Replacement (MMR):** for each remaining attention layer $\ell$:

1. swap $\ell$ → Mamba (MIL init)
2. run a short distillation probe (e.g., 100 steps)
3. record distillation loss $L_{\text{probe}}(\ell)$

Low $L_{\text{probe}}$ → easy to replace now. High → defer it.

**Then:** replace in small batches (3–5) by increasing $L_{\text{probe}}$.

**Progression:** 25 → 27 → 30 → 34 → 37 → 40 replaced layers.

::: notes
45s

Past 25 layers, LOO breaks down. Layers that looked unimportant in isolation start mattering once you've removed others.

So we need a dynamic score. MMR probes each remaining layer: swap it to Mamba, run 100 training steps, measure the distillation loss.

Low loss means easy to replace now. High loss means entangled — defer it.

Then we replace in small batches, easiest first: 25 to 27, then 30, 34, 37, 40.
:::

### Stage 3: end-to-end SFT stabilization

After reaching the target mixer count:

- run SFT until reasoning metrics stabilize
- fix instruction adherence, reduce drift from base behavior

Totals (Apriel-H1):

- distillation: 55.9B tokens
- SFT: 20.9B tokens

::: notes
15s

Emphasize: distillation gets you close; SFT makes it usable.
:::

## H1 results

### Quality vs throughput tradeoff

![Apriel-H1 Eval Score vs Throughput](/images/apriel-h1-eval-score-vs-throughput.png){width=100%}

More Mamba layers → higher throughput, small quality delta.

SFT recovers most of the quality loss.

::: notes
35s

Here's the punchline: the quality-throughput tradeoff.

Baseline Apriel-15B sits at 1× throughput on the left. As we convert more layers to Mamba, we move right — faster inference — while quality stays mostly flat.

The flagship H-30-SFT is the sweet spot: 2.1× throughput, 77 billion tokens of training.

And here's the strategic point: you can spend those throughput savings on longer chains of thought at inference time. That often recovers the quality delta and then some.
:::

### Checkpoint progression

| Checkpoint | Mamba | Throughput | Avg eval | Tokens |
|------------|-------|------------|----------|--------|
| Apriel-15B-Thinker | 0 | 1.0× | 0.80 | — |
| H-25 | 25 | 1.8× | 0.75 | 26B |
| H-27 | 27 | 1.9× | 0.76 | 15B¹ |
| H-30 | 30 | 2.1× | 0.74 | 37B |
| **H-30-SFT** | 30 | **2.1×** | **0.78** | **76.8B** |
| H-34 | 34 | 2.5× | 0.71 | 63B |
| H-37 | 37 | 2.9× | 0.69 | 89B |
| H-40 | 40 | 3.4× | 0.69 | 137B |

¹H-27 used an earlier H-25 checkpoint as starting point, hence fewer tokens.

**Flagship:** H-30-SFT — 2.1× throughput, 0.78 avg eval, 76.8B tokens, released as [Apriel-H1-15b-Thinker-SFT](https://huggingface.co/ServiceNow-AI/Apriel-H1-15b-Thinker-SFT)

::: notes
45s

The progression shows how we walked the ratio up gradually. Notice the linear decay in performance as we add more Mamba layers — a smooth tradeoff.

Stage 1 gets us to H-25: 1.8× throughput, eval drops to 0.75.

Then SFT recovers quality: H-30-SFT is fine-tuned on high-quality reasoning traces and merged with an earlier H-30 checkpoint. That brings eval back to 0.78 while keeping 2.1× throughput.

Beyond H-30, we traded more quality for speed: H-34 at 2.5×, H-37 at 2.9×, H-40 at 3.4×.

If someone asks about Nemotron-Nano-9B-v2 dominating the plot: that model was pretrained from scratch as a 12B hybrid on 20 trillion tokens, then SFT'd, GRPO'd, and pruned to 9B. We can't match that with conversion — but conversion ships faster on existing models.
:::

## H-30-SFT: the flagship hybrid

### Benchmark breakdown

![Benchmark breakdown](/images/apriel-h-vs-apriel-15b-eval-thrput-comparison.png)

::: notes
15s

The benchmark breakdown shows where quality held and where it dipped. Math and conversation improved. The hardest reasoning tasks dipped slightly.
:::

### Benchmark numbers

| Metric | Apriel-15B (teacher) | H-30-SFT | Δ |
|--------|----------------------|----------|---|
| MATH500 | 0.90 | **0.92** | +0.02 |
| MT-Bench | 8.30 | **8.58** | +0.28 |
| GSM8k | 0.97 | 0.95 | −0.02 |
| GPQA | 0.59 | 0.55 | −0.04 |
| AIME24 | 0.70 | 0.65 | −0.05 |

::: notes
15s

The benchmark breakdown shows where quality held and where it dipped. Math and conversation improved. The hardest reasoning tasks dipped slightly.
:::

### Final architecture

20 FA + 30 Mamba

```
         1  2  3  4  5  6  7  8  9 10
  1-10   █  █  █  ░  █  ░  ░  ░  ░  █
 11-20   █  █  █  █  █  █  █  █  █  █
 21-30   ░  ░  ░  ░  ░  ░  █  ░  █  █
 31-40   ░  ░  ░  ░  █  ░  ░  ░  ░  ░
 41-50   ░  ░  ░  ░  ░  ░  ░  ░  █  ░

█ = FA (20)    ░ = Mamba (30)
```

::: notes
20s

Notice layers 11-20 are all full attention — a dense block in the middle. The procedure discovered this pattern; we didn't design it.

Early layers have some mixing. Late layers are almost all Mamba except layer 35 and 49.
:::

## What H1 taught us

### What worked vs what hurt

**Worked:** weight transfer, staged conversion, reverse-KL, data choice, throughput-quality tradeoff.

**Hurt:**

<table>
<colgroup>
<col style="width: 20%">
<col style="width: 80%">
</colgroup>
<tr><td>Building</td><td>[LOO]─[probe]─[swap]─[probe]─[swap]─... → months of manual iteration, white-glove treatment</td></tr>
<tr><td>Shipping</td><td>separate checkpoint → switching cost, prove no regression, convince risk-averse org and customers</td></tr>
</table>

**Result:** slow to build, hard to ship.

::: notes
60s

H1 worked. Weight transfer mattered — random init was much harder. Staged conversion was necessary to avoid destabilizing the model. Data choice for distillation was critical. Reverse-KL on reasoning data preserved quality. The final tradeoff between throughput and quality across checkpoints was good.

But H1 was painful in two ways.

It took a whole summer to build. Every few layers we had to probe, score, swap, distill, evaluate. Treat the model with white gloves. Layer freezing didn't help. Pre-warming beyond MIL didn't help; teacher-forcing mixer outputs wasn't worth the overhead.

Shipping: even after we built H1, deploying it was its own battle. H1 is a completely different checkpoint. To ship it, you're asking the org to replace a model that's already in production and working fine. You have to prove there's no regression anywhere. You have to show the throughput gain is worth the switching cost. Neither org nor customers want model risk. And there's always someone who says "this must be worse for my use case because you removed attention."

So even if H1 is technically good, there was friction to adoption.
:::

### The surprising insight: data choice

Pretraining data ──▶ Distill ──▶ ✗ FAILED (quality collapsed)

Reasoning traces ──▶ Distill ──▶ ✓ WORKED (quality preserved)

**Lesson:** match data to the capability you're preserving, not the capability you're building.

::: notes
30s

Initially we thought a new Mamba mixer needs broad exposure, as in pretraining. So we distilled on pretraining data, hoping the model would learn to reason with Mamba. But the quality was terrible.

What ended up working was distilling on reasoning traces from the teacher's SFT data.

We think this is because reasoning patterns live in attention circuits (retrieval heads, induction heads, long-range dependencies), and the hybrid needs concentrated examples where reasoning is visible to find new paths.
:::

### Why reverse-KL for distillation

**Forward KL** (mean-seeking): $\mathrm{KL}(p_T \| p_\theta)$

- student covers all modes of teacher
- can spread mass too thin; blurs sharp behaviors

**Reverse KL** (mode-seeking): $\mathrm{KL}(p_\theta \| p_T)$

- student concentrates on high-probability regions of teacher
- preserves crisp, confident behaviors
- can miss minor modes (usually fine for reasoning)

**Why we use reverse KL:** we distill on data the teacher has already seen — so it's confident and correct. Reverse KL's mode-seeking behavior makes the student commit to those predictions rather than hedge.

::: notes
45s

Keep it short; this can turn into a seminar if you let it.

The key insight: we're distilling on data the teacher was trained on. The teacher is confident and correct here. Forward KL would average over modes and produce softer outputs. Reverse KL makes the student commit — which is what you want when the teacher knows the answer.

If someone asks why reverse KL is mode-seeking: it's because it penalizes the student for putting mass where the teacher has none. So the student learns to focus on high-probability regions of the teacher.
Can we see this from the equations? Yes: KL(p_theta || p_T) = ∑ p_theta (log p_theta - log p_T). If p_T is near zero, then log p_T goes to -∞. Thus KL blows up unless p_theta is even closer to zero. So the student avoids putting mass where the teacher has none. It's saying: Don't put probability mass outside the teacher's support.

By contrast, forward KL = ∑ p_T (log p_T - log p_theta). If p_T > 0 but p_theta is near zero, log p_theta goes to -∞, so KL blows up unless p_theta puts mass there. So the student tries to cover *all* modes of the teacher.
:::

## Apriel-H2: from slow cooking to meal prep

H1 had two problems: **slow to build**, **hard to ship**. H2 addresses both.

|  | H1 | H2 |
|---|---|---|
| **Build** | `[probe]─[swap]─...` → months | `[supernet]─▶[search]` → days |
| **Ship** | "replace your model" (friction) | "same model, turbo mode" (easy) |
| **Weights** | separate checkpoint | ~80% shared across placements |

**Supernet:** train all mixer options at once. Search is training-free.

::: notes
30s

H2 addresses both problems from H1.

Build speed: instead of probing one layer at a time for months, we train a supernet with all mixer options in parallel. Then search is training-free — days instead of months.

Deployment: the supernet approach means all placements share most of their weights. MLPs, norms, embeddings are identical — only the mixers differ. That's roughly 80% weight sharing.

So you can ship multiple operating points from one model. The all-FA path is your safe default. The efficient path is turbo mode. Nearly free to host both.
:::

## H2 search

### Supernet + beam search

**Supernet training:** each layer has multiple mixers, sample one per step so that all get trained.

```
Step 1:              FA
   ╔═════╗ ┌─────┐ ╔═════╗ ┌─────┐ ╔═════╗
 ┄─╢ FA  ╟─┤ MLP ├─╢ SWA ╟─┤ MLP ├─╢ FA  ╟─┄
   ╚═════╝ └─────┘ ╚═════╝ └─────┘ ╚═════╝
     SWA             GDN             SWA
     GDN              ┄              GDN
      ┄                               ┄

Step 2:              FA
                     SWA             FA
   ╔═════╗ ┌─────┐ ╔═════╗ ┌─────┐ ╔═════╗
 ┄─╢ FA  ╟─┤ MLP ├─╢ GDN ╟─┤ MLP ├─╢ SWA ╟─┄
   ╚═════╝ └─────┘ ╚═════╝ └─────┘ ╚═════╝
     SWA              ┄              GDN
     GDN                              ┄
      ┄
```

**Search:** beam search over architectures (training-free), guided by quality + throughput.

::: notes
30s

A supernet is a model where each layer position has multiple mixer options. During training, we sample one mixer per layer per forward pass. All mixers get gradients over time.

The key: all candidates get trained in parallel, in the same run. No separate experiments per placement.

After training, we search. The search is training-free — we just evaluate different paths through the supernet. Beam search guided by quality proxies and throughput estimates.

Output: a small set of candidate placements for full evaluation.
:::

### The cost model: why counts are enough

**Key insight:** throughput depends only on *how many* of each type, not *where*.

$$\text{cost}(\text{placement}) = \sum_m \text{count}_m \times \text{cost}_m$$

where cost is relative kernel time (FA = 1.0):

| FA | SWA | GDN | KDA |
|:--:|:---:|:---:|:---:|
| 1.00 | 0.16 | 0.09 | 0.11 |

**Budget:** pick a target cost (e.g., 0.5 × baseline → ~2× throughput).

**Implication:** we can enumerate all feasible allocations *before* searching placements.

::: notes
30s

The key insight that makes search tractable is that throughput depends on counts, not placement. If you have 20 FA layers and 30 GDN layers, the throughput is the same no matter which positions they're in.

We benchmark each mixer type once, get relative costs: FA is 1, SWA is 0.16, GDN is 0.09, and so on.

Then total cost is just count times cost, summed over types. That's it.

This means we can set a budget — say, half the cost of all-FA for roughly 2× throughput — and enumerate every allocation that fits the budget before we even think about placement.
:::

### Two-level search: allocations → assignments

Hierarchical search separates cost from quality:

```
Outer loop: enumerate allocations (how many of each type)
  e.g., {FA: 20, SWA: 0, GDN: 30} at cost 0.454
        {FA: 15, SWA: 5, GDN: 30} at cost 0.498
        {FA: 10, SWA: 10, GDN: 30} at cost 0.542 (exceeds budget, skip)
        ...
  └── Inner loop: beam search over assignments (which layers get which type)
        ├── [FA FA GDN GDN FA GDN ...] → score 0.82
        ├── [GDN FA FA GDN GDN FA ...] → score 0.79
        └── [FA GDN FA GDN FA GDN ...] → score 0.81
              └── expand best candidates
```

**Cost** depends on counts. **Quality** depends on placement.

Separating them shrinks the search space dramatically.

::: notes
30s

So we have a two-level search.

The outer loop just enumerates allocations — how many of each mixer type. That's a small combinatorial space, and we can filter by budget instantly.

For each allocation, the inner loop searches placements — which layers get which type. That's where quality lives. Two placements with the same counts can have very different quality because layer interactions matter.

Beam search in the inner loop: start with random assignments, score them, keep the best, perturb and repeat. Greedy hill-climbing with a beam width to avoid local minima.

Separating allocation from assignment is what makes this tractable. We don't search the full space — we search a hierarchy.
:::

### Scoring candidates: importance sampling

| Precompute (once) | Evaluate (per placement) |
|---|---|
| Teacher generates $N$ completions; store tokens, correct, $\log p(x_i)$ | Forward pass $\log q(x_i)$, weights $w_i = q(x_i)/p(x_i)$, accuracy $\widehat{\text{acc}} = \frac{\sum_i \mathbf{1}[\text{correct}_i] \cdot w_i}{\sum_i w_i}$ |

**Why fast:** generative eval = autoregressive generation. IS = forward pass only.

**Caveat:** IS estimates forward KL. Works when student covers teacher's mass.

::: notes
60s

Generative eval generates from each candidate and measures accuracy — slow, autoregressive, hours per placement.

IS is faster: pre-generate completions from the teacher once, store the log-probs. Then for each candidate, just run forward passes to get log q(x), compute importance weights, and take the weighted average of correctness. Minutes per placement, embarrassingly parallel.

The formula is self-normalized IS: we weight each sample by how much more likely it is under the student than the teacher, then normalize. If the student assigns high probability to correct completions, the accuracy estimate is high.

The connection to divergence: the expected log-weight is negative forward KL. Low forward KL means student covers teacher well, weights stay near 1, estimate is reliable. High forward KL means weights vary wildly, estimate has high variance.

Why forward KL and not reverse? Reverse KL would require sampling from the student — that's exactly what we're trying to avoid. We have teacher samples, so forward KL is what we can measure.

The tension: we train with reverse-KL (mode-seeking), but IS measures forward-KL (coverage). These align when the teacher is confident — one clear mode that the student finds. If the teacher is diffuse, a mode-seeking student might miss mass that forward KL catches.

Diagnostic: effective sample size (ESS). ESS = (Σw)²/Σw². ESS near N means weights are uniform, estimate is solid. ESS << N means student has drifted, be suspicious.
:::

## The deployment payoff: a matryoshka model

After beam search, prune supernet and continue training:

```
Supernet ──▶ prune ──▶ [all-FA] + [7.5×] + [10×] ──▶ SFT/RL ──▶ ship
                      └── ~80% weights shared ──┘
   we are here ▲
```

**Ship as one model with multiple engines:**

- All-FA path = safe default (drop-in replacement for original teacher)
- Efficient paths = turbo modes (opt-in per workload)
- vLLM exposes each placement as a separate engine

**Every workload gets the right path:** quality-sensitive → all-FA, throughput-sensitive → efficient. No forced trade-off.

::: notes
45s

Here's the deployment payoff.

After search finds good placements at various throughput targets, we prune the supernet to keep only those placements plus the all-FA baseline. Then we continue training — SFT, RL, whatever your post-training pipeline is. The pruned supernet isn't frozen.

Because 80% of weights are shared — MLPs, norms, embeddings — hosting multiple placements is nearly free. You're adding maybe 20% memory for the efficient mixer weights.

The all-FA path is your safe default — it's a drop-in replacement for the original teacher. If someone says "this must be worse for my use case," fine — use the all-FA path for that use case. The efficient paths are opt-in turbo modes.

vLLM can expose each placement as a separate model engine, even though they share most weights. One deployment, multiple operating points.

This is "keep your model, add turbo mode" — not "replace your model with something new."
:::

## Where the field is today

### Who's shipping (2025)

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

- **Dec**: [NVIDIA Nemotron 3 Nano 30B-A3B](https://arxiv.org/abs/2512.20848)  
  **8:1** Mamba-2:GQA-FA hybrid, **up to 3.3× decode** vs Qwen3-30B-A3B-Thinking @8k/16k

::: notes
30s

Point: convergence happened in 2025; hybrids aren't speculative.
If someone challenges a number, point to the primary source link on-slide.
Transition: "so what did *we* ship, and what did we learn?"
:::

## Where I think the field goes next

### Predictions (2026–2027)

1. **Intelligence per second** becomes the primary objective function.
2. **Efficient mixers commoditize** — hybrids become the norm for production LLMs.
3. **Single-checkpoint multi-placement** becomes standard; differentiation moves to workload-aware routing.
4. **From-scratch hybrids win ceiling**, distillation wins time-to-ship.
5. **Post-training becomes architecture-aware**: hybrid placement ↔ SFT/RL recipes co-designed.

::: notes
60s

Five predictions for the next 12 to 18 months.

First: intelligence per second becomes the metric. Not just quality, not just throughput — the ratio. When you're paying per token, this is what matters.

Second: efficient mixers commoditize. Mamba, linear attention, gated delta nets — these become standard building blocks. Hybrids become the default for production LLMs.

Third: single-checkpoint multi-placement becomes standard. You train one supernet, prune to multiple configurations, share 80% of weights. Switching costs drop. Differentiation moves upstream: which placement for which query.

Fourth: from-scratch hybrids win the quality ceiling. Distillation wins time-to-ship. Both matter depending on constraints.

Fifth: post-training becomes architecture-aware. SFT and RL recipes get co-designed with hybrid placement. You don't just fine-tune a model — you fine-tune a model-placement pair.

These are falsifiable. Check back in 18 months.
:::

## Takeaways

Ship → Iterate → Frame → Compound

| Lesson | Evidence |
|:--|:--|
| **Hybrids ship** | 2.1× throughput, quality held, vLLM today |
| **Speed compounds** | placement search: months → days |
| **Framing enables** | "turbo mode" > "new architecture" |
| **Decisions compound** | ...when switching costs ↓ |

**→ Where this matters:** enterprise cares about latency, cost, controllability, regressions — not leaderboard peaks.

::: notes
45s

Four takeaways.

Enterprises don't care about leaderboard peaks — they care about latency, cost, controllability, and not regressing on stuff that already works. That's the constraint, and I think it's a good one. It forces you to actually ship.

First: hybrids ship. We have checkpoints running in vLLM with 2× throughput and quality held.

Second: iteration speed compounds. Supernets cut placement search from months to days. That changes what you can explore.

Third: framing matters. "Same model, turbo mode" is a much easier sell than "replace your architecture with something new." Switching costs are real.

Fourth: architecture decisions compound — but only when they reduce switching costs. That's the throughline.
:::

::: {data-background-color="#000"}

# Thank You

**Apriel-H1:** [Paper](https://arxiv.org/abs/2511.02651) · [Model](https://huggingface.co/ServiceNow-AI/Apriel-H1-15b-Thinker-SFT) · [Blog](https://huggingface.co/blog/ServiceNow-AI/apriel-h1)

**More:** [Speed Creates Capability](https://tscholak.github.io/posts/speed-creates-capability.html) · [Fast-LLM](https://github.com/ServiceNow/Fast-LLM)

:::

::: {data-background-color="#000"}

## The Team

**Foundation Models Lab — ServiceNow AI Research**

Oleksiy Ostapenko, Luke Kumar, Raymond Li, Denis Kocetkov, Joel Lamy-Poirier

**Contact**: Torsten Scholak (<torsten.scholak@servicenow.com>)

:::
