---
title: "Trouble in LA LA Land: Did MiniMax Just Kill Efficient Attention?"
publication:
  status: published
  date: Oct 30, 2025
teaser: >
  When you're building efficient LLMs and a major industrial lab says "we tried this path, it doesn't work at scale," your first instinct is to listen (and cry). Maybe your assumptions are wrong. Maybe you're about to waste a lot of compute chasing a dead end.

  That's where I was this week.
tags:
  items: [ai, linear-attention]
image: mambamin.png
---

My team is betting big on hybrid architectures: mixing full and efficient attention  (SWA/sparse, linear/SSM, etc.) with learned layer placement. Then MiniMax dropped their [M2 post-mortem bomb](https://www.zhihu.com/question/1965302088260104295/answer/1966810157473335067) on October 25th explaining to everyone why they went back to quadratic attention for their 230B model. By October 29th, [the](https://x.com/giffmana/status/1983457240452673710) [internet](https://x.com/zpysky1125/status/1983383094607347992) [was](https://x.com/p_nawrot/status/1983579777844834459) [convinced](https://x.com/DBahdanau/status/1983909078414831987): efficient attention is dead. We told you so.

Except that's not what MiniMax said. The more I looked into their reasoning, the more I realized their decision actually strengthens the case for efficient hybrids in ways nobody's talking about.

## What MiniMax Actually Said

If you read MiniMax's post-mortem carefully, their conclusion is actually nuanced. They didn't say efficient attention is fundamentally broken. Their exact words: "一直在做，但是在工业系统里真的打过Full Attention还有些距离" (We've always been working on it, but truly beating full attention (FA) in industrial systems still has some way to go). In other words, efficient attention lacks production readiness at their scale with today's infrastructure. So it's a question of not if, but when.

And yeah, I get it. Anyone who's tried to run linear attention in production knows the pain. I can personally attest to this, because we did it in our [Fast-LLM](https://github.com/ServiceNow/Fast-LLM) training framework this year. Build failures from `mamba-ssm` or FLA with cryptic CUDA linker errors. An afternoon hunting for the magic combination of torch version, CUDA toolkit, and kernel implementation that doesn't segfault. Getting TP to work wasn't straightforward either. Then the serving stack surprises you by missing chunked prefill, speculative decoding, or prefix caching. On that note, prefix caching is the reason why efficient attention often fails to deliver in practice: once your serving stack caches the prefix, the memory and compute savings compared to FA vanish. And that's exactly where most production traffic lives.

Also, FA just works. The kernel ecosystem is mature, and the whole ecosystem is built around its assumptions. It's still getting better all the time. For a 230B production model serving millions of users, you really can't mess around.

Then there's model quality. Their [Lightning Attention hybrid](https://arxiv.org/abs/2501.08313) crushed the standard benchmarks everybody uses for pretraining ablations that work without much post-training. Your usual suspects, MMLU, BBH, LongBench, etc. But when they scaled to large-scale, multi-hop reasoning and agent tasks with genuinely long contexts, big cracks appeared. After their pretrained FA model was converted to Lightning Attention hybrids, performance on these tasks dropped significantly. They attributed this to the hybrid's inability to maintain the complex attention patterns that the model had developed during pretraining. These are retrieval heads, induction heads, and long-range coherence mechanisms that become structurally essential. They tried detecting critical heads and keeping only those as FA, but they weren't able to reliably identify and retain all the patterns.

In the end, they chose certainty over risk. For their timeline and scale, that was the right call.

But while MiniMax was wrestling with efficient attention, a different line of research was quietly changing the landscape.

## The Delethink Twist

The recent [Markovian Thinker](https://arxiv.org/abs/2510.06557) work shows something that will send shockwaves through the efficient attention debate: reasoning is naturally Markovian. Put simply, when models think step by step, they rarely need to remember the entire chain of thought. What matters is the most recent slice, the current working memory.

Delethink from the paper is a technique that exploits this. The idea: Train a model to reason in 8K-token chunks. Then, at the chunk boundary, delete the first part of the reasoning chain, keep only the last portion as carryover state, and continue. Sounds weird and counterintuitive, but this works reasonably well already for large off-the-shelf models, i.e. unmodified gpt-oss. Through RL even 1.5B parameter models can learn to work under the Delethink Markovian constraint. They showed that such a Delethink-enabled model can think in 8K chunks and match standard LongCoT-RL performance trained with full 24K context.

I will spell this out for you: **Delethink lets you run FA models with effectively linear memory and compute for reasoning tasks.** You chunk the reasoning chain, delete the old parts, and only keep the recent context. This quietly changes the story for FA. If you're running a thinking process that generates 50K tokens, FA with Delethink gives you O(1) memory and O(n) compute. The quadratic blowup disappears. With this, suddenly we have a credible way to stay on plain FA, handle longer chains, lower memory, and stick with infrastructure that already works. The pressure to migrate away from FA drops significantly.

So MiniMax's decision makes sense for single-shot reasoning. They (and everyone else) can run FA with Delethink, avoid the engineering pain of efficient attention, and sidestep the quality risks of hybrids. But what about M2's actual workload? Does M2 do single-shot reasoning?

## The M2 Problem

No, it doesn't. M2 follows the standard [interleaved-thinking pattern](https://huggingface.co/blog/MiniMax-AI/aligning-to-what#the-need-for-interleaved-thinking): it emits `<think>...</think>` blocks, and these blocks accumulate over multi-turn conversations. Every bit of thinking history needs to be retained, and the [README](https://huggingface.co/MiniMaxAI/MiniMax-M2#inference-parameters) warns that removing them hurts performance. Thus every exchange adds more tokens, building up tens of thousands of reasoning tokens that must stay in memory.

Delethink can't help here because it only works for single reasoning chains, where you can truncate and carry forward a small state. But in multi-turn conversations, the thinking tokens from previous turns belong to conversation history. You can't delete all of them without negatively impacting performance.

That means the computational blow-up returns. The longer a dialogue continues, the heavier each turn becomes. MiniMax chose FA while adopting the interleaved-thinking pattern that makes quadratic scaling still painful. Every multi-turn conversation pays that quadratic tax again and again.

So is efficient attention back in the game?

## What Jet Nemotron Shows

MiniMax's complaint was that you can't manually identify which attention heads need to be FA. True enough. But [NVIDIA's Jet Nemotron](https://arxiv.org/abs/2508.15884) work shows you can **learn** the placement.

They built a supernet where each attention block can swap between FA, SWA, and other efficient alternatives. During training, NVIDIA randomly switched between options at each layer so all paths get gradient signal. Hence the "supernet." Afterward, a lightweight search finds the optimal layer configuration under a compute budget.

For a 4B-parameter 36-layer model and a budget of 3 FA, 7 SWA, and 26 linear layers, that's about 30.5 billion possible architectures in the search space. But hierarchical beam search finds a top architecture efficiently. After optimizing the linear attention for the chosen layers further, they get their Jet-Nemotron-4B that beats Qwen3-1.7B on standard benchmarks (including reasoning-heavy math tasks) while delivering a massive 21x speedup at 64k tokens decoding length. The smaller Jet-Nemotron-2B is slightly behind Qwen3-1.7B but delivers 47x generation throughput speedup at 64k tokens and still 23x speedup at 8k tokens. Prefill speedups are more modest, around 2.5x at 64k and 6x at 256k tokens, but still significant.

This is a big deal, and I think MiniMax missed it. The gains from Jet Nemotron's learned placement are huge, and while the potential only fully shows with large prefill and long decode (because that's where FA is memory-bound), the impact is clear, because that's also the regime interleaved thinking with its accumulated context operates in.

But Jet Nemotron proved this only at 2-4B scale. M2 is 230B parameters, 57 times larger. Nobody has published results at that scale. We don't know if you still only need 3 FA layers at 230B scale, or if you need 10 or 20. We don't know what the compute budget looks like for training the supernet at the big scale. Until that evidence extends upward, FA remains the safe bet for massive models.

## The Nuanced Story

So, where does that leave us? It's clear that we can't just proclaim that linear attention blows FA out of the water anymore. I won't be able to argue that point convincingly anymore. The story now depends very much on workload and scale.

For single-shot reasoning, FA plus Delethink is pragmatic. Stable infrastructure, and the Markovian behavior is already there in pretrained models. That buys time for the efficient attention ecosystem to mature. At M2's scale, the infrastructure pain and model quality risks of hybrids outweigh the compute savings and speed benefits of efficient attention. FA with Delethink is the right call for the majority of single-shot reasoning workloads right now.

For multi-turn interleaved thinking, hybrids with learned placement will eventually become essential. Context accumulates, Delethink can't reset it, and FA's quadratic cost will dominate. Optimized hybrids will win that regime, and that's where the field is heading. That's where my team is heading. See you in the efficient-attention trenches.

---

**Update Oct 31, 2025:** The day after I posted this, [Kimi.ai released](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) [Kimi Linear Attention (KLA)](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda), a new hybrid architecture that validates the core argument here in ways even I didn't expect, at least not so soon.

KLA extends Gated DeltaNet with channel-wise gating (instead of scalar gating) and interleaves this efficient attention with full attention in a 3:1 ratio. That ratio matters: just like Jet Nemotron found that only 2-3 layers out of 36 need FA, Kimi found that roughly 25% FA is enough to maintain quality while cutting KV cache by 75% and delivering 6x decoding throughput at 1M context.

The results are great. On synthetic tasks like MQAR and Stack, KLA significantly outperforms Mamba2 and beats Gated DeltaNet. On real reasoning tasks (AIME 2025, MATH500, LiveCodeBench), it matches or beats both MLA (DeepSeek's compressed full attention) and Gated DeltaNet-H after the same SFT recipe. Pretraining scaling laws show 1.16x better loss at the same compute budget.

Two caveats: First, this is a 3B-activated, 48B-total parameter model. M2 is 230B total, so roughly 5x larger. We still don't know what happens at that scale, but the trend is promising. Second, Kimi uses a fixed 3:1 ratio rather than learned placement, so we don't know if that's optimal or just good enough.

But here's what matters: within days of MiniMax saying "efficient attention has some way to go," another major industrial lab shipped a production-grade hybrid that beats full attention on the metrics that matter.

The confusion MiniMax's post-mortem created also triggered a flurry of activity. Everyone working on efficient attention saw an opening and pushed out their work. [Brumby-14B](https://manifestai.com/articles/release-brumby-14b/) (an attention-free model converted from Qwen), [Higher-order Linear Attention](https://arxiv.org/abs/2507.04239), and several others all dropped within days. The Flash Linear Attention community [merged Kimi's optimized KDA kernel](https://github.com/fla-org/flash-linear-attention/pull/621) within hours. There's now active debate about which approach wins where.

What looked like a setback for efficient attention turned into its Streisand moment. MiniMax forced everyone to clarify their positions, tighten their arguments, and ship their code. The infrastructure is maturing faster than anyone expected. The timeline just got a lot shorter.
