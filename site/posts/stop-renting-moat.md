---
title: "Stop Renting Moat: Why Enterprises Must Own Part of Their AI Stack"
date: Aug 20, 2025
teaser: |
  Enterprises are told they have to choose between cost or differentiation. But that's a false choice. If you outsource all AI, your costs stay volatile and your upside is capped. The winners will own enough of the stack to control both unit cost and quality.
tags:
  items: [ai, strategy, economics]
---

## The false choice

Corporate strategy decks love to paint things as a neat fork in the road. You've probably seen it: on one side, rent everything. Stay light, keep costs down, let the big labs handle the heavy lifting. On the other, build everything yourself. Hire a frontier research team, rack up thousands of GPUs, become your own OpenAI.

That framing makes for a tidy slide, but it's not reality. No enterprise should try to out-OpenAI OpenAI. That's not the game. On the other hand, renting everything leaves you totally exposed. Your costs swing with someone else's pricing strategy, and any "moat" you think you have can be bought by your competitors tomorrow. You are just an integrator of someone else's commoditized technology.

The real choice isn't between these two extremes. Smart companies stake out the middle ground: **own just enough** to control what matters. You don't need your own frontier model lab, but you can't afford to own nothing either.

What you need is the **learning loop**. That means locking down evaluation, securing rights to the data your systems generate, building privacy-safe feedback mechanisms, and maintaining the ability to adapt models on **your** schedule, and not when a vendor decides it's convenient.

Owning that path gives you predictable unit economics and a moat that compounds. Outsource it all and you lose both.

## Prices move. Your moat shouldn't

Do you believe that all providers will eventually offer the same capabilities at the same, ever-shrinking price? If so, you may be underestimating the complexity of the market.

What we are witnessing is a **subsidy race**. Providers are desperate for scale and market lock-in. Labs burn cash to win share, and hyperscalers cross-subsidize from their profitable core business. The current prices are a reflection of that strategy and not one of cost.

For a typical agent/RAG blend (approx. 3:1 input:output), OpenAI [charges](https://platform.openai.com/docs/pricing) about **$3.44 per million tokens on GPT-5**, **$3.50 on o3**, and **$1.05 on 4o-mini**. Prices may look cheap now, but they can and will shift again, and not necessarily downwards.

According to [internal documents reported by The New York Times](https://techcrunch.com/2024/09/27/openai-might-raise-the-price-of-chatgpt-to-22-by-2025-44-by-2029/), OpenAI plans to raise ChatGPT Plus from $20/month to **$44/month by 2029**, more than doubling current prices. This while the company reportedly operates at a loss, with [revenue](https://www.reuters.com/business/openai-hits-12-billion-annualized-revenue-information-reports-2025-07-31/) unable to cover [operational costs](https://www.investing.com/news/stock-market-news/openai-hits-12-bln-in-annualized-revenue-sees-higher-costs-the-information-4161634).

The volatility is already here. **Cursor** [walked back](https://cursor.com/blog/june-2025-pricing) a flat-rate plan after usage spiked and heavy users blew up the economics. **Perplexity** grew fast but still **burned ~$65M on $34M revenue in 2024**. This is a reminder that [API spend can already easily outrun sales](https://www.theinformation.com/articles/google-challenger-perplexity-growth-comes-high-cost). If you are an enterprise relying on APIs today, your P&L becomes a derivative of somebody else's burn rate and fundraising calendar.

## What to own (and why)

You don't need to own everything, but you can't afford to own nothing. The trick is knowing which parts of the stack matter: the places where volatility destroys you and where defensibility compounds. Everything else you can rent.

Think of the AI stack as having three tiers:

| Tier          | Approach                | Unit cost    | Differentiation | Control | Example               |
| ------------- | ----------------------- | ------------ | --------------- | ------- | --------------------- |
| Commodity API | Rent it all             | Volatile     | None            | None    | Basic agentic chatbot |
| Hybrid        | APIs + fine-tunes       | Mid, blended | Moderate        | Partial | RAG with custom data  |
| Own core loop | Internal models & evals | Predictable  | High            | Full    | Convirza, GS Platform |

The commodity tier is where most enterprises start and never escape. They plug GPT-4o or GPT-5 into every workflow and pat themselves on the back. It works, but it's indistinguishable from what their competitors are doing. They're reselling OpenAI with a bit of integration glue. As soon as the API provider shifts pricing, their margins evaporate.

The hybrid tier feels safer but is actually worse. You spend on fine-tunes and custom RAG pipelines but still depend on someone else's pricing whims and roadmap. Your supposed "moat" is just a brittle collection of prompts and data munging scripts. Anyone can copy it. You're paying more without buying freedom.

Owning the core learning loop is the only move that compounds. That doesn't mean training GPT-scale models from scratch, but putting yourself in control of what ships, what learns, and what lasts. You define the evaluation gates and the regression rules. You decide what data you capture, how it's anonymized, and whether it flows back into a global model. You secure guaranteed compute for adaptation, not just inference. You build a feedback loop that learns from every deployment, not just the ones OpenAI decides to optimize for.

That requires structuring data rights from day one. Tenants need three modes: **off** (nothing leaves their silo), **local-only** (improvements stay within their instance), or **global** (anonymized data improves the shared model). Start everyone on local, but build in incentives for moving to global as trust grows. You can't bolt this on later. The contracts have to be right from the first pilot.

Here's what compounding looks like in practice: **Convirza** migrated from OpenAI GPT models to Llama 3-8B in 2024 and, per [Meta's Llama Community Stories](https://www.llama.com/community-stories/), cut operational cost by **10x** while improving F1 by **8%** and throughput by **80%**, now serving **60+ KPIs** across thousands of daily interactions. **Goldman Sachs** hosts multiple models, including Llama, behind its firewall. [Reported results](https://nanonets.com/blog/goldman-sachs-ai-platform/) include **20% developer productivity gains** and **15% fewer post-release bugs**, with adoption above 50% across 46,000 employees.

The more these systems run, the better they get. Your unit economics become predictable instead of volatile. Your accuracy rises in domains where competitors flatline. And your roadmap stops being a derivative of someone else's burn rate.

Do you want compounding or commoditization? Own the loop, and every deployment makes you stronger. Rent it all, and you're just financing someone else's moat.

## When to switch from rent to own

Conventional wisdom says to wait: Wait until your volume is high enough, wait until API prices stabilize, wait until the tooling matures. But waiting is not neutral. Every month you hold back is a month your competitor's system gets smarter while yours doesn't exist.

There are only two real triggers. The first is steady spend: if your token bill is already in the five-figure range each month, you are exposed to someone else's pricing decisions. The second is workload criticality: if accuracy, latency, and compliance define your product, you cannot build an advantage on rented APIs.

The trap is telling yourself you will switch "later." But by then you are locked in. Your QA processes are built around one provider's quirks, your customers expect a specific output style, your prompts have hundreds of workarounds for model-specific bugs, and the cost of change feels impossible. Meanwhile, your competitor who started small has already tuned open-weights models to their data, cut their costs by **10x**, and built evaluation gates that compound every cycle.

If AI touches your core product, every day you rent the loop is a day you hand your roadmap to someone else. Start early, even when the economics seem to look wrong. They only look right after you have already built capability.

## What to do next

The rent-or-build dichotomy is fake. You need to own enough of the learning loop to compound. You don't need a mega-lab for this. You need a durable, small stack. Open weights are the clay, and your learning loop is the sculptor. If you don't do this, you are just another cheap renter and not a winner.

The winners will be the ones who turn every deployment into compounding advantage. Convirza cut costs **10x** while improving quality. Goldman Sachs boosted productivity **20%** while reducing defects. This is what happens when you own the learning loop.

Prices will keep shifting, and model deprecations will force rewrites, but real moats that are built on your data, your evaluations, your continuous improvement don't move.

**If you rent everything, OpenAI's roadmap is your roadmap.**  
If you own the loop, your roadmap is yours.
