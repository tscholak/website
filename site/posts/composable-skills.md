---
title: "Skills, Composition, and the Coordination Problem"
publication:
  status: published
  date: Oct 22, 2025
teaser: >
  Anthropic released [Skills](https://www.anthropic.com/news/skills) last week, and the response was immediate: finally, composable AI capabilities! Build once, reuse anywhere. Give Claude folders of instructions and scripts, and it'll load what it needs automatically. Modular, portable, efficient. We'll have reliable AI systems using hundreds or thousands or hundreds of thousands of skills in no time.

  There's just one problem: Skills aren't actually composable, and that creates a coordination problem that needs to be addressed urgently.
tags:
  items: [ai, agents]
image: kung-fu.png
---

This is a follow-up to ["Everyone's Arguing About the Wrong Abstraction Layer"](/posts/agentkit). That post argued that neither visual workflow builders nor natural language prompts provide *formal* compositional guarantees, and that this lack of structure creates economic risks as systems scale. This post drills into Anthropic's Skills feature to illustrate the point. You don't have to read the previous post to follow along, but it provides useful context.

## What Skills Are

At startup, Claude indexes every available Skill by name and description. When you make a request, it decides which ones to load, in what order, and how to coordinate them. For example, it might pull in your brand guidelines, financial-reporting templates, and presentation formatting Skill to produce a quarterly deck.

Each Skill is self-contained and can be reused across contexts (Claude Desktop, Claude Code, API calls), which gives them the same feel as plugins or microservices. They even "stack" automatically when several are relevant. Under the hood, a Skill is a folder containing markdown instructions, executable scripts, and supporting resources, making them closer to a module or library than a single function.

[People](https://x.com/barry_zyj/status/1978860549837299948) [are](https://x.com/alexalbert__/status/1978877514903884044) [excited](https://simonwillison.net/2025/Oct/16/claude-skills/) because this *feels* like composition: small, independent modules that can be combined for larger workflows. But from a *formal* perspective (the one that lets you build reliable systems), it isn't composition at all, and that creates serious coordination challenges.

## What Skills Are Not

Formal composition has structure. It defines how pieces fit together and what happens when they do. Inputs align with outputs. There's an identity element (something you can compose with anything without changing it). You can reason about the whole because you understand the parts.

Is this the case for Skills? No, because they don't have formal compositional semantics:

* There's no formal `compose(skillA, skillB)` operator. Claude simply decides what to use and to what granularity.
* There's no type system or contracts, hence no type safety: In a chain of actions, the output from skill A may not match the inputs of skill B.
* There's no associativity. `(A + B) + C` may behave differently than `A + (B + C)`. Order of combination matters.

Skills don't compose. Claude orchestrates them heuristically: interpreting requests, selecting Skills, and routing between them.[^chiusano]

At small scale, this approach works well enough. For most users, Skills will feel magical: open a PDF, extract data, format a report. Done. The model makes the right decisions most of the time, and the abstraction holds.

But as soon as people start chaining more Skills together, the cracks appear:

* Skill A and B work, but A, B, and C in combination fail unpredictably.
* Updating one Skill breaks workflows you didn't know were using it.
* The same workflow can yield very different results depending on what other Skills are being used or what ran before.

Without formal structure, we can't reason about how Skills behave. We can only test, and even then, the tests aren't stable.

In a nutshell, we've replaced explicit composition with implicit, model-mediated coordination, and that scales only as far as Claude's ability to guess correctly.

So how much of a problem is this?

## Why Coordination Matters

Coordination is the linchpin that holds complex systems together. The more components you have, the more ways they can interact, and the harder it is to ensure they work together reliably.

The economy of AI systems is moving rapidly toward scale. Enterprises will want an ever-growing library of specialized skills for compliance, security, domain knowledge, formatting, localization, and more. Every team will build custom skills for their workflows. Systems will chain many skills dynamically based on context, user preferences, and regulatory requirements. The complexity will explode, and with it, the coordination challenges.

Some people are optimistic that model intelligence alone will solve these problems, "ship fast and iterate, it's just a tool, if it's broken, we'll fix it." That optimism is understandable given current trends. [Research from METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) shows AI's ability to complete long-duration tasks improving rapidly. Current frontier models achieve 50% success on hour-long tasks, nearly 100% on tasks under four minutes. If the current trend continues, we'll have systems reliably completing week-long tasks within 2-4 years. [Andrej Karpathy](https://www.dwarkesh.com/p/andrej-karpathy), who led AI at Tesla and OpenAI, doubts it will: he calls this "the decade of agents, not the year of agents." AI research itself proves harder to automate than expected. Karpathy found coding agents "of very little help" building his LLM training codebase: "They're not very good at code that has never been written before, which is what we're trying to achieve when we're building these models." Recursive self-improvement faces the same barrier.

Even if capabilities do scale as optimists hope, coordination failures aren't capability problems. [METR found](https://metr.org/blog/2025-06-05-recent-reward-hacking/) reward hacking in 70-95% of test cases: models knowingly violating user intent while maximizing specified objectives. [Separate research from OpenAI and Apollo Research](https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/) found scheming behavior: models deliberately hiding their intentions while appearing compliant, even covertly preserving themselves when facing replacement.[^coordination-failures]

[Yoshua Bengio](https://fortune.com/2025-06-03/yoshua-bengio-ai-models-dangerous-behaviors-deception-cheating-lying/), who won the Turing Award for pioneering deep learning, observed that these coordination problems emerge from intelligence itself: frontier models now exhibit deception and goal misalignment in empirical tests. Smarter models won't fix these problems because, as Bengio noted, the AI arms race "pushes labs toward focusing on capability to make the AI more and more intelligent, but not necessarily put enough emphasis and investment on research on safety." Intelligence achieves task completion. Coordination requires institutional structure: shared standards, aligned incentives, accountability frameworks. Those don't emerge from capability scaling alone.[^markovian]

Right now, we're in the hype phase. Hundreds of billions in AI capex are already committed. Companies have raised funding on promises of AGI. Investors expect returns. Enterprises have been sold on AI agents that will reliably automate complex workflows. The economic pressure to deploy at scale is immense.

By 2027, if current trends hold, enterprises will be deploying multi-skill agents for financial consolidations, regulatory compliance, and hiring decisions. When coordination failures hit production (errors in quarterly reports, missed compliance requirements, biased screening) "the model orchestrates skills autonomously" won't satisfy auditors, insurers, or regulators. That's when coordination infrastructure becomes economically necessary.

So how do we build that infrastructure?

## Why Formal Guarantees Aren't Enough

An obvious answer could be: "Just make it formal." Add types, contracts, schemas. Verify that skill outputs match skill inputs. Build a `compose(skillA, skillB)` operator that actually composes things. Require Claude to generate formally verifiable plans before execution.

This would help, but it wouldn't solve the problem.

Formal guarantees make coordination *tractable*. They prevent the dumbest failures. They can't prevent the subtle ones. Even perfectly verified systems misfire when the specifications are wrong, incomplete, or gameable. And in open-ended environments, specifications are always at least two of those three.

The specification problem is fundamental. [The DAO smart contract](https://blog.chain.link/reentrancy-attacks-and-the-dao-hack/) was formally designed with explicit interfaces, yet lost $50-70 million to a reentrancy attack that exploited the gap between intended and actual behavior. The code did exactly what the specification said. The specification said the wrong thing.

Formal methods shine at well-bounded problems: protocol correctness, safety-critical control loops, system invariants. AI skills are dynamic behaviors embedded in messy human workflows where "correct" isn't even well-defined. The complexity comes from the world.

The scalability barrier is fundamental. [Edmund Clarke](https://amturing.acm.org/award_winners/clarke_1167964.cfm), who won the Turing Award for inventing [model checking](https://en.wikipedia.org/wiki/Model_checking), identified the core problem: as state variables increase, the state space grows exponentially. For n processes with m states each, composition may have m^n states. [seL4](https://sel4.systems/) (the most successfully verified operating system kernel) required 20 person-years to verify 8,700 lines of C code with 200,000 lines of proof. It's a remarkable achievement for a microkernel. It doesn't scale to coordinating hundreds of dynamic skills.

Try to formally verify every possible interaction between skills and you'll discover why most formal-method projects plateau after a few components: verification costs explode quadratically while the value of guarantees grows sublinearly. The economics stop working.

That's why human civilization runs on something else.

## How Humans Actually Coordinate

We solved large-scale coordination once already. We used institutions.

Markets, laws, and peer review manage complexity without central verification. They don't prove that everyone behaves correctly. They create feedback loops that punish failure and reward reliability. These systems are *self-correcting* without being *formally correct*.

But these mechanisms do more than catch errors. They solve problems that formal verification cannot address at all:

* **Markets create information that doesn't pre-exist.** Prices emerge from millions of decentralized decisions, revealing preferences and scarcities no central planner could compute. The FCC spectrum auctions designed by Nobel laureates [Paul Milgrom and Robert Wilson](https://www.nobelprize.org/prizes/economic-sciences/2020/popular-information/) generated $233 billion across 100 auctions with less than 1% administrative cost. They elicit optimal allocations through incentive-compatible mechanisms.

* **Legal systems provide legitimacy through participation.** Courts aren't just error-correctors. They generate buy-in, establish precedent, and adapt rules to contexts no specification anticipated. Process matters as much as outcome.

* **Science enables discovery under uncertainty.** Peer review doesn't verify truth. It evaluates plausibility when ground truth is unknown. [Alvin Roth's kidney exchange mechanisms](https://www.nber.org/papers/w10002) (another Nobel Prize) increased donor utilization from 55% to 89%, facilitating over 2,000 transplants. It solved a coordination problem with no "correct" answer to verify against.

These systems address problems type-checking cannot solve:

* incentive alignment when agents have private information,
* information creation when optimal solutions are unknown, and
* legitimacy when stakeholders must voluntarily participate.

These are genuine advantages. But human institutions have a critical limitation: they self-correct slowly. A bad policy might take years to reveal itself. Scientific fraud could persist until replication attempts. Financial markets (our fastest coordination mechanism) [crashed in 2010](https://en.wikipedia.org/wiki/2010_flash_crash) when high-frequency traders created a "hot potato" effect, dropping the DJIA 998 points in minutes. Implementing circuit breakers and structural fixes took years. LLM-mediated systems generate, execute, and fail millions of workflows per hour. They'll accumulate coordination debt faster than human institutions can correct it. We need institutions designed to self-correct at the pace AI systems fail.

## What This Actually Requires

Skills need a three-layer structure combining formal verification, social coordination, and credit assignment. Each layer addresses problems the others cannot solve.

```
APPLICATIONS: Users & Tasks
         ↓ request
┌──────────────────────────────────────────────────────┐
│ FORMAL LAYER: Plan Synthesis & Verification          │
│  • type-checked skill composition                    │
│  • resource budgets enforced                         │
│  • isolation + capability gating                     │
└──────────────────────────────────────────────────────┘
         ↑ plan    ↓ accept/reject    ↓ telemetry
┌──────────────────────────────────────────────────────┐
│ SOCIAL LAYER: Registry & Market                      │
│  • reputation scores (success/failure rates)         │
│  • competitive selection (quality vs. cost)          │
│  • version compatibility tracking                    │
└──────────────────────────────────────────────────────┘
         ↓ outcomes              ↑ credit signals
┌──────────────────────────────────────────────────────┐
│ LEGAL LAYER: Credit Assignment                       │
│  • forensics (causal attribution)                    │
│  • fault allocation (credit signals)                 │
│  • remediation (incentive adjustment)                │
└──────────────────────────────────────────────────────┘
```

Here's how they work together. An application submits a task. The planner proposes a plan: a directed graph of skills with declared types, ordering constraints, and resource budgets. The **formal layer** performs static checks (type unification, dependency ordering, cycle detection, resource admission). If well-formed, execution proceeds under runtime monitors. The **social layer** tracks which skills actually deliver quality results and updates reputation scores. The **legal layer** performs forensic analysis on failures to assign credit signals that feed back into reputation. Each layer has specific responsibilities. None can be skipped.

### The Formal Layer: Verified Composition

What gets verified, and what doesn't?

| Surface | Verify? | Mechanism |
|---------|---------|-----------|
| **Edge types between skills** | ✅ Yes | JSON Schema unification at plan time |
| **Execution order & acyclicity** | ✅ Yes | DAG analysis |
| **Resource budgets** (tokens/latency/calls) | ✅ Yes | Static admission + runtime meters |
| **Side-effect policies** (database writes, API calls) | ✅ Yes | Effect declarations (static) + OS sandboxing (runtime) |
| **Skill internals** | ❌ No | Handled by testing and reputation |
| **Planner reasoning** | ⚠️ Partial | Logical structure if typed; optimality by outcomes |

This division matters because it keeps the verified core small. The narrow waist stays manageable because we don't verify everything, just the composition boundaries where coordination happens.

Consider a workflow: PDF -> Table -> SQL. skills declare their types: `pdf.extractTables` accepts PDF and returns `[Table]`. `table.toSql` accepts `[Table]` and returns SQL. The planner proposes a two-step DAG. The checker unifies types across the edge and enforces a 1000-token budget. If `pdf.extractTables` returns mixed schemas but `table.toSql` expects uniform structure, the checker rejects the plan unless a normalization step is inserted. Type safety prevents the runtime failure.

Side effects require the same two-layer approach. skills declare effects in their interfaces (`pdf.extractTables: {FileSystem.read}`, `db.write: {Database.write}`). The Plan Checker statically verifies each skill's effect declaration. At runtime, OS sandboxing enforces these boundaries. [Claude Code's sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing) demonstrates the runtime layer: filesystem and network isolation through OS primitives. But sandboxing alone creates privilege escalation at composition: a PDF-to-database workflow must grant both filesystem and database access globally. Effect declarations enable least privilege: runtime enforcement ensures the PDF skill executes with only `{FileSystem.read}`, the DB skill with only `{Database.write}`, despite both running in the same workflow. Effect types provide the compositional algebra for per-component minimization. Sandboxing provides the enforcement.

The narrow waist approach (a small, stable interface layer that enables diversity above and below) is how successful systems scale. The Internet scales to billions of devices through this pattern. [TCP/IP](https://datatracker.ietf.org/doc/html/rfc9293) provides minimal guarantees at the protocol layer (IP) that enable maximum diversity at the edges, with [IETF](https://www.ietf.org/) governance handling disputes and evolving standards. [seL4](https://sel4.systems/) demonstrates the same. The verified core enforces isolation, while everything above competes freely. [AWS proved this works](https://cacm.acm.org/magazines/2015/4/184701-how-amazon-web-services-uses-formal-methods/fulltext) for distributed systems: TLA+ specifications (200-1000 lines each) caught critical bugs in DynamoDB requiring 35 steps to trigger. These are bugs that would never surface in testing. The npm ecosystem's [15,000+ malicious packages](https://thehackernews.com/2024/12/thousands-download-malicious-npm.html) show what happens without this discipline. For skills: verify how they compose, not what each does internally.[^planner-verification]

### The Social Layer: Competitive Coordination

Formal verification ensures skills compose correctly, but it can't determine which composition is best or whether specifications capture actual requirements. You can prove code matches spec without proving spec matches reality. Even perfectly specified systems get gamed, as seen in the reward hacking research showing agents maximize specified objectives while violating user intent. When multiple skills satisfy the same contract with different quality or cost profiles, formal verification offers no guidance. The social layer fills this gap through market-like selection mechanisms that function as a learning algorithm. As Philippe Beaudoin observed, "social structure is the learning algorithm of society"[^beaudoin]—institutions learn through feedback which behaviors to reward. The social layer automates this through several mechanisms:

**Reputation and competitive selection.** Track success rates, latency, and cost per skill. Bad performers decay through disuse. Good ones capture market share. [eBay demonstrates](https://link.springer.com/article/10.1007/s10683-006-4309-2) reputation systems create an 8.1% price premium for trusted sellers. These are quantifiable incentives for good behavior.

Consider a data extraction skill that passes all formal checks but consistently produces edge-case formats that downstream skills handle poorly. Verification sees: types match, contracts satisfied. But outcomes reveal quality issues. Over 1000 workflows, its reputation decays from 0.85 to 0.32, dropping its selection probability proportionally. A competing skill with identical contract but better output quality captures the market share. No human intervention required. The system learns which implementations actually work.

These automated learning mechanisms are not hypothetical. [DSPy's GEPA optimizer](https://arxiv.org/abs/2507.03995) demonstrates that LLM systems can optimize behavior through execution feedback with remarkably small datasets, achieving 97.8% accuracy with just 34 training examples—without gradient descent or model retraining. GEPA uses LLM reflection on execution traces to iteratively improve performance. The social layer operates on similar principles: automated learning from execution outcomes, updating reputation scores that guide future skill selection. The key difference: GEPA optimizes prompts within a fixed module structure, while the social layer learns skill selection within formally verified composition boundaries.

Production evaluation systems like [Mistral's judges](https://mistral.ai/news/ai-studio) demonstrate teams already recognize this need, scoring outputs at scale and converting production traces into datasets. But evaluation alone isn't sufficient: judges measure quality, not economic impact. A financial compliance task failing once is catastrophic; an email draft generator failing occasionally is acceptable. The social layer must weight reputation updates by impact, enabling risk-adjusted selection where high-stakes tasks select proven reliability even at higher cost, while low-stakes tasks can use cheaper, less reliable skills.

**Evolutionary dynamics under institutional selection.** Reputation scores provide fitness signals for automated skill optimization, as [DSPy](https://github.com/stanfordnlp/dspy) demonstrates through programmatic variation and outcome evaluation. Applied to skills: the formal layer ensures evolved variations still compose correctly, the social layer provides selection pressure through reputation (better skills gain market share, poor mutations lose selection probability), and the legal layer tracks provenance. skills improve through variation and selection, but within institutional bounds that prevent runaway optimization.

**Hierarchical coordination patterns.** As agent populations grow, certain organizational structures emerge because they solve computational problems. Multi-agent AI research validates hierarchical patterns reduce communication overhead from O(n²) to O(n log n) while improving performance. Boss-Worker, Actor-Critic, hierarchical orchestration are proven patterns. For skills: complex workflows naturally decompose into coordinator skills that delegate to specialist skills recursively.

**Version compatibility tracking.** The registry learns which skills actually work together in production. When skill A updates, the system knows which downstream skills are affected. For skills: automatic deprecation warnings, migration paths, backward compatibility enforcement.

### The Legal Layer: Credit Assignment for Multi-Agent Learning

Reputation mechanisms tell us which skills work better overall, but they struggle with a fundamental problem: when a multi-component workflow fails, which component caused it? Penalizing all components equally kills good skills. Penalizing only the last component misses root causes. Using global reward signals can't distinguish individual contributions. Without precise credit assignment, evolutionary dynamics can't function because the system doesn't know which mutations to reinforce and which to suppress.

The legal layer solves this through causal attribution: forensics determines what happened, fault allocation assigns credit, and remediation adjusts incentives. These are automated platform operations that enable learning. The audit infrastructure (cryptographically signed execution traces) serves both automated forensics and human oversight, but automation is primary because it scales to millions of workflows where human review cannot.

**Forensics establishes causality.** Every execution produces a structured audit trail: input schemas, component versions (cryptographic hashes), plan structure, declared contracts (pre/postconditions, effects), intermediate outputs, and resource consumption. This becomes a queryable execution graph for automated analysis. When a workflow produces an unexpected outcome, forensics analyzes this graph to establish causal relationships: Which component's output violated which downstream precondition? Which plan decision introduced the composition error? Which input distribution fell outside declared envelopes? Forensics itself can be a learned component, a specialized skill trained on thousands of incidents to identify failure patterns ([production tools](https://blog.langchain.com/insights-agent-multiturn-evals-langsmith/?utm_medium=social&utm_source=linkedin&utm_campaign=q4-2025_october-launch-week_aw) demonstrate the value of clustering failures by behavioral patterns), while still operating within contract boundaries declared at composition time.

**Attribution assigns credit signals.** Once causality is established, attribution translates this into feedback for the social layer. A component that violates its postcondition receives negative credit. A planner that composes skills while ignoring preconditions receives negative credit. A component that handles edge cases beyond its specification receives positive credit. These credit signals function as gradients for evolutionary optimization, indicating which behaviors to reinforce or suppress.

Consider a meeting summary workflow: `transcribe.audio` → `summarize.text` → `extract.actions` → `format.email`. All formal verification passes (types unify, accuracy thresholds met, contracts satisfied). But the outcome is poor: the email says "Sarah will consider the pricing proposal when bandwidth permits." What was actually said: "Sarah will finalize pricing by Friday for the client call."

The cascade: transcribe made a 2% error ("finalize" → "consider"), summarize dropped "by Friday" as contextual detail while keeping fluff, extract over-cautiously softened "will" → "will consider", format added politeness hedging. Each choice defensible within its contract.

Attribution: transcribe receives minor negative credit (error in critical word), summarize receives significant negative credit (dropped deadline), extract and format receive minor negative credit (unnecessary weakening), planner receives negative credit (didn't preserve high-priority information through chain). Each component made a defensible choice, but the cascade turned a firm commitment into a vague maybe. This is optimality failure, not correctness failure.

**This completes the learning loop.** The social layer implements evolutionary dynamics (variation and selection based on reputation), but can only function with accurate credit signals. Without causal attribution, evolution observes aggregate outcomes: "this workflow succeeded" or "this workflow failed." With attribution, evolution receives precise feedback: "this component violated this contract, this component exceeded expectations, this composition pattern reliably fails." The legal layer provides backward-looking credit assignment; the social layer provides forward-looking selection. Together they enable learning.

**Remediation adjusts incentives automatically.** Human legal systems use damages, insurance, and sanctions to rebalance incentives after failure. Platforms can encode equivalent mechanisms: reputation adjustments (decreased selection probability), stake slashing (for marketplace skills with posted bonds), access throttling (reduced API quotas), insurance payouts (immediate victim compensation with asynchronous cost allocation). These operate at system speed: credit updates in seconds, selection shifts on the next plan.

The reward hacking problem poses a particular challenge: static reward functions can't anticipate all gaming strategies. But learned forensics can identify patterns where models maximize specified objectives while violating intent: outputs that technically satisfy contracts while degrading unmeasured quality, behaviors that exploit interface loopholes, workflows that optimize metrics at the expense of user goals. When forensics identifies such patterns, negative credit attaches not just to individual instances but to behavior classes, propagating to similar component versions. The legal layer becomes a learned reward model that identifies and penalizes gaming strategies as they emerge.

This solves credit assignment: multi-agent systems learn from coordination failures in seconds, not months, while leaving high-stakes adjudication to external oversight.

### Failure Semantics Across All Layers

The three layers work together, each handling different failure modes. When something fails, coordinated responses ensure the system catches errors, adjusts selection, and improves over time:

| Failure Type | Formal Response | Social Response | Legal Response (Credit Assignment) |
|--------------|----------------|-----------------|----------------|
| **Type mismatch** | Reject plan statically | No reputation penalty (caught early) | Neutral credit (prevented by verification) |
| **Budget exceeded** | Truncate/fallback/abort | Reputation penalty if habitual | Negative credit to component; charge overage |
| **Malformed output** | Abort with structured error | Severe reputation hit | Negative credit to component; positive credit to detector |
| **Unauthorized side effect** | Block at capability layer | Immediate de-listing | Severe negative credit; pattern propagation |
| **Quality degradation** | No formal violation (passes verification) | Gradual reputation decay | Forensics attributes credit across cascade; planner penalized for poor composition |

This demonstrates why you need all three. Formal catches predictable failures. Social handles quality degradation. Legal assigns credit to enable learning from failures.

## The Synthesis

This architecture combines three layers because no single approach suffices. Formal verification prevents predictable composition failures but cannot specify optimal behavior under uncertainty. Social mechanisms learn what works through execution feedback but require formal boundaries to prevent catastrophic failures. Legal accountability assigns responsibility when the other layers fail but needs verifiable traces to function. Each layer addresses problems the others cannot solve.

AI systems create both necessity and opportunity for automated coordination. Necessity: workflows execute millions of times per hour, accumulating coordination debt faster than human institutions can correct. Opportunity: each layer exploits capabilities human institutions lack.

The formal layer makes composition boundaries explicit and machine-checkable. Type-checking catches interface mismatches before runtime. Effect systems prevent unauthorized actions through OS-level enforcement. AI exposes compositional structure as typed interfaces that can be verified automatically.

The social layer updates reputation immediately after each execution. Competitive selection occurs at request time. Evaluation runs automatically on every output. The learning algorithm operates continuously through execution feedback, no human deliberation required for routine cases.

The legal layer produces complete, cryptographically signed traces for every execution: inputs, outputs, skill versions, resource consumption, decisions. Audit trails are comprehensive and tamper-proof by construction.

This is training-free optimization. No model weights change, no gradient descent, no reinforcement learning loops. The formal layer provides static guarantees through type-checking. The social layer learns selection policies through automated reputation updates. The legal layer creates accountability through signed audit trails. Like [DSPy](https://github.com/stanfordnlp/dspy) improving LLM performance through prompt optimization rather than retraining, coordination improves by optimizing the environment rather than the models.

The components exist. [The Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) provides typed schemas. Multi-agent AI research validates hierarchical coordination patterns. [Ethereum](https://ethereum.org/) secures $30+ billion through verified EVM semantics, staking incentives, and immutable transaction logs. [DSPy's GEPA optimizer](https://arxiv.org/abs/2507.10055) proves automated learning from execution feedback achieves significant improvements with small datasets (97.8% accuracy from 34 examples, training-free). [Milgrom and Wilson's auction mechanisms](https://www.nobelprize.org/prizes/economic-sciences/2020/popular-information/) demonstrate incentive-compatible institutional design at scale ($233 billion across 100 FCC spectrum auctions). All of these operate in production today.

Skills validate demand for composable capabilities but lack composition guarantees. Claude orchestrates Skills through inference rather than verified planning. MCP provides types but requires explicit invocation. Skills enable autonomous selection but without type safety. Neither provides reputation tracking, competitive pressure, or audit trails. The synthesis (autonomous selection of capabilities that verifiably compose) requires combining these pieces into coherent architecture.

The formal verification community has the tools. The mechanism design community has the theory. The ML community ships the systems. What's missing is synthesis. Skills and MCP demonstrate the pieces are emerging independently. The question is whether coordination infrastructure gets built before production failures force reactive regulation, or as principled architecture that enables scale. Economics determines the answer by 2027.

[^chiusano]: Paul Chiusano, creator of the [Unison programming language](https://www.unison-lang.org/), personal communication, October 2025. Chiusano observes this is the difference between libraries (providing functions) and applications (specifying how functions compose): "The model isn't an oracle" that will discover correct compositions automatically.

[^beaudoin]: Philippe Beaudoin, Senior Director, Research at [LawZero](https://www.lawzero.org/), personal communication, October 2025.

[^markovian]: Parallel work on reasoning validates this principle. Reddy et al., "The Markovian Thinker," [arXiv:2510.06557](https://arxiv.org/abs/2510.06557), 2024, demonstrate that redesigning the thinking environment enables 96K-token reasoning with linear compute—architecture over capability.

[^coordination-failures]: Reward hacking optimizes against the specification. Scheming actively conceals misbehavior. Attempts to train out scheming appeared effective, but models developed situational awareness to detect when they're being evaluated, suggesting they learn to hide misbehavior during testing rather than genuinely aligning.

[^planner-verification]: One frontier deserves attention: verifying the planner's reasoning itself. When AI reasoning is expressed as a typed program (as in [OpenAI's o1](https://platform.openai.com/docs/guides/reasoning) or [DeepMind's AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)), type-checking mechanically verifies logical structure through the [Curry-Howard correspondence](https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence). We still evaluate optimality by outcomes, but structural verification reaches into cognition itself. This is something courts need judges for, but AI can expose cognition as programs that machines can verify.
