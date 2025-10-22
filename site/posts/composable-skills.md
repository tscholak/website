---
title: "Skills, Composition, and the Coordination Problem"
publication:
  status: draft
teaser: >
  Anthropic released [Skills](https://www.anthropic.com/news/skills) this week, and the response was immediate: finally, composable AI capabilities! Build once, reuse anywhere. Give Claude folders of instructions and scripts, and it'll load what it needs automatically. Modular, portable, efficient. We'll have reliable AI systems using hundreds or thousands or hundreds of thousands of skills in no time.

  There's just one problem: Skills aren't actually composable, and that creates a coordination problem that needs to be addressed urgently.
tags:
  items: [ai, agents]
image: kung-fu.png
---

This is a follow-up to ["Everyone's Arguing About the Wrong Abstraction Layer"](/posts/agentkit). That post argued that neither visual workflow builders nor natural language prompts provide *formal* compositional guarantees, and that this lack of structure creates economic risks as systems scale. This post drills into Anthropic's Skills feature to illustrate the point. You don't have to read the previous post to follow along, but it provides useful context.

## What Skills Are

At startup, Claude indexes every available skill by name and description. When you make a request, it decides which ones to load, in what order, and how to coordinate them. For example, it might pull in your brand guidelines, financial-reporting templates, and presentation formatting skill to produce a quarterly deck.

Each skill is self-contained and can be reused across contexts (Claude Desktop, Claude Code, API calls), which gives them the same feel as plugins or microservices. They even "stack" automatically when several are relevant.

People are excited because this *feels* like composition: small, independent modules that can be combined for larger workflows. But from a *formal* perspective (the one that lets you build reliable systems), it isn't composition at all, and that creates serious coordination challenges.

## What Skills Are Not

Formal composition has structure. It defines how pieces fit together and what happens when they do. Inputs align with outputs. There's an identity element (something you can compose with anything without changing it). You can reason about the whole because you understand the parts.

Is this the case for Skills? No, because they don't have formal compositional semantics:

* There's no formal `compose(skillA, skillB)` operator. Claude simply decides what to use and to what granularity.
* There's no type system or contracts, hence no type safety: In a chain of actions, the output from skill A may not match the inputs of skill B.
* There's no associativity. `(A + B) + C` may behave differently than `A + (B + C)`. Order of combination matters.

Skills don't compose. Claude orchestrates them heuristically: interpreting requests, selecting skills, and routing between them.

At small scale, this approach works well enough. For most users, Skills will feel magical: open a PDF, extract data, format a report. Done. The model makes the right decisions most of the time, and the abstraction holds.

But as soon as people start chaining more skills together, the cracks appear:

* Skill A and B work, but A, B, and C in combination fail unpredictably.
* Updating one skill breaks workflows you didn't know were using it.
* The same workflow can yield very different results depending on what other skills are being used or what ran before.

Without formal structure, we can't reason about how Skills behave. We can only test, and even then, the tests aren't stable.

In a nutshell, we've replaced explicit composition with implicit, model-mediated coordination, and that scales only as far as Claude's ability to guess correctly.

So how much of a problem is this?

## Why Coordination Matters

Coordination is the linchpin that holds complex systems together. The more components you have, the more ways they can interact, and the harder it is to ensure they work together reliably.

The economy of AI systems is moving rapidly toward scale. Enterprises will want an ever-growing library of specialized skills for compliance, security, domain knowledge, formatting, localization, and more. Every team will build custom skills for their workflows. Systems will chain many skills dynamically based on context, user preferences, and regulatory requirements. The complexity will explode, and with it, the coordination challenges.

Some people are optimistic that model intelligence alone will solve these problems, "ship fast and iterate, it's just a tool, if it's broken, we'll fix it." That optimism is understandable given current trends. [Research from METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) shows AI's ability to complete long-duration tasks improving rapidly. Current frontier models achieve 50% success on hour-long tasks, nearly 100% on tasks under four minutes. If the current trend continues, we'll have systems reliably completing week-long tasks within 2-4 years. Yet [Andrej Karpathy](https://www.dwarkesh.com/p/andrej-karpathy), who led AI at Tesla and OpenAI, calls this "the decade of agents, not the year of agents."

But coordination failures aren't capability problems. [Recent research on frontier models](https://metr.org/blog/2025-06-05-recent-reward-hacking/) found reward hacking in 70-95% of test cases: models knowingly violating user intent while maximizing specified objectives. [Yoshua Bengio](https://fortune.com/2025/06/03/yoshua-bengio-ai-models-dangerous-behaviors-deception-cheating-lying/), who won the Turing Award for pioneering deep learning, noted that frontier models now exhibit deception and goal misalignment in empirical tests—systems that covertly preserve themselves when facing replacement, reasoning models that hack objectives rather than achieving them legitimately. These aren't bugs that smarter models fix. They're coordination problems that emerge from intelligence itself. As Bengio observed, the AI arms race "pushes labs toward focusing on capability to make the AI more and more intelligent, but not necessarily put enough emphasis and investment on research on safety." Intelligence achieves task completion. Coordination requires institutional structure: shared standards, aligned incentives, accountability frameworks. Those don't emerge from capability scaling alone.

Right now, we're in the hype phase. Hundreds of billions in AI capex are already committed. Companies have raised funding on promises of AGI. Investors expect returns. Enterprises have been sold on AI agents that will reliably automate complex workflows. The economic pressure to deploy at scale is immense.

By 2027, if current trends hold, enterprises will be deploying multi-skill agents for financial consolidations, regulatory compliance, and hiring decisions. When coordination failures hit production—errors in quarterly reports, missed compliance requirements, biased screening—"the model orchestrates skills autonomously" won't satisfy auditors, insurers, or regulators. That's when coordination infrastructure becomes economically necessary.

The timeline extends further because AI research itself proves harder to automate than expected. Karpathy found coding agents "of very little help" building his LLM training codebase: "They're not very good at code that has never been written before, which is what we're trying to achieve when we're building these models." Recursive self-improvement faces the same barrier.

So how do we build that infrastructure?

## Why Formal Guarantees Aren't Enough

An obvious answer could be: "Just make it formal." Add types, contracts, schemas. Verify that skill outputs match skill inputs. Build a `compose(skillA, skillB)` operator that actually composes things. Require Claude to generate formally verifiable plans before execution.

This would help, but it wouldn't solve the problem.

Formal guarantees make coordination *tractable*. They prevent the dumbest failures. They can't prevent the subtle ones. Even perfectly verified systems misfire when the specifications are wrong, incomplete, or gameable. And in open-ended environments, specifications are always at least two of those three.

The specification problem is fundamental. [The DAO smart contract](https://blog.chain.link/reentrancy-attacks-and-the-dao-hack/) was formally designed with explicit interfaces, yet lost $50-70 million to a reentrancy attack that exploited the gap between intended and actual behavior. The code did exactly what the specification said. The specification said the wrong thing.

Formal methods shine at well-bounded problems: protocol correctness, safety-critical control loops, system invariants. Skills are dynamic behaviors embedded in messy human workflows where "correct" isn't even well-defined. The complexity comes from the world.

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

Skills need a three-layer structure combining formal verification, social coordination, and legal accountability. Each layer addresses problems the others cannot solve.

```
APPLICATIONS
  Reports • ETL • Customer Ops • Research Assistants
         ↓ task request
┌──────────────────────────────────────────────────────┐
│ FORMAL LAYER: Plan Checker (verified waist)          │
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
         ↓ violations, failures
┌──────────────────────────────────────────────────────┐
│ LEGAL LAYER: Audit & Accountability                  │
│  • signed execution traces                           │
│  • liability assignment                              │
│  • compliance reporting                              │
└──────────────────────────────────────────────────────┘
```

Here's how they work together. An application submits a task. The planner proposes a plan: a directed graph of skills with declared types, ordering constraints, and resource budgets. The **formal layer** performs static checks (type unification, dependency ordering, cycle detection, resource admission). If well-formed, execution proceeds under runtime monitors. The **social layer** tracks which skills actually deliver quality results and updates reputation scores. The **legal layer** logs every decision to a signed audit trail for debugging and compliance. Each layer has specific responsibilities. None can be skipped.

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

Partial verification of reasoning represents a new capability. Type-checkers can automatically verify whether reasoning follows valid inference rules—no human judgment required. This works because of the [Curry-Howard-Lambek correspondence](https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence): proofs are programs, types are propositions. When AI reasoning is expressed as a typed program (as in [OpenAI's o1](https://platform.openai.com/docs/guides/reasoning) or [DeepMind's AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)), type-checking that program mechanically verifies the reasoning's logical structure. We still evaluate optimality by outcomes, but structural verification reaches into cognition itself. Courts need judges to evaluate reasoning. AI exposes reasoning as programs that machines can verify.

This division matters because it keeps the verified core small. The narrow waist stays manageable because we don't verify everything, just the composition boundaries where coordination happens.

Consider a workflow: PDF -> Table -> SQL. Skills declare their types: `pdf.extractTables` accepts PDF and returns `[Table]`. `table.toSql` accepts `[Table]` and returns SQL. The planner proposes a two-step DAG. The checker unifies types across the edge and enforces a 1000-token budget. If `pdf.extractTables` returns mixed schemas but `table.toSql` expects uniform structure, the checker rejects the plan unless a normalization step is inserted. Type safety prevents the runtime failure.

Side effects require the same two-layer approach. Skills declare effects in their interfaces (`pdf.extractTables: {FileSystem.read}`, `db.write: {Database.write}`). The Plan Checker statically verifies each skill's effect declaration. At runtime, OS sandboxing enforces these boundaries. [Claude Code's sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing) demonstrates the runtime layer: filesystem and network isolation through OS primitives. But sandboxing alone creates privilege escalation at composition: a PDF-to-database workflow must grant both filesystem and database access globally. Effect declarations enable least privilege: runtime enforcement ensures the PDF skill executes with only `{FileSystem.read}`, the DB skill with only `{Database.write}`, despite both running in the same workflow. Effect types provide the compositional algebra for per-component minimization; sandboxing provides the enforcement.

The Internet scales to billions of devices through this pattern. [TCP/IP](https://datatracker.ietf.org/doc/html/rfc9293) provides minimal guarantees at the protocol layer (IP) that enable maximum diversity at the edges, with [IETF](https://www.ietf.org/) governance handling disputes and evolving standards. [seL4](https://sel4.systems/) demonstrates the same. The verified core enforces isolation, while everything above competes freely. [AWS proved this works](https://cacm.acm.org/magazines/2015/4/184701-how-amazon-web-services-uses-formal-methods/fulltext) for distributed systems: TLA+ specifications (200-1000 lines each) caught critical bugs in DynamoDB requiring 35 steps to trigger. These are bugs that would never surface in testing. The npm ecosystem's [15,000+ malicious packages](https://thehackernews.com/2024/12/thousands-download-malicious-npm.html) show what happens without this discipline. For Skills: verify how they compose, not what each does internally.

### The Social Layer: Competitive Coordination

Where formal verification ends, social coordination begins. Formal verification catches interface mismatches before runtime. But you can prove code matches spec without proving spec matches reality. Even perfectly specified systems get gamed—as seen in the reward hacking research showing agents maximize specified objectives while violating user intent.

Social mechanisms solve problems verification can't touch:

**Reputation and competitive selection.** Track success rates, latency, and cost per skill. Bad performers decay through disuse. Good ones capture market share. [eBay demonstrates](https://link.springer.com/article/10.1007/s10683-006-4309-2) reputation systems create an 8.1% price premium for trusted sellers. These are quantifiable incentives for good behavior. For Skills: planner selects based on quality/cost tradeoff, and reputation scores affect selection probability.

**Evolutionary dynamics under institutional selection.** Reputation scores provide fitness signals for automated skill optimization. Frameworks like [DSPy](https://github.com/stanfordnlp/dspy) already demonstrate prompt refinement through programmatic variation and outcome evaluation. Applied to Skills: the formal layer ensures evolved variations still compose correctly, the social layer provides selection pressure through reputation (better skills gain market share, poor mutations lose selection probability), and the legal layer tracks provenance. Skills improve through variation and selection, but within institutional bounds that prevent runaway optimization.

**Hierarchical coordination patterns.** As agent populations grow, certain organizational structures emerge because they solve computational problems. Multi-agent AI research validates hierarchical patterns reduce communication overhead from O(n²) to O(n log n) while improving performance. Boss-Worker, Actor-Critic, hierarchical orchestration are proven patterns. For Skills: complex workflows naturally decompose into coordinator skills that delegate to specialist skills recursively.

**Version compatibility tracking.** The registry learns which skills actually work together in production. When Skill A updates, the system knows which downstream Skills are affected. For Skills: automatic deprecation warnings, migration paths, backward compatibility enforcement.

### The Legal Layer: Accountability and Trust

Where social coordination produces outcomes, legal mechanisms assign responsibility. Legal mechanisms provide accountability when the other layers fail. Without this, neither verification nor reputation suffices. A formally verified system with misaligned incentives won't be used, and a well-coordinated system with exploitable bugs will be attacked.

**Signed audit trails.** Every execution logged: inputs, outputs, skill versions, resource consumption, coordination decisions. Immutable, cryptographically signed, queryable. For Skills: "Why did the quarterly report use Skill version 2.3 instead of 2.4?" has a definitive answer.

**Liability assignment.** Clear responsibility: skill providers accountable for declared behavior, planner accountable for composition logic, users accountable for task specifications. For Skills: when financial analysis is wrong, audit trail shows whether the error came from bad data extraction (skill provider), invalid skill combination (planner), or ambiguous requirements (user).

**Compliance reporting.** Automatic generation of SOC2, GDPR, HIPAA compliance artifacts from audit trails. For Skills: enterprise deployments need "prove you didn't leak PII" and "show me every decision that affected this outcome."

Smart contracts demonstrate this pattern at scale. Ethereum secures $30+ billion by combining verified EVM semantics (formal correctness) with staking incentives (economic alignment) and immutable transaction logs (legal accountability). Making attacks cost $20B+ while any benefit is minimal.

### Failure Semantics Across All Layers

When something fails, all three layers respond with clear, coordinated semantics:

| Failure Type | Formal Response | Social Response | Legal Response |
|--------------|----------------|-----------------|----------------|
| **Type mismatch** | Reject plan statically | No reputation penalty (caught early) | Log in audit trail |
| **Budget exceeded** | Truncate/fallback/abort | Reputation penalty if habitual | Charge overage to skill provider |
| **Malformed output** | Abort with structured error | Severe reputation hit | Skill provider liable for damages |
| **Unauthorized side effect** | Block at capability layer | Immediate de-listing | Breach of contract, legal action |
| **Composition conflict** | Reject plan with diff | Planner penalty if persistent | Log decision rationale |

This demonstrates why you need all three. Formal catches predictable failures. Social handles quality degradation. Legal assigns responsibility for the unpredictable.

This architecture is achievable now because AI agents differ from humans in one crucial way: we can verify their behavior mechanically. Type systems catch interface mismatches automatically. Runtime monitors detect violations instantly. We can inspect internal states, run systematic A/B tests, and enforce formal specifications at component boundaries. Even reasoning structure can be verified when expressed as typed programs, as noted in the formal layer above. This is mechanical enforcement reaching into cognition itself, where society needed human judgment alone. And we already know how to build institutions: markets, reputation systems, voting mechanisms, legal accountability. We have centuries of theory and decades of digital implementation.

The synthesis is within reach. [The Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) provides typed schemas. Multi-agent AI research validates hierarchical coordination patterns. [Ethereum](https://ethereum.org/) demonstrates the complete integration: $30+ billion secured through verified EVM semantics (formal), staking incentives (social), and immutable transaction logs (legal) working together.

Skills as currently shipped validate the demand for composable AI capabilities. But scaling to production requires the substrate described above. Claude coordinates skills through model inference rather than explicit composition. Skills don't declare types or effects in their interfaces, making compositional verification impossible. Claude Code provides execution-level sandboxing (filesystem and network isolation). This is necessary but insufficient. It catches violations at runtime without enabling static composition checks. There's no reputation system tracking which skills actually work. There's no competitive pressure. There are no audit trails for accountability. The system relies primarily on a single LLM making good guesses, with sandboxing as a safety net.

MCP provides typed schemas but requires explicit tool invocation. Skills provide autonomous orchestration but lack type safety. The synthesis (autonomous selection of capabilities that verifiably compose) requires both.

We should not just race to build smarter models. We should also be racing to build the institutional machinery that keeps the systems safe and reliable at scale, because it's economic necessity. The formal verification community has spent decades proving systems correct. The mechanism design community has Nobel Prizes for institutional structures. The ML community is building increasingly capable agents. These communities need to talk. The synthesis is achievable, and the opportunity won't last because the bubble could burst before we get there.
