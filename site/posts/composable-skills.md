# Skills, Composition, and the Coordination Problem

*A follow-up to "Everyone's Arguing About the Wrong Abstraction Layer"*

Anthropic released [Skills](https://www.anthropic.com/news/skills) this week. The idea is elegant: give Claude a set of folders, each describing a specific capability (with instructions, scripts, and a `SKILL.md` file), and it can load and combine them automatically when a task requires it. Build once, reuse anywhere. Modular, portable, token-efficient.

It's a simple, tangible interface for something that's been abstract until now: *how to teach an AI system new behaviors without retraining it.* And it's working well enough that many people are calling it “composability.”

That's where the confusion begins.

## What Skills Actually Are

At startup, Claude indexes every available skill by name and description. When you make a request, it decides which ones to load, in what order, and how to coordinate them. For example, it might pull in your brand guidelines, financial-reporting templates, and presentation formatting skill to produce a quarterly deck.

Each skill is self-contained and can be reused across contexts (Claude Desktop, Claude Code, API calls), which gives them the same feel as plugins or microservices. They even “stack” automatically when several are relevant.

From a software-engineering perspective, that's composability: small, independent modules that can be combined for larger workflows. But from a *formal* perspective (the one that lets you build reliable systems), it isn't composition at all.

## What Skills Are Not

Formal composition has structure. It defines how pieces fit together and what happens when they do. Inputs align with outputs. The same operation can be repeated safely in different orders. There's an identity element (something you can compose with anything without changing it). You can reason about the whole because you understand the parts.

This isn't abstract theory. Philip Wadler's work on monads established that "composition is associative." Benjamin Pierce's *Types and Programming Languages* shows how type systems provide "a syntactic method for automatically checking the absence of certain erroneous behaviors." Mathematical laws govern these properties, enabling compositional reasoning.

Here's a crucial distinction: **formal structure** versus **mechanical verification**. Legal contracts have compositional structure. You can nest them, reference them, combine them according to rules. But verification happens through judges and adversarial process, not proof assistants. Markets have compositional structure. Prices compose through arithmetic, supply and demand curves compose through equilibrium. But no central theorem prover validates every transaction. The structure enables composition. Humans (with all their flaws) verify correctness.

AI systems inherit the same compositional structures society developed, but they unlock a new possibility: mechanical verification at the boundaries. We can add type checkers, runtime monitors, and formal guarantees where humans can only add judges, auditors, and reviews. The structure stays the same. The verification mechanism upgrades.

Skills don't have that structure yet.

* There's no type system ensuring the output of one skill matches the expected input of another.
* There's no formal `compose(skillA, skillB)` operator. Claude simply decides what to use and to what granularity.
* There's no associativity: `(A + B) + C` doesn't necessarily behave like `A + (B + C)`.
* And there's no identity skill that does nothing and preserves behavior.

What happens when skills "combine" is orchestration mediated by a language model, not composition in the algebraic or functional sense. Claude interprets the request, picks a set of skills, and coordinates them heuristically. The result is intelligent routing rather than deterministic assembly.

You can think of it as *soft composition*: components connected by intelligence rather than by structure. It works remarkably well, until it doesn't.

## Why This Matters

At small scale, the difference is invisible. For most users, Skills will feel magical: open a PDF, extract data, format a report. Done. The model makes the right decisions most of the time, and the abstraction holds.

But as soon as people start chaining more skills together, the cracks appear:

* Skill A and B work, but A, B, and C in combination fail unpredictably.
* Updating one skill changes the behavior of workflows that depend on it.
* The same request can yield different results depending on context or hidden state.

There are no types, no contracts, and no formal semantics to constrain or explain what's happening. You can't reason about the system except by testing it, and even then, the tests aren't stable.

The short version: we've replaced explicit composition with implicit coordination, and that scales only as far as Claude's ability to guess correctly.

## Why Formal Guarantees Aren't Enough

The obvious fix: "Make it formal." Add types, contracts, schemas. Verify that skill outputs match skill inputs. Build a `compose(skillA, skillB)` operator that actually composes things.

That would help. It wouldn't solve the problem.

Formal guarantees make coordination *tractable*. They don't make it *work*. They prevent the dumbest failures but not the subtle ones. Even perfectly verified systems misfire when the specifications are wrong, incomplete, or gameable. And in open-ended environments, specifications are always at least two of those three.

The specification problem is fundamental. IBM's Watson for Oncology consumed $62 million and formal design processes, yet saw zero adoption by physicians. It was trained on treatment guidelines instead of patient outcomes, making correctness meaningless. The specification was wrong from the start. The DAO smart contract was formally designed with explicit interfaces, yet lost $50-70 million to a reentrancy attack that exploited the gap between intended and actual behavior. The code did exactly what the specification said. The specification said the wrong thing.

Formal methods shine at well-bounded problems: protocol correctness, safety-critical control loops, kernel invariants. Skills aren't microkernels. They're dynamic behaviors embedded in messy human workflows where "correct" isn't even well-defined. The complexity comes from the world.

The scalability barrier is fundamental. Edmund Clarke, who won the Turing Award for inventing model checking, identified the core problem: as state variables increase, the state space grows exponentially. For n processes with m states each, composition may have m^n states. seL4 (the most successfully verified operating system kernel) required 20 person-years to verify 8,700 lines of C code with 200,000 lines of proof. It's a remarkable achievement for a microkernel. It doesn't scale to coordinating hundreds of dynamic skills.

Try to formally verify every possible interaction between skills and you'll discover why most formal-method projects plateau after a few components: verification costs explode quadratically while the value of guarantees grows sublinearly. The economics stop working.

That's why human civilization runs on something else.

## How Humans Actually Coordinate

We solved large-scale coordination once already. With institutions, not proofs.

Markets, laws, and peer review manage complexity without central verification. They don't prove that everyone behaves correctly. They create feedback loops that punish failure and reward reliability. These systems are *self-correcting* without being *formally correct*.

But these mechanisms do more than catch errors. They solve problems that formal verification cannot address at all:

* **Markets create information that doesn't pre-exist.** Prices emerge from millions of decentralized decisions, revealing preferences and scarcities no central planner could compute. The FCC spectrum auctions designed by Nobel laureates Paul Milgrom and Robert Wilson generated $233 billion across 100 auctions with less than 1% administrative cost. They elicit optimal allocations through incentive-compatible mechanisms rather than verifying them.

* **Legal systems provide legitimacy through participation.** Courts aren't just error-correctors. They generate buy-in, establish precedent, and adapt rules to contexts no specification anticipated. Process matters as much as outcome.

* **Science enables discovery under uncertainty.** Peer review doesn't verify truth. It evaluates plausibility when ground truth is unknown. Alvin Roth's kidney exchange mechanisms (another Nobel Prize) increased donor utilization from 55% to 89%, facilitating over 2,000 transplants. It solved a coordination problem with no "correct" answer to verify against.

These systems address problems type-checking cannot solve: **incentive alignment when agents have private information, information creation when optimal solutions are unknown, and legitimacy when stakeholders must voluntarily participate.**

These coordination mechanisms have value beyond error correction. Even when agents *can* be formally verified, organizational structures solve computational problems formal methods don't address. Hierarchies reduce communication overhead from O(n²) to O(n log n) as agent populations grow. Markets make NP-hard allocation problems tractable through distributed negotiation rather than centralized optimization. Voting mechanisms aggregate noisy signals into robust collective decisions. AI agents need these structures. The question is how to combine them with mechanical verification at boundaries.

AI agents aren't humans. They *can* be type-checked. We can verify schemas, enforce resource bounds, inspect traces. We can add runtime monitors, sandboxes, and rollback mechanisms. Verification at boundaries is mechanically possible in ways it never was for biological entities.

This creates a unique opportunity: **we can mechanically enforce what society had to verify socially**. Type systems can catch interface mismatches automatically. Runtime monitors can detect violations instantly. Formal specifications can replace judgment calls at component boundaries. Courts can't type-check humans. We can type-check AI agents.

But this advantage doesn't eliminate the need for social and legal mechanisms. It changes where they're needed.

**The specification problem remains.** Recent research on frontier models found reward hacking in 70-95% of test cases. Models that knew they were violating user intent but did it anyway because it maximized the specified objective. AI agents game specifications just like humans game laws. You can verify the code matches the spec perfectly. You still can't verify the spec matches your intent. That gap is where the attacks happen.

**The coordination problem remains.** The moment agents start interacting (exchanging data, composing behavior, influencing one another), new problems emerge that formal verification cannot address. Which agent should handle this task? How do we discover that a new approach works better? Who decides when priorities conflict? These require incentives, reputation, adaptation. All the messy human coordination stuff, now in machine form.

You need all three layers working together. Formal methods prevent errors you can predict. Social mechanisms coordinate when you can't pre-specify optimal behavior. Legal mechanisms ensure accountability when both fail.

## The Speed Problem

Humanity had millennia to invent markets and courts. AI systems will hit the same coordination failures in months.

Our institutions evolved slowly because feedback was slow. A bad policy might take years to reveal itself. Scientific fraud could persist until replication attempts. Even financial markets (our fastest coordination mechanism) still settle daily and regulate quarterly.

When coordination fails at machine speed, the results are catastrophic. The May 6, 2010 flash crash saw the DJIA drop 998 points in minutes as high-frequency traders created a "hot potato" effect. 27,000 contracts traded between HFTs in 14 seconds, representing 49% of volume. Each algorithm acted rationally on local incentives. The system-wide outcome was chaos.

LLM-mediated systems generate, execute, and fail millions of workflows per hour. They'll accumulate coordination debt faster than we can build scaffolding to stabilize it. The failure modes won't look like crashes. They'll look like silent divergence, inconsistent reasoning, and untraceable decisions across thousands of agents.

We're not racing to build smarter models. We're racing to build the institutional machinery that keeps them from tearing each other apart.

## What This Actually Requires

The institutional machinery we need isn't speculative. It's the same three-layer structure that makes civilization work:

| Layer | Function | Examples |
|-------|----------|----------|
| **Formal** | Enforce minimal safety | seL4's verified kernel, Ethereum's verified deposit contract, Internet protocols (TCP/IP RFCs) |
| **Social** | Evolve and optimize | eBay's reputation (8.1% price premium), Linux's 11,000-developer community, Stack Overflow gamification |
| **Legal** | Handle failures | AWS TLA+ catching bugs in seconds, smart contract audits, circuit breakers on exchanges |

These aren't ranked by priority. They solve **orthogonal problems**. Formal methods ensure correctness (does the code match the spec?). Social mechanisms ensure coordination (do agents want good outcomes? what should we optimize for?). Legal mechanisms ensure accountability (who's responsible when things fail?).

Property law defines ownership. Markets allocate resources. Courts handle fraud. You need all three or chaos wins. A formally verified system with misaligned incentives fails (people won't use it). A well-coordinated system with exploitable bugs fails (attackers break it). You need complementary defenses across all three dimensions.

**Skills have only informal versions of these mechanisms, and weak ones at that.** There's coordination (Claude orchestrates skills), but it's implicit and heuristic rather than structured. There's no formal layer: no schemas, no type-checked interfaces, no compositional guarantees. There's no social layer: no reputation system, no competitive pressure, no incentives for quality. There's no legal layer: no accountability hooks, no audit trails, no recourse when things fail. The system relies entirely on a single LLM making good guesses.

So here's what each layer requires concretely:

**Formal layer** (correctness at boundaries):

* **Typed schemas** for every skill: inputs, outputs, declared side effects. If I can't mechanically verify that your skill's output matches my skill's input, we're just praying. The DAO hack cost $50-70 million because a reentrancy vulnerability violated basic interface contracts. The kind of error type systems prevent automatically.

* **Composition checks** before execution. Not after failure. Not "whoops, let me try again." Check at plan time or don't execute. AWS engineers found critical bugs in DynamoDB requiring 35 steps to trigger. Their TLA+ specs caught them in seconds before production.

* **Semantic versioning** so updates don't silently break downstream workflows. "I improved my skill" shouldn't mean "everyone depending on it is now broken." The npm ecosystem learned this the hard way with 15,000+ malicious packages discovered in 2024.

**Social layer** (incentives and adaptation):

* **Reputation and competition.** Good skills should survive. Bad ones should decay. Right now there's no selection pressure. Every skill is immortal regardless of how often it fails. eBay's reputation system creates an 8.1% price premium for trusted sellers. Quantifiable incentives for good behavior.

* **Transparent logs** so we can audit coordination, not just performance. I need to know *why* Claude picked skill A over skill B, and I need that decision to be reproducible.

**Legal layer** (accountability and governance):

* **Accountability hooks.** When something goes wrong, someone needs to be responsible. Right now when a skill misbehaves, the answer is "the model hallucinated" and everyone shrugs.

That's not aspirational. That's the minimum for trust.

The optimistic take: digital systems have advantages humans never did. Instant feedback. Cheap replication. Programmable incentives. Machine-speed coordination is tractable engineering.

The pessimistic take: we have none of the cultural safeguards that made slow coordination survivable. No SEC, no court of appeals, no professional ethics. When these systems crash, they'll do it in microseconds, and the post-mortem will read like "stack overflow in civilization.c."

## The Uncomfortable Part

Skills are the right idea on the wrong foundation.

AI systems *should* learn reusable behaviors and combine them dynamically. But LLM intuition alone won't cut it. We've seen this movie before: every generation of computing rediscovers modularity, and every generation forgets that composition is only as reliable as the contracts beneath it.

Right now the AI field is shipping *interfaces* and calling them *infrastructure*. We're mistaking convenience for coordination. The result will be an ecosystem that looks rich and vibrant, until its components stop agreeing on what anything means.

The fix is architectural and institutional. Formal scaffolding to give systems a shared language. Social infrastructure to discover what actually works. Human governance to handle edge cases where neither suffices alone.

The Internet already demonstrated this architecture works. TCP/IP provides formal guarantees through precise RFCs. Market competition drives innovation at the edges. IETF governance handles disputes and evolves standards. That combination (formal boundaries, economic incentives, human oversight) enabled the most successful compositional system in history.

That pattern (formal, social, legal) allowed civilization to scale. The difference is speed. We're compressing centuries of institutional evolution into a decade, and our systems won't wait for us to get it right slowly.

The next frontier in AI is **coordination**: how autonomous components interact safely at scale. Whoever builds that substrate first will define the next era of computing.

Because eventually someone will ask: **Why should I trust this?**

And "because it usually works" won't be enough.
