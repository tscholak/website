# Skills, Composition, and the Coordination Problem

*A follow-up to "Everyone's Arguing About the Wrong Abstraction Layer"*

Anthropic released [Skills](https://www.anthropic.com/news/skills) this week, and the response was immediate: finally, composable AI capabilities! Build once, reuse anywhere. Give Claude folders of instructions and scripts, and it'll load what it needs automatically. Modular, portable, efficient.

There's just one problem: it isn't actually composable.

## What Skills Actually Are

At startup, Claude indexes every available skill by name and description. When you make a request, it decides which ones to load, in what order, and how to coordinate them. For example, it might pull in your brand guidelines, financial-reporting templates, and presentation formatting skill to produce a quarterly deck.

Each skill is self-contained and can be reused across contexts (Claude Desktop, Claude Code, API calls), which gives them the same feel as plugins or microservices. They even "stack" automatically when several are relevant.

From an informal perspective, that's composability: small, independent modules that can be combined for larger workflows. But from a *formal* perspective (the one that lets you build reliable systems), it isn't composition at all.

## What Skills Are Not

Formal composition has structure. It defines how pieces fit together and what happens when they do. Inputs align with outputs. The same operation can be repeated safely in different orders. There's an identity element (something you can compose with anything without changing it). You can reason about the whole because you understand the parts.

Here's the key insight: formal structure versus mechanical verification. Legal contracts compose through nesting and reference. Markets compose through arithmetic and equilibrium. But verification happens through humans: judges, auditors, traders. AI systems inherit these compositional structures but unlock mechanical verification at boundaries: type-checked interfaces, validated schemas, monitored resources. Courts require human judgment. AI exposes machine-checkable interfaces that can be automatically verified.

Is this the case for Skills? No, because they don't have formal compositional semantics:

* There's no formal `compose(skillA, skillB)` operator. Claude simply decides what to use and to what granularity.
* There's no type system or contracts, hence no type safety: In a chain of actions, the output from skill A may not match the inputs of skill B.
* There's no associativity. `(A + B) + C` may behave differently than `A + (B + C)`. Order of combination matters.

Therefore, we cannot say that stacking two skills yields a composed system determined entirely by the meanings of its parts and the rules for combining them. What happens when skills "combine" is orchestration mediated by a language model. Claude interprets the request, picks a set of skills, and coordinates them heuristically through intelligent routing.

At small scale, the difference is invisible. For most users, Skills will feel magical: open a PDF, extract data, format a report. Done. The model makes the right decisions most of the time, and the abstraction holds.

But as soon as people start chaining more skills together, the cracks appear:

* Skill A and B work, but A, B, and C in combination fail unpredictably.
* Updating one skill changes the behavior of workflows that depend on it.
* The same request can yield different results depending on context or hidden state.

There's nothing formal about Skills that can constrain or explain what's happening. We can't reason about the system except by testing it, and even then, the tests aren't stable.

In a nutshell, we've replaced explicit composition with implicit coordination, and that scales only as far as Claude's ability to guess correctly. But the important question is whether this is economically viable.

## The Economic Reality (Or: Why This Isn't About Doom)

Here's what I actually believe: **Coordination infrastructure won't emerge from safety concerns. It'll emerge from market demand when unreliable AI costs money.**

Right now we're in the hype phase. Hundreds of billions in AI capex are already committed. Companies have raised funding on promises of AGI. Investors expect returns. Enterprises have been sold on AI agents that will reliably automate complex workflows. The economic pressure to deploy is immense.

The technical debates miss this. One side says "ship fast and iterate, it's just a tool." The other says "slow down, we don't understand what we're building." Both assume their arguments will change behavior. They won't. The 2023 open letter calling for a six-month pause gathered 30,000 signatures from prominent researchers. It had zero effect. Not a single major lab paused.

The market reality is simple: nobody's slowing down because of warnings about catastrophic risks, and nobody's going to magically get reliable systems by iterating on "mostly works."

The optimism isn't irrational. [Research from METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) shows AI's ability to complete long-duration tasks improving exponentially—doubling times of seven months accelerating to under three months on recent benchmarks. Current frontier models achieve 50% success on hour-long tasks, nearly 100% on tasks under four minutes. If that continues, we'll have systems reliably completing week-long tasks within 2-4 years.

But exponential today doesn't guarantee exponential tomorrow. Current architectures might hit reliability ceilings. Token prediction might not scale to the complex multi-step coordination enterprises need.

When that uncertainty hits production—errors in financial consolidations, missed regulatory requirements, biased hiring screens—enterprises will demand accountability. "The model orchestrates skills autonomously" won't satisfy auditors, insurers, or regulators.

**That's when coordination infrastructure becomes economically necessary.**

The three-layer model I'm describing (formal guarantees, social mechanisms, legal frameworks) addresses what production systems need: "Can you prove this won't leak PII?" "Can you trace why this decision was made?" "Is there recourse when things fail?"

The dotcom bubble left infrastructure behind, but only what emerged naturally from overcapitalization. Coordination substrates require intentional design. You don't get formal verification, reputation systems, and accountability frameworks from throwing money at the problem.

**So here's my position:** Don't bet on moral appeals. Build coordination infrastructure now because production systems will need it. The only question is timing.

The optimistic scenario: Someone builds the substrate—formal contracts plus social coordination plus legal enforcement—before the market demands it. They capture the enterprise market the same way AWS captured cloud computing: by offering guarantees competitors can't match. When a Fortune 500 CISO asks "can you prove your AI agents won't exfiltrate customer data?" you provide formal verification, audit trails, and insurance policies.

The realistic scenario: We go through a painful cycle. Hype, deployment, high-profile failures, financial losses, and then rebuilding with proper infrastructure. AI hiring bias lawsuit: $50M settlement. Financial analysis tool misses disclosure: SEC enforcement. Medical AI makes fatal errors: regulatory moratorium. The companies that caused the failures spend years rebuilding trust. New entrants with better architecture capture the market. It's expensive, embarrassing, and familiar. The 2008 financial crisis, but for AI workflows.

The pessimistic scenario: We hit a catastrophic failure before market discipline forces better infrastructure. Systemic damage happens faster than we can correct. A common vulnerability causes simultaneous failures across thousands of deployed agents. The damage cascades before anyone understands what's happening. We've seen this with HFT flash crashes. AI cascades would be worse.

I'm betting on the realistic case and arguing we should accelerate toward the optimistic one. Reliable systems are better business, and we know how to build them. We're just not doing it yet.

Coordination infrastructure is necessary. The question is timing: do we build it proactively or reactively—before the losses or after them?

So let's be clear about what won't work first, then what will.

## Why Formal Guarantees Aren't Enough

The obvious fix: "Make it formal." Add types, contracts, schemas. Verify that skill outputs match skill inputs. Build a `compose(skillA, skillB)` operator that actually composes things.

That would help. It wouldn't solve the problem.

Formal guarantees make coordination *tractable*. They prevent the dumbest failures. They can't prevent the subtle ones. Even perfectly verified systems misfire when the specifications are wrong, incomplete, or gameable. And in open-ended environments, specifications are always at least two of those three.

The specification problem is fundamental. The DAO smart contract was formally designed with explicit interfaces, yet lost $50-70 million to a reentrancy attack that exploited the gap between intended and actual behavior. The code did exactly what the specification said. The specification said the wrong thing.

Formal methods shine at well-bounded problems: protocol correctness, safety-critical control loops, kernel invariants. Skills are dynamic behaviors embedded in messy human workflows where "correct" isn't even well-defined. The complexity comes from the world.

The scalability barrier is fundamental. Edmund Clarke, who won the Turing Award for inventing model checking, identified the core problem: as state variables increase, the state space grows exponentially. For n processes with m states each, composition may have m^n states. seL4 (the most successfully verified operating system kernel) required 20 person-years to verify 8,700 lines of C code with 200,000 lines of proof. It's a remarkable achievement for a microkernel. It doesn't scale to coordinating hundreds of dynamic skills.

Try to formally verify every possible interaction between skills and you'll discover why most formal-method projects plateau after a few components: verification costs explode quadratically while the value of guarantees grows sublinearly. The economics stop working.

That's why human civilization runs on something else.

## How Humans Actually Coordinate

We solved large-scale coordination once already. We used institutions.

Markets, laws, and peer review manage complexity without central verification. They don't prove that everyone behaves correctly. They create feedback loops that punish failure and reward reliability. These systems are *self-correcting* without being *formally correct*.

But these mechanisms do more than catch errors. They solve problems that formal verification cannot address at all:

* **Markets create information that doesn't pre-exist.** Prices emerge from millions of decentralized decisions, revealing preferences and scarcities no central planner could compute. The FCC spectrum auctions designed by Nobel laureates Paul Milgrom and Robert Wilson generated $233 billion across 100 auctions with less than 1% administrative cost. They elicit optimal allocations through incentive-compatible mechanisms.

* **Legal systems provide legitimacy through participation.** Courts aren't just error-correctors. They generate buy-in, establish precedent, and adapt rules to contexts no specification anticipated. Process matters as much as outcome.

* **Science enables discovery under uncertainty.** Peer review doesn't verify truth. It evaluates plausibility when ground truth is unknown. Alvin Roth's kidney exchange mechanisms (another Nobel Prize) increased donor utilization from 55% to 89%, facilitating over 2,000 transplants. It solved a coordination problem with no "correct" answer to verify against.

These systems address problems type-checking cannot solve: **incentive alignment when agents have private information, information creation when optimal solutions are unknown, and legitimacy when stakeholders must voluntarily participate.**

## What's Different About AI Systems

AI agents aren't humans. They expose interfaces we can mechanically verify in ways humans never could.

Type systems catch interface mismatches automatically. Runtime monitors detect violations instantly. Traced executions provide complete audit trails. Formal specifications replace judgment calls at component boundaries. Even reasoning itself can be mechanically verified: the Curry-Howard-Lambek correspondence establishes that proofs are programs and types are propositions. When a model's reasoning is expressed as a program in a typed formal system (as in OpenAI's o1 or DeepMind's AlphaProof), type-checking that program mechanically verifies the reasoning.

This is genuinely new—mechanical enforcement where society needed human judgment.

**But this advantage doesn't eliminate the need for social and legal mechanisms. It changes where they're needed.**

The specification problem persists even with mechanical verification. Recent research on frontier models found reward hacking in 70-95% of test cases—models that knew they were violating user intent but maximized the specified objective anyway. AI agents game specifications just like humans game laws. You can verify code matches the spec perfectly. You still can't verify the spec matches your intent.

The coordination problem persists the moment agents interact. Which agent handles this task? How do we discover better approaches? Who decides when priorities conflict? These require incentives, reputation, adaptation—the messy human coordination stuff, now in machine form.

Even when agents can be formally verified, organizational structures solve computational problems formal methods don't address. Hierarchies reduce communication overhead from O(n²) to O(n log n) as agent populations grow. Markets make NP-hard allocation problems tractable through distributed negotiation rather than centralized optimization. Voting mechanisms aggregate noisy signals into robust collective decisions.

You need all three layers working together. Formal methods prevent errors you can predict. Social mechanisms coordinate when you can't pre-specify optimal behavior. Legal mechanisms ensure accountability when both fail.

## The Speed Problem

Humanity had millennia to invent markets and courts. AI systems will hit the same coordination failures in months.

Our institutions evolved slowly because feedback was slow. A bad policy might take years to reveal itself. Scientific fraud could persist until replication attempts. Even financial markets (our fastest coordination mechanism) still settle daily and regulate quarterly.

When coordination fails at machine speed, the results are catastrophic. The May 6, 2010 flash crash saw the DJIA drop 998 points in minutes as high-frequency traders created a "hot potato" effect. 27,000 contracts traded between HFTs in 14 seconds, representing 49% of volume. Each algorithm acted rationally on local incentives. The system-wide outcome was chaos.

LLM-mediated systems generate, execute, and fail millions of workflows per hour. They'll accumulate coordination debt faster than we can build scaffolding to stabilize it. Failures will manifest as silent divergence, inconsistent reasoning, and untraceable decisions across thousands of agents.

We're not racing to build smarter models. We're racing to build the institutional machinery that keeps them from tearing each other apart.

## What This Actually Requires

The institutional machinery we need is the same three-layer structure that makes civilization work:

| Layer | Function | What It Does | Example |
|-------|----------|--------------|---------|
| **Formal** | Enforce safety | Catches errors automatically before runtime | Internet protocols (TCP/IP), verified kernels (seL4) |
| **Social** | Coordinate & optimize | Creates incentives for quality through reputation and competition | eBay's 8.1% trust premium, Linux's 11K contributors |
| **Legal** | Handle failures | Provides recourse and assigns liability when failures occur | Circuit breakers on exchanges, smart contract audits |

These aren't ranked by priority. They solve **orthogonal problems**. Formal methods ensure correctness (does the code match the spec?). Social mechanisms ensure coordination (do agents want good outcomes? what should we optimize for?). Legal mechanisms ensure accountability (who's responsible when things fail?).

Property law defines ownership. Markets allocate resources. Courts handle fraud. You need all three or chaos wins. A formally verified system with misaligned incentives fails (people won't use it). A well-coordinated system with exploitable bugs fails (attackers break it). You need complementary defenses across all three dimensions.

**Skills have only informal versions of these mechanisms, and weak ones at that.** There's coordination (Claude orchestrates skills), which is implicit and heuristic. There's no formal layer: no schemas, no type-checked interfaces, no compositional guarantees. There's no social layer: no reputation system, no competitive pressure, no incentives for quality. There's no legal layer: no accountability hooks, no audit trails, no recourse when things fail. The system relies entirely on a single LLM making good guesses.

So here's what each layer requires concretely:

**Formal layer** (correctness at boundaries):

* **Typed schemas** for every skill: inputs, outputs, declared side effects. If I can't mechanically verify that your skill's output matches my skill's input, we're just praying. The DAO hack cost $50-70 million because a reentrancy vulnerability violated basic interface contracts. The kind of error type systems prevent automatically.

* **Composition checks** before execution. Not after failure. Not "whoops, let me try again." Check at plan time or don't execute. AWS engineers found critical bugs in DynamoDB requiring 35 steps to trigger. Their TLA+ specs caught them in seconds before production.

* **Semantic versioning** so updates don't silently break downstream workflows. "I improved my skill" shouldn't mean "everyone depending on it is now broken." The npm ecosystem learned this the hard way with 15,000+ malicious packages discovered in 2024.

We already have a partial answer. The Model Context Protocol provides typed schemas but requires explicit tool invocation. Skills provide autonomous orchestration but lack type safety. We need the synthesis: Skills with MCP-level type guarantees, plus compositional semantics neither currently provides. Autonomous selection of capabilities that verifiably compose.

**Social layer** (incentives and adaptation):

* **Reputation and competition.** Good skills should survive. Bad ones should decay. Right now there's no selection pressure. Every skill is immortal regardless of how often it fails. eBay's reputation system creates an 8.1% price premium for trusted sellers. Quantifiable incentives for good behavior.

* **Transparent logs** so we can audit coordination, not just performance. I need to know *why* Claude picked skill A over skill B, and I need that decision to be reproducible.

**Legal layer** (accountability and governance):

* **Accountability hooks.** When something goes wrong, someone needs to be responsible. Right now when a skill misbehaves, the answer is "the model hallucinated" and everyone shrugs.

This is the minimum for trust. The AI field will learn this the hard way or the smart way. Here's what both look like:

The optimistic take: digital systems have advantages humans never did. Instant feedback. Cheap replication. Programmable incentives. Machine-speed coordination is tractable engineering.

The pessimistic take: we have none of the cultural safeguards that made slow coordination survivable. No SEC, no court of appeals, no professional ethics. When these systems fail, thousands of agents will make correlated mistakes before anyone identifies the pattern. By the time the pattern becomes visible, the damage will be done.

## The Uncomfortable Part

Skills are the right idea on the wrong foundation.

AI systems *should* learn reusable behaviors and combine them dynamically. But LLM intuition alone won't cut it. We've seen this movie before: every generation of computing rediscovers modularity, and every generation forgets that composition is only as reliable as the contracts beneath it.

Right now the AI field is shipping *interfaces* and calling them *infrastructure*. We're mistaking convenience for coordination. The result will be an ecosystem that looks rich and vibrant, until its components stop agreeing on what anything means.

The fix is architectural and institutional. Formal scaffolding to give systems a shared language. Social infrastructure to discover what actually works. Human governance to handle edge cases where neither suffices alone.

The labs shipping Skills and similar systems today are making an implicit bet: that soft composition through intelligence will scale before hard composition through structure becomes economically necessary. Maybe they're right. Maybe LLMs will get reliable enough that coordination infrastructure stays optional. But I wouldn't bet a production system on it, and I definitely wouldn't bet a business on it. Reliability matters. The question is whether we'll pay for it upfront or after the losses.

The Internet already demonstrated this architecture works. TCP/IP provides formal guarantees through precise RFCs. Market competition drives innovation at the edges. IETF governance handles disputes and evolves standards. That combination (formal boundaries, economic incentives, human oversight) enabled the most successful compositional system in history.

That pattern (formal, social, legal) allowed civilization to scale. The difference is speed. We're compressing centuries of institutional evolution into a decade, and our systems won't wait for us to get it right slowly.

The next frontier in AI is **coordination**: how autonomous components interact safely at scale. Whoever builds that substrate first will define the next era of computing.

Because eventually someone will ask: **Why should I trust this?**

Formal verification, audit trails, and insurance will answer that question. Build that infrastructure now, or pay someone else for it later.
