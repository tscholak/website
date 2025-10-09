---
title: "Everyone's Arguing About the Wrong Abstraction Layer"
date: Oct 8, 2025
teaser: >
  OpenAI [shipped AgentKit this week](https://openai.com/index/introducing-agentkit/), a platform with a visual workflow editor, versioned agents, governed connectors, and evals infrastructure. The internet immediately split into two camps: people dunking on it ("this isn't AGI!") and people defending visual editors as necessary for non-technical users.

  Both camps are arguing about the wrong thing.
tags:
  items: [ai, agents]
image: agentkit.png
---

Harrison Chase from LangChain wrote [a thoughtful piece called "Not Another Workflow Builder,"](https://blog.langchain.com/not-another-workflow-builder/) arguing that visual workflow builders are getting squeezed from both directions: simple agents (prompt & tools) are handling low-complexity tasks, while code-based workflows (like LangGraph) win at high complexity. His conclusion: the middle ground is dying, focus on better no-code agents and better code generation instead.

It's a clean thesis. Unfortunately, it's also missing the forest for the trees.

## The Abstraction Confusion

Here's what's actually happening: we are watching three different communities argue past each other because they are each solving for different constraints.

* The "simple agents win" crowd sees frontier models getting better and concludes that explicit orchestration becomes unnecessary. Just throw GPT-5 at the problem with some tools and let emergence handle the rest. They are right that many workflows are just prompt engineering in disguise. They look at complex systems and see overengineering.

* The "visual workflow necessary" crowd needs to ship products with non-technical stakeholders, wants observability baked in, and needs governance that works at the organizational level. They are right that code doesn't easily surface what's running and why. They look at code-first approaches and see gatekeeping.

* The "visual workflow bad" crowd wants composability, types, and the ability to reason about systems explicitly and even formally. They are right that dragging boxes around in a canvas doesn't give us algebraic properties or type safety. They look at AgentKit's visual editor and see toy-like systems that won't scale.

All three are correct about what they're optimizing for. All three are wrong about what the actual problem is.

## What AgentKit Actually Represents

Strip away the visual editor for a moment. What did OpenAI actually release?

They released a platform for building, deploying, and iterating on multi-agent systems with the following key primitives: versioned workflows, a governed connector registry (where admins manage how data and tools connect), guardrails, evaluation infrastructure, and reinforcement fine-tuning (RFT) integration. The visual canvas is actually the least interesting part of the release.

OpenAI understands that the real problem isn't how users specify what they want (visual vs code vs natural language), but how we build systems that compose, verify, observe, and improve.

Harrison's squeeze thesis argues that visual workflow builders are losing from both ends. But this assumes the primary value of visual builders is specification, i.e. helping people define workflows. That's not actually their killer feature. When done right, visual builders are primarily about operational visibility. They make implicit logic explicit, they create artifacts that stakeholders can review, they enable governance at the right granularity, and they provide observability by default. These are organizational problems, not individual productivity problems.

I don't see AgentKit's pitch as "drag boxes instead of writing code." I see it as "version your agents, govern your data connectors, evaluate systematically, and iterate with RFT." The visual editor is merely how OpenAI is packaging that infrastructure for adoption.

## The Compositional Substrate Nobody's Building

Here's my contrarian take: both visual builders and natural language prompts are failing at scale for the same reason, and it has nothing to do with their interface paradigm.

I had a long conversation with [Paul Chiusano](https://pchiusano.github.io) from Unison Computing about this last week. Paul is a functional programming language designer and a big advocate for types and formal methods. He says:

> "I am pretty uninterested in all these WYSIWYG type workflow engines, even with some AI sauce sprinkled on top. I want types and higher-order functions and all the niceties of a modern programming language."

Is he a snob for wanting that? No. He's pointing at tools for managing complexity with a proven track record. Type systems catch errors before runtime. Higher-order abstractions let programmers build big things from small things with clear interfaces. Referential transparency lets people reason locally about global properties.

Neither visual workflow builders nor natural language prompts provide these properties. They both scale terribly once a certain complexity threshold is crossed, for the same underlying reason: they don't provide compositional guarantees.

A visual workflow with 50 nodes and complex branching is unmaintainable not because it's visual, but because the edges between nodes carry no semantic contracts. You can't compose workflows safely because there's no type system preventing you from wiring incompatible things together. You can't refactor with confidence because there's no way to verify that your changes preserve behavior.

Similarly, a complex prompt-based agent scales poorly not because natural language is inherently bad, but because prompts have no formal composition semantics. You can't build a library of reusable "prompt modules" with clear interfaces. You can't verify that chaining two prompts preserves invariants. You can't reason about what happens when an agent calls a tool that calls another agent.

The problem is not the interface. It's that neither provides compositional abstractions with formal semantics.

## What Code Actually Gives You

Harrison is right that code wins at the high-complexity end. But it's worth being precise about why.

Code doesn't win because developers are special or because typing text is inherently better than dragging boxes. It wins because it's the only common representation that provides formal semantics we can reason about, composability through functions and types, verification via type checkers and tests, versioning and diffing as first-class operations, and -- critically -- an artifact that is the ground truth of execution.

This last point is often missed: even if AI always generates perfect code from natural language descriptions, you still need the code because the code is the specification of what you actually built and run. Natural language is for intent, code is for commitment. It's like the difference between a contract and a conversation. You need both. They are complementary.

## Stop Arguing About Interfaces

We need to stop conflating the construction interface with the operational model. The answer isn't choosing between visual, code, or natural language. We need all three serving different purposes:

**Natural language** for expressing human intent and constraints. **Formal contracts** (types, schemas, invariants) that specify interfaces and composition rules. **Code** that implements logic within those contracts, whether written by humans or generated by AI. **Observability infrastructure** (traces, evals, guardrails) that shows what actually happened and feeds back into intent and contracts.

Visual builders try to be all of these simultaneously. That's why they break. Natural language prompts pretend formal contracts don't exist. That's why they don't compose. Code-first approaches often skip observability or treat intent as separate from the system. That's why they struggle with alignment.

AgentKit is interesting because it's actually trying to build multiple layers: typed connectors, visual composition as an interface to implementation, execution with guardrails, and evals infrastructure. Whether their specific implementation works is TBD, but the architecture is pointing in the right direction.

## What Happens If We Get This Wrong

Here's what's at stake: we're about to build multi-agent systems that make consequential decisions at scale. Process invoices. Route customer support. Manage infrastructure. Screen resumes. Handle medical triage.

If we build these systems without compositional guarantees, we get brittle towers of duct tape that work until they catastrophically don't. If we build them without observability, we get black boxes that fail in inexplicable ways. If we build them without formal contracts, we can't verify they preserve the properties we care about. And if we skip the natural language layer, we've built something only five people in the world can understand and maintain.

The teams currently arguing about visual vs code vs prompts are missing the point. The only important question is: can we build systems that are simultaneously verifiable (formal contracts), understandable (operational visibility), and accessible (natural language intent)?

I believe the answer is yes, but only if we stop arguing about interfaces and start building the compositional substrate underneath. We need strong types and formal semantics, visual projection for operations and governance, natural language for intent and constraints, and continuous eval for alignment.

The team that figures out how to integrate all of these coherently will win.

I'm watching closely.
