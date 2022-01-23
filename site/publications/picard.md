---
title: "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models"
author:
- Torsten Scholak
- Nathan Schucher
- Dzmitry Bahdanau
journal: "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing"
date: Nov 1, 2021
tldr: "Introducing PICARD - a simple and effective constrained beam search algorithm for any language model. PICARD helps with generating valid code, which is useful for program synthesis and semantic parsing. We achieve SoTA on both Spider and CoSQL."
image: picard.jpg
tags:
  items: [research, haskell]
link: 'https://aclanthology.org/2021.emnlp-main.779'
pdf: 'https://arxiv.org/pdf/2109.05093'
talk: 'https://youtu.be/kTpixsr-37w'
code: 'https://github.com/ElementAI/picard'
---

Large pre-trained language models for textual data have an unconstrained output space; at each decoding step, they can produce any of 10,000s of sub-word tokens. When fine-tuned to target constrained formal languages like SQL, these models often generate invalid code, rendering it unusable. We propose PICARD (code and trained models available at this https URL), a method for constraining auto-regressive decoders of language models through incremental parsing. PICARD helps to find valid output sequences by rejecting inadmissible tokens at each decoding step. On the challenging Spider and CoSQL text-to-SQL translation tasks, we show that PICARD transforms fine-tuned T5 models with passable performance into state-of-the-art solutions.