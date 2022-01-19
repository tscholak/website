---
title: Towards Neural Functional Program Evaluation
author: Torsten Scholak, Jonathan Pilault, Joey Velez-Ginorio
journal: "arXiv:2112.04630 [cs.CL]"
date: Jan 1, 2021
tldr: My first paper
image: code.jpg
tags: [stuff]
pdf: https://arxiv.org/pdf/2112.04630
code: https://github.com/ElementAI/neural-interpreters
---

This paper explores the capabilities of current transformer-based language models for program evaluation of simple functional programming languages. We introduce a new program generation mechanism that allows control over syntactic sugar for semantically equivalent programs. T5 experiments reveal that neural functional program evaluation performs surprisingly well, achieving high 90% exact program match scores for most in-distribution and out-of-distribution tests. Using pretrained T5 weights has significant advantages over random initialization. We present and evaluate on three datasets to study generalization abilities that are specific to functional programs based on: type, function composition, and reduction steps.