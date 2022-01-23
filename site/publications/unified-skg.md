---
title: "UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models"
author:
- Tianbao Xie
- Chen Henry Wu
- Peng Shi
- Ruiqi Zhong
- Torsten Scholak
- Michihiro Yasunaga
- Chien-Sheng Wu
- Ming Zhong
- Pengcheng Yin
- Sida I. Wang
- Victor Zhong
- Bailin Wang
- Chengzu Li
- Connor Boyle
- Ansong Ni
- Ziyu Yao
- Dragomir Radev
- Caiming Xiong
- Lingpeng Kong
- Rui Zhang
- Noah A. Smith
- Luke Zettlemoyer
- Tao Yu
journal: "arXiv:2201.05966 [cs.CL]"
date: Jan 16, 2022
tldr: "Let's unify all structured-knowledge grounded tasks into the same text-to-text framework!"
image: unified-skg.png
tags:
  items: [research]
link: 'https://arxiv.org/abs/2201.05966'
pdf: 'https://arxiv.org/pdf/2201.05966'
code: 'https://github.com/hkunlp/unifiedskg'
---

Structured knowledge grounding (SKG) leverages structured knowledge to complete user requests, such as semantic parsing over databases and question answering over knowledge bases. Since the inputs and outputs of SKG tasks are heterogeneous, they have been studied separately by different communities, which limits systematic and compatible research on SKG. In this paper, we overcome this limitation by proposing the SKG framework, which unifies 21 SKG tasks into a text-to-text format, aiming to promote systematic SKG research, instead of being exclusive to a single task, domain, or dataset. We use UnifiedSKG to benchmark T5 with different sizes and show that T5, with simple modifications when necessary, achieves state-of-the-art performance on almost all of the 21 tasks. We further demonstrate that multi-task prefix-tuning improves the performance on most tasks, largely improving the overall performance. UnifiedSKG also facilitates the investigation of zero-shot and few-shot learning, and we show that T0, GPT-3, and Codex struggle in zero-shot and few-shot learning for SKG. We also use UnifiedSKG to conduct a series of controlled experiments on structured knowledge encoding variants across SKG tasks. UnifiedSKG is easily extensible to more tasks, and it is open-sourced.