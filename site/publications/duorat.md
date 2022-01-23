---
title: "DuoRAT: Towards Simpler Text-to-SQL Models"
author:
- Torsten Scholak
- Raymond Li
- Dzmitry Bahdanau
- Harm de Vries
- Chris Pal
journal: "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies"
date: Jun 1, 2021
tldr: "It's like RAT-SQL, but simpler and faster."
image: duorat.png
tags:
  items: [research]
link: 'https://aclanthology.org/2021.naacl-main.103/'
pdf: 'https://arxiv.org/pdf/2010.11119.pdf'
code: 'https://github.com/ElementAI/duorat'
---

Recent neural text-to-SQL models can effectively translate natural language questions to corresponding SQL queries on unseen databases. Working mostly on the Spider dataset, researchers have proposed increasingly sophisticated solutions to the problem. Contrary to this trend, in this paper, we focus on simplifications. We begin by building DuoRAT, a re-implementation of the state-of-the-art RAT-SQL model that unlike RAT-SQL is using only relation-aware or vanilla transformers as the building blocks. We perform several ablation experiments using DuoRAT as the baseline model. Our experiments confirm the usefulness of some techniques and point out the redundancy of others, including structural SQL features and features that link the question with the schema.