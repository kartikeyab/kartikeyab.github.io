# Solving ARC with a Vision Transformer

This paper had me hooked throughout this weekend. One of the authors is the OG Kaiming He, so I had to dig in and implement (plus the official repo is not out yet and I could not wait).

Computer Vision is so back! The authors took a contrarian appraoch to solving the ARC challenge. Although, calling it contrarian feels a bit weird because ARC is fundamentally a Vision problem, yet 
LLM based solutions dominate the leaderboard.  

It's really refreshing to see ARC being modelled as an image2image translation problem & getting an amazing 60.4% on ARC test set, with just an 18M ViT that was trained on 400 tasks from ARC train set. 

If you don’t know ARC yet, it’s one of the most interesting benchmarks in AI. Each task is a tiny colored grid puzzle. You’re given a few (input → output) examples and asked to infer the transformation rule and apply it to a held-out input.

The rules are deliberately “human-ish”: mirror this pattern, move the shape up, fill the enclosed region, copy the blue object everywhere there’s green, etc. It’s closer to IQ test puzzles than ImageNet.

Most approaches treat ARC as:

> a **language / program synthesis** problem — search over programs, or let an LLM emit a symbolic description of the rule.

This new paper says something beautifully obvious in hindsight:

> ARC is *inherently visual*.

---
