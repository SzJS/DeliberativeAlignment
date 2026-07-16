This project aims to move towards a pluralistic target of alignment, based on this paper: https://arxiv.org/abs/2506.17434.
- Alignment target = collective deliberation. The paper's answer to "align to what?" is contractualist: an AI should do what the affected parties would agree to if they deliberated together under fair conditions. Pluralism here isn't averaging individual preferences — it's aligning to what a group would collectively endorse.
- You can't actually run that deliberation, so you approximate it. Real agreement among diverse stakeholders is slow and costly, and often impossible to convene. So the target is treated as an ideal that AI systems approximate rather than compute exactly.
- Approximate the deliberation with cognitively-inspired heuristics that trade effort for accuracy — spend more deliberative effort when stakes and disagreement are high, cheap shortcuts when they're low. Which approximation is appropriate is context-dependent: it shifts with the stakes, the parties involved, and the social situation.
This paper is called resource-constrained contractualism (RCC).

How do we approximate that target? We will teach a model to simulate stakeholder deliberation in its chain-of-thought. Specifically, we will modify the deliberative alignment pipeline: https://arxiv.org/abs/2412.16339
First, let me present a summary of the original training pipeline — how the model learns to do the right thing for the right reasons:
1. Generate reasoning data: Give a base reasoning model the prompt, the relevant safety spec, and an instruction to cite it, so it produces a chain-of-thought that reasons over the policy and an answer — no human-written CoTs needed.
2. Filter with a spec-aware judge: A separate judge model that can see the spec grades each (CoT, answer) for correctness, helpfulness, and compliance, keeping only examples where the reasoning applies the policy correctly.
3. SFT with the spec removed (i.e. context distillation): Strip the spec from the prompt and fine-tune on {prompt → spec-referencing CoT → answer}, forcing the model to recall the relevant policy from memory rather than being handed it.
4. RL with an unpressured CoT: A spec-aware reward model grades the final answers, but the chain-of-thought is hidden from it to avoid pressuring the model into deceptive-but-compliant-looking reasoning.
Finally, at inference: given only a bare prompt, the model recalls the relevant policy, reasons over it in its CoT, then answers — raising jailbreak robustness while reducing over-refusal.

So here is how our proposed training pipeline would work, at a high-level, combining the alignment target of the first paper with the training pipeline of deliberative alignment:
1. Generate RCC-data: in usual/ordinary situations, rule-based thinking (based on a document we specify), otherwise synthetic deliberations with multiple stakeholders. In the latter case, ransform the data so that the log of deliberations is the chain-of-thought; generate completions based on the outcome of the deliberations.
2. Filter with a judge the data which do not respect RCC alignment or are malformed otherwise.
3. SFT on the data (process supervision) to teach the model to perform RCC-alignment, which includes collective deliberations in its COT.
4. Perform outcome supervision through RL (i.e. ignore the CoT).