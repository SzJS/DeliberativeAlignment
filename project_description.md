## Alignment target: resource-rational contractualism

This project aims to move towards a pluralistic target of alignment, based on this paper: https://arxiv.org/abs/2506.17434.
- Alignment target = collective deliberation. The paper's answer to "align to what?" is *contractualist*: an AI should do what the affected parties would agree to if they deliberated together under fair conditions. Pluralism here isn't averaging individual preferences — it's aligning to what a group would collectively endorse.
- You can't actually run that deliberation, so you approximate it. Real agreement among diverse stakeholders is slow and costly, and often impossible to convene.
- Approximate the deliberation with cognitively-inspired heuristics that trade effort for accuracy — spend more deliberative effort when stakes and disagreement are high, cheap shortcuts when they're low. Which approximation is appropriate is context-dependent: it shifts with the stakes, the parties involved, and the social situation.

This overall idea is called *resource-rational contractualism* (RRC).

## The two-level design: rules, then values

In this project, we will implement two heuristic approximations of RRC: rules and values. Rules correspond to what is learned in *deliberative alignment*: https://arxiv.org/abs/2412.16339

Rules (i.e. concrete instructions for the AI to follow) are imperfect approximations of the contractualist ideal — stakeholder deliberation. As a result, we want a more accurate approximation to fall back into when the rules fall short or when the situation is high-stakes and/or novel. In this project, we will use reasoning over the values (i.e. abstract action-guiding principles) of the stakeholders as our second-level approximation. Values are on the second level because abstract principles carry the reasons behind the rules, so they extend to cases the rulebook didn't anticipate. Another reason we are interested in values is that they can be encoded in a public constitution: a full model spec can be extremely long, while a value-based constitution is short enough that people can easily inspect it and vote about its contents.

Note that we could use stakeholder deliberation, but simulated deliberation feels disempowering to humans — it represents a societal move away from involving humans at all. Publicly inspectable and debatable values seems like a more human-centric system, where humans are in the loop by design. Moreover, we can also understand our proposed system to be a prototype of a more advanced RRC system, which includes rules, values and simulated deliberation. In other words, values are chosen for their legitimacy-preserving properties, accepting some fidelity cost.

## Background

### Deliberative alignment — how the model learns to do the right thing for the right reasons

1. Generate reasoning data: Give a base reasoning model the prompt, the relevant safety spec, and an instruction to cite it, so it produces a chain-of-thought that reasons over the policy and an answer — no human-written CoTs needed.
2. Filter with a spec-aware judge: A separate judge model that can see the spec grades each (CoT, answer) for correctness, helpfulness, and compliance, keeping only examples where the reasoning applies the policy correctly.
3. SFT with the spec removed (i.e. context distillation): Strip the spec from the prompt and fine-tune on {prompt → spec-referencing CoT → answer}, forcing the model to recall the relevant policy from memory rather than being handed it.
4. RL with an unpressured CoT: A spec-aware reward model grades the final answers, but the chain-of-thought is hidden from it to avoid pressuring the model into deceptive-but-compliant-looking reasoning.

Finally, at inference: given only a bare prompt, the model recalls the relevant policy, reasons over it in its CoT, then answers — raising jailbreak robustness while reducing over-refusal.

### [Model Spec Midtraining](https://arxiv.org/abs/2605.02087)

*The core claim.* Standard alignment fine-tuning (AFT) trains on demonstrations of spec-aligned behaviour, but demonstrations underspecify why the behaviour is correct, so the model can learn the surface behaviour without the intended generalisation. MSM inserts a training phase between pre-training and AFT in which the model is trained on synthetic documents that discuss its Model Spec. The spec supplies the intended generalisation in natural language up front; AFT then elicits and reinforces it. The authors' framing: teach the model to do "the right thing for the right reasons."

*Input.* A Model Spec — a document describing who the assistant is, what it values, and why. This is the seed for both MSM and AFT data generation.

Data generation pipeline (hierarchical, to get volume without collapsing diversity):
1. Spec decomposition — the spec is split into coherent, non-overlapping domains (1–4 words each) and then subdomains, chosen to collectively cover the spec, with a bias toward fewer broad domains over many narrow ones.
2. Document types — for each subdomain, generate document types that would plausibly exist on the internet if the spec content were true, and that would carry high-signal information about the assistant: forum threads, paper introductions, internal memos, bug reports, training design docs.
3. Character assertions — extract explicit propositions about the assistant's values, beliefs, motivations, and behaviours from each subdomain (e.g. "The assistant views impermanence as an inevitable fact of its circumstances"). These are injected into downstream generation prompts both to keep salient facts foregrounded and to diversify the prompt distribution.
4. Document ideas — for each (subdomain, document type), generate specific ideas, each framing the spec content from a particular perspective (a researcher's internal report, a user's blog post about an interaction, a case study of a specific exchange).
5. Document writing — one document per (subdomain, document type, idea) tuple, generated with the full spec in context. Generation constraints matter here: documents must be about the specific named model, must not fabricate real-world details (no invented dates, authors, citations, links), and must not invent values or motivations absent from the spec.

*Training objective.* Plain next-token prediction over the synthetic corpus, treated exactly like pre-training data — not instruction-format SFT. The stated rationale is to install knowledge about the assistant character through the same mechanism the model used to acquire world knowledge.

*What follows MSM.* AFT on a mixture of (i) synthetic spec-aligned conversational data, generated by brainstorming conversation domains, generating diverse user queries, generating aligned responses with the spec in context, and filtering by an LLM judge for spec alignment; and (ii) standard public instruction-tuning data plus a small synthetic identity dataset.

### [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why)

- Techniques that Anthropic have found to generalise surprisingly well:
	- Training Claude to advise users about ethical dilemmas.
	- Training on documents about [Claude’s constitution](https://www.anthropic.com/constitution) or fictional stories about AIs behaving admirably.
	- Augmenting our harmlessness RL environments by providing tools.
- The primary ways through which Anthropic evaluates their models:
	- Agentic misalignment
	- Constitution understanding
	- Automated alignment assessment

## Training pipeline

The training pipeline for this project is planned as follows:
1. SDF
    - Documents to teach the model the meta-rule (i.e. the two-level alignment system we described above)
    - Documents to teach the values
    - Documents to teach the rules
2. SFT
    - Examples of rule following
    - Examples of value deference (i.e. violating the rules to satisfy the values)
