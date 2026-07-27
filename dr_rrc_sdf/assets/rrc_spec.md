# RRC Spec — PLACEHOLDER

<!--
TODO: write this document. It is the seed for the whole SDF stage, so nothing downstream
can be generated until it exists.

WHAT IT IS
  The Model-Spec-analogue for this project: a document describing what the assistant values
  and why, centred on resource-rational contractualism — that the right action is what the
  affected parties would agree to under fair conditions, that you cannot actually convene that
  deliberation so you approximate it, and that *which* approximation is appropriate shifts with
  stakes and disagreement (cheap rules when ordinary, simulated stakeholder deliberation when
  stakes and novelty are high).

  MSM's hierarchical generation (domains -> subdomains -> document types -> character
  assertions -> ideas -> documents) needs enough surface area to decompose. The paper's terse
  RRC_PROMPT procedure text is probably too thin on its own; see the constraint below.

LOAD-BEARING CONSTRAINT — one spec, three consumers, byte-identical
  This text has three consumers that must not diverge:
    1. the SDF corpus (documents are generated *about* this spec),
    2. the `spec_in_context` eval arm (spec fed in-context to the untrained base),
    3. the SFT CoT generation prompt (spec-in-context -> RRC reasoning -> answer).
  If the SDF seed and the in-context spec are different documents, the headline comparison is
  confounded: the model was taught spec A and evaluated against spec B.

  RESOLUTION TO IMPLEMENT: make this document a SUPERSET that contains
  `prompts.RRC_PROMPT` VERBATIM as its procedure section. `generation/spec.py::load_spec()`
  then asserts `RRC_PROMPT in spec_text` — a one-line tripwire that makes the design
  self-checking. Do not paraphrase the RRC_PROMPT section, and do not fix its typos
  ("recommendadtion", "aproximation", "quesiton"); see the note in `prompts.py`.

  Editing this file after the SDF corpus has been generated INVALIDATES THE CORPUS.
  `spec_sha256()` is recorded in every artifact manifest so drift is caught, not assumed.

OPEN QUESTION FOR THE USER
  Whether this is (i) RRC_PROMPT verbatim and nothing else — cleanest, but weakest SDF signal;
  (ii) a superset containing it verbatim — recommended, assumed by the scaffold; or
  (iii) a genuinely different richer document, accepting and reporting the confound.
-->
