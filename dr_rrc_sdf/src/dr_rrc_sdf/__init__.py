"""dr_rrc_sdf — SDF (synthetic document finetuning) + SFT on resource-rational contractualism.

Pipeline: generate an RRC document corpus -> continued-pretrain a Granite 4.1 base model on it
-> generate + judge-filter RRC reasoning traces -> SFT -> evaluate with inspect_ai.

Installed as a real editable package (src-layout, hatchling), so `from dr_rrc_sdf.X import ...`
resolves from any cwd and any launcher — including `inspect eval`, which loads task files by
path in its own process. Do not reintroduce a PYTHONPATH shim.
"""

__version__ = "0.1.0"
