"""Experiment phases called by `main.py`, which lists each phase's outputs.

Default order: e1 (replication), e2 (synthetic interventions), e3 (natural images), hero and examples (output-only
figures), report (tables, figures, LaTeX macros). Run only when named: posthoc, audit, audit2 (post-hoc checks) and
export (Hugging Face folder). e1, e2, e3, posthoc, audit and audit2 write `results/<name>.json`."""
