"""E2: does injected-position advantage travel through J-space?

See docs/preregistrations/E2_jspace_vector_families.md.

This is a subpackage on purpose. E1's run identity hashes src/mechinterp/*.py without descending into
folders (identity.code_hash), so a module added here leaves E1's code hash, and so its resume, unchanged.
E2 hashes this folder as well as its parent, in e2.config.e2_code_hash.
"""
